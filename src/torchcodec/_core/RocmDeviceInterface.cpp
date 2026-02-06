// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <hip/hip_runtime.h>
#include <torch/types.h>
#include <mutex>

#include "RocmDeviceInterface.h"
#include "DeviceInterface.h"
#include "FFMPEGCommon.h"
#include "RocDecCache.h"
#include "RocmCommon.h"

extern "C" {
#include <libavutil/pixdesc.h>
}

namespace facebook::torchcodec {

namespace {

// Register ROCm device interface with the default "ffmpeg" variant.
// Note: ROCm uses torch::kCUDA device type (PyTorch uses "cuda" for both NVIDIA and AMD).
// This doesn't conflict with CUDA's default variant because ENABLE_CUDA and ENABLE_ROCM
// are mutually exclusive build flags - only one backend is compiled at a time.
static bool g_rocm = registerDeviceInterface(
    DeviceInterfaceKey(torch::kCUDA),  // Uses default variant "ffmpeg"
    [](const torch::Device& device) {
      return new RocmDeviceInterface(device);
    });

// C callbacks for rocDecode parser
static int handleVideoSequenceCallback(
    void* pUserData,
    RocdecVideoFormat* videoFormat) {
  auto decoder = static_cast<RocmDeviceInterface*>(pUserData);
  return decoder->handleVideoSequence(videoFormat);
}

static int handlePictureDecodeCallback(
    void* pUserData,
    RocdecPicParams* picParams) {
  auto decoder = static_cast<RocmDeviceInterface*>(pUserData);
  return decoder->handlePictureDecode(picParams);
}

static int handlePictureDisplayCallback(
    void* pUserData,
    RocdecParserDispInfo* dispInfo) {
  auto decoder = static_cast<RocmDeviceInterface*>(pUserData);
  return decoder->handlePictureDisplay(dispInfo);
}

static UniqueRocDecDecoder createDecoder(RocdecVideoFormat* videoFormat) {
  // Decoder creation parameters
  RocDecoderCreateInfo decoderParams = {};
  decoderParams.bit_depth_minus_8 = videoFormat->bit_depth_luma_minus8;
  decoderParams.chroma_format = videoFormat->chroma_format;
  
  // Choose output format based on bit depth
  // For 10-bit videos, use P016 (16-bit semi-planar)
  // For 8-bit videos, use NV12 (8-bit semi-planar)
  if (videoFormat->bit_depth_luma_minus8 > 0) {
    // 10-bit or higher: use P016 format
    decoderParams.output_format = rocDecVideoSurfaceFormat_P016;
  } else {
    // 8-bit: use NV12 format
    decoderParams.output_format = rocDecVideoSurfaceFormat_NV12;
  }
  decoderParams.codec_type = videoFormat->codec;
  decoderParams.height = videoFormat->coded_height;
  decoderParams.width = videoFormat->coded_width;
  decoderParams.max_height = videoFormat->coded_height;
  decoderParams.max_width = videoFormat->coded_width;
  decoderParams.target_height =
      videoFormat->display_area.bottom - videoFormat->display_area.top;
  decoderParams.target_width =
      videoFormat->display_area.right - videoFormat->display_area.left;
  decoderParams.num_decode_surfaces = videoFormat->min_num_decode_surfaces;
  decoderParams.num_output_surfaces = 1;
  decoderParams.display_rect.left = videoFormat->display_area.left;
  decoderParams.display_rect.right = videoFormat->display_area.right;
  decoderParams.display_rect.top = videoFormat->display_area.top;
  decoderParams.display_rect.bottom = videoFormat->display_area.bottom;

  rocDecDecoderHandle* decoder = new rocDecDecoderHandle();
  
  rocDecStatus result = rocDecCreateDecoder(decoder, &decoderParams);
  
  TORCH_CHECK(
      result == ROCDEC_SUCCESS, 
      "Failed to create rocDecode decoder. Status: ", result,
      ". This may indicate that rocDecode is not properly installed or ",
      "the video format is not supported by your GPU's VCN hardware decoder.");
  return UniqueRocDecDecoder(decoder, RocDecDecoderDeleter{});
}

std::optional<rocDecVideoCodec> validateCodecSupport(AVCodecID codecId) {
  switch (codecId) {
    case AV_CODEC_ID_H264:
      return rocDecVideoCodec_AVC;
    case AV_CODEC_ID_HEVC:
      return rocDecVideoCodec_HEVC;
    case AV_CODEC_ID_AV1:
      return rocDecVideoCodec_AV1;
    case AV_CODEC_ID_VP9:
      return rocDecVideoCodec_VP9;
    default:
      return std::nullopt;
  }
}

std::optional<rocDecVideoChromaFormat> validateChromaSupport(
    const AVPixFmtDescriptor* desc) {
  TORCH_CHECK(desc != nullptr, "desc can't be null");

  if (desc->nb_components == 1) {
    return rocDecVideoChromaFormat_Monochrome;
  } else if (desc->nb_components >= 3 && !(desc->flags & AV_PIX_FMT_FLAG_RGB)) {
    if (desc->log2_chroma_w == 0 && desc->log2_chroma_h == 0) {
      return rocDecVideoChromaFormat_444;
    } else if (desc->log2_chroma_w == 1 && desc->log2_chroma_h == 1) {
      return rocDecVideoChromaFormat_420;
    } else if (desc->log2_chroma_w == 1 && desc->log2_chroma_h == 0) {
      return rocDecVideoChromaFormat_422;
    }
  }

  return std::nullopt;
}

bool nativeRocDecodeSupport(const SharedAVCodecContext& codecContext) {
  auto codecType = validateCodecSupport(codecContext->codec_id);
  if (!codecType.has_value()) {
    return false;
  }

  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(codecContext->pix_fmt);
  if (!desc) {
    return false;
  }

  auto chromaFormat = validateChromaSupport(desc);
  if (!chromaFormat.has_value()) {
    return false;
  }

  RocdecDecodeCaps caps = {};
  caps.codec_type = codecType.value();
  caps.chroma_format = chromaFormat.value();
  caps.bit_depth_minus_8 = desc->comp[0].depth - 8;

  rocDecStatus result = rocDecGetDecoderCaps(&caps);
  if (result != ROCDEC_SUCCESS) {
    return false;
  }

  if (!caps.is_supported) {
    return false;
  }

  auto coded_width = static_cast<unsigned int>(codecContext->coded_width);
  auto coded_height = static_cast<unsigned int>(codecContext->coded_height);
  if (coded_width < caps.min_width || coded_height < caps.min_height ||
      coded_width > caps.max_width || coded_height > caps.max_height) {
    return false;
  }

  // Check if NV12 (8-bit) or P016 (10-bit) output is supported
  bool supportsNV12Output =
      (caps.output_format_mask >> rocDecVideoSurfaceFormat_NV12) & 1;
  bool supportsP016Output =
      (caps.output_format_mask >> rocDecVideoSurfaceFormat_P016) & 1;
  
  // For 10-bit videos, we need P016 support
  if (desc->comp[0].depth > 8 && !supportsP016Output) {
    return false;
  }
  // For 8-bit videos, we need NV12 support
  if (desc->comp[0].depth == 8 && !supportsNV12Output) {
    return false;
  }

  return true;
}

} // namespace

RocmDeviceInterface::RocmDeviceInterface(const torch::Device& device)
    : DeviceInterface(device) {
  TORCH_CHECK(g_rocm, "RocmDeviceInterface was not registered!");
  TORCH_CHECK(
      device_.type() == torch::kCUDA, "Unsupported device: ", device_.str());

  // Get the actual device index (handles -1 case by querying current device)
  int deviceIndex = getDeviceIndex(device_);
  
  // Update device_ to have the explicit index
  device_ = torch::Device(device_.type(), deviceIndex);

  // Set HIP device before initializing PyTorch context
  hipError_t err = hipSetDevice(deviceIndex);
  TORCH_CHECK(
      err == hipSuccess,
      "Failed to set HIP device ",
      deviceIndex,
      ": ",
      hipGetErrorString(err));

  // Create a persistent HIP stream for async memory operations
  err = hipStreamCreate(&copyStream_);
  TORCH_CHECK(
      err == hipSuccess,
      "Failed to create HIP stream: ",
      hipGetErrorString(err));

  initializeRocmContextWithPytorch(device_);
  
  // Check if rocDecode library is available first
  rocDecodeAvailable_ = loadRocDecodeLibrary();
  
  // Note: RPP context will be initialized lazily when needed for color conversion
  // This avoids potential issues if RPP is not properly installed
}

RocmDeviceInterface::~RocmDeviceInterface() {
  try {
    if (decoder_) {
      flush();
      releasePreviousFrame();
      
      // Set the device before accessing cache to ensure HIP context is valid
      hipSetDevice(device_.index());
      RocDecCache::getCache(device_).returnDecoder(
          &videoFormat_, std::move(decoder_));
    }

    if (videoParser_) {
      rocDecDestroyVideoParser(videoParser_);
      videoParser_ = nullptr;
    }

    // Only return RPP context if it was initialized
    if (rppCtx_) {
      hipSetDevice(device_.index());
      returnRppStreamContextToCache(device_, std::move(rppCtx_));
    }

    // Destroy the HIP stream
    if (copyStream_) {
      hipStreamDestroy(copyStream_);
      copyStream_ = nullptr;
    }
  } catch (const std::exception& e) {
    // Don't rethrow from destructor
  }
}

void RocmDeviceInterface::initialize(
    const AVStream* avStream,
    const UniqueDecodingAVFormatContext& avFormatCtx,
    [[maybe_unused]] const SharedAVCodecContext& codecContext) {
  // Store codec context for fallback color range info
  codecContext_ = codecContext;
  
  if (!rocDecodeAvailable_ || !nativeRocDecodeSupport(codecContext)) {
    cpuFallback_ = createDeviceInterface(torch::kCPU);
    TORCH_CHECK(
        cpuFallback_ != nullptr, "Failed to create CPU device interface");
    cpuFallback_->initialize(avStream, avFormatCtx, codecContext);
    cpuFallback_->initializeVideo(
        VideoStreamOptions(), {}, /*resizedOutputDims=*/std::nullopt);
    return;
  }

  TORCH_CHECK(avStream != nullptr, "AVStream cannot be null");
  timeBase_ = avStream->time_base;
  frameRateAvgFromFFmpeg_ = avStream->r_frame_rate;

  const AVCodecParameters* codecPar = avStream->codecpar;
  TORCH_CHECK(codecPar != nullptr, "CodecParameters cannot be null");

  initializeBSF(codecPar, avFormatCtx);

  // Create parser
  RocdecParserParams parserParams = {};
  auto codecType = validateCodecSupport(codecPar->codec_id);
  TORCH_CHECK(
      codecType.has_value(),
      "This should never happen, we should be using the CPU fallback by now.");
  parserParams.codec_type = codecType.value();
  parserParams.max_num_decode_surfaces = 8;
  parserParams.max_display_delay = 0;
  parserParams.user_data = this;
  parserParams.pfn_sequence_callback = handleVideoSequenceCallback;
  parserParams.pfn_decode_picture = handlePictureDecodeCallback;
  parserParams.pfn_display_picture = handlePictureDisplayCallback;

  rocDecStatus result = rocDecCreateVideoParser(&videoParser_, &parserParams);
  
  TORCH_CHECK(
      result == ROCDEC_SUCCESS, 
      "Failed to create rocDecode video parser. Status: ", result,
      ". This may indicate that rocDecode is not properly installed.");
}

void RocmDeviceInterface::initializeBSF(
    const AVCodecParameters* codecPar,
    const UniqueDecodingAVFormatContext& avFormatCtx) {
  TORCH_CHECK(codecPar != nullptr, "codecPar cannot be null");
  TORCH_CHECK(avFormatCtx != nullptr, "AVFormatContext cannot be null");
  TORCH_CHECK(
      avFormatCtx->iformat != nullptr,
      "AVFormatContext->iformat cannot be null");
  std::string filterName;

  switch (codecPar->codec_id) {
    case AV_CODEC_ID_H264: {
      const std::string formatName = avFormatCtx->iformat->long_name
          ? avFormatCtx->iformat->long_name
          : "";

      if (formatName == "QuickTime / MOV" ||
          formatName == "FLV (Flash Video)" ||
          formatName == "Matroska / WebM" || formatName == "raw H.264 video") {
        filterName = "h264_mp4toannexb";
      }
      break;
    }

    case AV_CODEC_ID_HEVC: {
      const std::string formatName = avFormatCtx->iformat->long_name
          ? avFormatCtx->iformat->long_name
          : "";

      if (formatName == "QuickTime / MOV" ||
          formatName == "FLV (Flash Video)" ||
          formatName == "Matroska / WebM" || formatName == "raw HEVC video") {
        filterName = "hevc_mp4toannexb";
      }
      break;
    }

    default:
      break;
  }

  if (filterName.empty()) {
    return;
  }

  const AVBitStreamFilter* avBSF = av_bsf_get_by_name(filterName.c_str());
  TORCH_CHECK(
      avBSF != nullptr, "Failed to find bitstream filter: ", filterName);

  AVBSFContext* avBSFContext = nullptr;
  int retVal = av_bsf_alloc(avBSF, &avBSFContext);
  TORCH_CHECK(
      retVal >= AVSUCCESS,
      "Failed to allocate bitstream filter: ",
      getFFMPEGErrorStringFromErrorCode(retVal));

  bitstreamFilter_.reset(avBSFContext);

  retVal = avcodec_parameters_copy(bitstreamFilter_->par_in, codecPar);
  TORCH_CHECK(
      retVal >= AVSUCCESS,
      "Failed to copy codec parameters: ",
      getFFMPEGErrorStringFromErrorCode(retVal));

  retVal = av_bsf_init(bitstreamFilter_.get());
  TORCH_CHECK(
      retVal == AVSUCCESS,
      "Failed to initialize bitstream filter: ",
      getFFMPEGErrorStringFromErrorCode(retVal));
}

int RocmDeviceInterface::handleVideoSequence(
    RocdecVideoFormat* videoFormat) {
  
  TORCH_CHECK(videoFormat != nullptr, "Invalid video format");

  videoFormat_ = *videoFormat;

  if (videoFormat_.min_num_decode_surfaces == 0) {
    videoFormat_.min_num_decode_surfaces = 20;
  }

  if (!decoder_) {
    decoder_ = RocDecCache::getCache(device_).getDecoder(videoFormat);

    if (!decoder_) {
      decoder_ = createDecoder(videoFormat);
    } else {
    }

    TORCH_CHECK(decoder_, "Failed to get or create decoder");
  }

  return static_cast<int>(videoFormat_.min_num_decode_surfaces);
}

int RocmDeviceInterface::sendPacket(ReferenceAVPacket& packet) {
  
  if (cpuFallback_) {
    return cpuFallback_->sendPacket(packet);
  }

  TORCH_CHECK(
      packet.get() && packet->data && packet->size > 0,
      "sendPacket received an empty packet");

  AutoAVPacket filteredAutoPacket;
  ReferenceAVPacket filteredPacket(filteredAutoPacket);
  ReferenceAVPacket& packetToSend = applyBSF(packet, filteredPacket);

  RocdecSourceDataPacket rocDecPacket = {};
  rocDecPacket.payload = packetToSend->data;
  rocDecPacket.payload_size = packetToSend->size;
  rocDecPacket.flags = ROCDEC_PKT_TIMESTAMP;
  rocDecPacket.pts = packetToSend->pts;

  int result = sendRocDecPacket(rocDecPacket);
  return result;
}

int RocmDeviceInterface::sendEOFPacket() {
  if (cpuFallback_) {
    return cpuFallback_->sendEOFPacket();
  }

  RocdecSourceDataPacket rocDecPacket = {};
  rocDecPacket.flags = ROCDEC_PKT_ENDOFSTREAM;
  eofSent_ = true;

  return sendRocDecPacket(rocDecPacket);
}

int RocmDeviceInterface::sendRocDecPacket(
    RocdecSourceDataPacket& rocDecPacket) {
  rocDecStatus result = rocDecParseVideoData(videoParser_, &rocDecPacket);
  return result == ROCDEC_SUCCESS ? AVSUCCESS : AVERROR_EXTERNAL;
}

ReferenceAVPacket& RocmDeviceInterface::applyBSF(
    ReferenceAVPacket& packet,
    ReferenceAVPacket& filteredPacket) {
  if (!bitstreamFilter_) {
    return packet;
  }

  int retVal = av_bsf_send_packet(bitstreamFilter_.get(), packet.get());
  TORCH_CHECK(
      retVal >= AVSUCCESS,
      "Failed to send packet to bitstream filter: ",
      getFFMPEGErrorStringFromErrorCode(retVal));

  retVal = av_bsf_receive_packet(bitstreamFilter_.get(), filteredPacket.get());
  TORCH_CHECK(
      retVal >= AVSUCCESS,
      "Failed to receive packet from bitstream filter: ",
      getFFMPEGErrorStringFromErrorCode(retVal));

  return filteredPacket;
}

int RocmDeviceInterface::handlePictureDecode(RocdecPicParams* picParams) {
  
  TORCH_CHECK(picParams != nullptr, "Invalid picture parameters");
  TORCH_CHECK(decoder_, "Decoder not initialized before picture decode");
  
  rocDecStatus result = rocDecDecodeFrame(*decoder_.get(), picParams);
  
  return (result == ROCDEC_SUCCESS);
}

int RocmDeviceInterface::handlePictureDisplay(
    RocdecParserDispInfo* dispInfo) {
  readyFrames_.push(*dispInfo);
  return 1;
}

int RocmDeviceInterface::receiveFrame(UniqueAVFrame& avFrame) {
  
  if (cpuFallback_) {
    return cpuFallback_->receiveFrame(avFrame);
  }

  if (readyFrames_.empty()) {
    return eofSent_ ? AVERROR_EOF : AVERROR(EAGAIN);
  }

  RocdecParserDispInfo dispInfo = readyFrames_.front();
  readyFrames_.pop();

  RocdecProcParams procParams = {};
  procParams.progressive_frame = dispInfo.progressive_frame;
  procParams.top_field_first = dispInfo.top_field_first;

  void* framePtr[3] = {nullptr, nullptr, nullptr};
  unsigned int pitch = 0;

  // Get video frame from decoder
  // Note: rocDecode manages its own internal streams
  rocDecStatus result = rocDecGetVideoFrame(
      *decoder_.get(), dispInfo.picture_index, framePtr, &pitch, &procParams);
  
  if (result != ROCDEC_SUCCESS) {
    return AVERROR_EXTERNAL;
  }

  avFrame = convertRocmFrameToAVFrame(framePtr, pitch, dispInfo);

  return AVSUCCESS;
}

void RocmDeviceInterface::releasePreviousFrame() {
  // Note: rocDecode does not require explicit frame release
  // The frame resources are automatically managed by the decoder
}

UniqueAVFrame RocmDeviceInterface::convertRocmFrameToAVFrame(
    void* framePtr[3],
    unsigned int pitch,
    const RocdecParserDispInfo& dispInfo) {
  
  TORCH_CHECK(framePtr != nullptr, "Invalid ROCm frame pointer");

  int width = videoFormat_.display_area.right - videoFormat_.display_area.left;
  int height = videoFormat_.display_area.bottom - videoFormat_.display_area.top;
  
  TORCH_CHECK(width > 0 && height > 0, "Invalid frame dimensions");
  TORCH_CHECK(pitch >= static_cast<unsigned int>(width), "Pitch must be >= width");

  UniqueAVFrame avFrame(av_frame_alloc());
  TORCH_CHECK(avFrame.get() != nullptr, "Failed to allocate AVFrame");

  avFrame->width = width;
  avFrame->height = height;
  
  // Set pixel format based on decoder output format
  if (videoFormat_.bit_depth_luma_minus8 > 0) {
    // 10-bit: P016LE (16-bit per component, little-endian)
    // rocDecode outputs P016 format for 10-bit videos
    avFrame->format = AV_PIX_FMT_P016LE;
  } else {
    // 8-bit: NV12
    avFrame->format = AV_PIX_FMT_NV12;
  }
  
  avFrame->pts = dispInfo.pts;

  setDuration(avFrame, computeSafeDuration(frameRateAvgFromFFmpeg_, timeBase_));

  // Set colorspace information
  switch (videoFormat_.video_signal_description.matrix_coefficients) {
    case 1:
      avFrame->colorspace = AVCOL_SPC_BT709;
      break;
    case 6:
      avFrame->colorspace = AVCOL_SPC_SMPTE170M;
      break;
    default:
      avFrame->colorspace = AVCOL_SPC_SMPTE170M;
      break;
  }

  // Set color range from rocDecode's parsed VUI parameters
  // If rocDecode explicitly indicates full range, trust it
  // Otherwise, fall back to container-level metadata from FFmpeg codec context
  // because rocDecode might not have parsed VUI parameters (video_full_range_flag=0 by default)
  
  if (videoFormat_.video_signal_description.video_full_range_flag) {
    // rocDecode explicitly parsed full range from VUI - trust it
    avFrame->color_range = AVCOL_RANGE_JPEG;  // Full range (0-255)
  } else if (codecContext_) {
    // rocDecode didn't indicate full range - could be:
    // 1. VUI absent from bitstream, OR
    // 2. VUI present but video_signal_type_present_flag=0, OR  
    // 3. VUI present and explicitly indicates studio range
    // Fall back to container-level metadata which FFmpeg parses reliably
    avFrame->color_range = codecContext_->color_range;
  } else {
    // No codec context available (shouldn't happen) - default to studio
    avFrame->color_range = AVCOL_RANGE_MPEG;  // Studio range (16-235)
  }

  // CRITICAL: We must copy the frame data from rocDecode's internal buffer
  // because rocDecode will reuse/free this buffer for the next frame.
  // The AVFrame will be used later in Python and must own its memory.
  
  // For NV12: pitch is the stride/alignment, width is the actual image width (1 byte per pixel)
  // For P016: pitch is the stride/alignment, width is the actual image width (2 bytes per pixel)
  // We need to allocate based on pitch to maintain alignment
  int bytesPerPixel = (videoFormat_.bit_depth_luma_minus8 > 0) ? 2 : 1;
  int ySize = pitch * height;
  int uvSize = pitch * (height / 2);  // UV plane also uses pitch
  size_t totalSize = static_cast<size_t>(ySize + uvSize);

  uint8_t* hipBuffer = nullptr;
  hipError_t err = hipMalloc(reinterpret_cast<void**>(&hipBuffer), totalSize);
  TORCH_CHECK(
      err == hipSuccess,
      "Failed to allocate HIP memory for frame: ",
      hipGetErrorString(err));

  TORCH_CHECK(framePtr[0] != nullptr, "rocDecode returned null Y plane pointer");
  TORCH_CHECK(framePtr[1] != nullptr, "rocDecode returned null UV plane pointer");

  // Copy Y plane asynchronously
  err = hipMemcpy2DAsync(
      hipBuffer,                                    // dst
      pitch,                                        // dst pitch
      reinterpret_cast<uint8_t*>(framePtr[0]),     // src - Y plane
      pitch,                                        // src pitch
      width * bytesPerPixel,                        // width in bytes
      height,                                       // height in rows
      hipMemcpyDeviceToDevice,
      copyStream_);
  TORCH_CHECK(
      err == hipSuccess,
      "Failed to copy Y plane: ",
      hipGetErrorString(err));

  // Copy UV plane asynchronously
  err = hipMemcpy2DAsync(
      hipBuffer + ySize,                           // dst
      pitch,                                        // dst pitch
      reinterpret_cast<uint8_t*>(framePtr[1]),     // src - UV plane
      pitch,                                        // src pitch
      width * bytesPerPixel,                        // width in bytes
      height / 2,                                   // height in rows
      hipMemcpyDeviceToDevice,
      copyStream_);
  TORCH_CHECK(
      err == hipSuccess,
      "Failed to copy UV plane: ",
      hipGetErrorString(err));

  // Synchronize the stream to ensure copies complete
  err = hipStreamSynchronize(copyStream_);
  TORCH_CHECK(
      err == hipSuccess,
      "Failed to synchronize stream: ",
      hipGetErrorString(err));

  avFrame->data[0] = hipBuffer;
  avFrame->data[1] = hipBuffer + ySize;
  avFrame->data[2] = nullptr;
  avFrame->data[3] = nullptr;
  // Use pitch for linesize since that's how we allocated and copied
  avFrame->linesize[0] = pitch;
  avFrame->linesize[1] = pitch;
  avFrame->linesize[2] = 0;
  avFrame->linesize[3] = 0;

  // Set up cleanup callback to free our HIP buffer when the frame is destroyed
  avFrame->opaque_ref = av_buffer_create(
      nullptr,
      0,
      [](void* opaque, [[maybe_unused]] uint8_t* data) {
        hipError_t freeErr = hipFree(opaque);
        if (freeErr != hipSuccess) {
          // Clear the error so subsequent operations can continue
          hipGetLastError();
        }
      },
      hipBuffer,
      0);
  TORCH_CHECK(
      avFrame->opaque_ref != nullptr,
      "Failed to create frame memory cleanup reference");

  return avFrame;
}

void RocmDeviceInterface::flush() {
  if (cpuFallback_) {
    cpuFallback_->flush();
    return;
  }

  sendEOFPacket();
  eofSent_ = false;

  std::queue<RocdecParserDispInfo> emptyQueue;
  std::swap(readyFrames_, emptyQueue);
}

UniqueAVFrame RocmDeviceInterface::transferCpuFrameToGpuNV12(
    UniqueAVFrame& cpuFrame) {
  // Transfer CPU-decoded frame to GPU memory
  TORCH_CHECK(cpuFrame != nullptr, "CPU frame cannot be null");

  int width = cpuFrame->width;
  int height = cpuFrame->height;

  // Convert to NV12 first
  UniqueAVFrame nv12CpuFrame(av_frame_alloc());
  TORCH_CHECK(nv12CpuFrame != nullptr, "Failed to allocate NV12 CPU frame");

  nv12CpuFrame->format = AV_PIX_FMT_NV12;
  nv12CpuFrame->width = width;
  nv12CpuFrame->height = height;

  int ret = av_frame_get_buffer(nv12CpuFrame.get(), 0);
  TORCH_CHECK(
      ret >= 0,
      "Failed to allocate NV12 CPU frame buffer: ",
      getFFMPEGErrorStringFromErrorCode(ret));

  SwsFrameContext swsFrameContext(
      width,
      height,
      static_cast<AVPixelFormat>(cpuFrame->format),
      width,
      height);

  if (!swsContext_ || prevSwsFrameContext_ != swsFrameContext) {
    // Use swsFlags=0 (point sampling) instead of SWS_BILINEAR to better match
    // the direct CPU decoding path, which also uses swsFlags=0.
    // This reduces color conversion differences when comparing CPU fallback against direct CPU.
    // Note: CUDA beta uses SWS_BILINEAR, which is why it needs PSNR > 25 dB threshold.
    swsContext_ = createSwsContext(
        swsFrameContext, cpuFrame->colorspace, AV_PIX_FMT_NV12, 0);
    prevSwsFrameContext_ = swsFrameContext;
  }

  int convertedHeight = sws_scale(
      swsContext_.get(),
      cpuFrame->data,
      cpuFrame->linesize,
      0,
      height,
      nv12CpuFrame->data,
      nv12CpuFrame->linesize);
  TORCH_CHECK(
      convertedHeight == height, "sws_scale failed for CPU->NV12 conversion");

  // Allocate GPU memory and copy
  int ySize = width * height;
  int uvSize = ySize / 2;
  size_t totalSize = static_cast<size_t>(ySize + uvSize);

  uint8_t* hipBuffer = nullptr;
  hipError_t err = hipMalloc(reinterpret_cast<void**>(&hipBuffer), totalSize);
  TORCH_CHECK(
      err == hipSuccess, "Failed to allocate HIP memory: ", hipGetErrorString(err));

  UniqueAVFrame gpuFrame(av_frame_alloc());
  TORCH_CHECK(gpuFrame != nullptr, "Failed to allocate GPU AVFrame");

  gpuFrame->format = AV_PIX_FMT_NV12;  // Use NV12 format for ROCm
  gpuFrame->width = width;
  gpuFrame->height = height;
  gpuFrame->data[0] = hipBuffer;
  gpuFrame->data[1] = hipBuffer + ySize;
  gpuFrame->linesize[0] = width;
  gpuFrame->linesize[1] = width;

  err = hipMemcpy2D(
      gpuFrame->data[0],
      gpuFrame->linesize[0],
      nv12CpuFrame->data[0],
      nv12CpuFrame->linesize[0],
      width,
      height,
      hipMemcpyHostToDevice);
  TORCH_CHECK(
      err == hipSuccess, "Failed to copy Y plane to GPU: ", hipGetErrorString(err));

  err = hipMemcpy2D(
      gpuFrame->data[1],
      gpuFrame->linesize[1],
      nv12CpuFrame->data[1],
      nv12CpuFrame->linesize[1],
      width,
      height / 2,
      hipMemcpyHostToDevice);
  TORCH_CHECK(
      err == hipSuccess, "Failed to copy UV plane to GPU: ", hipGetErrorString(err));

  ret = av_frame_copy_props(gpuFrame.get(), cpuFrame.get());
  TORCH_CHECK(
      ret >= 0,
      "Failed to copy frame properties: ",
      getFFMPEGErrorStringFromErrorCode(ret));

  gpuFrame->opaque_ref = av_buffer_create(
      nullptr,
      0,
      [](void* opaque, [[maybe_unused]] uint8_t* data) {
        hipFree(opaque);
      },
      hipBuffer,
      0);
  TORCH_CHECK(
      gpuFrame->opaque_ref != nullptr,
      "Failed to create GPU memory cleanup reference");

  return gpuFrame;
}

void RocmDeviceInterface::convertAVFrameToFrameOutput(
    UniqueAVFrame& avFrame,
    FrameOutput& frameOutput,
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  UniqueAVFrame gpuFrame =
      cpuFallback_ ? transferCpuFrameToGpuNV12(avFrame) : std::move(avFrame);

  // Accept both NV12 (8-bit) and P010LE (10-bit) formats
  TORCH_CHECK(
      gpuFrame->format == AV_PIX_FMT_NV12 || 
      gpuFrame->format == AV_PIX_FMT_P010LE ||
      gpuFrame->format == AV_PIX_FMT_P010BE ||
      gpuFrame->format == AV_PIX_FMT_P016LE ||
      gpuFrame->format == AV_PIX_FMT_P016BE,
      "Expected NV12, P010, or P016 format frame from ROCm interface, got format: ",
      gpuFrame->format);

  validatePreAllocatedTensorShape(preAllocatedOutputTensor, gpuFrame);

  
  // Lazily initialize RPP context on first use
  if (!rppCtx_) {
    rppCtx_ = getRppStreamContext(device_);
  }

  // Use copyStream_ which was passed to convertRocmFrameToAVFrame for frame copy
  // The color conversion will use rppCtx_->stream internally
  frameOutput.data = convertNV12FrameToRGB(
      gpuFrame, device_, rppCtx_, copyStream_, preAllocatedOutputTensor);

  // Synchronize the RPP stream to ensure color conversion completes
  // before the gpuFrame (NV12 buffer) is destroyed
  hipError_t err = hipStreamSynchronize(rppCtx_->stream);
  TORCH_CHECK(
      err == hipSuccess,
      "Failed to synchronize RPP stream: ",
      hipGetErrorString(err));
  
}

std::string RocmDeviceInterface::getDetails() {
  std::string details = "ROCm Device Interface.";
  if (cpuFallback_) {
    details += " Using CPU fallback.";
    if (!rocDecodeAvailable_) {
      details += " rocDecode not available!";
    }
  } else {
    details += " Using AMD VCN (rocDecode).";
  }
  return details;
}

} // namespace facebook::torchcodec
