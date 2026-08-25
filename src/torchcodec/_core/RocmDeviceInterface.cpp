// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <hip/hip_runtime.h>
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
static bool g_rocm = register_device_interface(
    DeviceInterfaceKey(c10::kCUDA),  // Uses default variant "ffmpeg"
    [](const StableDevice& device) {
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
  
  // We explicitly request NV12 format, which means 10bit videos will be
  // automatically converted to 8bits by rocDecode itself. That is, the raw frames
  // we get back from rocDecGetVideoFrame will already be in 8bit format.
  decoderParams.output_format = rocDecVideoSurfaceFormat_NV12;
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
  
  STD_TORCH_CHECK(
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
  STD_TORCH_CHECK(desc != nullptr, "desc can't be null");

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

  // Decoder is created with rocDecVideoSurfaceFormat_NV12; 10-bit streams are
  // converted to 8-bit NV12 by rocDecode. Require NV12 output for all depths.
  bool supportsNV12Output =
      (caps.output_format_mask >> rocDecVideoSurfaceFormat_NV12) & 1;
  if (!supportsNV12Output) {
    return false;
  }

  return true;
}

} // namespace

RocmDeviceInterface::RocmDeviceInterface(const StableDevice& device)
    : DeviceInterface(device) {
  STD_TORCH_CHECK(g_rocm, "RocmDeviceInterface was not registered!");
  STD_TORCH_CHECK(
      device_.type() == c10::kCUDA, "Unsupported device: must be CUDA (for ROCm)");

  // Get the actual device index (handles -1 case by querying current device)
  int deviceIndex = get_device_index(device_);

  // Update device_ to have the explicit index
  device_ = StableDevice(device_.type(), deviceIndex);

  // Set HIP device before initializing PyTorch context
  hipError_t err = hipSetDevice(deviceIndex);
  STD_TORCH_CHECK(
      err == hipSuccess,
      "Failed to set HIP device ",
      deviceIndex,
      ": ",
      hipGetErrorString(err));

  // Create a persistent HIP stream for async memory operations
  err = hipStreamCreate(&rocdecStream_);
  STD_TORCH_CHECK(
      err == hipSuccess,
      "Failed to create HIP stream: ",
      hipGetErrorString(err));

  initializeRocmContextWithPytorch(device_);

  rppCtx_ = getRppStreamContext(device_);
}

RocmDeviceInterface::~RocmDeviceInterface() {
  if (decoder_) {
    try {
      flush();
    } catch (const std::exception&) {
      // Don't rethrow from destructor
    }

    // Set the device before accessing cache to ensure HIP context is valid
    hipError_t hipErr = hipSetDevice(static_cast<int>(device_.index()));
    STD_TORCH_CHECK(
        hipErr == hipSuccess,
        "hipSetDevice failed in ~RocmDeviceInterface (before returnDecoder): ",
        hipGetErrorString(hipErr));
    RocDecCache::getCache(device_).returnDecoder(
        &videoFormat_, std::move(decoder_));
  }

  if (videoParser_) {
    rocDecDestroyVideoParser(videoParser_);
    videoParser_ = nullptr;
  }

  if (rppCtx_) {
    hipError_t hipErr = hipSetDevice(static_cast<int>(device_.index()));
    STD_TORCH_CHECK(
        hipErr == hipSuccess,
        "hipSetDevice failed in ~RocmDeviceInterface (before returnRppStreamContextToCache): ",
        hipGetErrorString(hipErr));
    returnRppStreamContextToCache(device_, std::move(rppCtx_));
  }

  if (rocdecStream_) {
    hipError_t hipErr = hipStreamDestroy(rocdecStream_);
    STD_TORCH_CHECK(
        hipErr == hipSuccess,
        "hipStreamDestroy failed in ~RocmDeviceInterface: ",
        hipGetErrorString(hipErr));
    rocdecStream_ = nullptr;
  }
}

void RocmDeviceInterface::initialize(const SharedAVCodecContext& codec_context) {
  // Store codec context for fallback color range info
  codecContext_ = codec_context;

  // Select chroma upsampling for the RPP NV12->RGB kernel to match FFmpeg's swscale
  // output. FFmpeg's vertical chroma handling for >8-bit (scaled) content differs by
  // major version, so the >8-bit choice is version-gated:
  //   - 8-bit:            nearest-neighbor (matches FFmpeg's unscaled fast path)
  //   - 10-bit, FFmpeg 4: linear
  //   - 10-bit, FFmpeg>4: bicubic (B=0, C=0.6); see FFmpeg9_YUV_to_RGB_spec.md
  // The matching .so is loaded per FFmpeg major version, so a compile-time check is
  // correct here. FFmpeg 4 == libavcodec 58.
  const AVPixFmtDescriptor* desc =
      av_pix_fmt_desc_get(codec_context->pix_fmt);
  if (desc && desc->comp[0].depth > 8) {
#if LIBAVCODEC_VERSION_MAJOR <= 58
    chromaUpsampling_ = ChromaUpsampling::kLinear;
#else
    chromaUpsampling_ = ChromaUpsampling::kCubic;
#endif
  } else {
    chromaUpsampling_ = ChromaUpsampling::kNearestNeighbor;
  }
}

void RocmDeviceInterface::initialize_video_decoding(
    const AVStream* av_stream,
    const UniqueDecodingAVFormatContext& av_format_ctx,
    [[maybe_unused]] const VideoStreamOptions& video_stream_options) {
  STD_TORCH_CHECK(codecContext_ != nullptr, "Must call initialize() first");

  if (!nativeRocDecodeSupport(codecContext_)) {
    cpuFallback_ = create_device_interface(c10::kCPU);
    STD_TORCH_CHECK(
        cpuFallback_ != nullptr, "Failed to create CPU device interface");
    cpuFallback_->initialize(codecContext_);
    cpuFallback_->initialize_video_decoding(
        av_stream, av_format_ctx, video_stream_options);
    return;
  }

  STD_TORCH_CHECK(av_stream != nullptr, "AVStream cannot be null");
  timeBase_ = av_stream->time_base;
  frameRateAvgFromFFmpeg_ = av_stream->r_frame_rate;

  const AVCodecParameters* codecPar = av_stream->codecpar;
  STD_TORCH_CHECK(codecPar != nullptr, "CodecParameters cannot be null");

  initializeBSF(codecPar, av_format_ctx);

  // Create parser
  RocdecParserParams parserParams = {};
  auto codecType = validateCodecSupport(codecPar->codec_id);
  STD_TORCH_CHECK(
      codecType.has_value(),
      "This should never happen, we should be using the CPU fallback by now.");
  parserParams.codec_type = codecType.value();
  parserParams.max_num_decode_surfaces = 8;
  parserParams.max_display_delay = 0;
  parserParams.user_data = this;
  // Callback setup, all are triggered by the parser within a call
  // to rocDecParseVideoData
  parserParams.pfn_sequence_callback = handleVideoSequenceCallback;
  parserParams.pfn_decode_picture = handlePictureDecodeCallback;
  parserParams.pfn_display_picture = handlePictureDisplayCallback;

  rocDecStatus result = rocDecCreateVideoParser(&videoParser_, &parserParams);

  STD_TORCH_CHECK(
      result == ROCDEC_SUCCESS,
      "Failed to create rocDecode video parser. Status: ", result);
}

void RocmDeviceInterface::initializeBSF(
    const AVCodecParameters* codecPar,
    const UniqueDecodingAVFormatContext& avFormatCtx) {
  STD_TORCH_CHECK(codecPar != nullptr, "codecPar cannot be null");
  STD_TORCH_CHECK(avFormatCtx != nullptr, "AVFormatContext cannot be null");
  STD_TORCH_CHECK(
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
  STD_TORCH_CHECK(
      avBSF != nullptr, "Failed to find bitstream filter: ", filterName);

  AVBSFContext* avBSFContext = nullptr;
  int retVal = av_bsf_alloc(avBSF, &avBSFContext);
  STD_TORCH_CHECK(
      retVal >= AVSUCCESS,
      "Failed to allocate bitstream filter: ",
      get_ffmpeg_error_string_from_error_code(retVal));

  bitstreamFilter_.reset(avBSFContext);

  retVal = avcodec_parameters_copy(bitstreamFilter_->par_in, codecPar);
  STD_TORCH_CHECK(
      retVal >= AVSUCCESS,
      "Failed to copy codec parameters: ",
      get_ffmpeg_error_string_from_error_code(retVal));

  retVal = av_bsf_init(bitstreamFilter_.get());
  STD_TORCH_CHECK(
      retVal == AVSUCCESS,
      "Failed to initialize bitstream filter: ",
      get_ffmpeg_error_string_from_error_code(retVal));
}

int RocmDeviceInterface::handleVideoSequence(
    RocdecVideoFormat* videoFormat) {
  
  STD_TORCH_CHECK(videoFormat != nullptr, "Invalid video format");

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

    STD_TORCH_CHECK(decoder_, "Failed to get or create decoder");
  }

  return static_cast<int>(videoFormat_.min_num_decode_surfaces);
}

int RocmDeviceInterface::send_packet(ReferenceAVPacket& av_packet) {
  
  if (cpuFallback_) {
    return cpuFallback_->send_packet(av_packet);
  }

  STD_TORCH_CHECK(
      av_packet.get() && av_packet->data && av_packet->size > 0,
      "sendPacket received an empty av_packet");

  AutoAVPacket filteredAutoPacket;
  ReferenceAVPacket filteredPacket(filteredAutoPacket);
  ReferenceAVPacket& packetToSend = applyBSF(av_packet, filteredPacket);

  RocdecSourceDataPacket rocDecPacket = {};
  rocDecPacket.payload = packetToSend->data;
  rocDecPacket.payload_size = packetToSend->size;
  rocDecPacket.flags = ROCDEC_PKT_TIMESTAMP;
  rocDecPacket.pts = packetToSend->pts;

  int result = sendRocDecPacket(rocDecPacket);
  return result;
}

int RocmDeviceInterface::send_eof_packet() {
  if (cpuFallback_) {
    return cpuFallback_->send_eof_packet();
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
  STD_TORCH_CHECK(
      retVal >= AVSUCCESS,
      "Failed to send packet to bitstream filter: ",
      get_ffmpeg_error_string_from_error_code(retVal));

  retVal = av_bsf_receive_packet(bitstreamFilter_.get(), filteredPacket.get());
  STD_TORCH_CHECK(
      retVal >= AVSUCCESS,
      "Failed to receive packet from bitstream filter: ",
      get_ffmpeg_error_string_from_error_code(retVal));

  return filteredPacket;
}

int RocmDeviceInterface::handlePictureDecode(RocdecPicParams* picParams) {
  
  STD_TORCH_CHECK(picParams != nullptr, "Invalid picture parameters");
  STD_TORCH_CHECK(decoder_, "Decoder not initialized before picture decode");
  
  rocDecStatus result = rocDecDecodeFrame(*decoder_.get(), picParams);
  
  return (result == ROCDEC_SUCCESS);
}

int RocmDeviceInterface::handlePictureDisplay(
    RocdecParserDispInfo* dispInfo) {
  readyFrames_.push(*dispInfo);
  return 1;
}

int RocmDeviceInterface::receive_frame(UniqueAVFrame& av_frame) {
  
  if (cpuFallback_) {
    return cpuFallback_->receive_frame(av_frame);
  }

  if (readyFrames_.empty()) {
    // No frame found, instruct caller to try again later after sending more
    // packets, or to stop if EOF was already sent.
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

  av_frame = convertRocmFrameToAVFrame(framePtr, pitch, dispInfo);

  return AVSUCCESS;
}

UniqueAVFrame RocmDeviceInterface::convertRocmFrameToAVFrame(
    void* framePtr[3],
    unsigned int pitch,
    const RocdecParserDispInfo& dispInfo) {
  
  STD_TORCH_CHECK(framePtr != nullptr, "Invalid ROCm frame pointer");

  int width = videoFormat_.display_area.right - videoFormat_.display_area.left;
  int height = videoFormat_.display_area.bottom - videoFormat_.display_area.top;
  
  STD_TORCH_CHECK(width > 0 && height > 0, "Invalid frame dimensions");
  STD_TORCH_CHECK(pitch >= static_cast<unsigned int>(width), "Pitch must be >= width");

  UniqueAVFrame av_frame(av_frame_alloc());
  STD_TORCH_CHECK(av_frame.get() != nullptr, "Failed to allocate AVFrame");

  av_frame->width = width;
  av_frame->height = height;
  // Decoder requests NV12; 10-bit content is output as 8-bit NV12 from rocDecode.
  av_frame->format = AV_PIX_FMT_NV12;
  av_frame->pts = dispInfo.pts;

  //We compute the duration based on average frame rate info, so
  // so if the video has variable frame rate, the durations may be off
  set_duration(*av_frame, compute_safe_duration(frameRateAvgFromFFmpeg_, timeBase_));

  // Set colorspace information
  // rocDecode parses matrix_coefficients from the bitstream, but for some codecs
  // (notably AV1) it may report 0 (unspecified) even when the container has
  // valid colorspace metadata. In that case, fall back to FFmpeg's codec context.
  // Per ITU-T specs, both 0 and 2 mean unspecified/reserved.
  int matrixCoeffs = videoFormat_.video_signal_description.matrix_coefficients;

  if ((matrixCoeffs == 0 || matrixCoeffs == 2) && codecContext_ &&
      codecContext_->colorspace != AVCOL_SPC_UNSPECIFIED) {
    // rocDecode reports unspecified, but FFmpeg has valid colorspace from container
    // Map FFmpeg's AVColorSpace enum to the colorspace we'll use
    switch (codecContext_->colorspace) {
      case AVCOL_SPC_BT709:
        av_frame->colorspace = AVCOL_SPC_BT709;
        break;
      case AVCOL_SPC_SMPTE170M:
      case AVCOL_SPC_BT470BG:
        av_frame->colorspace = AVCOL_SPC_SMPTE170M;
        break;
      default:
        // Default to SMPTE170M for unknown colorspaces
        av_frame->colorspace = AVCOL_SPC_SMPTE170M;
        break;
    }
  } else {
    // Use rocDecode's parsed matrix_coefficients
    switch (matrixCoeffs) {
      case 1:
        av_frame->colorspace = AVCOL_SPC_BT709;
        break;
      case 6:
        av_frame->colorspace = AVCOL_SPC_SMPTE170M;
        break;
      default:
        av_frame->colorspace = AVCOL_SPC_SMPTE170M;
        break;
    }
  }

  // Set color range from rocDecode's parsed VUI parameters
  // If rocDecode explicitly indicates full range, trust it
  // Otherwise, fall back to container-level metadata from FFmpeg codec context
  // because rocDecode might not have parsed VUI parameters (video_full_range_flag=0 by default)
  if (videoFormat_.video_signal_description.video_full_range_flag) {
    // rocDecode explicitly parsed full range from VUI - trust it
    av_frame->color_range = AVCOL_RANGE_JPEG;  // Full range (0-255)
  } else if (codecContext_) {
    // rocDecode didn't indicate full range - could be:
    // 1. VUI absent from bitstream, OR
    // 2. VUI present but video_signal_type_present_flag=0, OR
    // 3. VUI present and explicitly indicates studio range
    // Fall back to container-level metadata which FFmpeg parses reliably
    av_frame->color_range = codecContext_->color_range;
  } else {
    // No codec context available (shouldn't happen) - default to studio
    av_frame->color_range = AVCOL_RANGE_MPEG;  // Studio range (16-235)
  }

  // NV12 has only 2 valid planes
  av_frame->data[0] = reinterpret_cast<uint8_t*>(framePtr[0]);
  av_frame->data[1] = reinterpret_cast<uint8_t*>(framePtr[1]);
  av_frame->data[2] = nullptr;
  av_frame->data[3] = nullptr;
  // Use pitch for linesize since that's how we allocated and copied
  av_frame->linesize[0] = pitch;
  av_frame->linesize[1] = pitch;
  av_frame->linesize[2] = 0;
  av_frame->linesize[3] = 0;

  return av_frame;
}

void RocmDeviceInterface::flush() {
  if (cpuFallback_) {
    cpuFallback_->flush();
    return;
  }

  send_eof_packet();
  eofSent_ = false;

  std::queue<RocdecParserDispInfo> emptyQueue;
  std::swap(readyFrames_, emptyQueue);
}

UniqueAVFrame RocmDeviceInterface::transferCpuFrameToGpuNV12(
    UniqueAVFrame& cpuFrame) {
  // This is called in the context of the CPU fallback: the frame was decoded on
  // the CPU, and in this function we convert that frame into NV12 format and
  // send it to the GPU.
  STD_TORCH_CHECK(cpuFrame != nullptr, "CPU frame cannot be null");

  int width = cpuFrame->width;
  int height = cpuFrame->height;

  // Convert to NV12 first
  UniqueAVFrame nv12CpuFrame(av_frame_alloc());
  STD_TORCH_CHECK(nv12CpuFrame != nullptr, "Failed to allocate NV12 CPU frame");

  nv12CpuFrame->format = AV_PIX_FMT_NV12;
  nv12CpuFrame->width = width;
  nv12CpuFrame->height = height;

  int ret = av_frame_get_buffer(nv12CpuFrame.get(), 0);
  STD_TORCH_CHECK(
      ret >= 0,
      "Failed to allocate NV12 CPU frame buffer: ",
      get_ffmpeg_error_string_from_error_code(ret));

  // Always recreate the sws context (simpler than caching)
  // Use swsFlags=0 (point sampling) instead of SWS_BILINEAR to better match
  // the direct CPU decoding path, which also uses swsFlags=0.
  swsContext_ = UniqueSwsContext(sws_getContext(
      width, height, static_cast<AVPixelFormat>(cpuFrame->format),
      width, height, AV_PIX_FMT_NV12,
      0, nullptr, nullptr, nullptr));

  int convertedHeight = sws_scale(
      swsContext_.get(),
      cpuFrame->data,
      cpuFrame->linesize,
      0,
      height,
      nv12CpuFrame->data,
      nv12CpuFrame->linesize);
  STD_TORCH_CHECK(
      convertedHeight == height, "sws_scale failed for CPU->NV12 conversion");

  // Allocate GPU memory and copy
  int ySize = width * height;
  int uvSize = ySize / 2;
  size_t totalSize = static_cast<size_t>(ySize + uvSize);

  uint8_t* hipBuffer = nullptr;
  hipError_t err = hipMalloc(reinterpret_cast<void**>(&hipBuffer), totalSize);
  STD_TORCH_CHECK(
      err == hipSuccess, "Failed to allocate HIP memory: ", hipGetErrorString(err));

  UniqueAVFrame gpuFrame(av_frame_alloc());
  STD_TORCH_CHECK(gpuFrame != nullptr, "Failed to allocate GPU AVFrame");

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
  STD_TORCH_CHECK(
      err == hipSuccess, "Failed to copy Y plane to GPU: ", hipGetErrorString(err));

  STD_TORCH_CHECK(
        height % 2 == 0,
        "height must be even. Please report on TorchCodec repo.");
  err = hipMemcpy2D(
      gpuFrame->data[1],
      gpuFrame->linesize[1],
      nv12CpuFrame->data[1],
      nv12CpuFrame->linesize[1],
      width,
      height / 2,
      hipMemcpyHostToDevice);
  STD_TORCH_CHECK(
      err == hipSuccess, "Failed to copy UV plane to GPU: ", hipGetErrorString(err));

  ret = av_frame_copy_props(gpuFrame.get(), cpuFrame.get());
  STD_TORCH_CHECK(
      ret >= 0,
      "Failed to copy frame properties: ",
      get_ffmpeg_error_string_from_error_code(ret));

  gpuFrame->opaque_ref = av_buffer_create(
      nullptr,
      0,
      [](void* opaque, [[maybe_unused]] uint8_t* data) {
        hipError_t hipErr = hipFree(opaque);
        STD_TORCH_CHECK(
            hipErr == hipSuccess,
            "hipFree failed in transferCpuFrameToGpuNV12 buffer free: ",
            hipGetErrorString(hipErr));
      },
      hipBuffer,
      0);
  STD_TORCH_CHECK(
      gpuFrame->opaque_ref != nullptr,
      "Failed to create GPU memory cleanup reference");

  return gpuFrame;
}

void RocmDeviceInterface::convert_av_frame_to_frame_output(
    const AVFrame& av_frame,
    FrameOutput& frame_output,
    std::optional<torch::stable::Tensor> pre_allocated_output_tensor) {
  // For CPU fallback, convert CPU frame to GPU NV12
  // For rocDecode, av_frame already has GPU pointers - use it directly (no copy/clone needed)
  UniqueAVFrame converted_frame;
  if (cpuFallback_) {
    // Clone CPU frame then transfer to GPU
    UniqueAVFrame cpu_frame = UniqueAVFrame(av_frame_alloc());
    av_frame_ref(cpu_frame.get(), &av_frame);
    converted_frame = transferCpuFrameToGpuNV12(cpu_frame);
  }

  // Use converted frame if we did conversion, otherwise use input directly
  const AVFrame& gpu_frame = converted_frame ? *converted_frame : av_frame;

  STD_TORCH_CHECK(
      gpu_frame.format == AV_PIX_FMT_NV12,
      "Expected NV12 frame from rocDecode (hardware path), got format: ",
      gpu_frame.format);

  validatePreAllocatedTensorShape(pre_allocated_output_tensor, gpu_frame);

  frame_output.data = convertNV12FrameToRGB(
      gpu_frame, device_, rppCtx_, rocdecStream_, chromaUpsampling_,
      pre_allocated_output_tensor);

  // Synchronize the RPP stream to ensure color conversion completes
  // before the gpuFrame (NV12 buffer) is destroyed
  hipError_t err = hipStreamSynchronize(rppCtx_->stream);
  STD_TORCH_CHECK(
      err == hipSuccess,
      "Failed to synchronize RPP stream: ",
      hipGetErrorString(err));
  
}

std::string RocmDeviceInterface::get_details() {
  std::string details = "ROCm Device Interface.";
  if (cpuFallback_) {
    details += " Using CPU fallback.";
  } else {
    details += " Using AMD VCN (rocDecode).";
  }
  return details;
}

} // namespace facebook::torchcodec
