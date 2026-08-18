// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "RocmCommon.h"
#include "Cache.h" // for PerGpuCache

#include <hip/hip_runtime.h>

extern "C" {
#include <libavutil/pixfmt.h>
}

namespace facebook::torchcodec {

namespace {

// Map FFmpeg metadata to rppt_yuv_to_rgb (RpptColorStandard / RpptColorRange in rppdefs.h).
RpptColorStandard avColorSpaceToRppColStandard(AVColorSpace space) {
  switch (space) {
    case AVCOL_SPC_FCC:
      return RpptColorStandard_FCC;
    case AVCOL_SPC_BT470BG:
      return RpptColorStandard_BT470BG;
    case AVCOL_SPC_SMPTE170M:
      return RpptColorStandard_BT601;
    case AVCOL_SPC_SMPTE240M:
      return RpptColorStandard_SMPTE240M;
    case AVCOL_SPC_BT2020_NCL:
      return RpptColorStandard_BT2020_NCL;
    case AVCOL_SPC_BT2020_CL:
      return RpptColorStandard_BT2020_CL;
    case AVCOL_SPC_BT709:
    default:
      return RpptColorStandard_BT709;
  }
}

// FFmpeg: JPEG = full; MPEG / unspecified = studio.
RpptColorRange avColorRangeToRppColorRange(AVColorRange range) {
  if (range == AVCOL_RANGE_JPEG) {
    return RpptColorRange_FULL;
  }
  return RpptColorRange_STUDIO;
}

// Set to -1 to have an infinitely sized cache. Set it to 0 to disable caching.
// Set to a positive number to have a cache of that size.
const int MAX_CONTEXTS_PER_GPU_IN_CACHE = -1;

PerGpuCache<RppContext, RppContextDeleter> g_cached_rpp_ctxs(
    MAX_ROCM_GPUS,
    MAX_CONTEXTS_PER_GPU_IN_CACHE);

} // namespace

void initializeRocmContextWithPytorch(const StableDevice& device) {
  // It is important for pytorch itself to create the HIP context. If ffmpeg
  // creates the context it may not be compatible with pytorch.
  // This is a dummy tensor to initialize the HIP context.
  torch::stable::Tensor dummyTensorForHipInitialization =
      torch::stable::empty({1}, kStableUInt8, std::nullopt, device);
  torch::stable::zero_(dummyTensorForHipInitialization);
}

torch::stable::Tensor convertNV12FrameToRGB(
    const AVFrame& avFrame,
    const StableDevice& device,
    const UniqueRppContext& rppCtx,
    hipStream_t rocdecStream,
    ChromaUpsampling chromaUpsampling,
    std::optional<torch::stable::Tensor> preAllocatedOutputTensor) {

  auto frameDims = FrameDims(avFrame.height, avFrame.width);
  
  STD_TORCH_CHECK(
      avFrame.format == AV_PIX_FMT_NV12,
      "convertNV12FrameToRGB on ROCm expects NV12 (AV_PIX_FMT_NV12); "
      "rppt_yuv_to_rgb is 8-bit only. Got format: ",
      av_get_pix_fmt_name(static_cast<AVPixelFormat>(avFrame.format))
          ? av_get_pix_fmt_name(static_cast<AVPixelFormat>(avFrame.format))
          : "unknown",
      " (format code: ",
      avFrame.format,
      ")");

  torch::stable::Tensor dst;
  if (preAllocatedOutputTensor.has_value()) {
    dst = preAllocatedOutputTensor.value();
    STD_TORCH_CHECK(
        dst.scalar_type() == kStableUInt8,
        "ROCm NV12→RGB requires a uint8 HWC output tensor.");
  } else {
    dst = allocate_empty_hwc_tensor(frameDims, device, OutputDtype::UINT8);
  }

  // We need to make sure rocDecode has finished decoding a frame before
  // color-converting it with HIP kernels.
  // So we make the RPP stream wait for rocDecode to finish.
  hipStream_t rppStream = rppCtx->stream;
  
  hipEvent_t rocdecodeDoneEvent;
  hipError_t err = hipEventCreate(&rocdecodeDoneEvent);
  STD_TORCH_CHECK(
      err == hipSuccess,
      "hipEventCreate failed: ",
      hipGetErrorString(err));
  
  err = hipEventRecord(rocdecodeDoneEvent, rocdecStream);
  STD_TORCH_CHECK(
      err == hipSuccess,
      "hipEventRecord failed: ",
      hipGetErrorString(err));
  
  err = hipStreamWaitEvent(rppStream, rocdecodeDoneEvent, 0);
  STD_TORCH_CHECK(
      err == hipSuccess,
      "hipStreamWaitEvent failed: ",
      hipGetErrorString(err));
  
  err = hipEventDestroy(rocdecodeDoneEvent);
  STD_TORCH_CHECK(
      err == hipSuccess,
      "hipEventDestroy failed: ",
      hipGetErrorString(err));

  err = hipStreamGetFlags(rppCtx->stream, &rppCtx->streamFlags);
  STD_TORCH_CHECK(
      err == hipSuccess,
      "hipStreamGetFlags failed: ",
      hipGetErrorString(err));

  // NV12: Y plane then interleaved UV (device pointers from rocDecode).
  uint8_t* yuvData[2] = {avFrame.data[0], avFrame.data[1]};

  rppStatus_t setStreamStatus =
      rppSetStream(rppCtx->handle, rppCtx->stream);
  STD_TORCH_CHECK(
      setStreamStatus == rppStatusSuccess,
      "rppSetStream failed. Status: ",
      setStreamStatus);

  RpptDesc srcDesc{};
  srcDesc.numDims = 4;
  srcDesc.offsetInBytes = 0;
  srcDesc.dataType = RpptDataType::U8;
  srcDesc.n = 1;
  srcDesc.c = 1;
  srcDesc.h = static_cast<Rpp32u>(frameDims.height);
  srcDesc.w = static_cast<Rpp32u>(frameDims.width);
  srcDesc.layout = RpptLayout::NHWC;
  srcDesc.strides.nStride = srcDesc.c * srcDesc.w * srcDesc.h;
  srcDesc.strides.hStride = srcDesc.c * srcDesc.w;
  srcDesc.strides.wStride = srcDesc.c;
  srcDesc.strides.cStride = 1;

  const Rpp32u rgbRowBytes =
      static_cast<Rpp32u>(dst.strides()[0]) * static_cast<Rpp32u>(dst.element_size());
  RpptDesc dstDesc{};
  dstDesc.numDims = 4;
  dstDesc.offsetInBytes = 0;
  dstDesc.dataType = RpptDataType::U8;
  dstDesc.n = 1;
  dstDesc.c = 3;
  dstDesc.h = static_cast<Rpp32u>(frameDims.height);
  dstDesc.w = static_cast<Rpp32u>(frameDims.width);
  dstDesc.layout = RpptLayout::NHWC;
  dstDesc.strides.nStride = rgbRowBytes * dstDesc.h;
  dstDesc.strides.hStride = rgbRowBytes;
  dstDesc.strides.wStride = 3;
  dstDesc.strides.cStride = 1;

  const RpptColorStandard colStandard =
      avColorSpaceToRppColStandard(static_cast<AVColorSpace>(avFrame.colorspace));
  const RpptColorRange colorRange =
      avColorRangeToRppColorRange(static_cast<AVColorRange>(avFrame.color_range));

  using RppYuvToRgbFn = RppStatus (*)(
      RppPtr_t, RppPtr_t, RpptDescPtr, RppPtr_t, RpptDescPtr,
      Rpp32u, Rpp32u, Rpp32u, Rpp32u, Rpp32u,
      RpptColorStandard, RpptColorRange, rppHandle_t, RppBackend);

  RppYuvToRgbFn yuvToRgbFn;
  const char* modeName;
  switch (chromaUpsampling) {
    case ChromaUpsampling::kCubic:
      yuvToRgbFn = rppt_yuv_to_rgb_cubic_v;
      modeName = "cubic";
      break;
    case ChromaUpsampling::kLinear:
      yuvToRgbFn = rppt_yuv_to_rgb_linear_v;
      modeName = "linear";
      break;
    case ChromaUpsampling::kNearestNeighbor:
    default:
      yuvToRgbFn = rppt_yuv_to_rgb;
      modeName = "nearest-neighbor";
      break;
  }
  std::cout << "Using " << modeName
            << " chroma upsampling for NV12 to RGB conversion on ROCm."
            << std::endl;

  RppStatus status = yuvToRgbFn(
      yuvData[0],
      yuvData[1],
      &srcDesc,
      dst.mutable_data_ptr<uint8_t>(),
      &dstDesc,
      static_cast<Rpp32u>(avFrame.linesize[0]),
      static_cast<Rpp32u>(avFrame.linesize[1]),
      rgbRowBytes,
      static_cast<Rpp32u>(frameDims.width),
      static_cast<Rpp32u>(frameDims.height),
      colStandard,
      colorRange,
      rppCtx->handle,
      RPP_HIP_BACKEND);
  STD_TORCH_CHECK(
      status == RPP_SUCCESS,
      "Failed to convert NV12 to RGB (",
      modeName,
      "). Status: ",
      status);
  return dst;
}

UniqueRppContext getRppStreamContext(const StableDevice& device) {
  [[maybe_unused]] int deviceIndex = get_device_index(device);

  UniqueRppContext rppCtx = g_cached_rpp_ctxs.get(device);
  if (rppCtx) {
    return rppCtx;
  }

  // Create a new RPP context manually with custom deleter
  RppContext* ctx = new RppContext();
  ctx->stream = nullptr;
  ctx->handle = nullptr;
  int batchSize = 1;

  hipError_t err = hipStreamCreate(&ctx->stream);
  STD_TORCH_CHECK(
      err == hipSuccess,
      "Failed to create HIP stream: ",
      hipGetErrorString(err));

  rppStatus_t status = rppCreate(
      &ctx->handle,
      batchSize,
      0,  // numThreads = 0 for HIP backend
      ctx->stream,
      RPP_HIP_BACKEND);
  STD_TORCH_CHECK(
      status == rppStatusSuccess,
      "Failed to create RPP handle. Status: ",
      status);

  rppCtx = UniqueRppContext(ctx, RppContextDeleter());
  return rppCtx;
}

void returnRppStreamContextToCache(
    const StableDevice& device,
    UniqueRppContext rppCtx) {
  if (rppCtx) {
    g_cached_rpp_ctxs.add_if_cache_has_capacity(device, std::move(rppCtx));
  }
}

void validatePreAllocatedTensorShape(
    const std::optional<torch::stable::Tensor>& preAllocatedOutputTensor,
    const AVFrame& avFrame) {
  // Note that ROCm does not yet support transforms, so the only possible
  // frame dimensions are the raw decoded frame's dimensions.
  auto frameDims = FrameDims(avFrame.height, avFrame.width);

  if (preAllocatedOutputTensor.has_value()) {
    auto shape = preAllocatedOutputTensor.value().sizes();
    STD_TORCH_CHECK(
        (shape.size() == 3) && (shape[0] == frameDims.height) &&
            (shape[1] == frameDims.width) && (shape[2] == 3),
        "Expected tensor of shape ",
        frameDims.height,
        "x",
        frameDims.width,
        "x3, got shape with ",
        shape.size(),
        " dimensions");
  }
}

int get_device_index(const StableDevice& device) {
  // PyTorch uses int8_t as its StableDeviceIndex, but FFmpeg and HIP
  // libraries use int. So we use int, too.
  int deviceIndex = static_cast<int>(device.index());
  STD_TORCH_CHECK(
      deviceIndex >= -1 && deviceIndex < MAX_ROCM_GPUS,
      "Invalid device index = ",
      deviceIndex);

  if (deviceIndex == -1) {
    STD_TORCH_CHECK(
        hipGetDevice(&deviceIndex) == hipSuccess,
        "Failed to get current HIP device.");
  }
  return deviceIndex;
}

} // namespace facebook::torchcodec
