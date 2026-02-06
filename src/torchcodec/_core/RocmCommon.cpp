// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "RocmCommon.h"
#include "Cache.h" // for PerGpuCache

#include <hip/hip_runtime.h>
#include <rpp/rpp.h>

namespace facebook::torchcodec {

namespace {

// Set to -1 to have an infinitely sized cache. Set it to 0 to disable caching.
// Set to a positive number to have a cache of that size.
const int MAX_CONTEXTS_PER_GPU_IN_CACHE = -1;

PerGpuCache<RppContext, RppContextDeleter> g_cached_rpp_ctxs(
    MAX_ROCM_GPUS,
    MAX_CONTEXTS_PER_GPU_IN_CACHE);

} // namespace

void initializeRocmContextWithPytorch(const torch::Device& device) {
  // It is important for pytorch itself to create the HIP context. If ffmpeg
  // creates the context it may not be compatible with pytorch.
  // This is a dummy tensor to initialize the HIP context.
  torch::Tensor dummyTensorForHipInitialization = torch::zeros(
      {1}, torch::TensorOptions().dtype(torch::kUInt8).device(device));
}

torch::Tensor convertNV12FrameToRGB(
    UniqueAVFrame& avFrame,
    const torch::Device& device,
    const UniqueRppContext& rppCtx,
    hipStream_t hipStream,
    std::optional<torch::Tensor> preAllocatedOutputTensor) {
  auto frameDims = FrameDims(avFrame->height, avFrame->width);
  
  // Check bit depth first to allocate the correct tensor type
  bool is8bit = (avFrame->format == AV_PIX_FMT_NV12);
  bool is10bit = (avFrame->format == AV_PIX_FMT_P010LE || avFrame->format == AV_PIX_FMT_P010BE ||
                  avFrame->format == AV_PIX_FMT_P016LE || avFrame->format == AV_PIX_FMT_P016BE);
  
  TORCH_CHECK(
      is8bit || is10bit,
      "convertNV12FrameToRGB expects NV12, P010, or P016 format, but got format: ",
      av_get_pix_fmt_name(static_cast<AVPixelFormat>(avFrame->format)) ? 
          av_get_pix_fmt_name(static_cast<AVPixelFormat>(avFrame->format)) : "unknown",
      " (format code: ", avFrame->format, ")");
  
  torch::Tensor dst;
  if (preAllocatedOutputTensor.has_value()) {
    dst = preAllocatedOutputTensor.value();
  } else {
    // Allocate uint8 tensor for 8-bit, uint16 tensor for 10-bit
    if (is8bit) {
      dst = allocateEmptyHWCTensor(frameDims, device);  // uint8
    } else {
      // Allocate uint16 tensor for 10-bit video (RGB48)
      dst = torch::empty(
          {frameDims.height, frameDims.width, 3},
          torch::TensorOptions().dtype(torch::kUInt16).device(device));
    }
  }

  // We need to make sure rocDecode has finished decoding a frame before
  // color-converting it with HIP kernels.
  // So we make the RPP stream wait for rocDecode to finish.
  hipStream_t rppStream = rppCtx->stream;
  
  hipEvent_t rocdecodeDoneEvent;
  hipError_t err = hipEventCreate(&rocdecodeDoneEvent);
  TORCH_CHECK(
      err == hipSuccess,
      "hipEventCreate failed: ",
      hipGetErrorString(err));
  
  err = hipEventRecord(rocdecodeDoneEvent, hipStream);
  TORCH_CHECK(
      err == hipSuccess,
      "hipEventRecord failed: ",
      hipGetErrorString(err));
  
  err = hipStreamWaitEvent(rppStream, rocdecodeDoneEvent, 0);
  TORCH_CHECK(
      err == hipSuccess,
      "hipStreamWaitEvent failed: ",
      hipGetErrorString(err));
  
  err = hipEventDestroy(rocdecodeDoneEvent);
  TORCH_CHECK(
      err == hipSuccess,
      "hipEventDestroy failed: ",
      hipGetErrorString(err));

  err = hipStreamGetFlags(rppCtx->stream, &rppCtx->streamFlags);
  TORCH_CHECK(
      err == hipSuccess,
      "hipStreamGetFlags failed: ",
      hipGetErrorString(err));

  // NV12/P010 format: Y plane followed by interleaved UV plane
  uint8_t* yuvData[2] = {avFrame->data[0], avFrame->data[1]};
  
  // Convert to RGB using appropriate HIP kernel based on bit depth
  // NV12 is semi-planar 8-bit: Y plane (luma) + interleaved UV plane (chroma)
  // P010 is semi-planar 10-bit stored as 16-bit (right-aligned): Y plane + interleaved UV plane
  if (is8bit) {
    // 8-bit: Use Nv12ToColor24 (NV12 → RGB24, uint8)
    // bytesPerPixel = 1 for uint8
    Nv12ToColor24<RGB24>(
        yuvData[0],                                  // Y plane
        avFrame->linesize[0],                        // Y plane stride (pitch in bytes)
        static_cast<uint8_t*>(dst.data_ptr()),       // RGB output (uint8)
        dst.stride(0),                               // RGB stride in bytes (width * 3 * 1)
        frameDims.width,                             // Width
        frameDims.height,                            // Height
        frameDims.height,                            // v_pitch: number of Y rows
        avFrame->colorspace,                         // Color space (BT.601/BT.709/etc)
        avFrame->color_range,                        // Color range (studio/full)
        rppStream);                                  // HIP stream
  } else {
    // 10-bit: Use P016ToColor48 (P016 → RGB48, uint16)
    // This preserves 10-bit precision in 16-bit RGB output
    // rocDecode outputs P016 format (10-bit stored in 16-bit per component)
    // P016ToColor48 kernel converts directly to RGB48 (uint16 output)
    // bytesPerPixel = 2 for uint16
    // dst.stride(0) is in elements for uint16 tensor, need to multiply by 2 for bytes
    int rgbPitchInBytes = dst.stride(0) * 2;  // Convert element stride to byte stride
    
    P016ToColor48<RGB48>(
        yuvData[0],                                  // Y plane (16-bit per pixel, P016 format)
        avFrame->linesize[0],                        // Y plane stride (pitch in bytes)
        static_cast<uint8_t*>(dst.data_ptr()),       // RGB output (uint16 cast to uint8*)
        rgbPitchInBytes,                             // RGB stride in bytes (width * 3 * 2)
        frameDims.width,                             // Width
        frameDims.height,                            // Height
        frameDims.height,                            // v_pitch: number of Y rows
        avFrame->colorspace,                         // Color space
        avFrame->color_range,                        // Color range (studio/full)
        rppStream);                                  // HIP stream
  }
  
  return dst;
}

UniqueRppContext getRppStreamContext(const torch::Device& device) {
  int deviceIndex = getDeviceIndex(device);

  UniqueRppContext rppCtx = g_cached_rpp_ctxs.get(device);
  if (rppCtx) {
    // Clear any prior HIP errors before reusing cached context
    hipError_t priorErr = hipGetLastError();
    
    // Synchronize the RPP stream to ensure it's in a clean state
    hipError_t syncErr = hipStreamSynchronize(rppCtx->stream);
    if (syncErr != hipSuccess) {
      // Don't use this corrupted context, create a new one
      rppCtx.reset();
    } else {
      return rppCtx;
    }
  }

  // Create a new RPP context manually with custom deleter
  RppContext* ctx = new RppContext();
  
  // Initialize RPP handle
  rppStatus_t status = rppCreate(&ctx->handle, 1);  // batch size = 1
  TORCH_CHECK(
      status == rppStatusSuccess,
      "Failed to create RPP handle. Status: ",
      status);
  
  // Wrap in unique_ptr with custom deleter
  rppCtx = UniqueRppContext(ctx, RppContextDeleter());

  // Get current HIP stream
  hipError_t err = hipStreamCreate(&rppCtx->stream);
  TORCH_CHECK(
      err == hipSuccess,
      "Failed to create HIP stream: ",
      hipGetErrorString(err));

  return rppCtx;
}

void returnRppStreamContextToCache(
    const torch::Device& device,
    UniqueRppContext rppCtx) {
  if (rppCtx) {
    g_cached_rpp_ctxs.addIfCacheHasCapacity(device, std::move(rppCtx));
  }
}

void validatePreAllocatedTensorShape(
    const std::optional<torch::Tensor>& preAllocatedOutputTensor,
    const UniqueAVFrame& avFrame) {
  // Note that ROCm does not yet support transforms, so the only possible
  // frame dimensions are the raw decoded frame's dimensions.
  auto frameDims = FrameDims(avFrame->height, avFrame->width);

  if (preAllocatedOutputTensor.has_value()) {
    auto shape = preAllocatedOutputTensor.value().sizes();
    TORCH_CHECK(
        (shape.size() == 3) && (shape[0] == frameDims.height) &&
            (shape[1] == frameDims.width) && (shape[2] == 3),
        "Expected tensor of shape ",
        frameDims.height,
        "x",
        frameDims.width,
        "x3, got ",
        shape);
  }
}

int getDeviceIndex(const torch::Device& device) {
  // PyTorch uses int8_t as its torch::DeviceIndex, but FFmpeg and HIP
  // libraries use int. So we use int, too.
  int deviceIndex = static_cast<int>(device.index());
  TORCH_CHECK(
      deviceIndex >= -1 && deviceIndex < MAX_ROCM_GPUS,
      "Invalid device index = ",
      deviceIndex);

  if (deviceIndex == -1) {
    TORCH_CHECK(
        hipGetDevice(&deviceIndex) == hipSuccess,
        "Failed to get current HIP device.");
  }
  return deviceIndex;
}

} // namespace facebook::torchcodec
