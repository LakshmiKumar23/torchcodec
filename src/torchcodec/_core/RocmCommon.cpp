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
  torch::Tensor dst;
  if (preAllocatedOutputTensor.has_value()) {
    dst = preAllocatedOutputTensor.value();
  } else {
    dst = allocateEmptyHWCTensor(frameDims, device);
  }

  // Verify the frame is in NV12 format (semi-planar YUV420)
  // rocDecode outputs frames in NV12 format directly.
  // Unlike CUDA, ROCm doesn't use FFmpeg's hardware frames context,
  // so we check avFrame->format directly.
  TORCH_CHECK(
      avFrame->format == AV_PIX_FMT_NV12,
      "convertNV12FrameToRGB expects NV12 format, but got format: ",
      av_get_pix_fmt_name(static_cast<AVPixelFormat>(avFrame->format)) ? 
          av_get_pix_fmt_name(static_cast<AVPixelFormat>(avFrame->format)) : "unknown",
      " (format code: ", avFrame->format, "). ",
      "Frames should be converted to NV12 before calling this function.");

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

  // NV12 format: Y plane followed by interleaved UV plane
  uint8_t* yuvData[2] = {avFrame->data[0], avFrame->data[1]};
  
  // Convert NV12 to RGB using HIP kernel
  // NV12 is semi-planar: Y plane (luma) + interleaved UV plane (chroma)
  // Use RGB24 (3 bytes per pixel) to match the 3-channel tensor
  Nv12ToColor24<RGB24>(
      yuvData[0],                                  // Y plane
      avFrame->linesize[0],                        // Y plane stride (pitch)
      static_cast<uint8_t*>(dst.data_ptr()),       // RGB output
      dst.stride(0),                               // RGB stride in bytes (for uint8, stride is already in bytes)
      frameDims.width,                             // Width
      frameDims.height,                            // Height
      frameDims.height,                            // v_pitch: number of Y rows (NOT UV stride!)
      avFrame->colorspace,                         // Color space (BT.601/BT.709/etc)
      rppStream);                                  // HIP stream
  
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
