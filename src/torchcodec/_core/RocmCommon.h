// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <hip/hip_runtime.h>
#include <rpp/rpp.h>
#include <torch/types.h>

#include "FFMPEGCommon.h"
#include "Frame.h"

#include "rocdecode_kernels/colorspace_kernels.h"

extern "C" {
#include <libavutil/pixdesc.h>
// Note: FFmpeg doesn't have native hwcontext_hip/rocm support
// Handle hardware acceleration using rocDecode
}

namespace facebook::torchcodec {

// PyTorch can handle up to 128 GPUs
constexpr int MAX_ROCM_GPUS = 128;

void initializeRocmContextWithPytorch(const torch::Device& device);

// ROCm context structure
struct RppContext {
  rppHandle_t handle;
  hipStream_t stream;
  unsigned int streamFlags;
};

// Custom deleter for RppContext
struct RppContextDeleter {
  void operator()(RppContext* ctx) const {
    if (ctx) {
      if (ctx->handle) {
        rppDestroy(ctx->handle);
      }
      delete ctx;
    }
  }
};

// Unique pointer type for RPP context
using UniqueRppContext = std::unique_ptr<RppContext, RppContextDeleter>;

torch::Tensor convertNV12FrameToRGB(
    UniqueAVFrame& avFrame,
    const torch::Device& device,
    const UniqueRppContext& rppCtx,
    hipStream_t hipStream,
    std::optional<torch::Tensor> preAllocatedOutputTensor = std::nullopt);

UniqueRppContext getRppStreamContext(const torch::Device& device);
void returnRppStreamContextToCache(
    const torch::Device& device,
    UniqueRppContext rppCtx);

void validatePreAllocatedTensorShape(
    const std::optional<torch::Tensor>& preAllocatedOutputTensor,
    const UniqueAVFrame& avFrame);

int getDeviceIndex(const torch::Device& device);

} // namespace facebook::torchcodec
