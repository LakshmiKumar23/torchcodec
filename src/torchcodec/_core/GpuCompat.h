// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

/* clang-format off */
// Note: [Compiling one source for both CUDA and HIP]
//
// torchcodec's GPU color-conversion kernels and stream/event helpers are shared
// verbatim between the CUDA (NVDEC) and ROCm (rocDecode) backends. Rather than
// maintaining a hipified copy of those sources, or running them through
// hipify-perl at build time, we compile the *same* files for both backends and
// alias the handful of CUDA runtime symbols they use onto their HIP
// equivalents here.
//
// The CUDA spelling is the canonical one throughout torchcodec. That is
// deliberate: it keeps the CUDA path byte-for-byte unchanged, so this header is
// invisible unless you're building for ROCm, and the shared sources cannot
// drift between the two backends.
//
// This is NOT a general-purpose CUDA -> HIP shim. It aliases only what
// torchcodec actually uses. When a shared source starts using a new runtime
// symbol, add it here explicitly. That keeps the ported surface small and
// reviewable, which is the whole point of not using hipify.
//
// The aliases are declared at global scope, matching where <cuda_runtime.h>
// declares the originals, so this header is a true drop-in replacement for it.
//
// USE_ROCM and USE_CUDA are set by CMakeLists.txt; the build system compiles
// exactly one GPU backend at a time, so they are mutually exclusive. USE_ROCM
// is required independently of this header: PyTorch hipifies its own
// aoti_torch/c/shim.h, so the ROCm wheel guards
// aoti_torch_get_current_cuda_stream (used by CUDACommon.h) with
// `#ifdef USE_ROCM` where the CUDA wheel guards it with `#ifdef USE_CUDA`.
// Do not define USE_CUDA on a ROCm build to try to reach it - the ROCm header
// does not test that macro.
/* clang-format on */

#if defined(USE_ROCM)

// The HIP headers refuse to compile unless a platform is selected. hipcc
// defines this itself when compiling HIP (-x hip), and CMake's hip::host
// target carries it as an INTERFACE definition, but a plain host C++ TU that
// merely includes this header gets neither. Define it ourselves so the header
// is self-contained, while still honouring an explicit platform choice.
#if !defined(__HIP_PLATFORM_AMD__) && !defined(__HIP_PLATFORM_NVIDIA__)
#define __HIP_PLATFORM_AMD__
#endif

// hip_runtime.h pulls in device-side machinery and is only valid under hipcc;
// host translation units want the API-only header.
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#else
#include <hip/hip_runtime_api.h>
#endif

// Note that hipError_t is [[nodiscard]] where cudaError_t is not, so call
// sites that intentionally drop a status (cleanup paths, destructors) must
// cast to (void) explicitly. That cast is a no-op for CUDA.
using cudaStream_t = hipStream_t;
using cudaEvent_t = hipEvent_t;
using cudaError_t = hipError_t;

constexpr hipError_t cudaSuccess = hipSuccess;
constexpr unsigned int cudaEventDisableTiming = hipEventDisableTiming;

inline cudaError_t cudaGetDevice(int* device) {
  return hipGetDevice(device);
}

inline const char* cudaGetErrorString(cudaError_t error) {
  return hipGetErrorString(error);
}

inline cudaError_t cudaEventCreateWithFlags(
    cudaEvent_t* event,
    unsigned int flags) {
  return hipEventCreateWithFlags(event, flags);
}

inline cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
  return hipEventRecord(event, stream);
}

inline cudaError_t cudaEventDestroy(cudaEvent_t event) {
  return hipEventDestroy(event);
}

inline cudaError_t cudaEventSynchronize(cudaEvent_t event) {
  return hipEventSynchronize(event);
}

inline cudaError_t cudaStreamWaitEvent(
    cudaStream_t stream,
    cudaEvent_t event,
    unsigned int flags) {
  return hipStreamWaitEvent(stream, event, flags);
}

#else

#include <cuda_runtime.h>

#endif // USE_ROCM
