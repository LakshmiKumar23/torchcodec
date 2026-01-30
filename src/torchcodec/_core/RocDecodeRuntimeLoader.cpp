// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifdef FBCODE_CAFFE2
// No need to do anything on fbcode. rocDecode is available there, we can take a
// hard dependency on it.
// The FBCODE_CAFFE2 macro is defined in the upstream fbcode build of torch, so
// we can rely on it, that's what torch does too.

namespace facebook::torchcodec {
bool loadRocDecodeLibrary() {
  return true;
}
} // namespace facebook::torchcodec
#else

#include "RocDecodeRuntimeLoader.h"

#include "rocdecode_include/rocdecode.h"
#include "rocdecode_include/rocparser.h"

#include <torch/types.h>
#include <cstdio>
#include <mutex>

#if defined(WIN64) || defined(_WIN64)
#include <windows.h>
typedef HMODULE tHandle;
#else
#include <dlfcn.h>
typedef void* tHandle;
#endif

namespace facebook::torchcodec {

/* clang-format off */
// This file defines the logic to load the rocDecode library **at runtime**,
// along with the corresponding rocDecode functions that we'll need.
//
// We do this because we *do not want* to link (statically or dynamically)
// against librocdecode.so: it is not always available on the users machine! If
// we were to link against librocdecode.so, that would mean that our
// libtorchcodec_coreN.so would try to look for it when loaded at import time.
// And if it's not on the users machine, that causes `import torchcodec` to
// fail.
//
// So, we don't link against librocdecode.so. But we still want to call its
// functions. We dynamically load function
// pointers at runtime using dlopen/dlsym.
//
// This follows the rocDecode documentation and examples:
// https://rocm.docs.amd.com/projects/rocDecode/en/latest
//
// At runtime, within the ROCm interface code we have a fallback mechanism
// to switch back to the CPU backend if any of the rocDecode functions are not
// available, or if librocdecode.so itself couldn't be found.

// Function types for rocDecode API (not pointer types - pointers added via bindFunction)
typedef rocDecStatus tRocDecCreateVideoParser(RocdecVideoParser*, RocdecParserParams*);
typedef rocDecStatus tRocDecParseVideoData(RocdecVideoParser, RocdecSourceDataPacket*);
typedef rocDecStatus tRocDecDestroyVideoParser(RocdecVideoParser);
typedef rocDecStatus tRocDecGetDecoderCaps(RocdecDecodeCaps*);
typedef rocDecStatus tRocDecCreateDecoder(rocDecDecoderHandle*, RocDecoderCreateInfo*);
typedef rocDecStatus tRocDecDestroyDecoder(rocDecDecoderHandle);
typedef rocDecStatus tRocDecDecodeFrame(rocDecDecoderHandle, RocdecPicParams*);
typedef rocDecStatus tRocDecGetVideoFrame(rocDecDecoderHandle, int, void**, unsigned int*, RocdecProcParams*);
/* clang-format on */

// Global function pointers - will be dynamically loaded
static tRocDecCreateVideoParser* dl_rocDecCreateVideoParser = nullptr;
static tRocDecParseVideoData* dl_rocDecParseVideoData = nullptr;
static tRocDecDestroyVideoParser* dl_rocDecDestroyVideoParser = nullptr;
static tRocDecGetDecoderCaps* dl_rocDecGetDecoderCaps = nullptr;
static tRocDecCreateDecoder* dl_rocDecCreateDecoder = nullptr;
static tRocDecDestroyDecoder* dl_rocDecDestroyDecoder = nullptr;
static tRocDecDecodeFrame* dl_rocDecDecodeFrame = nullptr;
static tRocDecGetVideoFrame* dl_rocDecGetVideoFrame = nullptr;

static tHandle g_rocdecode_handle = nullptr;
static std::mutex g_rocdecode_mutex;

bool isLoaded() {
  return (
      g_rocdecode_handle && dl_rocDecCreateVideoParser &&
      dl_rocDecParseVideoData && dl_rocDecDestroyVideoParser &&
      dl_rocDecGetDecoderCaps && dl_rocDecCreateDecoder &&
      dl_rocDecDestroyDecoder && dl_rocDecDecodeFrame &&
      dl_rocDecGetVideoFrame);
}

template <typename T>
T* bindFunction(const char* functionName) {
#if defined(WIN64) || defined(_WIN64)
  return reinterpret_cast<T*>(GetProcAddress(g_rocdecode_handle, functionName));
#else
  return reinterpret_cast<T*>(dlsym(g_rocdecode_handle, functionName));
#endif
}

bool _loadLibrary() {
  // Helper that just calls dlopen or equivalent on Windows.
#if defined(WIN64) || defined(_WIN64)
#ifdef UNICODE
  static LPCWSTR rocdecodeDll = L"rocdecode.dll";
#else
  static LPCSTR rocdecodeDll = "rocdecode.dll";
#endif
  g_rocdecode_handle = LoadLibrary(rocdecodeDll);
  if (g_rocdecode_handle == nullptr) {
    return false;
  }
#else
  g_rocdecode_handle = dlopen("librocdecode.so", RTLD_NOW);
  if (g_rocdecode_handle == nullptr) {
    g_rocdecode_handle = dlopen("librocdecode.so.1", RTLD_NOW);
  }
  if (g_rocdecode_handle == nullptr) {
    return false;
  }
#endif

  return true;
}

bool loadRocDecodeLibrary() {
  // Loads rocDecode library and all required function pointers.
  // Returns true on success, false on failure.
  std::lock_guard<std::mutex> lock(g_rocdecode_mutex);

  if (isLoaded()) {
    return true;
  }

  if (!_loadLibrary()) {
    return false;
  }

  // Load all function pointers. They'll be set to nullptr if not found.
  dl_rocDecCreateVideoParser =
      bindFunction<tRocDecCreateVideoParser>("rocDecCreateVideoParser");
  dl_rocDecParseVideoData =
      bindFunction<tRocDecParseVideoData>("rocDecParseVideoData");
  dl_rocDecDestroyVideoParser =
      bindFunction<tRocDecDestroyVideoParser>("rocDecDestroyVideoParser");
  dl_rocDecGetDecoderCaps =
      bindFunction<tRocDecGetDecoderCaps>("rocDecGetDecoderCaps");
  dl_rocDecCreateDecoder =
      bindFunction<tRocDecCreateDecoder>("rocDecCreateDecoder");
  dl_rocDecDestroyDecoder =
      bindFunction<tRocDecDestroyDecoder>("rocDecDestroyDecoder");
  dl_rocDecDecodeFrame =
      bindFunction<tRocDecDecodeFrame>("rocDecDecodeFrame");
  dl_rocDecGetVideoFrame =
      bindFunction<tRocDecGetVideoFrame>("rocDecGetVideoFrame");

  return isLoaded();
}

} // namespace facebook::torchcodec

extern "C" {

rocDecStatus rocDecCreateVideoParser(
    RocdecVideoParser* videoParser,
    RocdecParserParams* parserParams) {
  TORCH_CHECK(
      facebook::torchcodec::dl_rocDecCreateVideoParser,
      "rocDecCreateVideoParser called but rocDecode not loaded!");
  return facebook::torchcodec::dl_rocDecCreateVideoParser(
      videoParser, parserParams);
}

rocDecStatus rocDecParseVideoData(
    RocdecVideoParser videoParser,
    RocdecSourceDataPacket* packet) {
  TORCH_CHECK(
      facebook::torchcodec::dl_rocDecParseVideoData,
      "rocDecParseVideoData called but rocDecode not loaded!");
  return facebook::torchcodec::dl_rocDecParseVideoData(videoParser, packet);
}

rocDecStatus rocDecDestroyVideoParser(RocdecVideoParser videoParser) {
  TORCH_CHECK(
      facebook::torchcodec::dl_rocDecDestroyVideoParser,
      "rocDecDestroyVideoParser called but rocDecode not loaded!");
  return facebook::torchcodec::dl_rocDecDestroyVideoParser(videoParser);
}

rocDecStatus rocDecGetDecoderCaps(RocdecDecodeCaps* caps) {
  TORCH_CHECK(
      facebook::torchcodec::dl_rocDecGetDecoderCaps,
      "rocDecGetDecoderCaps called but rocDecode not loaded!");
  return facebook::torchcodec::dl_rocDecGetDecoderCaps(caps);
}

rocDecStatus rocDecCreateDecoder(
    rocDecDecoderHandle* decoderHandle,
    RocDecoderCreateInfo* createInfo) {
  TORCH_CHECK(
      facebook::torchcodec::dl_rocDecCreateDecoder,
      "rocDecCreateDecoder called but rocDecode not loaded!");
  return facebook::torchcodec::dl_rocDecCreateDecoder(decoderHandle, createInfo);
}

rocDecStatus rocDecDestroyDecoder(rocDecDecoderHandle decoderHandle) {
  TORCH_CHECK(
      facebook::torchcodec::dl_rocDecDestroyDecoder,
      "rocDecDestroyDecoder called but rocDecode not loaded!");
  return facebook::torchcodec::dl_rocDecDestroyDecoder(decoderHandle);
}

rocDecStatus rocDecDecodeFrame(
    rocDecDecoderHandle decoderHandle,
    RocdecPicParams* picParams) {
  TORCH_CHECK(
      facebook::torchcodec::dl_rocDecDecodeFrame,
      "rocDecDecodeFrame called but rocDecode not loaded!");
  return facebook::torchcodec::dl_rocDecDecodeFrame(decoderHandle, picParams);
}

rocDecStatus rocDecGetVideoFrame(
    rocDecDecoderHandle decoderHandle,
    int pictureIndex,
    void** framePtr,
    unsigned int* pitch,
    RocdecProcParams* procParams) {
  TORCH_CHECK(
      facebook::torchcodec::dl_rocDecGetVideoFrame,
      "rocDecGetVideoFrame called but rocDecode not loaded!");
  return facebook::torchcodec::dl_rocDecGetVideoFrame(
    decoderHandle, pictureIndex, framePtr, pitch, procParams);
}

} // extern "C"

#endif // FBCODE_CAFFE2
