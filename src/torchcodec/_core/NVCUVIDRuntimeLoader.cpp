// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifdef FBCODE_CAFFE2
#include "StableABICompat.h"

// No need to do anything on fbcode. NVCUVID is available there, we can take a
// hard dependency on it.
// The FBCODE_CAFFE2 macro is defined in the upstream fbcode build of torch, so
// we can rely on it, that's what torch does too.

namespace facebook::torchcodec {
bool load_nvcuvid_library() {
  return true;
}
} // namespace facebook::torchcodec
#elif !defined(USE_ROCM)

#include "NVCUVIDRuntimeLoader.h"
#include "StableABICompat.h"

#include "nvcuvid_include/cuviddec.h"
#include "nvcuvid_include/nvcuvid.h"

#include <cstdio>
#include <mutex>
#include "StableABICompat.h"

#if defined(WIN64) || defined(_WIN64)
#include <windows.h>
typedef HMODULE tHandle;
#else
#include <dlfcn.h>
typedef void* tHandle;
#endif

namespace facebook::torchcodec {

/* clang-format off */
// This file defines the logic to load the NVCUVID library **at runtime**,
// along with the corresponding NVCUVID functions that we'll need.
//
// We do this because we *do not want* to link (statically or dynamically)
// against libnvcuvid.so: it is not always available on the users machine! If we
// were to link against libnvcuvid.so, that would mean that our
// libtorchcodec_coreN.so would try to look for it when loaded at import time.
// And if it's not on the users machine, that causes `import torchcodec` to
// fail. Source: that's what we did, and we got user reports.
//
// So, we don't link against libnvcuvid.so. But we still want to call its
// functions. So here's how it's done, we'll use cuvidCreateVideoParser as an
// example, but it works the same for all. We are largely following the
// instructions from the NVCUVID docs:
// https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvdec-video-decoder-api-prog-guide/index.html#dynamic-loading-nvidia-components
//
// This:
// typedef CUresult CUDAAPI tcuvidCreateVideoParser(CUvideoparser*, CUVIDPARSERPARAMS*);
// defines tcuvidCreateVideoParser, which is the *type* of a *function*.
// We define such a function of that type just below with:
// static tcuvidCreateVideoParser* dl_cuvidCreateVideoParser = nullptr;
// "dl" is for "dynamically loaded. For now dl_cuvidCreateVideoParser is
// nullptr, but later it will be a proper function [pointer] that can be called
// with dl_cuvidCreateVideoParser(...);
//
// For that to happen we need to call load_nvcuvid_library(): in there, we first
// dlopen(libnvcuvid.so) which loads the .so somewhere in memory. Then we call
// dlsym(...), which binds dl_cuvidCreateVideoParser to its actual address: it
// literally sets the value of the dl_cuvidCreateVideoParser pointer to the
// address of the actual code section. If all went well, by now, we can safely
// call dl_cuvidCreateVideoParser(...);
// All of that happens at runtime *after* import time, when the first instance
// of the NVDEC CUDA interface is created, i.e. only when the user requests
// CUDA decoding.
//
// At the bottom of this file we have an `extern "C"` section with function
// definitions like:
//
// CUresult CUDAAPI cuvidCreateVideoParser(
//  CUvideoparser* videoParser,
//  CUVIDPARSERPARAMS* parserParams)  {...}
//
// These are the actual functions that are compiled against and called by the
// NVDEC CUDA interface code. Crucially, these functions signature match exactly
// the NVCUVID functions (as defined in cuviddec.h). Inside of
// cuvidCreateVideoParser(...) we simply call the dl_cuvidCreateVideoParser
// function [pointer] that we dynamically loaded earlier.
//
// At runtime, within the NVDEC CUDA interface code we have a fallback mechanism
// to switch back to the CPU backend if any of the NVCUVID functions are not
// available, or if libnvcuvid.so itself couldn't be found. This is what FFmpeg
// does too.


// Function pointers types
typedef CUresult CUDAAPI tcuvidCreateVideoParser(CUvideoparser*, CUVIDPARSERPARAMS*);
typedef CUresult CUDAAPI tcuvidParseVideoData(CUvideoparser, CUVIDSOURCEDATAPACKET*);
typedef CUresult CUDAAPI tcuvidDestroyVideoParser(CUvideoparser);
typedef CUresult CUDAAPI tcuvidGetDecoderCaps(CUVIDDECODECAPS*);
typedef CUresult CUDAAPI tcuvidCreateDecoder(CUvideodecoder*, CUVIDDECODECREATEINFO*);
typedef CUresult CUDAAPI tcuvidDestroyDecoder(CUvideodecoder);
typedef CUresult CUDAAPI tcuvidDecodePicture(CUvideodecoder, CUVIDPICPARAMS*);
typedef CUresult CUDAAPI tcuvidMapVideoFrame(CUvideodecoder, int, unsigned int*, unsigned int*, CUVIDPROCPARAMS*);
typedef CUresult CUDAAPI tcuvidUnmapVideoFrame(CUvideodecoder, unsigned int);
typedef CUresult CUDAAPI tcuvidMapVideoFrame64(CUvideodecoder, int, unsigned long long*, unsigned int*, CUVIDPROCPARAMS*);
typedef CUresult CUDAAPI tcuvidUnmapVideoFrame64(CUvideodecoder, unsigned long long);
/* clang-format on */

// Global function pointers - will be dynamically loaded
static tcuvidCreateVideoParser* dl_cuvidCreateVideoParser = nullptr;
static tcuvidParseVideoData* dl_cuvidParseVideoData = nullptr;
static tcuvidDestroyVideoParser* dl_cuvidDestroyVideoParser = nullptr;
static tcuvidGetDecoderCaps* dl_cuvidGetDecoderCaps = nullptr;
static tcuvidCreateDecoder* dl_cuvidCreateDecoder = nullptr;
static tcuvidDestroyDecoder* dl_cuvidDestroyDecoder = nullptr;
static tcuvidDecodePicture* dl_cuvidDecodePicture = nullptr;
static tcuvidMapVideoFrame* dl_cuvidMapVideoFrame = nullptr;
static tcuvidUnmapVideoFrame* dl_cuvidUnmapVideoFrame = nullptr;
static tcuvidMapVideoFrame64* dl_cuvidMapVideoFrame64 = nullptr;
static tcuvidUnmapVideoFrame64* dl_cuvidUnmapVideoFrame64 = nullptr;

static tHandle g_nvcuvid_handle = nullptr;
static std::mutex g_nvcuvid_mutex;

bool is_loaded() {
  return (
      g_nvcuvid_handle && dl_cuvidCreateVideoParser && dl_cuvidParseVideoData &&
      dl_cuvidDestroyVideoParser && dl_cuvidGetDecoderCaps &&
      dl_cuvidCreateDecoder && dl_cuvidDestroyDecoder &&
      dl_cuvidDecodePicture && dl_cuvidMapVideoFrame &&
      dl_cuvidUnmapVideoFrame && dl_cuvidMapVideoFrame64 &&
      dl_cuvidUnmapVideoFrame64);
}

template <typename T>
T* bind_function(const char* function_name) {
#if defined(WIN64) || defined(_WIN64)
  return reinterpret_cast<T*>(GetProcAddress(g_nvcuvid_handle, function_name));
#else
  return reinterpret_cast<T*>(dlsym(g_nvcuvid_handle, function_name));
#endif
}

bool load_library() {
  // Helper that just calls dlopen or equivalent on Windows. In a separate
  // function because of the #ifdef uglyness.
#if defined(WIN64) || defined(_WIN64)
#ifdef UNICODE
  static LPCWSTR nvcuvidDll = L"nvcuvid.dll";
#else
  static LPCSTR nvcuvidDll = "nvcuvid.dll";
#endif
  g_nvcuvid_handle = LoadLibrary(nvcuvidDll);
  if (g_nvcuvid_handle == nullptr) {
    return false;
  }
#else
  g_nvcuvid_handle = dlopen("libnvcuvid.so", RTLD_NOW);
  if (g_nvcuvid_handle == nullptr) {
    g_nvcuvid_handle = dlopen("libnvcuvid.so.1", RTLD_NOW);
  }
  if (g_nvcuvid_handle == nullptr) {
    return false;
  }
#endif

  return true;
}

bool load_nvcuvid_library() {
  // Loads NVCUVID library and all required function pointers.
  // Returns true on success, false on failure.
  std::lock_guard<std::mutex> lock(g_nvcuvid_mutex);

  if (is_loaded()) {
    return true;
  }

  if (!load_library()) {
    return false;
  }

  // Load all function pointers. They'll be set to nullptr if not found.
  dl_cuvidCreateVideoParser =
      bind_function<tcuvidCreateVideoParser>("cuvidCreateVideoParser");
  dl_cuvidParseVideoData =
      bind_function<tcuvidParseVideoData>("cuvidParseVideoData");
  dl_cuvidDestroyVideoParser =
      bind_function<tcuvidDestroyVideoParser>("cuvidDestroyVideoParser");
  dl_cuvidGetDecoderCaps =
      bind_function<tcuvidGetDecoderCaps>("cuvidGetDecoderCaps");
  dl_cuvidCreateDecoder =
      bind_function<tcuvidCreateDecoder>("cuvidCreateDecoder");
  dl_cuvidDestroyDecoder =
      bind_function<tcuvidDestroyDecoder>("cuvidDestroyDecoder");
  dl_cuvidDecodePicture =
      bind_function<tcuvidDecodePicture>("cuvidDecodePicture");
  dl_cuvidMapVideoFrame =
      bind_function<tcuvidMapVideoFrame>("cuvidMapVideoFrame");
  dl_cuvidUnmapVideoFrame =
      bind_function<tcuvidUnmapVideoFrame>("cuvidUnmapVideoFrame");
  dl_cuvidMapVideoFrame64 =
      bind_function<tcuvidMapVideoFrame64>("cuvidMapVideoFrame64");
  dl_cuvidUnmapVideoFrame64 =
      bind_function<tcuvidUnmapVideoFrame64>("cuvidUnmapVideoFrame64");

  return is_loaded();
}

} // namespace facebook::torchcodec

extern "C" {

CUresult CUDAAPI cuvidCreateVideoParser(
    CUvideoparser* video_parser,
    CUVIDPARSERPARAMS* parser_params) {
  STD_TORCH_CHECK(
      facebook::torchcodec::dl_cuvidCreateVideoParser,
      "cuvidCreateVideoParser called but NVCUVID not loaded!");
  return facebook::torchcodec::dl_cuvidCreateVideoParser(
      video_parser, parser_params);
}

CUresult CUDAAPI cuvidParseVideoData(
    CUvideoparser video_parser,
    CUVIDSOURCEDATAPACKET* cuvid_packet) {
  STD_TORCH_CHECK(
      facebook::torchcodec::dl_cuvidParseVideoData,
      "cuvidParseVideoData called but NVCUVID not loaded!");
  return facebook::torchcodec::dl_cuvidParseVideoData(
      video_parser, cuvid_packet);
}

CUresult CUDAAPI cuvidDestroyVideoParser(CUvideoparser video_parser) {
  STD_TORCH_CHECK(
      facebook::torchcodec::dl_cuvidDestroyVideoParser,
      "cuvidDestroyVideoParser called but NVCUVID not loaded!");
  return facebook::torchcodec::dl_cuvidDestroyVideoParser(video_parser);
}

CUresult CUDAAPI cuvidGetDecoderCaps(CUVIDDECODECAPS* caps) {
  STD_TORCH_CHECK(
      facebook::torchcodec::dl_cuvidGetDecoderCaps,
      "cuvidGetDecoderCaps called but NVCUVID not loaded!");
  return facebook::torchcodec::dl_cuvidGetDecoderCaps(caps);
}

CUresult CUDAAPI cuvidCreateDecoder(
    CUvideodecoder* decoder,
    CUVIDDECODECREATEINFO* decoder_params) {
  STD_TORCH_CHECK(
      facebook::torchcodec::dl_cuvidCreateDecoder,
      "cuvidCreateDecoder called but NVCUVID not loaded!");
  return facebook::torchcodec::dl_cuvidCreateDecoder(decoder, decoder_params);
}

CUresult CUDAAPI cuvidDestroyDecoder(CUvideodecoder decoder) {
  STD_TORCH_CHECK(
      facebook::torchcodec::dl_cuvidDestroyDecoder,
      "cuvidDestroyDecoder called but NVCUVID not loaded!");
  return facebook::torchcodec::dl_cuvidDestroyDecoder(decoder);
}

CUresult CUDAAPI
cuvidDecodePicture(CUvideodecoder decoder, CUVIDPICPARAMS* pic_params) {
  STD_TORCH_CHECK(
      facebook::torchcodec::dl_cuvidDecodePicture,
      "cuvidDecodePicture called but NVCUVID not loaded!");
  return facebook::torchcodec::dl_cuvidDecodePicture(decoder, pic_params);
}

#if !defined(__CUVID_DEVPTR64) || defined(__CUVID_INTERNAL)
// We need to protect the definition of the 32bit versions under the above
// conditions (see cuviddec.h). Defining them unconditionally would cause
// conflict compilation errors when cuviddec.h redefines those to the 64bit
// versions.
CUresult CUDAAPI cuvidMapVideoFrame(
    CUvideodecoder decoder,
    int pix_index,
    unsigned int* frame_ptr,
    unsigned int* pitch,
    CUVIDPROCPARAMS* proc_params) {
  STD_TORCH_CHECK(
      facebook::torchcodec::dl_cuvidMapVideoFrame,
      "cuvidMapVideoFrame called but NVCUVID not loaded!");
  return facebook::torchcodec::dl_cuvidMapVideoFrame(
      decoder, pix_index, frame_ptr, pitch, proc_params);
}

CUresult CUDAAPI
cuvidUnmapVideoFrame(CUvideodecoder decoder, unsigned int frame_ptr) {
  STD_TORCH_CHECK(
      facebook::torchcodec::dl_cuvidUnmapVideoFrame,
      "cuvidUnmapVideoFrame called but NVCUVID not loaded!");
  return facebook::torchcodec::dl_cuvidUnmapVideoFrame(decoder, frame_ptr);
}
#endif

CUresult CUDAAPI cuvidMapVideoFrame64(
    CUvideodecoder decoder,
    int pix_index,
    unsigned long long* frame_ptr,
    unsigned int* pitch,
    CUVIDPROCPARAMS* proc_params) {
  STD_TORCH_CHECK(
      facebook::torchcodec::dl_cuvidMapVideoFrame64,
      "cuvidMapVideoFrame64 called but NVCUVID not loaded!");
  return facebook::torchcodec::dl_cuvidMapVideoFrame64(
      decoder, pix_index, frame_ptr, pitch, proc_params);
}

CUresult CUDAAPI
cuvidUnmapVideoFrame64(CUvideodecoder decoder, unsigned long long frame_ptr) {
  STD_TORCH_CHECK(
      facebook::torchcodec::dl_cuvidUnmapVideoFrame64,
      "cuvidUnmapVideoFrame64 called but NVCUVID not loaded!");
  return facebook::torchcodec::dl_cuvidUnmapVideoFrame64(decoder, frame_ptr);
}

} // extern "C"

#endif // FBCODE_CAFFE2 / !USE_ROCM

#if defined(USE_ROCM)

#include "NVCUVIDRuntimeLoader.h"
#include "NvcuvidCompat.h"
#include "StableABICompat.h"

#include <dlfcn.h>
#include <mutex>

namespace facebook::torchcodec {

/* clang-format off */
// [Loading rocDecode at runtime]
//
// This is the ROCm counterpart of the NVCUVID loader above, and it exists for
// exactly the same reason, restated for AMD: librocdecode.so.1 is packaged
// separately from the HIP runtime, so a machine that has a torchcodec wheel and
// a working GPU does not necessarily have it installed. If
// libtorchcodec_coreN.so carried a link-time dependency on it, ld.so would have
// to resolve that when the extension module is loaded, and `import torchcodec`
// itself would fail on such a machine - not the GPU decode call, the import,
// which would also take CPU decoding down with it. That is the exact failure
// mode the NVCUVID loader was written to avoid after user reports, and linking
// rocDecode would simply reintroduce it on the AMD side. Lazy binding is not a
// way out: RTLD_LAZY defers *symbol* resolution, but a DT_NEEDED object still
// has to be found at load time.
//
// So the ROCm build does not link librocdecode either. We dlopen() it on first
// use, bind the entry points we need, and load_nvcuvid_library() returns false
// if any of that fails. That false is the whole point: the caller in
// BetaCudaDeviceInterface treats it as "no hardware decoder here" and falls
// back to CPU decoding, so a missing rocDecode costs the user speed rather than
// the ability to import the library at all.
//
// The mechanism is the same as the CUDA side (see the long note above), applied
// one layer lower. There, the dlsym'd pointers are called by extern "C" cuvid*
// definitions in this file. Here, the dlsym'd pointers are called by extern "C"
// rocDec* definitions in this file, and the cuvid* entry points that the shared
// NVDEC code actually compiles against are defined on top of *those* in the
// USE_ROCM block at the end of BetaCudaDeviceInterface.cpp. Splitting it that
// way keeps the two questions apart: this file answers "how do we call rocDecode
// without linking it", and that one answers "how does NVCUVID map onto
// rocDecode".
//
// Only the eight entry points the NVDEC code reaches are bound. rocDecode's
// remaining public functions (rocDecGetDecodeStatus, rocDecReconfigureDecoder,
// rocDecGetErrorName) are deliberately absent: binding a symbol we never call
// would only add a way for the load to fail for no reason.

// Function pointer types
typedef rocDecStatus ROCDECAPI trocDecCreateVideoParser(RocdecVideoParser*, RocdecParserParams*);
typedef rocDecStatus ROCDECAPI trocDecParseVideoData(RocdecVideoParser, RocdecSourceDataPacket*);
typedef rocDecStatus ROCDECAPI trocDecDestroyVideoParser(RocdecVideoParser);
typedef rocDecStatus ROCDECAPI trocDecGetDecoderCaps(RocdecDecodeCaps*);
typedef rocDecStatus ROCDECAPI trocDecCreateDecoder(rocDecDecoderHandle*, RocDecoderCreateInfo*);
typedef rocDecStatus ROCDECAPI trocDecDestroyDecoder(rocDecDecoderHandle);
typedef rocDecStatus ROCDECAPI trocDecDecodeFrame(rocDecDecoderHandle, RocdecPicParams*);
typedef rocDecStatus ROCDECAPI trocDecGetVideoFrame(rocDecDecoderHandle, int, void*[3], uint32_t*, RocdecProcParams*);
/* clang-format on */

// Global function pointers - will be dynamically loaded
static trocDecCreateVideoParser* dl_rocDecCreateVideoParser = nullptr;
static trocDecParseVideoData* dl_rocDecParseVideoData = nullptr;
static trocDecDestroyVideoParser* dl_rocDecDestroyVideoParser = nullptr;
static trocDecGetDecoderCaps* dl_rocDecGetDecoderCaps = nullptr;
static trocDecCreateDecoder* dl_rocDecCreateDecoder = nullptr;
static trocDecDestroyDecoder* dl_rocDecDestroyDecoder = nullptr;
static trocDecDecodeFrame* dl_rocDecDecodeFrame = nullptr;
static trocDecGetVideoFrame* dl_rocDecGetVideoFrame = nullptr;

static void* g_rocdecode_handle = nullptr;
static std::mutex g_rocdecode_mutex;

bool is_loaded() {
  return (
      g_rocdecode_handle && dl_rocDecCreateVideoParser &&
      dl_rocDecParseVideoData && dl_rocDecDestroyVideoParser &&
      dl_rocDecGetDecoderCaps && dl_rocDecCreateDecoder &&
      dl_rocDecDestroyDecoder && dl_rocDecDecodeFrame &&
      dl_rocDecGetVideoFrame);
}

template <typename T>
T* bind_function(const char* function_name) {
  return reinterpret_cast<T*>(dlsym(g_rocdecode_handle, function_name));
}

bool load_library() {
  // librocdecode.so.1 is the soname, which is what a runtime-only install
  // ships. The unversioned name only exists when the development package is
  // present, so it is the fallback rather than the first choice.
  g_rocdecode_handle = dlopen("librocdecode.so.1", RTLD_NOW);
  if (g_rocdecode_handle == nullptr) {
    g_rocdecode_handle = dlopen("librocdecode.so", RTLD_NOW);
  }
  return g_rocdecode_handle != nullptr;
}

bool load_nvcuvid_library() {
  // Loads rocDecode and all required function pointers.
  // Returns true on success, false on failure.
  //
  // The NVCUVID name is kept because this is the hook the shared NVDEC code
  // calls. On ROCm there is no NVCUVID to load, and this loads its stand-in.
  std::lock_guard<std::mutex> lock(g_rocdecode_mutex);

  if (is_loaded()) {
    return true;
  }

  if (!load_library()) {
    return false;
  }

  // Load all function pointers. They'll be set to nullptr if not found.
  dl_rocDecCreateVideoParser =
      bind_function<trocDecCreateVideoParser>("rocDecCreateVideoParser");
  dl_rocDecParseVideoData =
      bind_function<trocDecParseVideoData>("rocDecParseVideoData");
  dl_rocDecDestroyVideoParser =
      bind_function<trocDecDestroyVideoParser>("rocDecDestroyVideoParser");
  dl_rocDecGetDecoderCaps =
      bind_function<trocDecGetDecoderCaps>("rocDecGetDecoderCaps");
  dl_rocDecCreateDecoder =
      bind_function<trocDecCreateDecoder>("rocDecCreateDecoder");
  dl_rocDecDestroyDecoder =
      bind_function<trocDecDestroyDecoder>("rocDecDestroyDecoder");
  dl_rocDecDecodeFrame = bind_function<trocDecDecodeFrame>("rocDecDecodeFrame");
  dl_rocDecGetVideoFrame =
      bind_function<trocDecGetVideoFrame>("rocDecGetVideoFrame");

  return is_loaded();
}

} // namespace facebook::torchcodec

extern "C" {

rocDecStatus ROCDECAPI rocDecCreateVideoParser(
    RocdecVideoParser* parser_handle,
    RocdecParserParams* params) {
  STD_TORCH_CHECK(
      facebook::torchcodec::dl_rocDecCreateVideoParser,
      "rocDecCreateVideoParser called but rocDecode not loaded!");
  return facebook::torchcodec::dl_rocDecCreateVideoParser(
      parser_handle, params);
}

rocDecStatus ROCDECAPI rocDecParseVideoData(
    RocdecVideoParser parser_handle,
    RocdecSourceDataPacket* packet) {
  STD_TORCH_CHECK(
      facebook::torchcodec::dl_rocDecParseVideoData,
      "rocDecParseVideoData called but rocDecode not loaded!");
  return facebook::torchcodec::dl_rocDecParseVideoData(parser_handle, packet);
}

rocDecStatus ROCDECAPI
rocDecDestroyVideoParser(RocdecVideoParser parser_handle) {
  STD_TORCH_CHECK(
      facebook::torchcodec::dl_rocDecDestroyVideoParser,
      "rocDecDestroyVideoParser called but rocDecode not loaded!");
  return facebook::torchcodec::dl_rocDecDestroyVideoParser(parser_handle);
}

rocDecStatus ROCDECAPI rocDecGetDecoderCaps(RocdecDecodeCaps* decode_caps) {
  STD_TORCH_CHECK(
      facebook::torchcodec::dl_rocDecGetDecoderCaps,
      "rocDecGetDecoderCaps called but rocDecode not loaded!");
  return facebook::torchcodec::dl_rocDecGetDecoderCaps(decode_caps);
}

rocDecStatus ROCDECAPI rocDecCreateDecoder(
    rocDecDecoderHandle* decoder_handle,
    RocDecoderCreateInfo* decoder_create_info) {
  STD_TORCH_CHECK(
      facebook::torchcodec::dl_rocDecCreateDecoder,
      "rocDecCreateDecoder called but rocDecode not loaded!");
  return facebook::torchcodec::dl_rocDecCreateDecoder(
      decoder_handle, decoder_create_info);
}

rocDecStatus ROCDECAPI
rocDecDestroyDecoder(rocDecDecoderHandle decoder_handle) {
  STD_TORCH_CHECK(
      facebook::torchcodec::dl_rocDecDestroyDecoder,
      "rocDecDestroyDecoder called but rocDecode not loaded!");
  return facebook::torchcodec::dl_rocDecDestroyDecoder(decoder_handle);
}

rocDecStatus ROCDECAPI rocDecDecodeFrame(
    rocDecDecoderHandle decoder_handle,
    RocdecPicParams* pic_params) {
  STD_TORCH_CHECK(
      facebook::torchcodec::dl_rocDecDecodeFrame,
      "rocDecDecodeFrame called but rocDecode not loaded!");
  return facebook::torchcodec::dl_rocDecDecodeFrame(decoder_handle, pic_params);
}

rocDecStatus ROCDECAPI rocDecGetVideoFrame(
    rocDecDecoderHandle decoder_handle,
    int pic_idx,
    void* dev_mem_ptr[3],
    uint32_t* horizontal_pitch,
    RocdecProcParams* vid_postproc_params) {
  STD_TORCH_CHECK(
      facebook::torchcodec::dl_rocDecGetVideoFrame,
      "rocDecGetVideoFrame called but rocDecode not loaded!");
  return facebook::torchcodec::dl_rocDecGetVideoFrame(
      decoder_handle,
      pic_idx,
      dev_mem_ptr,
      horizontal_pitch,
      vid_postproc_params);
}

} // extern "C"

#endif // USE_ROCM
