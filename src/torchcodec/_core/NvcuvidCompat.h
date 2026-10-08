// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>

/* clang-format off */
// Note: [Compiling BetaCudaDeviceInterface for both NVDEC and rocDecode]
//
// This is the NVCUVID counterpart of GpuCompat.h, and it follows the same rule:
// the CUDA spelling is the canonical one throughout torchcodec, and the ROCm
// build aliases it onto the equivalent rocDecode type. BetaCudaDeviceInterface
// is written against NVCUVID, exactly as it always was; BetaRocmDeviceInterface
// derives from it and overrides map_frame(), the one place the two APIs
// genuinely diverge. This header is what lets the shared half of that class
// compile on a machine with no CUDA toolkit.
//
// AMD clearly modelled rocDecode on NVCUVID, so most of these aliases are
// between structs that are field-for-field identical in name, order and type:
//
// - CUVIDEOFORMAT / RocdecVideoFormat match exactly, including the nested
//   anonymous frame_rate, display_area, display_aspect_ratio and
//   video_signal_description structs and the video_format:3 /
//   video_full_range_flag:1 bitfield. ROCm only *appends* a reconfig_options
//   field.
//
// - CUVIDPARSERDISPINFO / RocdecParserDispInfo match except for the name of
//   the last field: timestamp vs pts. Same for CUVIDSOURCEDATAPACKET /
//   RocdecSourceDataPacket (whose flags and payload_size are also narrower on
//   ROCm, which doesn't matter: we only ever assign small values to them).
//   The get_timestamp()/set_timestamp() helpers below are the only place that
//   rename shows up.
//
// - CUVIDPICPARAMS / RocdecPicParams have nothing in common internally
//   (rocDecode's is VA-API-shaped). A picture's parameters come from the
//   parser callback and go straight back into cuvidDecodePicture() /
//   rocDecDecodeFrame(), so for the most part only the type name matters. The
//   one field torchcodec reads is the output surface index, CurrPicIdx vs
//   curr_pic_idx, behind get_curr_pic_idx().
//
// - CUvideoparser, CUvideodecoder, RocdecVideoParser and rocDecDecoderHandle
//   are all `void *`. Nothing portable can be done with one except hand it
//   back to the backend that made it.
//
// - cudaVideoChromaFormat and cudaVideoSurfaceFormat assign the same values to
//   the same formats as their rocDecode counterparts, and those values double
//   as the bit indices of the decoders' supported-format masks. rocDecode adds
//   planar 4:2:0 / 4:2:2 formats at 4..7 plus a Native sentinel at 8; we
//   deliberately never select those, so that a ROCm decode lands on the same
//   surface layout as a CUDA one and the shared color-conversion kernels stay
//   bit-exact across backends.
//
// - cudaVideoCodec is aliased too, but it is the one alias that must never be
//   read as a conversion. NVCUVID numbers HEVC 8 where rocDecode numbers JPEG
//   8, so casting a value from one enum to the other is a silent wrong-codec
//   bug. The alias exists only because shared code *names* the type:
//   DecoderCapsCache's key, get_decoder_caps()'s parameters, and
//   validate_codec_support()'s return type. On each backend the name resolves
//   to that backend's own enum, and no value ever crosses, because the only
//   two producers are validate_codec_support() - which maps from AVCodecID and
//   belongs to whichever backend is compiled - and video_format->codec, which
//   is already native here since CUVIDEOFORMAT *is* RocdecVideoFormat.
//
// CUVIDDECODECREATEINFO, CUVIDPARSERPARAMS, CUVIDPROCPARAMS and
// CUVIDDECODECAPS are the four that cannot be aliased at all; see the note on
// the carrier structs further down.
/* clang-format on */

#if defined(USE_ROCM)

// Must come first: rocdecode.h includes <hip/hip_runtime.h> directly, which
// hard-errors unless a HIP platform has been selected. GpuCompat.h is where we
// select it.
#include "GpuCompat.h"

#include <rocdecode/rocdecode.h>
#include <rocdecode/rocparser.h>

// Declared at global scope, matching where the NVCUVID headers declare the
// originals, so this header is a drop-in replacement for them.
using CUVIDEOFORMAT = RocdecVideoFormat;
using CUVIDEOFORMATEX = RocdecVideoFormatEx;
using CUVIDPARSERDISPINFO = RocdecParserDispInfo;
using CUVIDPICPARAMS = RocdecPicParams;
using CUVIDSOURCEDATAPACKET = RocdecSourceDataPacket;
using CUvideoparser = RocdecVideoParser;
using CUvideodecoder = rocDecDecoderHandle;
using CUvideotimestamp = RocdecTimeStamp;
using cudaVideoChromaFormat = rocDecVideoChromaFormat;
using cudaVideoSurfaceFormat = rocDecVideoSurfaceFormat;

// NVCUVID's device pointer type. rocDecode hands out plain void*, so an
// integer wide enough to hold one is all the shared code needs.
using CUdeviceptr = uintptr_t;

constexpr cudaVideoChromaFormat cudaVideoChromaFormat_Monochrome =
    rocDecVideoChromaFormat_Monochrome;
constexpr cudaVideoChromaFormat cudaVideoChromaFormat_420 =
    rocDecVideoChromaFormat_420;
constexpr cudaVideoChromaFormat cudaVideoChromaFormat_422 =
    rocDecVideoChromaFormat_422;
constexpr cudaVideoChromaFormat cudaVideoChromaFormat_444 =
    rocDecVideoChromaFormat_444;

constexpr cudaVideoSurfaceFormat cudaVideoSurfaceFormat_NV12 =
    rocDecVideoSurfaceFormat_NV12;
constexpr cudaVideoSurfaceFormat cudaVideoSurfaceFormat_P016 =
    rocDecVideoSurfaceFormat_P016;
constexpr cudaVideoSurfaceFormat cudaVideoSurfaceFormat_YUV444 =
    rocDecVideoSurfaceFormat_YUV444;
constexpr cudaVideoSurfaceFormat cudaVideoSurfaceFormat_YUV444_16Bit =
    rocDecVideoSurfaceFormat_YUV444_16Bit;

constexpr uint32_t CUVID_PKT_ENDOFSTREAM = ROCDEC_PKT_ENDOFSTREAM;
constexpr uint32_t CUVID_PKT_TIMESTAMP = ROCDEC_PKT_TIMESTAMP;

// See the note above on why naming this type is safe and converting it is not.
using cudaVideoCodec = rocDecVideoCodec;

// The enumerators torchcodec names, mapped by meaning rather than by value:
// these are the six codecs validate_codec_support() can return. The mapping is
// not the identity - NVCUVID numbers H264 4 and HEVC 8, rocDecode numbers AVC 3
// and HEVC 4 - which is exactly why it has to be spelled out once here instead
// of being cast anywhere. With this table validate_codec_support() needs no
// per-backend variant: it keeps returning the CUDA spelling and each build
// resolves it to its own native enumerator.
constexpr cudaVideoCodec cudaVideoCodec_MPEG4 = rocDecVideoCodec_MPEG4;
constexpr cudaVideoCodec cudaVideoCodec_H264 = rocDecVideoCodec_AVC;
constexpr cudaVideoCodec cudaVideoCodec_HEVC = rocDecVideoCodec_HEVC;
constexpr cudaVideoCodec cudaVideoCodec_AV1 = rocDecVideoCodec_AV1;
constexpr cudaVideoCodec cudaVideoCodec_VP8 = rocDecVideoCodec_VP8;
constexpr cudaVideoCodec cudaVideoCodec_VP9 = rocDecVideoCodec_VP9;

// The calling-convention macro the NVCUVID callback signatures carry. Both
// expand to nothing on Linux, but keep the indirection so the shared code
// never has to care which header it came from.
#define CUDAAPI ROCDECAPI

// Every NVCUVID entry point returns CUresult; every rocDecode one returns
// rocDecStatus. Both are enums whose success value is 0, and the shared code
// only ever compares against success or prints the value in an error message.
using CUresult = rocDecStatus;
constexpr CUresult CUDA_SUCCESS = ROCDEC_SUCCESS;

// The driver-API stream handle. GpuCompat.h aliases the runtime-API one
// (cudaStream_t) to the same HIP type, which is what makes the
// reinterpret_cast in map_frame() a no-op here rather than a reinterpretation.
using CUstream = hipStream_t;

// NVCUVID's only decoder creation flag that torchcodec sets. rocDecode has no
// equivalent knob - it always picks the hardware path - so this is inert.
constexpr unsigned long cudaVideoCreate_Default = 0;

/* clang-format off */
// Note: [The four carrier structs]
//
// Unlike everything above, these four have no field names in common with
// their rocDecode counterparts. They carry exactly the same information, but
// NVIDIA spells it ulWidth / CodecType / bitDepthMinus8 where rocDecode
// spells it width / codec_type / bit_depth_minus_8. A type alias would
// therefore break every assignment in create_decoder(),
// initialize_video_decoding(), DecoderCapsCache and map_frame() - the NVDEC
// code we are deliberately not touching.
//
// So on ROCm they are declared here as plain carrier structs wearing NVIDIA's
// field names, holding only the fields torchcodec actually sets or reads, and
// typed as NVIDIA types them so that every assignment, cast and comparison in
// the shared code behaves identically on both backends. No AMD driver ever
// sees one: the forwarders at the end of BetaCudaDeviceInterface.cpp copy them
// field-by-field into the real rocDecode struct at the point of the call.
//
// Keep these in sync with what the shared code touches. A field that is never
// assigned is a field the forwarder would have to invent a value for, so the
// omissions are deliberate - notably ulIntraDecodeOnly, DeinterlaceMode,
// vidLock and target_rect on the create-info, which torchcodec leaves zeroed.
/* clang-format on */

// IN fields are read by the forwarder; OUT fields are written by it.
struct CUVIDDECODECAPS {
  cudaVideoCodec eCodecType; // IN
  cudaVideoChromaFormat eChromaFormat; // IN
  unsigned int nBitDepthMinus8; // IN
  unsigned char bIsSupported; // OUT
  unsigned short nOutputFormatMask; // OUT: bit per cudaVideoSurfaceFormat
  unsigned int nMaxWidth; // OUT
  unsigned int nMaxHeight; // OUT
  unsigned int nMaxMBCount; // OUT: see the forwarder, rocDecode has no such cap
  unsigned short nMinWidth; // OUT
  unsigned short nMinHeight; // OUT
};

struct CUVIDDECODECREATEINFO {
  unsigned long ulWidth;
  unsigned long ulHeight;
  unsigned long ulNumDecodeSurfaces;
  cudaVideoCodec CodecType;
  cudaVideoChromaFormat ChromaFormat;
  unsigned long ulCreationFlags;
  unsigned long bitDepthMinus8;
  unsigned long ulMaxWidth;
  unsigned long ulMaxHeight;

  struct {
    short left;
    short top;
    short right;
    short bottom;
  } display_area;

  cudaVideoSurfaceFormat OutputFormat;
  unsigned long ulTargetWidth;
  unsigned long ulTargetHeight;
  unsigned long ulNumOutputSurfaces;
};

// The callback types are rocparser.h's own: they are declared with the same
// names and, once CUVIDEOFORMAT and friends are aliased above, the same
// signatures. The parser callbacks in BetaCudaDeviceInterface need no
// adaptation at all.
struct CUVIDPARSERPARAMS {
  cudaVideoCodec CodecType;
  unsigned int ulMaxNumDecodeSurfaces;
  unsigned int ulMaxDisplayDelay;
  void* pUserData;
  PFNVIDSEQUENCECALLBACK pfnSequenceCallback;
  PFNVIDDECODECALLBACK pfnDecodePicture;
  PFNVIDDISPLAYCALLBACK pfnDisplayPicture;
  CUVIDEOFORMATEX* pExtVideoInfo;
};

// RocdecProcParams has progressive_frame and top_field_first but neither
// unpaired_field nor output_stream, which is why map_frame() takes the stream
// as an argument instead of passing it through here. This struct is only ever
// filled by the base map_frame(), which BetaRocmDeviceInterface overrides.
struct CUVIDPROCPARAMS {
  int progressive_frame;
  int top_field_first;
  int unpaired_field;
  CUstream output_stream;
};

// The NVCUVID entry points, as implemented for rocDecode. Definitions live in
// the USE_ROCM block at the end of BetaCudaDeviceInterface.cpp; seven of the
// nine are 1:1 forwards, and that block explains the two that are not.
CUresult cuvidGetDecoderCaps(CUVIDDECODECAPS* caps);
CUresult cuvidCreateDecoder(
    CUvideodecoder* decoder,
    CUVIDDECODECREATEINFO* create_info);
CUresult cuvidDestroyDecoder(CUvideodecoder decoder);
CUresult cuvidCreateVideoParser(
    CUvideoparser* parser,
    CUVIDPARSERPARAMS* parser_params);
CUresult cuvidDestroyVideoParser(CUvideoparser parser);
CUresult cuvidParseVideoData(
    CUvideoparser parser,
    CUVIDSOURCEDATAPACKET* packet);
CUresult cuvidDecodePicture(CUvideodecoder decoder, CUVIDPICPARAMS* pic_params);
CUresult cuvidMapVideoFrame(
    CUvideodecoder decoder,
    int pic_index,
    CUdeviceptr* frame_ptr,
    unsigned int* pitch,
    CUVIDPROCPARAMS* proc_params);
CUresult cuvidUnmapVideoFrame(CUvideodecoder decoder, CUdeviceptr frame_ptr);

#else

#include "nvcuvid_include/cuviddec.h"
#include "nvcuvid_include/nvcuvid.h"

#endif // USE_ROCM

namespace facebook::torchcodec {

// The one field NVCUVID and rocDecode disagree on the name of. Both structs
// carry it in the same position with the same width; only the spelling
// differs, so these accessors are the whole of the papering-over.
#if defined(USE_ROCM)

inline int64_t get_timestamp(const CUVIDPARSERDISPINFO& disp_info) {
  return static_cast<int64_t>(disp_info.pts);
}

inline int64_t get_timestamp(const CUVIDSOURCEDATAPACKET& packet) {
  return static_cast<int64_t>(packet.pts);
}

inline void set_timestamp(CUVIDSOURCEDATAPACKET& packet, int64_t timestamp) {
  packet.pts = static_cast<RocdecTimeStamp>(timestamp);
}

inline void set_timestamp(CUVIDPARSERDISPINFO& disp_info, int64_t timestamp) {
  disp_info.pts = static_cast<RocdecTimeStamp>(timestamp);
}

inline int get_curr_pic_idx(const CUVIDPICPARAMS& pic_params) {
  return pic_params.curr_pic_idx;
}

#else

inline int64_t get_timestamp(const CUVIDPARSERDISPINFO& disp_info) {
  return static_cast<int64_t>(disp_info.timestamp);
}

inline int64_t get_timestamp(const CUVIDSOURCEDATAPACKET& packet) {
  return static_cast<int64_t>(packet.timestamp);
}

inline void set_timestamp(CUVIDSOURCEDATAPACKET& packet, int64_t timestamp) {
  packet.timestamp = static_cast<CUvideotimestamp>(timestamp);
}

inline void set_timestamp(CUVIDPARSERDISPINFO& disp_info, int64_t timestamp) {
  disp_info.timestamp = static_cast<CUvideotimestamp>(timestamp);
}

inline int get_curr_pic_idx(const CUVIDPICPARAMS& pic_params) {
  return pic_params.CurrPicIdx;
}

#endif // USE_ROCM

} // namespace facebook::torchcodec
