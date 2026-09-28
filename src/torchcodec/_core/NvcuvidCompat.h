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
//   (rocDecode's is VA-API-shaped), but no torchcodec code ever reads a field:
//   a picture's parameters come from the parser callback and go straight back
//   into cuvidDecodePicture() / rocDecDecodeFrame(). Only the type name
//   matters.
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
// NOT aliased, because the values genuinely differ: cudaVideoCodec. NVCUVID
// numbers HEVC 8, where rocDecode numbers JPEG 8 - a cast between them is a
// silent wrong-codec bug. Each backend maps from AVCodecID itself, and the
// native codec enum never crosses into shared code.
//
// Also not aliased: CUVIDDECODECREATEINFO, CUVIDPARSERPARAMS, CUVIDPROCPARAMS
// and CUVIDDECODECAPS, whose rocDecode equivalents carry the same information
// under snake_case names and with a few fields added or missing. Those are
// named only by the code that sets up or drives a decoder, which each backend
// owns outright; none of them appear in this class's shared members or in any
// signature crossing between the two.
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

#endif // USE_ROCM

} // namespace facebook::torchcodec
