// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "BetaRocmDeviceInterface.h"

#include "CUDACommon.h"
#include "DeviceInterface.h"
#include "FFMPEGCommon.h"
#include "NvcuvidCompat.h"
#include "StableABICompat.h"

namespace facebook::torchcodec {

namespace {

static bool g_rocm_rocdec = register_device_interface(
    DeviceInterfaceKey(kStableCUDA, /*variant=*/"default"),
    [](const StableDevice& device) {
      return new BetaRocmDeviceInterface(device);
    });

} // namespace

BetaRocmDeviceInterface::BetaRocmDeviceInterface(const StableDevice& device)
    : BetaCudaDeviceInterface(device) {
  STD_TORCH_CHECK(g_rocm_rocdec, "BetaRocmDeviceInterface was not registered!");
}

std::string BetaRocmDeviceInterface::get_details() {
  // Deliberately not deferring to the base: it would name NVDEC, which is not
  // what ran.
  std::string details = "rocDecode ROCm Device Interface.";
  if (decoding_on_cpu_) {
    details += " Using CPU fallback.";
    if (!nvcuvid_available_) {
      details += " rocDecode not available!";
    }
  } else {
    details += " Using rocDecode.";
  }
  return details;
}

/* clang-format off */
// Note: [Mapping a rocDecode surface]
//
// The one place the two decode APIs genuinely disagree. cuvidMapVideoFrame()
// hands back a single device pointer and a single pitch describing the whole
// surface, and the shared code in convert_cuda_frame_to_av_frame() walks from
// there to the chroma planes arithmetically:
//
//     plane_stride = pitch * round_up_to_even(coded_height)
//     plane i      = frame_ptr + i * plane_stride
//
// rocDecGetVideoFrame() instead fills an array of three pointers and an array
// of three pitches, one per plane, and the documentation promises nothing
// about how those relate to one another.
//
// We take it at its word: the pointers and pitches go into the MappedSurface
// untouched, and each consumer reads the entry for the plane it wants. Nothing
// reconstructs one plane's address from another's, so none of what rocDecode
// declines to promise has to hold.
//
// Those planes are not where NVDEC's stacking would put them. On Linux
// rocDecode fills these pointers from
// VADRMPRIMESurfaceDescriptor.layers[i].offset[0] (roc_decoder.cpp), so the
// layout is the VA-API driver's, and it pads the luma plane to its own
// alignment: a 200-row HEVC frame gets a 208-row luma plane. That alignment
// belongs to the driver and the GPU, so it is not a constant to bake in - and
// not one we need, since nothing here steps from one plane to the next.
//
// That leaves only the two things we go on to dereference to check: that a
// plane we will read exists, and that its pitch is usable.
//
// Two smaller asymmetries, both benign:
//
// - RocdecProcParams has no output_stream field. rocDecGetVideoFrame() is
//   documented to block until the picture is decoded, so the surface contents
//   are ready on return, for any stream. The base class then records
//   nvdec_surface_ready_ on the output stream; an event recorded over
//   already-complete work is trivially satisfied, so consumers stay correct.
//   That is why `stream` is unused here.
//
// - RocdecProcParams also has no unpaired_field. Nothing to pass it to, and
//   the interlaced paths it guards are not ones we exercise.
/* clang-format on */

int BetaRocmDeviceInterface::map_frame(
    const CUVIDPARSERDISPINFO& disp_info,
    [[maybe_unused]] cudaStream_t stream,
    MappedSurface& surface) {
  RocdecProcParams proc_params = {};
  proc_params.progressive_frame = disp_info.progressive_frame;
  proc_params.top_field_first = disp_info.top_field_first;

  void* plane_ptr[3] = {nullptr, nullptr, nullptr};
  uint32_t plane_pitch[3] = {0, 0, 0};

  rocDecStatus result = rocDecGetVideoFrame(
      *decoder_.get(),
      disp_info.picture_index,
      plane_ptr,
      plane_pitch,
      &proc_params);
  if (result != ROCDEC_SUCCESS) {
    return AVERROR_EXTERNAL;
  }

  // 4:4:4 surfaces carry three full-size planes; NV12 and P016 carry a luma
  // plane and one interleaved chroma plane.
  bool is_444 = surface_format_ == cudaVideoSurfaceFormat_YUV444 ||
      surface_format_ == cudaVideoSurfaceFormat_YUV444_16Bit;
  int num_planes = is_444 ? 3 : 2;

  for (int plane = 0; plane < num_planes; ++plane) {
    STD_TORCH_CHECK(
        plane_ptr[plane] != nullptr && plane_pitch[plane] > 0,
        "rocDecode mapped a picture but returned no plane ",
        plane,
        " of a ",
        num_planes,
        "-plane surface. This should never happen, please report.");
    surface.planes[plane] =
        static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(plane_ptr[plane]));
    surface.pitch[plane] = plane_pitch[plane];
  }
  return AVSUCCESS;
}

/* clang-format off */
// Note: [Why the ROCm surface read has to be synchronous]
//
// The base class records surface_read_done_ here and has receive_frame() order
// the next mapping after it. That is enough on NVDEC, where a surface stays
// mapped until cuvidUnmapVideoFrame() and the decoder will not write to a
// mapped surface.
//
// rocDecode has no unmap entry point: it recycles its internal VAAPI surface
// pool on its own schedule, and rocDecDecodeFrame() submissions are not ordered
// against any torch stream. So the base class's event gate is vacuous here -
// nothing stops rocDecode overwriting the surface while our copy out of it is
// still queued.
//
// It only bites when the consumer runs on a stream other than the decoder's:
// same-stream use orders the copy before the reuse by construction. The "Blocks"
// APIs let a caller read frame.planes on its own stream, which is exactly that
// case, and it shows up as torn frames - partially-overwritten surfaces, not
// swapped buffers.
//
// So we wait on the host: once this returns, the copy has landed and rocDecode
// is free to reclaim the surface. That costs one host sync per frame, which is
// the price of a decoder that will not tell us when it reuses its surfaces.
/* clang-format on */

void BetaRocmDeviceInterface::record_surface_read(cudaStream_t stream) {
  BetaCudaDeviceInterface::record_surface_read(stream);
  surface_read_done_.synchronize();
}

} // namespace facebook::torchcodec
