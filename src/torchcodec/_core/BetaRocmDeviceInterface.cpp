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
// Measured on gfx942 / ROCm 10.1, the planes do land in a single allocation
// with a uniform pitch, so collapsing the three pointers down to one loses
// nothing. What they do *not* do is sit coded_height rows apart. A 480x272
// stream comes back with the chroma plane exactly 272 rows in, but a 200-row
// HEVC stream comes back with it 208 rows in, and a 360-row one 368 rows in.
//
// That padding is not rocDecode's. On Linux it fills these pointers straight
// from VADRMPRIMESurfaceDescriptor.layers[i].offset[0] (roc_decoder.cpp), so
// the layout is whatever the VA-API driver allocated - here, the luma plane
// rounded up to a 16-row boundary. Which means the alignment is a property of
// the driver and the GPU, not a constant we can bake in: hard-coding "round up
// to 16" would work on this machine and silently rot on the next one.
//
// So we do not compute the offset at all. We take the offset the driver
// reports, divide it by the pitch to get the row count it implies, and hand
// that to the base class through plane_rows(). The shared arithmetic then
// reproduces the driver's own layout exactly, whatever alignment it chose.
//
// The checks below are what is left over: the parts of the layout that, if
// they ever changed, could not be expressed through this seam at all, because
// the base class carries a single pointer and a single pitch for the whole
// surface. Non-uniform pitches or a chroma offset that is not a whole number
// of rows would both mean the frame cannot be described that way. Getting
// those wrong would show up as discoloured frames rather than an error - the
// exact class of silent bug that sharing the CUDA kernels is meant to rule
// out - so we fail loudly at the seam instead.
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
    CUdeviceptr& frame_ptr,
    unsigned int& pitch) {
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

  STD_TORCH_CHECK(
      plane_ptr[0] != nullptr && plane_pitch[0] > 0,
      "rocDecode mapped a picture but returned no luma plane. This should "
      "never happen, please report.");

  // 4:4:4 surfaces carry three full-size planes; NV12 and P016 carry a luma
  // plane and one interleaved chroma plane.
  bool is_444 = surface_format_ == cudaVideoSurfaceFormat_YUV444 ||
      surface_format_ == cudaVideoSurfaceFormat_YUV444_16Bit;
  int num_planes = is_444 ? 3 : 2;

  auto luma_address = reinterpret_cast<uintptr_t>(plane_ptr[0]);
  STD_TORCH_CHECK(
      plane_ptr[1] != nullptr,
      "rocDecode mapped a picture but returned no chroma plane. This should "
      "never happen, please report.");
  uintptr_t plane_stride =
      reinterpret_cast<uintptr_t>(plane_ptr[1]) - luma_address;

  // The base class describes the whole surface with one pointer and one pitch,
  // which it can only do if the planes are evenly spaced and equally pitched.
  STD_TORCH_CHECK(
      plane_stride % plane_pitch[0] == 0,
      "rocDecode placed the chroma plane ",
      plane_stride,
      " bytes after the luma plane, which is not a whole number of ",
      plane_pitch[0],
      "-byte rows. The shared color-conversion path cannot describe such a "
      "surface; please report this.");

  for (int plane = 1; plane < num_planes; ++plane) {
    STD_TORCH_CHECK(
        plane_ptr[plane] != nullptr,
        "rocDecode returned a null pointer for plane ",
        plane,
        " of a ",
        num_planes,
        "-plane surface. This should never happen, please report.");
    STD_TORCH_CHECK(
        plane_pitch[plane] == plane_pitch[0],
        "rocDecode returned per-plane pitches that differ (plane 0: ",
        plane_pitch[0],
        ", plane ",
        plane,
        ": ",
        plane_pitch[plane],
        "). The shared color-conversion path assumes a uniform pitch; "
        "please report this, as decoding with it would corrupt chroma.");
    STD_TORCH_CHECK(
        reinterpret_cast<uintptr_t>(plane_ptr[plane]) - luma_address ==
            plane_stride * static_cast<uintptr_t>(plane),
        "rocDecode laid plane ",
        plane,
        " out ",
        reinterpret_cast<uintptr_t>(plane_ptr[plane]) - luma_address,
        " bytes after the luma plane, but its own plane spacing is ",
        plane_stride,
        ", so it was expected at ",
        plane_stride * static_cast<uintptr_t>(plane),
        ". The shared color-conversion path assumes evenly spaced planes; "
        "please report this, as decoding with it would corrupt chroma.");
  }

  // Tell the base class where the driver actually put the planes, so its
  // offset arithmetic lands on them.
  mapped_plane_rows_ = static_cast<int>(plane_stride / plane_pitch[0]);

  // Safe to narrow to the NVCUVID shape now that the layout is verified.
  frame_ptr = static_cast<CUdeviceptr>(luma_address);
  pitch = plane_pitch[0];
  return AVSUCCESS;
}

int BetaRocmDeviceInterface::plane_rows() const {
  // Before the first mapping there is nothing to report and nothing that needs
  // it, so defer to the coded-height formula until the driver has told us
  // otherwise. This is re-learned on every mapping rather than cached once,
  // because a mid-stream resolution change gets new surfaces with a new
  // alignment.
  return mapped_plane_rows_ > 0 ? mapped_plane_rows_
                                : BetaCudaDeviceInterface::plane_rows();
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
