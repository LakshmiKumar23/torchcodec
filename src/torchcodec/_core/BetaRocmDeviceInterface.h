// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "BetaCudaDeviceInterface.h"

namespace facebook::torchcodec {

/* clang-format off */
// The AMD hardware-decode interface. It decodes with rocDecode instead of
// NVDEC, and is deliberately almost empty: everything except the frame mapping
// is inherited from BetaCudaDeviceInterface unchanged.
//
// That works because the ROCm build compiles BetaCudaDeviceInterface.cpp too.
// The NVCUVID types it is written against are aliased onto their rocDecode
// counterparts by NvcuvidCompat.h, and eight of the nine NVCUVID entry points
// it calls are satisfied by the forwarders at the end of that same .cpp. So
// the parser callbacks, the send/receive state machine, the display-order
// re-ordering, the decoder cache, the cropping and the color conversion are
// not merely equivalent between the two backends - they are the same code,
// which is what makes a ROCm decode bit-exact with a CUDA one.
//
// Inheriting from the CUDA class rather than from a device-neutral sibling is
// the shape the TorchCodec maintainers asked for, and it is also why a ROCm
// device is spelled kStableCUDA throughout: under a ROCm build of PyTorch,
// torch.device("cuda") *is* the AMD GPU.
//
// What is left below is the ninth entry point. See Note: [The one
// hardware-decoder seam] in BetaCudaDeviceInterface.h for why it cannot be a
// forwarder, and Note: [Mapping a rocDecode surface] in the .cpp for what the
// override has to check.
/* clang-format on */

class BetaRocmDeviceInterface : public BetaCudaDeviceInterface {
 public:
  explicit BetaRocmDeviceInterface(const StableDevice& device);

  std::string get_details() override;

 protected:
  int map_frame(
      const CUVIDPARSERDISPINFO& disp_info,
      cudaStream_t stream,
      MappedSurface& surface) override;

  void record_surface_read(cudaStream_t stream) override;
};

} // namespace facebook::torchcodec
