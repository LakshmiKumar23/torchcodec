// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

namespace facebook::torchcodec {

// Dynamically load the rocDecode library at runtime.
// This allows TorchCodec to work on systems without rocDecode installed,
// falling back to CPU decoding when the library is not available.
//
// We do NOT want to link statically or
// dynamically against librocdecode.so at build time. Instead, we load it
// at runtime using dlopen/dlsym.
//
// rocDecode documentation:
// https://rocm.docs.amd.com/projects/rocDecode/en/latest
//
// Returns true if rocDecode library is successfully loaded, false otherwise.
bool loadRocDecodeLibrary();

} // namespace facebook::torchcodec
