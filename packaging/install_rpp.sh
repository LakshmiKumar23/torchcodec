#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Installs the ROCm Performance Primitives (RPP) library
#
# RPP is not preinstalled by default in the CI runners. Both the build
# and the test jobs need it (we don't ship it in the wheel), so we install it
# from the ROCm dnf repo.

set -euo pipefail

# Unlike rocDecode, RPP is a HIP compute library with no VA-API dependency: at
# runtime it only needs the HIP runtime, which the torch-ROCm wheel provides. A
# proper `dnf install` works where AMD's amdgpu-graphics repo is configured (the
# ROCm test runners). On the plain build image that repo is absent; there we only
# need to *compile* against rpp.h/librpp.so, so fall back to downloading just
# those RPMs and installing them with --nodeps (the HIP deps are satisfied by the
# ROCm environment used to build torch).
install_rpp_build_only() {
    dnf install -y "dnf-command(download)" >/dev/null 2>&1 || dnf install -y dnf-plugins-core
    rpm_dir="$(mktemp -d)"
    dnf download --destdir "${rpm_dir}" amdrocm-rpp amdrocm-rpp-devel
    rpm -Uvh --nodeps "${rpm_dir}"/amdrocm-rpp*.rpm
}

dnf install -y --refresh amdrocm-rpp-devel \
    || install_rpp_build_only
