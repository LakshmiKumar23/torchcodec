#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Installs the rocDecode library
#
# rocDecode is not preinstalled by default in the CI runners. Both the build
# and the test jobs need it (we don't ship it in the wheel), so we install it
# from the ROCm dnf repo.

set -euo pipefail

# rocDecode requires HIP runtime and other ROCm dependencies. On CI runners
# with full ROCm, install from the repo directly. On build-only images that
# lack the full amdgpu stack, install with --nodeps (dependencies are satisfied
# by the ROCm environment used to build torch).
install_rocdecode_build_only() {
    dnf install -y "dnf-command(download)" >/dev/null 2>&1 || dnf install -y dnf-plugins-core
    rpm_dir="$(mktemp -d)"
    dnf download --destdir "${rpm_dir}" rocdecode rocdecode-devel
    rpm -Uvh --nodeps "${rpm_dir}"/rocdecode*.rpm
}

dnf install -y --refresh rocdecode-devel \
    || install_rocdecode_build_only
