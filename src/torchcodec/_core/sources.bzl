# Single source of truth for torchcodec's _core C++ source file lists.
#
# These lists are consumed by BOTH build systems so that adding or removing a
# source file only requires editing this one file:
# - CMake (the GitHub/OSS build) reads it from CMakeLists.txt in this directory.
# - Buck (the internal Meta build) loads it from src/BUCK.
#
# This file must stay syntactically valid as BOTH Starlark (so Buck can `load()`
# it) and Python (CMake execs it with the Python interpreter to extract the
# lists). Only plain list literals and list comprehensions are allowed here: no
# functions, no `load()`, no `select()`, no f-strings, no Starlark-only or
# Python-only constructs.
#
# Filenames are listed WITHOUT a directory prefix: they live next to this file
# in src/torchcodec/_core/. CMake runs from this directory and uses them as-is;
# Buck prepends the "torchcodec/_core/" path prefix at load time.
#
# The lists are intentionally fine-grained building blocks so each build system
# can compose exactly the targets it needs without changing what gets compiled.
# The two builds group these differently (e.g. CMake compiles FFMPEGCommon.cpp
# into the core library while Buck builds it as a separate target), so do not
# merge the lists.

# CPU sources for the core decoder library. Does NOT include FFMPEGCommon.cpp
# (see ffmpeg_common_sources): the Buck build compiles that into a separate
# "core_common" target.
decoder_core_sources = [
    "AVIOContextHolder.cpp",
    "TensorIO.cpp",
    "FilterGraph.cpp",
    "Frame.cpp",
    "DeviceInterface.cpp",
    "CpuDeviceInterface.cpp",
    "Demuxer.cpp",
    "PacketDecoder.cpp",
    "AudioCommon.cpp",
    "ColorConverter.cpp",
    "AudioConverter.cpp",
    "SingleStreamDecoder.cpp",
    "Encoder.cpp",
    "ValidationUtils.cpp",
    "Transform.cpp",
    "Metadata.cpp",
    "SwScale.cpp",
    "WavDecoder.cpp",
    "NVDECCacheConfig.cpp",
    "Logging.cpp",
]

# FFmpeg glue. Part of the core library under CMake; a standalone "core_common"
# target under Buck.
ffmpeg_common_sources = [
    "FFMPEGCommon.cpp",
]

io_sources = [
    "FileIO.cpp",
]

# CUDA sources, added to the core library only for CUDA-enabled builds.
decoder_core_cuda_sources = [
    "CudaDeviceInterface.cpp",
    "BetaCudaDeviceInterface.cpp",
    "NVDECCache.cpp",
    "CUDACommon.cpp",
    "NVCUVIDRuntimeLoader.cpp",
    "color_conversion.cpp",
    "color_conversion.cu",
]

# CUDA sources with no ROCm counterpart, subtracted from the ROCm list below.
# CudaDeviceInterface.cpp is the non-beta, FFmpeg-hwaccel CUDA interface; the
# ROCm port covers the beta NVDEC interface only.
_cuda_only_sources = [
    "CudaDeviceInterface.cpp",
]

# ROCm sources, added to the core library only for ROCm-enabled builds.
#
# Derived from decoder_core_cuda_sources rather than restated, so the two builds
# cannot drift: a file added there is compiled by the ROCm build too unless it is
# named in _cuda_only_sources, which is a deliberate choice rather than an
# oversight.
#
# The shared files are compiled as HIP here: the CUDA runtime names they use
# resolve through CUDACompat.h, and the NVCUVID entry points they call are
# implemented on top of rocDecode in the USE_ROCM block at the end of
# BetaCudaDeviceInterface.cpp. Sharing the sources - rather than maintaining a
# parallel ROCm implementation - is what makes the two backends bit-exact, since
# they then run the same kernels over the same surfaces.
#
# NVCUVIDRuntimeLoader.cpp is shared for the same reason it exists on CUDA: we
# must not link the vendor decode library, because it is not always installed on
# a machine that has the wheel, and a link-time dependency turns that into an
# `import torchcodec` failure. Its ROCm branch dlopen()s librocdecode.so.1 and
# reports failure the same way, which is what lets the interface fall back to
# CPU decoding instead of crashing.
decoder_core_rocm_sources = [
    src
    for src in decoder_core_cuda_sources
    if src not in _cuda_only_sources
] + [
    "BetaRocmDeviceInterface.cpp",
]

file_like_context_sources = [
    "FileLikeIO.cpp",
]

# PyTorch custom ops registration.
custom_ops_sources = [
    "custom_ops.cpp",
]

image_sources = [
    "DecodeJpeg.cpp",
    "DecodeJpegCuda.cpp",
    "DecodePng.cpp",
    "DecodeWebp.cpp",
    "DecodeGif.cpp",
    "DecodeAvif.cpp",
    "EncodePng.cpp",
    "EncodeJpeg.cpp",
    "EncodeJpegCuda.cpp",
]

image_ops_sources = [
    "image_custom_ops.cpp",
]

heic_sources = [
    "DecodeHeic.cpp",
]

heic_ops_sources = [
    "heic_custom_ops.cpp",
]

# Vendored giflib (decode-only subset, MIT licensed). Compiled directly from
# source into the image library, so the GIF decoder needs no external dependency.
# See giflib/README for the license and local mods.
giflib_sources = [
    "giflib/dgif_lib.c",
    "giflib/gifalloc.c",
    "giflib/gif_hash.c",
    "giflib/openbsd-reallocarray.c",
]

# pybind11 bindings (file-like support). Built into the single, FFmpeg-free
# libtorchcodec_pybind_ops (alongside io_sources and file_like_context_sources)
# and used by both the image encoders and the FFmpeg encoders/decoders.
pybind_ops_sources = [
    "pybind_ops.cpp",
]
