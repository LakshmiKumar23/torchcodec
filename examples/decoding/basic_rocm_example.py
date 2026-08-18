# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""
Accelerated video decoding on AMD GPUs with ROCm and rocDecode
================================================================

TorchCodec can use supported AMD hardware (GPUs with VCN - Video Core Next) to speed-up
video decoding. This is called "ROCm Decoding" and it uses AMD's
`rocDecode hardware decoder <https://rocm.docs.amd.com/projects/rocDecode/en/latest>`_
and HIP kernels to respectively decompress and convert to RGB.
ROCm Decoding can be faster than CPU Decoding for the actual decoding step and also for
subsequent transform steps like scaling, cropping or rotating. This is because the decode step leaves
the decoded tensor in GPU memory so the GPU doesn't have to fetch from main memory before
running the transform steps. Encoded packets are often much smaller than decoded frames so
ROCm decoding also uses less PCI-e bandwidth.

Installing TorchCodec with ROCm Enabled
---------------------------------------

Refer to the installation guide in the `README <https://github.com/pytorch/torchcodec#installing-rocm-enabled-torchcodec>`_.

"""

# %%
# Checking if PyTorch has ROCm enabled
# -------------------------------------
#
# .. note::
#
#    This tutorial requires FFmpeg libraries and PyTorch compiled with ROCm support.
#
#
import torch

print(f"{torch.__version__=}")
print(f"{torch.cuda.is_available()=}")  # Returns True for ROCm as well
print(f"{torch.cuda.get_device_properties(0)=}")


# %%
# Downloading the video
# -------------------------------------
#
# We will use the following video which has the following properties:
#
# - Codec: H.264
# - Resolution: 960x540
# - FPS: 29.97
# - Pixel format: YUV420P
#
# .. raw:: html
#
#    <video style="max-width: 100%" controls>
#      <source src="https://download.pytorch.org/torchaudio/tutorial-assets/stream-api/NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4_small.mp4" type="video/mp4">
#    </video>
import urllib.request

video_file, _ = urllib.request.urlretrieve(
    "https://download.pytorch.org/torchaudio/tutorial-assets/stream-api/NASAs_Most_Scientifically_Complex_Space_Observatory_Requires_Precision-MP4_small.mp4",
    "video.mp4",
)


# %%
# ROCm Decoding using VideoDecoder
# -------------------------------------
#
# To use ROCm decoder with hardware acceleration (rocDecode), you need to pass
# in a cuda device to the decoder. Note: PyTorch uses "cuda" for both NVIDIA
# CUDA and AMD ROCm devices.
#
# TorchCodec will automatically use AMD's rocDecode hardware decoder (via VCN)
# when available, and fall back to CPU decoding if the video format is not
# supported by the hardware.
#
from torchcodec.decoders import VideoDecoder

decoder = VideoDecoder(video_file, device="cuda")  # "cuda" works for ROCm
print(f"Decoder created successfully: {decoder}")
print(f"Decoder Backend: {decoder.cpu_fallback._backend}")
frame = decoder[0]

# %%
#
# The video frames are decoded and returned as tensor of NCHW format.

print(frame.shape, frame.dtype)

# %%
#
# The video frames are left on the GPU memory.

print(frame.data.device)


# %%
# Checking for CPU Fallback
# -------------------------------------
#
# In some cases, ROCm decoding may fall back to CPU decoding. This can happen
# when the video codec or format is not supported by the VCN hardware decoder,
# or when rocDecode wasn't found.
#
# TorchCodec provides the :class:`~torchcodec.decoders.CpuFallbackStatus` class
# to help you detect when this fallback occurs.
#
# You can access the fallback status via the
# :attr:`~torchcodec.decoders.VideoDecoder.cpu_fallback` attribute:

decoder = VideoDecoder(video_file, device="cuda")

# Check and print the CPU fallback status
print(decoder.cpu_fallback)


# %%
# Visualizing Frames
# -------------------------------------
#
# Let's look at the frames decoded by ROCm decoder and compare them
# against equivalent results from the CPU decoders.
timestamps = [12, 19, 45, 131, 180]
cpu_decoder = VideoDecoder(video_file, device="cpu")
rocm_decoder = VideoDecoder(video_file, device="cuda")
cpu_frames = cpu_decoder.get_frames_played_at(timestamps).data
rocm_frames = rocm_decoder.get_frames_played_at(timestamps).data


def plot_cpu_and_rocm_frames(cpu_frames: torch.Tensor, rocm_frames: torch.Tensor):
    try:
        import matplotlib.pyplot as plt
        from torchvision.transforms.v2.functional import to_pil_image
    except ImportError:
        print("Cannot plot, please run `pip install torchvision matplotlib`")
        return
    n_rows = len(timestamps)
    fig, axes = plt.subplots(n_rows, 2, figsize=[12.8, 16.0])
    for i in range(n_rows):
        axes[i][0].imshow(to_pil_image(cpu_frames[i].to("cpu")))
        axes[i][1].imshow(to_pil_image(rocm_frames[i].to("cpu")))

    axes[0][0].set_title("CPU decoder", fontsize=24)
    axes[0][1].set_title("ROCm decoder", fontsize=24)
    plt.setp(axes, xticks=[], yticks=[])
    plt.tight_layout()
    plt.savefig('rocm_cpu_comparison.png', dpi=150, bbox_inches='tight')
    print("Plot saved to rocm_cpu_comparison.png")


plot_cpu_and_rocm_frames(cpu_frames, rocm_frames)

# %%
#
# They look visually similar to the human eye but there may be subtle
# differences because GPU math is not bit-exact with respect to CPU math.
#
frames_equal = torch.equal(cpu_frames.to("cuda"), rocm_frames)
mean_abs_diff = torch.mean(
    torch.abs(cpu_frames.float().to("cuda") - rocm_frames.float())
)
max_abs_diff = torch.max(torch.abs(cpu_frames.to("cuda").float() - rocm_frames.float()))
print(f"{frames_equal=}")
print(f"{mean_abs_diff=}")
print(f"{max_abs_diff=}")

# %%
# ROCm-Specific Information
# -------------------------------------
#
# You can check which AMD GPU is being used and get additional information:
if torch.cuda.is_available():
    print(f"ROCm device name: {torch.cuda.get_device_name(0)}")
    print(f"ROCm device count: {torch.cuda.device_count()}")
    print(f"Current ROCm device: {torch.cuda.current_device()}")