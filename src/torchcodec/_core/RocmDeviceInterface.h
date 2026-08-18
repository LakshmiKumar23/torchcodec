// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

// ROCm device interface that provides direct control over AMD VCN
// hardware decoder via rocDecode while keeping FFmpeg for demuxing.
// Implementation inspired by NVDEC integration and DALI's approach.
//
// rocDecode Library: https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode
// AMD VCN (Video Core Next) Architecture docs:
// https://rocm.docs.amd.com/projects/rocDecode/en/latest

#pragma once

#include "RocmCommon.h"
#include "Cache.h"
#include "DeviceInterface.h"
#include "FFMPEGCommon.h"
#include "RocDecCache.h"

#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <vector>

// rocDecode API headers
#include <rocdecode/rocdecode.h>
#include <rocdecode/rocparser.h>

namespace facebook::torchcodec {

class RocmDeviceInterface : public DeviceInterface {
 public:
  explicit RocmDeviceInterface(const torch::Device& device);
  virtual ~RocmDeviceInterface();

  void initialize(
      const AVStream* avStream,
      const UniqueDecodingAVFormatContext& avFormatCtx,
      const SharedAVCodecContext& codecContext) override;

  void convertAVFrameToFrameOutput(
      UniqueAVFrame& avFrame,
      FrameOutput& frameOutput,
      std::optional<torch::Tensor> preAllocatedOutputTensor) override;

  int sendPacket(ReferenceAVPacket& packet) override;
  int sendEOFPacket() override;
  int receiveFrame(UniqueAVFrame& avFrame) override;
  void flush() override;

  // rocDecode callback functions (must be public for C callbacks)
  int handleVideoSequence(RocdecVideoFormat* videoFormat);
  int handlePictureDecode(RocdecPicParams* picParams);
  int handlePictureDisplay(RocdecParserDispInfo* dispInfo);

  std::string getDetails() override;

 private:
  int sendRocDecPacket(RocdecSourceDataPacket& rocDecPacket);

  void initializeBSF(
      const AVCodecParameters* codecPar,
      const UniqueDecodingAVFormatContext& avFormatCtx);
  
  // Apply bitstream filter, returns filtered packet or original if no filter
  // needed.
  ReferenceAVPacket& applyBSF(
      ReferenceAVPacket& packet,
      ReferenceAVPacket& filteredPacket);

  UniqueAVFrame convertRocmFrameToAVFrame(
      void* framePtr[3],
      unsigned int pitch,
      const RocdecParserDispInfo& dispInfo);

  // Transfer CPU-decoded frame to GPU memory in NV12 format (for fallback)
  UniqueAVFrame transferCpuFrameToGpuNV12(UniqueAVFrame& cpuFrame);

  RocdecVideoParser videoParser_ = nullptr;
  UniqueRocDecDecoder decoder_;
  RocdecVideoFormat videoFormat_ = {};

  std::queue<RocdecParserDispInfo> readyFrames_;

  bool eofSent_ = false;

  AVRational timeBase_ = {0, 1};
  AVRational frameRateAvgFromFFmpeg_ = {0, 1};

  UniqueAVBSFContext bitstreamFilter_;

  // RPP context for color conversion
  UniqueRppContext rppCtx_;

  // HIP stream for async memory operations
  hipStream_t rocdecStream_ = nullptr;

  // CPU fallback for unsupported formats
  std::unique_ptr<DeviceInterface> cpuFallback_;

  // Store codec context for fallback color range info
  SharedAVCodecContext codecContext_;

  // Chroma upsampling mode: NN for 8-bit, bilinear for 10-bit
  ChromaUpsampling chromaUpsampling_ = ChromaUpsampling::kNearestNeighbor;
  
  // Software scaling context for format conversion (fallback)
  UniqueSwsContext swsContext_;
  SwsFrameContext prevSwsFrameContext_;
};

} // namespace facebook::torchcodec

/* clang-format off */
// Note: [General design, sendPacket, receiveFrame, frame ordering and rocDecode callbacks]
//
// This interface provides hardware-accelerated video decoding on AMD GPUs using
// rocDecode Library (https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode), which provides access to
// AMD's VCN (Video Core Next) hardware decoder.
//
// Architecture Design:
// 
// At a high level, this decoding interface mimics the FFmpeg send/receive
// architecture while using rocDecode for hardware acceleration:
// - sendPacket(AVPacket) sends an AVPacket from the FFmpeg demuxer to the
//   rocDecode parser.
// - receiveFrame(AVFrame) is a non-blocking call:
//   - if a frame is ready **in display order**, it returns it. By display
//   order, we mean that receiveFrame() must return frames with increasing pts
//   values when called successively.
//   - if no frame is ready, it returns AVERROR(EAGAIN) to indicate the
//   caller should send more packets.
//
// Frame Re-ordering and rocDecode Callbacks:
// ==========================================
// SendPacket(AVPacket)'s job is to pass down the packet to the rocDecode
// parser by calling RocdecParseVideoData(packet). When
// RocdecParseVideoData(packet) is called, it may trigger callbacks:
//
// - handleVideoSequence(videoFormat): triggered once at the start of the
//   stream, and possibly later if the stream properties change (e.g.
//   resolution). This is where we create/configure the decoder.
//
// - handlePictureDecode(picParams): triggered **in decode order** when the
//   parser has accumulated enough data to decode a frame. We send that frame to
//   the VCN hardware for **async** decoding via RocdecDecodeFrame().
//
// - handlePictureDisplay(dispInfo): triggered **in display order** when a
//   frame is ready to be "displayed" (returned). At that point, the parser also
//   gives us the pts of that frame. We store (a reference to) that frame in a
//   FIFO queue: readyFrames_.
//
// When receiveFrame(AVFrame) is called, if readyFrames_ is not empty, we pop
// the front of the queue, which is the next frame in display order, and get it
// from the decoder by calling rocdecGetVideoFrame(). If readyFrames_ is empty we
// return EAGAIN to indicate the caller should send more packets.
//
// Note on Frame Lifetime:
// =======================
// We release the previous frame
// before getting a new one to avoid holding too many frames in decoder memory.
//
// Supported Codecs (via AMD VCN hardware):
// =========================================
// - H.265 (HEVC) - 8 bit and 10 bit
// - H.264 (AVC) - 8 bit
// - AV1 - 8 bit and 10 bit
// - VP9 - 8 bit and 10 bit
//
// Hardware Requirements:
// ======================
// - AMD GPU with gfx908 or higher (RDNA 2+, CDNA 2+)
// - ROCm 7.13.0 or later
// - libva-amdgpu-dev (VA-API AMD implementation)
// - mesa-amdgpu-va-drivers
//
// Color Conversion:
// =================
// Decoded frames are in NV12 format from VCN hardware. We use RPP (ROCm
// Performance Primitives) for GPU-accelerated NV12->RGB conversion, keeping
// all data in GPU memory.
//
// CPU Fallback:
// =============
// If rocDecode is unavailable or the video format is not supported by VCN
// hardware, we automatically fall back to CPU decoding (similar to NVDEC
// implementation).
//
/* clang-format on */
