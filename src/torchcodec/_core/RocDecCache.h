// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <map>
#include <memory>
#include <mutex>

#include <hip/hip_runtime.h>
#include <torch/types.h>

#include <rocdecode/rocdecode.h>
#include <rocdecode/rocparser.h>

namespace facebook::torchcodec {

// This file implements a cache for rocDecode decoders.
// Cache key based on
// decoder parameters specific to AMD VCN hardware.

struct RocDecDecoderDeleter {
  void operator()(rocDecDecoderHandle* decoderPtr) const {
    if (decoderPtr && *decoderPtr) {
      rocDecDestroyDecoder(*decoderPtr);
      delete decoderPtr;
    }
  }
};

using UniqueRocDecDecoder =
    std::unique_ptr<rocDecDecoderHandle, RocDecDecoderDeleter>;

// A per-device cache for rocDecode decoders. There is one instance of this
// class per GPU device, and it is accessed through the static getCache() method.
class RocDecCache {
 public:
  static RocDecCache& getCache(const torch::Device& device);

  // Get decoder from cache - returns nullptr if none available
  UniqueRocDecDecoder getDecoder(RocdecVideoFormat* videoFormat);

  // Return decoder to cache - returns true if added to cache
  bool returnDecoder(
      RocdecVideoFormat* videoFormat,
      UniqueRocDecDecoder decoder);

 private:
  // Cache key struct: a decoder can be reused and taken from the cache only if
  // all these parameters match.
  struct CacheKey {
    rocDecVideoCodec codecType;
    uint32_t width;
    uint32_t height;
    rocDecVideoChromaFormat chromaFormat;
    uint32_t bitDepthMinus8;
    uint8_t numDecodeSurfaces;

    CacheKey() = delete;

    explicit CacheKey(RocdecVideoFormat* videoFormat)
        : codecType(videoFormat->codec),
          width(videoFormat->coded_width),
          height(videoFormat->coded_height),
          chromaFormat(videoFormat->chroma_format),
          bitDepthMinus8(videoFormat->bit_depth_luma_minus8),
          numDecodeSurfaces(videoFormat->min_num_decode_surfaces) {}

    CacheKey(const CacheKey&) = default;
    CacheKey& operator=(const CacheKey&) = default;

    bool operator<(const CacheKey& other) const {
      return std::tie(
                 codecType,
                 width,
                 height,
                 chromaFormat,
                 bitDepthMinus8,
                 numDecodeSurfaces) <
          std::tie(
                 other.codecType,
                 other.width,
                 other.height,
                 other.chromaFormat,
                 other.bitDepthMinus8,
                 other.numDecodeSurfaces);
    }
  };

  RocDecCache() = default;
  ~RocDecCache() = default;

  std::map<CacheKey, UniqueRocDecDecoder> cache_;
  std::mutex cacheLock_;

  // Max number of cached decoders, per device
  static constexpr int MAX_CACHE_SIZE = 20;
};

} // namespace facebook::torchcodec
