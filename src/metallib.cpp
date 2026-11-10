// Copyright 2026 air2glsl contributors
// SPDX-License-Identifier: MPL-2.0
#include "metallib.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace air2glsl {
namespace {

uint32_t read_u32(const std::vector<uint8_t> &data, size_t offset) {
    if (offset + 4 > data.size()) {
        throw std::runtime_error("read past end of file");
    }
    return static_cast<uint32_t>(data[offset]) |
           (static_cast<uint32_t>(data[offset + 1]) << 8) |
           (static_cast<uint32_t>(data[offset + 2]) << 16) |
           (static_cast<uint32_t>(data[offset + 3]) << 24);
}

bool matches(const std::vector<uint8_t> &data, size_t offset,
             const char *magic, size_t length) {
    return offset + length <= data.size() &&
           std::memcmp(data.data() + offset, magic, length) == 0;
}

}  // namespace

std::vector<uint8_t> read_file(const std::string &path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("cannot open " + path);
    }
    const std::streamsize size = file.tellg();
    if (size <= 0) {
        throw std::runtime_error("empty file: " + path);
    }
    std::vector<uint8_t> data(static_cast<size_t>(size));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char *>(data.data()), size)) {
        throw std::runtime_error("short read: " + path);
    }
    return data;
}

uint32_t sbin_section_count(const std::vector<uint8_t> &data) {
    if (!matches(data, 0, "SBIN", 4)) {
        throw std::runtime_error("not an SBIN container");
    }
    return read_u32(data, 8);
}

std::vector<size_t> find_metallibs(const std::vector<uint8_t> &data) {
    const uint32_t expected = sbin_section_count(data);
    std::vector<size_t> offsets;
    for (size_t i = 4; i + 4 <= data.size(); ++i) {
        if (data[i] == 'M' && matches(data, i, "MTLB", 4)) {
            offsets.push_back(i);
            i += 3;
        }
    }
    if (offsets.size() != expected) {
        throw std::runtime_error(
            "SBIN header declares " + std::to_string(expected) +
            " sections but the file contains " +
            std::to_string(offsets.size()) + " MTLB headers");
    }
    return offsets;
}

std::vector<ShaderBlob> extract_bitcode(const std::vector<uint8_t> &data,
                                        const std::vector<size_t> &metallibs) {
    std::vector<ShaderBlob> blobs;
    for (size_t m = 0; m < metallibs.size(); ++m) {
        const size_t begin = metallibs[m];
        const size_t end = (m + 1 < metallibs.size()) ? metallibs[m + 1]
                                                      : data.size();
        size_t index_in_section = 0;
        for (size_t i = begin; i + 20 <= end; ++i) {
            if (read_u32(data, i) != kBitcodeWrapperMagic) {
                continue;
            }
            const uint32_t offset = read_u32(data, i + 8);
            const uint32_t size = read_u32(data, i + 12);
            const size_t bitcode = i + offset;
            if (size == 0 || bitcode + size > data.size() ||
                !matches(data, bitcode, "BC\xC0\xDE", 4)) {
                // Not a real wrapper; could be a coincidence in the payload.
                continue;
            }
            ShaderBlob blob;
            blob.wrapper_offset = i;
            blob.bitcode_offset = bitcode;
            blob.size = size;
            blob.metallib_index = m;
            blob.index_in_metallib = index_in_section++;
            blobs.push_back(blob);
            i += 15;  // skip the rest of the wrapper header
        }
    }
    return blobs;
}

}  // namespace air2glsl