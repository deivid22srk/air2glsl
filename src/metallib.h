// Copyright 2026 air2glsl contributors
// SPDX-License-Identifier: MPL-2.0
//
// ShaderData.bin ("SBIN") container reader.
//
// The container is Ubisoft's, the sections inside it are Apple's:
//
//   SBIN  <u32 version> <u32 section_count> <16-byte uuid>
//   then `section_count` sections, each holding a .metallib ("MTLB") whose
//   payload is Apple "Air" LLVM bitcode.
//
// Two layers are parsed here:
//
//  * `find_metallibs` locates the MTLB headers. It relies on the count in the
//    SBIN header to validate the scan, so a stray "MTLB" inside shader data
//    cannot silently inflate the results.
//  * `extract_bitcode` finds Apple's bitcode wrapper. A .metallib stores each
//    function's bitcode wrapped in a small header:
//
//        u32 magic  = 0x0B17C0DE
//        u32 version
//        u32 offset (from the start of the wrapper to the bitcode)
//        u32 size
//        u32 cpu_type (0xFFFFFFFF for Air)
//        ... followed by the payload, which begins with "BC\xC0\xDE"
//
//    The wrapper magic is looked up directly rather than by walking the
//    tagged stream (the MTLB uses NAME/OFFT/ENDT tags) because the wrapper is
//    the authoritative record of the payload's offset and size, and it also
//    gives us the cpu_type field.
//
// Sources: the .metallib layout was popularised by the reverse-engineering
// write-ups at <https://worthdoingbadly.com/metalbitcode/>; the LLVM bitcode
// magic is documented in <llvm/Bitcode/BitcodeWriter.h>.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace air2glsl {

/// Apple's bitcode wrapper magic, `0x0B17C0DE`, as stored little-endian.
inline constexpr uint32_t kBitcodeWrapperMagic = 0x0B17C0DEu;

/// A bitcode payload found inside a .metallib.
struct ShaderBlob {
    /// Byte offset of the wrapper in the containing file.
    size_t wrapper_offset = 0;
    /// Byte offset of the `BC C0 DE` magic in the containing file.
    size_t bitcode_offset = 0;
    /// Size of the bitcode, in bytes.
    uint32_t size = 0;
    /// Index of the .metallib this blob came from.
    size_t metallib_index = 0;
    /// Sequential index within its .metallib.
    size_t index_in_metallib = 0;
};

/// Reads a whole file into memory. Throws `std::runtime_error` on failure.
std::vector<uint8_t> read_file(const std::string &path);

/// Locates the "MTLB" headers in an SBIN container and returns their offsets.
/// Throws if the file is not an SBIN container.
std::vector<size_t> find_metallibs(const std::vector<uint8_t> &data);

/// Extracts every bitcode payload in `data`, in file order.
///
/// `metallibs` must be the result of [find_metallibs], so each blob can be
/// attributed to a section.
std::vector<ShaderBlob> extract_bitcode(const std::vector<uint8_t> &data,
                                         const std::vector<size_t> &metallibs);

/// The `section_count` field of the SBIN header. Used to validate scans.
uint32_t sbin_section_count(const std::vector<uint8_t> &data);

}  // namespace air2glsl
