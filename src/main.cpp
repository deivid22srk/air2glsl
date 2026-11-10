// Copyright 2026 air2glsl contributors
// SPDX-License-Identifier: MPL-2.0
//
// air2glsl: translates Apple "Air" shader bitcode (as shipped in
// ShaderData.bin-style containers) into Vulkan GLSL.
//
// Usage:
//   air2glsl stats <ShaderData.bin>
//   air2glsl dump  <ShaderData.bin> <out-dir> [--limit N]
//
// `stats` reports the container and IR facts. `dump` writes one .glsl per
// shader plus a report of what failed and why.

#include "emit.h"
#include "metallib.h"

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace llvm;

std::unique_ptr<Module> parse(std::vector<uint8_t> &bytes) {
    StringRef data(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    auto buffer = MemoryBuffer::getMemBuffer(data, "shader", false);
    static LLVMContext context;
    auto module = parseBitcodeFile(buffer->getMemBufferRef(), context);
    if (!module) {
        consumeError(module.takeError());
        return nullptr;
    }
    return std::move(*module);
}

void usage() {
    std::fprintf(stderr,
                 "usage: air2glsl stats <ShaderData.bin>\n"
                 "       air2glsl dump  <ShaderData.bin> <out-dir> "
                 "[--limit N]\n"
                 "       air2glsl meta  <ShaderData.bin> [--limit N]\n");
}

int do_stats(const std::string &path) {
    auto data = air2glsl::read_file(path);
    const auto metallibs = air2glsl::find_metallibs(data);
    const auto blobs = air2glsl::extract_bitcode(data, metallibs);
    std::printf("container:            %s\n", path.c_str());
    std::printf("sections (MTLB):      %zu\n", metallibs.size());
    std::printf("bitcode payloads:     %zu\n", blobs.size());

    size_t parsed = 0, failed = 0;
    std::map<std::string, size_t> stages;
    size_t functions = 0, blocks = 0, instructions = 0;
    for (const auto &blob : blobs) {
        std::vector<uint8_t> bytes(data.begin() + blob.bitcode_offset,
                                   data.begin() + blob.bitcode_offset + blob.size);
        std::unique_ptr<Module> module = parse(bytes);
        if (!module) {
            ++failed;
            continue;
        }
        ++parsed;
        Function *entry = nullptr;
        const std::string stage = air2glsl::find_stage(*module, entry);
        stages[stage.empty() ? "unknown" : stage]++;
        for (const Function &fn : *module) {
            if (fn.isDeclaration()) continue;
            ++functions;
            for (const BasicBlock &bb : fn) {
                ++blocks;
                instructions += bb.size();
            }
        }
    }
    std::printf("parsed modules:       %zu (%zu failed)\n", parsed, failed);
    std::printf("functions:            %zu\n", functions);
    std::printf("basic blocks:         %zu\n", blocks);
    std::printf("instructions:         %zu\n", instructions);
    for (const auto &entry : stages) {
        std::printf("stage %-9s       %zu\n", entry.first.c_str(), entry.second);
    }
    return 0;
}

int do_meta(const std::string &path, size_t limit, long section) {
    auto data = air2glsl::read_file(path);
    const auto metallibs = air2glsl::find_metallibs(data);
    const auto blobs = air2glsl::extract_bitcode(data, metallibs);
    std::map<std::string, size_t> shown;
    size_t examined = 0;
    for (const auto &blob : blobs) {
        if (examined >= limit) break;
        if (section >= 0 && blob.metallib_index != static_cast<size_t>(section)) {
            continue;
        }
        ++examined;
        std::vector<uint8_t> bytes(data.begin() + blob.bitcode_offset,
                                   data.begin() + blob.bitcode_offset + blob.size);
        std::unique_ptr<Module> module = parse(bytes);
        if (!module) continue;
        Function *entry = nullptr;
        const std::string stage = air2glsl::find_stage(*module, entry);
        if (stage.empty()) continue;
        if (shown[stage] >= 1) continue;
        shown[stage]++;
        std::printf("=== %s shader (payload %zu) ===\n", stage.c_str(),
                    examined);
        for (const std::string &line : air2glsl::describe_entry_io(*module)) {
            std::printf("%s\n", line.c_str());
        }
        std::printf("\n");
    }
    return 0;
}

int do_dump(const std::string &path, const std::string &out_dir, size_t limit) {
    auto data = air2glsl::read_file(path);
    const auto metallibs = air2glsl::find_metallibs(data);
    const auto blobs = air2glsl::extract_bitcode(data, metallibs);
    std::filesystem::create_directories(out_dir);

    std::ofstream report(out_dir + "/report.tsv");
    report << "index\tstage\tstatus\tdetail\n";
    size_t ok = 0, bad = 0, examined = 0;
    std::map<std::string, size_t> reasons;
    std::map<std::string, size_t> stages_ok;
    for (const auto &blob : blobs) {
        if (limit != 0 && examined >= limit) break;
        ++examined;
        std::vector<uint8_t> bytes(data.begin() + blob.bitcode_offset,
                                   data.begin() + blob.bitcode_offset + blob.size);
        std::unique_ptr<Module> module = parse(bytes);
        air2glsl::Translation translation;
        if (!module) {
            translation.error = "bitcode parse failed";
        } else {
            translation = air2glsl::translate(*module);
        }
        // The stage goes in the file name so external tools (glslangValidator,
        // spirv-cross) can pick it without a side channel.
        const std::string name =
            "shader_" + std::to_string(blob.metallib_index) + "_" +
            std::to_string(blob.index_in_metallib) + "_" +
            (translation.stage.empty() ? "unknown" : translation.stage);
        if (translation.ok) {
            std::ofstream out(out_dir + "/" + name + ".glsl");
            out << translation.source;
            ++ok;
            stages_ok[translation.stage]++;
            report << examined << "\t" << translation.stage << "\tok\t-\n";
        } else {
            ++bad;
            reasons[translation.error]++;
            report << examined << "\t" << translation.stage << "\tfailed\t"
                   << translation.error << "\n";
        }
    }
    std::printf("examined: %zu\nok:       %zu\nfailed:   %zu\n", examined, ok,
                bad);
    for (const auto &entry : stages_ok) {
        std::printf("  ok %-9s %zu\n", entry.first.c_str(), entry.second);
    }
    std::printf("failure reasons:\n");
    for (const auto &entry : reasons) {
        std::printf("  %6zu  %s\n", entry.second, entry.first.c_str());
    }
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        usage();
        return 2;
    }
    const std::string command = argv[1];
    try {
        if (command == "stats") {
            return do_stats(argv[2]);
        }
        if (command == "meta") {
            size_t limit = 400;
            long section = -1;
            for (int i = 3; i + 1 < argc; ++i) {
                if (std::strcmp(argv[i], "--limit") == 0) {
                    limit = std::strtoul(argv[i + 1], nullptr, 10);
                }
                if (std::strcmp(argv[i], "--section") == 0) {
                    section = std::strtol(argv[i + 1], nullptr, 10);
                }
            }
            return do_meta(argv[2], limit, section);
        }
        if (command == "dump") {
            if (argc < 4) {
                usage();
                return 2;
            }
            size_t limit = 0;
            for (int i = 4; i + 1 < argc; ++i) {
                if (std::strcmp(argv[i], "--limit") == 0) {
                    limit = std::strtoul(argv[i + 1], nullptr, 10);
                }
            }
            return do_dump(argv[2], argv[3], limit);
        }
    } catch (const std::exception &e) {
        std::fprintf(stderr, "air2glsl: %s\n", e.what());
        return 1;
    }
    usage();
    return 2;
}