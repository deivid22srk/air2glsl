// Copyright 2026 air2glsl contributors
// SPDX-License-Identifier: MPL-2.0
//
// Air (LLVM IR) to Vulkan GLSL translation.
//
// Design notes, because the choices are not obvious:
//
//  * Target is Vulkan GLSL, not GLSL ES. Metal hands shaders raw GPU
//    addresses ("argument buffers" contain pointers, and the shader follows
//    them), and the only GLSL feature that can express that is
//    GL_EXT_buffer_reference with buffer_reference types. Vulkan GLSL is also
//    where GL_EXT_nonuniform_qualifier lives, which the texture path needs.
//
//  * Control flow uses a dispatcher (`_pc` + `switch` in a loop) instead of
//    structured if/else/while. Air from Ubisoft's HLSL pipeline is heavily
//    optimised and partly flattened, so reconstructing structure is a separate
//    problem; the dispatcher makes any CFG expressible, which lets this
//    prototype measure the parts that are actually hard (types, GEP, argument
//    buffers, intrinsics). Structured output is a later optimisation.
//
//  * Phi nodes are removed with LLVM's reg2mem pass before emitting, so the
//    emitter only has to handle loads and stores.
//
//  * Entry-point inputs are stubbed in this version: the translated body is
//    emitted as a function that the generated `main()` calls with zeroes. The
//    goal of v0 is to prove that the *body* translates and compiles.

#pragma once

#include <llvm/IR/Module.h>

#include <string>

namespace air2glsl {

/// Output of a translation attempt.
struct Translation {
    /// The GLSL source, when `ok` is true.
    std::string source;
    /// Stage, for the caller's bookkeeping: "fragment", "vertex", "kernel" or
    /// "unknown".
    std::string stage;
    /// Name of the entry point that was translated.
    std::string entry_name;
    /// Human-readable reason when `ok` is false.
    std::string error;
    bool ok = false;
};

/// Translates the entry point of `module` to Vulkan GLSL.
///
/// `module` is modified: the entry function is demoted with reg2mem and
/// internal helper functions are inlined first.
Translation translate(const llvm::Module &module);

/// Detects the Air stage ("vertex", "fragment", "kernel") of `module` from the
/// `!air.*` named metadata, and sets `entry` to the entry function. Returns an
/// empty string when the module has no Air entry point.
std::string find_stage(const llvm::Module &module, llvm::Function *&entry);

}  // namespace air2glsl