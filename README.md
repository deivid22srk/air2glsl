# air2glsl — Apple "Air" shader bitcode → Vulkan GLSL

Prototype translator for the shaders shipped inside iOS games as precompiled
Metal binaries. It exists to answer one question with measurements instead of
guesswork: **how much work is it to run a Metal-only game's shaders on a
non-Apple GPU?**

Background: `docs/ASSASSINS_CREED_MIRAGE.md` in
[HyperHLE-Reborn](https://github.com/deivid22srk/HyperHLE-Reborn).

## What it does

1. Parses the `SBIN` container used by Assassin's Creed Mirage
   (`ShaderData.bin`, 5,924 `MTLB` sections).
2. Extracts the LLVM bitcode wrapped inside each section (Apple's
   `0x0B17C0DE` wrapper).
3. Detects the entry point and stage from Air's `!air.vertex` /
   `!air.fragment` / `!air.kernel` metadata.
4. Emits Vulkan GLSL for the shader body.
5. Reports per-shader failures with a reason, so progress is measurable.

## What the translation looks like

An Air argument buffer is a struct of pointers, and the shader follows them:

```llvm
%struct__desc = type { ptr addrspace(1), ptr addrspace(1), i64 }
%..unpack14 = load ptr addrspace(1), ptr addrspace(2) %..elt13, align 8
%21 = call { <4 x float>, i8 } @air.sample_texture_2d_array.v4f32(
        ptr addrspace(1) %..unpack14, ptr addrspace(2) %.1.unpack, ...)
```

which becomes address arithmetic on a `uint64_t` handle, a cast to a
`buffer_reference` type when memory is actually touched, and a documented
helper for the resource:

```glsl
_t10 = Ref0((_arg4 + 48L)).v0;            // load a pointer from an arg buffer
_t12 = s1(air_sample_2d_array(_t11, _t7, _arg1, 0, 0.0, 0.0), 0);
```

Control flow becomes a dispatcher, so that any CFG translates:

```glsl
int _pc = 0;
for (int _guard = 0; _guard < 65536; ++_guard) {
    switch (_pc) {
        case 0: { _t4 = (_arg0 * 2.0); _pc = 3; break; }
        case 3: { if (_t4 > 1.0) { _pc = 7; } else { _pc = 5; } break; }
        ...
    }
    if (_pc < 0) break;
}
```

## Design choices

- **Vulkan GLSL, not GLSL ES.** Metal passes raw GPU addresses around: argument
  buffers contain pointers and the shader follows them. The only GLSL feature
  that expresses that is `GL_EXT_buffer_reference`. Texture access also needs
  `GL_EXT_nonuniform_qualifier`.
- **One `buffer_reference` type per pointee type** (`struct { T v0; }`). Byte
  offsets come from the IR's own `DataLayout`, so the emitter never has to
  reproduce Apple's struct layout; the member is always at offset 0.
- **A `switch`-based dispatcher for control flow.** Air from the app is heavily
  optimised and partly flattened, so reconstructing structured `if`/`while` is a
  separate problem. The dispatcher handles any CFG, which lets this prototype
  measure the parts that are actually hard (types, GEP, argument buffers,
  intrinsics).
- **Phi nodes become variables** assigned on the edge that jumps to their block.
- **Stage inputs and outputs are generated from Air's metadata**, not stubbed:
  `gl_FragCoord`, `gl_Position`, `gl_VertexIndex`, the workgroup built-ins,
  varyings at their locations, and argument buffers as host-supplied addresses.
- **Resource handles carry a descriptor index in their low 16 bits**, which is
  the contract `air_sample_2d_array` documents. A real driver resolves the
  handle through the resource descriptor heap instead.

## Usage

```sh
make                       # needs LLVM 18 (llvm-config), see Makefile

# container + IR statistics
./air2glsl stats /path/to/ShaderData.bin

# write one .glsl per shader plus report.tsv
./air2glsl dump /path/to/ShaderData.bin out/ --limit 200

# compile every generated shader and count successes
tools/validate.sh out/
```

## Status (measured on Assassin's Creed Mirage 1.3.0)

| step | result |
| --- | --- |
| Container → bitcode | 5,924 / 5,924 payloads, 0 parse failures |
| Shaders by stage | 3,309 compute, 2,208 fragment, 407 vertex |
| Translate to Vulkan GLSL | 5,796 / 5,924 (97.8 %) |
| Compile with `glslangValidator` | 5,796 / 5,796 (100 %) |

v0 proves the pipeline end to end and reports the failure distribution. It is
not a usable driver component yet: stage I/O is stubbed, resource handles are a
documented convention instead of a real descriptor heap, atomics and
compare-sampling are missing, and the control-flow dispatcher is slow. See
`NOTES.md` for the full list, the measured sizes, and the order to fix them.