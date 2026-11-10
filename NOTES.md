# Status and measurements

Everything here was measured on the real `ShaderData.bin` from
Assassin's Creed Mirage 1.3.0, on 2026-11-10, with LLVM 18 and
`glslangValidator 15.1.0` (`--target-env vulkan1.3`).

## Container and IR

```
sections (MTLB):      5924
bitcode payloads:     5924        (matches the SBIN header count)
parsed modules:       5924 (0 failed)
functions:            6240
basic blocks:         146645
instructions:         7056274
stage fragment        2208
stage kernel          3309
stage vertex          407
```

The stage split is worth noting on its own: 56 % of this game's shaders are
compute. That matches the `MTLIndirectCommandBufferDescriptor` /
`MTLComputePipelineDescriptor` usage found in the executable, and it means a
driver for this title has to get compute + indirect dispatch right, not just
rasterisation.

## Translation

```
examined: 5924
ok:       5796   (97.8%)
  fragment 2198
  kernel   3191
  vertex    407
```

Failure reasons, all reported per shader by the tool:

| count | reason |
| --- | --- |
| 90 | atomic read-modify-write on a texture buffer (`air.atomic_fetch_add_explicit_texture_buffer_1d`) |
| 11 | `air.get_depth_texture_3d` |
| 7 | `air.write_texture_3d.u.v4i32` |
| 6 | `llvm.fshl` (funnel shift) |
| 5 | assorted (`fast_tanh`, `simd_shuffle`, float buffer atomics, ...) |
| ~9 | remaining local-address cases and one `cmpxchg` |

## Compilation

```
compiled:  5796
failed:       0
```

Every shader the translator accepts now produces valid Vulkan GLSL according to
`glslangValidator --target-env vulkan1.3`. Getting there needed four real fixes
that are worth recording, because they are the kind of thing that only shows up
against real content:

1. **Type punning in locals.** The game stores floats through pointers to
   `[4 x i32]` arrays. GLSL cannot reinterpret bytes in a local, so any alloca
   whose address is used for a different type is moved into a host-allocated
   scratch buffer (`AIR_SCRATCH_BYTES`) and addressed with `buffer_reference`,
   which *can* address bytes.
2. **`threadgroup` globals.** Air gives `addrspace(3)` globals an undef
   initialiser and expects the driver to supply the address, which is exactly
   what the global-address block now models.
3. **Shadow sampling.** `textureGather` on a `sampler2DArrayShadow` takes the
   coordinate and the reference value as separate arguments, not a packed
   `vec4`, and a shadow sampler cannot be built from a separate texture and
   sampler, so depth-compare paths use a combined shadow sampler binding.
4. **`textureQueryLod`** cannot take a sampler constructed from a separate
   texture and sampler at all (glslang rejects it), and for an array sampler it
   takes only the coordinate, not the layer. The clamped-LOD intrinsic therefore
   uses one combined-sampler binding and drops the layer from the query.

## What v0 does not do (and the order to fix it)

1. ~~Stage I/O is stubbed~~ — **done**. `main()` is generated from Air's own
   metadata: `air.position` becomes `gl_FragCoord` (fragment) or the first
   return field becomes `gl_Position` (vertex), `air.vertex_id` /
   `air.instance_id` become `gl_VertexIndex` / `gl_InstanceIndex`, workgroup
   builtins (`thread_position_in_threadgroup`, `threadgroup_position_in_grid`,
   ...) become the matching `gl_*` built-ins, varyings become
   `layout(location = N) in/out`, and argument buffers become host-supplied
   addresses in `g_globals.io[]`. Integer varyings get `flat`, as GLSL requires.
   `air2glsl meta <ShaderData.bin> --section N` prints the parsed bindings.

   Two details found by testing against real content: the number inside
   `user(texcoord2)` is *not* a location (one fragment shader here has both
   `user(instanceid0)` and `user(texcoord0)`), so varyings are numbered by order
   of first appearance keyed by their full name; and `instanceid` is not a
   varying at all, so it is passed as a constant.
2. **Resource handles are a promise, not a heap.** `air_sample_2d_array` assumes
   the low 16 bits of a Metal resource handle hold the descriptor index the host
   bound. A real driver has to build the descriptor heaps the app fills:
   `smp_desc_heap_ab`, `res_desc_heap_ab` and `top_level_global_ab` appear as
   `air.indirect_buffer` arguments with `air.address_space 2` and an *empty*
   `air.struct_type_info`, i.e. their layout is defined by the runtime, not the
   IR. This is the descriptor-buffer/descriptor-indexing problem, and it is the
   biggest remaining piece of the graphics work.
3. **Atomics on texture buffers.** Global atomics (`atomicrmw`,
   `air.atomic.global.*`) now translate to GLSL atomics on `buffer_reference`
   members, which glslang accepts. The 90 remaining ones operate on *texture
   buffers* (a typed buffer view), which needs `imageAtomic*` through an
   `iimageBuffer` and a read-modify-write model of the value type.
4. **Compare/gather and gradient sampling.** 360 shaders use depth-compare
   sampling (`sampler2DShadow`, `textureGatherCompare`) and 75 use explicit
   gradients (`textureGrad`). Both are direct GLSL built-ins; they were simply
   not wired up in v0.
6. **Performance.** The `switch` dispatcher is correct but slow (each block is a
   case, every edge is a jump). Before this is usable in a game it needs
   structured output (`if`/`for`), which is a CFG-structuring pass over a
   reducible graph — well-understood work, but real work.
7. **Pipeline state.** The game also loads `MTLBinaryArchive` pipeline archives.
   Those are opaque compiled pipelines; the workable approach is to ignore them
   and build pipelines from the translated shaders at draw time.
8. **MetalFX** has no Vulkan equivalent and would have to be stubbed to an
   identity scaler to reach a first frame.

## Reproducing

```sh
make                                  # LLVM 18
./air2glsl stats ShaderData.bin
./air2glsl dump  ShaderData.bin out/
JOBS=8 tools/validate.sh out/
```

`ShaderData.bin` is Apple's and Ubisoft's content; it is not in this repository.
Extract it from your own copy of the app.