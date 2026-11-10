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
ok:       4891   (82.6%)
  fragment 1818
  kernel   2704
  vertex    369
```

Failure reasons, all reported per shader by the tool:

| count | reason |
| --- | --- |
| 280 | address of local memory escapes (a pointer to an alloca is passed on) |
| 189 | `air.gather_compare_depth_2d_array.f32` |
| 171 | `air.sample_compare_depth_2d_array.f32` |
| 75 | `air.sample_texture_2d_array_grad.v4f32` |
| 68 | atomic reads/writes on texture buffers |
| 51 | `air.write_texture_buffer_1d.v4f32` |
| 45 | `atomicrmw` |
| 35 | `air.calculate_clamped_lod_texture_2d_array` |
| 22 | `air.atomic.global.add.s.i32` |
| 22 | assorted (`llvm.fshl`, `fast_tanh`, `simd_shuffle`, `cmpxchg`, ...) |

## Compilation

```
compiled:  4889
failed:     2
```

So 4,889 of the 4,891 generated files are valid Vulkan GLSL according to
`glslangValidator`. The two failures are stores through a `buffer_reference`
whose member type was inferred from the stored value rather than the pointee
(a known v0 shortcut).

That 99.96 % figure is the useful one for planning: the *emitter* is not the
hard part. What remains is everything the emitter deliberately stubs.

## What v0 does not do (and the order to fix it)

1. **Stage I/O is stubbed.** The generated `main()` calls the translated body
   with zeroes. Air's `!air.vertex` / `!air.fragment` metadata already describes
   every input and output (`air.position`, `air.fragment_input`,
   `air.render_target`, `air.arg_name` → `sv_position0`, `texcoord0`, ...), so
   this is mechanical: emit `gl_Position`/`gl_FragCoord`/varyings with the
   locations the metadata gives, then drop the stub arguments.
2. **Resource handles are a promise, not a heap.** `air_sample_2d_array` assumes
   the low 16 bits of a Metal resource handle hold the descriptor index the host
   bound. A real driver has to build the descriptor heaps the app fills:
   `smp_desc_heap_ab`, `res_desc_heap_ab` and `top_level_global_ab` appear as
   `air.indirect_buffer` arguments with `air.address_space 2` and an *empty*
   `air.struct_type_info`, i.e. their layout is defined by the runtime, not the
   IR. This is the descriptor-buffer/descriptor-indexing problem, and it is the
   biggest remaining piece of the graphics work.
3. **Atomics.** 45 `atomicrmw` plus 90 atomic resource operations. GLSL needs
   SSBO/atomic-counter members or `imageAtomic*`; garbage-free translation needs
   a scratch buffer model, not a cast.
4. **Compare/gather and gradient sampling.** 360 shaders use depth-compare
   sampling (`sampler2DShadow`, `textureGatherCompare`) and 75 use explicit
   gradients (`textureGrad`). Both are direct GLSL built-ins; they were simply
   not wired up in v0.
5. **Local address escapes.** 280 shaders take the address of a local. The fix
   is to place such locals in a scratch SSBO addressed with `buffer_reference`,
   which the addressing model already supports.
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