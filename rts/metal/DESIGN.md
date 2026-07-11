# Futhark Metal backend — internal design conventions

This file pins the conventions shared between the kernel code generator
(`src/Futhark/CodeGen/ImpGen/GPU/ToOpenCL.hs`, `TargetMetal` case), the MSL
prelude (`rts/metal/prelude.metal`), the C runtime backend
(`rts/c/backends/metal.h`), and the metal-cpp shim (`rts/metal/shim.cpp`).
Change any of these conventions in all places at once or not at all.

## Overall shape

- The generated program is the standard single Futhark C99 file. MSL kernel
  source is embedded as `gpu_program[]` strings and compiled at context setup
  via `MTLDevice::newLibrary` (runtime JIT, like OpenCL/NVRTC).
- metal-cpp requires C++; the generated C file cannot be C++. So there are two
  translation units:
  1. `prog.c` — everything Futhark normally generates, incl.
     `rts/c/backends/metal.h` implementing the `gpu.h` contract as thin
     wrappers over the `futmtl_*` C API.
  2. `prog.metal.cpp` — a verbatim copy of `rts/metal/shim.cpp`, written next
     to `prog.c` at compile time. The only TU that includes metal-cpp; defines
     `NS_PRIVATE_IMPLEMENTATION` etc. Exposes `extern "C"` `futmtl_*`.
  Build: `cc -std=c99 -c prog.c` ; `c++ -std=c++17 -c prog.metal.cpp` ;
  link with `-framework Metal -framework Foundation -framework QuartzCore -lc++`.
  (Precedent: the ISPC backend's two-compiler build in `Actions.hs`.)
- metal-cpp headers are vendored under `rts/metal/metal-cpp/` and located at
  build time via `-I`; the `FUTHARK_METAL_CPP` environment variable overrides
  the include path.

## Kernel ABI (the load-bearing part)

Kernel argument *order* is fixed by the shared machinery
(`Backends/GPU.hs::genKernelFunction` builds `args[]`/`args_sizes[]`;
`rts/c/gpu.h` builtins do the same): `numFailureParams(safety)` failure args
first, then the kernel's value/memory args in `kernelUses` order.

The MSL side mirrors this order and relies on **automatic buffer index
assignment**: kernel parameters that are device/constant pointers get
`[[buffer(0..)]]` in declaration order when no explicit attribute is given.
Therefore:

- Parameter i of the MSL kernel (counting only buffer-typed params, in order)
  corresponds to `args[i]` at launch, bound to buffer index i.
- Dynamic shared memory is a `threadgroup` parameter — a *threadgroup* slot,
  not a buffer slot — declared first, always present on every kernel
  (generated and builtin):
    `__local uint64_t *shared_mem_aligned`   (macro-expands to
    `threadgroup uint64_t *shared_mem_aligned`, auto threadgroup index 0)
  The runtime always calls `setThreadgroupMemoryLength(max(bytes,16), 0)`.
- Failure params (present per `KernelSafety`, 0–3 of them, in this order):
    `__global int *global_failure`
    `__constant int *failure_is_an_option_p` (+ unpack, see scalars below)
    `__global int64_t *global_failure_args`
- Scalar args are passed with `setBytes` on the host and declared as
  **constant pointers** (C-parseable; MSL binds setBytes data to
  `constant T*` args just as it does `constant T&`):
    `__constant uint32_t *x_bits_p` + body unpack `float x = ...(*x_bits_p);`
  Bool/unit scalars use `unsigned char` storage like the OpenCL target.
  Hand-written RTS kernels (transpose.cl, copy.cl) use the prelude's
  `SCALAR_KARG(T)` macro instead (`constant T&` under Metal, plain `T`
  elsewhere) — same binding semantics, one buffer slot each.
  (By-value scalar kernel parameters are illegal MSL; spec §5.2.)
- Memory args: `__global unsigned char *name`. **Codegen must never
  emit `const device` pointers**: the launch discriminator below maps
  read-only bindings to setBytes, so read-only-ness must imply
  "scalar" (constant address space).
- Thread-index builtins are extra trailing parameters, spelled via
  name-position macros so the C AST stays plain C; the prelude expands them
  to attributed MSL parameters:
    `uint3 FUTHARK_TID_PARAM`    -> `__futhark_ltid [[thread_position_in_threadgroup]]`
    `uint3 FUTHARK_TBLOCK_PARAM` -> `__futhark_tblock [[threadgroup_position_in_grid]]`
    `uint3 FUTHARK_TBSIZE_PARAM` -> `__futhark_tbsize [[threads_per_threadgroup]]`
  These come *after* all buffer params and consume no buffer indices.

Launch mapping in the runtime (`gpu_launch_kernel` in metal.h → shim):
- For each of the `num_args` entries, the shim consults per-kernel reflection
  (captured at pipeline creation with binding info): buffer bindings with
  read-only access (`constant` args — scalars) get
  `setBytes(args[i], args_sizes[i], i)`; writable buffer bindings
  (non-const `device` args — memory) hold a `gpu_mem*` in `args[i]` and get
  `setBuffer(mem->buf, 0, i)`.
- `grid` from `gpu_launch_kernel` is in *thread blocks*:
  `dispatchThreadgroups(MTLSize(grid), threadsPerThreadgroup=MTLSize(block))`.

## Prelude macro layer (`rts/metal/prelude.metal`)

Starts with `#include <metal_stdlib>` and `using namespace metal;`, then
`#define FUTHARK_METAL` is emitted before it by the code generator. Defines:

- Address spaces (same syntactic position as OpenCL, so the shared codegen
  and multi-dialect RTS files work unchanged):
    `#define __global device`, `#define __local threadgroup`,
    `#define __private thread`, `#define __constant constant`
- `FUTHARK_KERNEL` -> `kernel void`;
  `FUTHARK_KERNEL_SIZED(x,y,z)` -> `[[max_total_threads_per_threadgroup(x*y*z)]] kernel void`
- Thread indexing (reads the builtin params above, so *every kernel signature
  must include the three FUTHARK_*_PARAM parameters*):
    `get_local_id(d)`, `get_tblock_id(d)`, `get_local_size(d)` as macros over
    `__futhark_ltid` / `__futhark_tblock` / `__futhark_tbsize`;
    `LOCKSTEP_WIDTH` = 1 for now (conservative).
- Barriers/fences with OpenCL spellings:
    `CLK_LOCAL_MEM_FENCE` / `CLK_GLOBAL_MEM_FENCE` flag constants;
    `barrier(flags)` -> `threadgroup_barrier(mem_flags::mem_threadgroup)` or
    `threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup)`
    when the global flag is set (inline function; flags always constant);
    `mem_fence_local()` / `mem_fence_global()` via `atomic_thread_fence`
    with `memory_order_seq_cst` (a relaxed fence is a spec'd no-op).
    **This sets the backend's floor at Metal 3.2 on Apple silicon** —
    older macOS and Intel/AMD Macs are out of scope.
- Integer typedefs (`int8_t` … `uint64_t`) mapped to MSL types (`long` is
  64-bit in MSL).

## Atomics

Same call-name convention as the OpenCL/CUDA targets:
`atomic_<op>_<type>_<space>(ptr, ...)` with `<space>` in `{global, shared}`,
e.g. `atomic_add_i32_global(p, x)`, `atomic_cmpxchg_i32_shared(p, cmp, val)`.
Implementations live in the `FUTHARK_METAL` sections of `rts/c/atomics*.h`
(these files are already multi-dialect with FUTHARK_OPENCL/FUTHARK_CUDA
sections) over `atomic_fetch_*_explicit` with `memory_order_relaxed`,
casting the incoming `volatile device T*` to `device atomic_*T*`.

Supported natively: 32-bit int/uint ops, 32-bit cmpxchg/xchg, f32 add
(via native `atomic_float` fetch-add where available, else a CAS loop on
`atomic_uint` with bitcasts). 8/16-bit ops keep the existing emulation on top
of 32-bit cmpxchg where the RTS already does that. **64-bit atomics are not
provided**; the code generator rejects them at compile time with a clear
error. f64 anything is rejected (Apple GPUs have no double).

## Failure protocol / no `goto`

MSL has no `goto`. The OpenCL/CUDA failure protocol uses forward gotos to the
next `ErrorSync` barrier in kernels that both (a) contain failure points
(asserts/bounds checks) and (b) contain barriers. For Metal:

- Kernels with failure points but no barriers: unchanged (plain `return`
  after recording the failure) — this covers ordinary bounds-checked map
  kernels.
- Kernels with failure points *and* barriers: compile-time error naming the
  kernel and suggesting `--unsafe`. Revisit post-v1 (structured
  `if (!local_failure)` guarding) if it bites in practice.
- No labels are ever emitted for Metal (labels without goto are useless and
  possibly unparseable).

## Runtime semantics (`metal.h` + `shim.cpp`)

- `gpu_mem` = `futmtl_mem*` (wraps `MTL::Buffer*`); `gpu_kernel` =
  `futmtl_kernel*` (wraps `MTL::ComputePipelineState*` + arg-kind table from
  reflection).
- Buffers are `storageModeShared` (Apple Silicon UMA). host<->device memcpy =
  `memcpy` against `buf->contents()` with a sync of pending GPU work when
  needed (`sync` flag semantics as in `opencl.h`).
- One `MTLCommandBuffer` per kernel launch / device copy, committed
  immediately. The context tracks the last committed command buffer;
  `futhark_context_sync` waits on it, checks `MTLCommandBuffer` status/error,
  then performs the standard `global_failure` readback (clone the logic in
  `cuda.h`'s `futhark_context_sync`).
- Program build at context setup: concatenate `gpu_program[]`, prepend
  tuning-param macros as preprocessor defines via `MTLCompileOptions`
  (mirror `cuda_nvrtc_mk_build_options`), `newLibrary`, then one
  `newComputePipelineState` per kernel at `gpu_create_kernel` (with the
  reflection option to capture argument info).
- Device limits for `gpu_init_log` / heuristics:
  `max_thread_block_size` = `maxTotalThreadsPerThreadgroup`,
  `max_shared_memory` = `maxThreadgroupMemoryLength`,
  `lockstep_width` = `threadExecutionWidth`, `max_grid_size` = 2^31-1,
  sensible constants for cache/registers like other backends.
- Profiling hooks may be stubbed (zeros) in v1; `futhark bench` measures wall
  clock and does not need them.
- No binary caching in v1 (`cfg->cache_fname` accepted but ignored).

## Compile-time rejections (clear errors, not silent miscompiles)

- f64 anywhere in kernel types.
- 64-bit atomics.
- Failure points in kernels with barriers (see above).
- `SegHist` never reaches the Metal target in this project's scope; if it
  does, the generic lowering emits atomics that hit the above rejections.
