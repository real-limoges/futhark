// Start of prelude.metal
//
// The Metal analog of rts/opencl/prelude.cl and rts/cuda/prelude.cu:
// defines the dialect-portability layer that lets the shared kernel
// code generator (ToOpenCL.hs) and the multi-dialect RTS device code
// (transpose.cl, copy.cl, atomics*.h, scalar defs) target MSL.
//
// The conventions here are load-bearing and documented in
// rts/metal/DESIGN.md; keep them in sync with the TargetMetal cases in
// src/Futhark/CodeGen/ImpGen/GPU/ToOpenCL.hs and with the argument
// binding logic in rts/metal/shim.cpp.

#include <metal_stdlib>

using namespace metal;

#define SCALAR_FUN_ATTR static inline
#define FUTHARK_FUN_ATTR static

// MSL provides int8_t..uint64_t (64-bit aliases since Metal 2.2),
// size_t and ptrdiff_t (MSL spec 4.1, section 2.1, Table 2.1), but it
// does NOT provide uintptr_t/intptr_t.  The 8/16-bit atomics
// emulation in rts/c/atomics{8,16}.h needs them for pointer
// arithmetic.  Device/threadgroup pointers are 64-bit.
typedef uint64_t uintptr_t;
typedef int64_t intptr_t;

// metal_stdlib only defines the suffixed float constants (M_PI_F
// etc., MSL spec section 6.6, Table 6.5); map the C spelling used by
// the shared RTS code to the float constant before rts/c/scalar.h's
// "#ifndef M_PI" fallback (a double literal) kicks in.
#define M_PI M_PI_F

// MSL kernel arguments cannot be plain by-value scalars: every kernel
// argument must be a resource (pointer/reference into device or
// constant memory, texture, sampler, threadgroup pointer, ...; MSL
// spec section 5.2).  The hand-written RTS kernels (transpose.cl,
// copy.cl) therefore declare their scalar arguments through this
// macro: a constant reference for Metal (bound with setBytes on the
// host, consuming a [[buffer(n)]] slot like any other buffer), and a
// plain by-value scalar for the other dialects (the fallback
// definition lives in those files).
#define SCALAR_KARG(T) constant T&

// MSL address-space qualifiers occupy the same syntactic position as
// OpenCL's, so the OpenCL spellings in shared/generated code can be
// mapped directly.
#define __global device
#define __local threadgroup
#define __private thread
#define __constant constant
#define __write_only
#define __read_only

#ifndef LOCKSTEP_WIDTH
#define LOCKSTEP_WIDTH 1
#endif

// MSL has no built-in index *functions*; indices come from attributed
// kernel parameters. Every Futhark kernel signature carries these
// parameters under fixed names (generated kernels emit them via the
// FUTHARK_*_PARAM macros below; hand-written RTS kernels get them from
// SHARED_MEM_PARAM), which makes the OpenCL-style getters expressible
// as macros.
#define FUTHARK_TID_PARAM __futhark_ltid [[thread_position_in_threadgroup]]
#define FUTHARK_TBLOCK_PARAM __futhark_tblock [[threadgroup_position_in_grid]]
#define FUTHARK_TBSIZE_PARAM __futhark_tbsize [[threads_per_threadgroup]]
#define FUTHARK_NTBLOCKS_PARAM __futhark_ntblocks [[threadgroups_per_grid]]
#define FUTHARK_SHARED_MEM_PARAM_NAME shared_mem_aligned [[threadgroup(0)]]

#define get_local_id(d) ((int)__futhark_ltid[d])
#define get_tblock_id(d) ((int)__futhark_tblock[d])
#define get_local_size(d) ((int)__futhark_tbsize[d])
#define get_num_tblocks(d) ((int)__futhark_ntblocks[d])
#define get_global_id(d) (get_tblock_id(d) * get_local_size(d) + get_local_id(d))
#define get_global_size(d) (get_num_tblocks(d) * get_local_size(d))

#define CLK_LOCAL_MEM_FENCE 1
#define CLK_GLOBAL_MEM_FENCE 2

// 'flags' is always a compile-time constant, so the branch folds away.
static inline void barrier(int flags) {
  if (flags & CLK_GLOBAL_MEM_FENCE) {
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  } else {
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

static inline void barrier_local() {
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Plain memory fences (not barriers): must be safe in divergent
// control flow, so threadgroup_barrier is not an option.
//
// atomic_thread_fence exists since Metal 3.2 (Apple silicon only;
// MSL spec 4.1, section 6.16.3):
//   void atomic_thread_fence(mem_flags flags, memory_order order,
//                            thread_scope scope = thread_scope_device)
// NOTE: with memory_order_relaxed the fence "has no effects" per the
// spec, so a relaxed fence would be a silent no-op.  We use
// memory_order_seq_cst (valid on atomic_thread_fence since Metal 3.2,
// even though atomic *operations* are relaxed-only until Metal 4.1),
// which gives at least the ordering OpenCL's mem_fence provides.
// This makes Metal 3.2 the minimum language version for the backend.
static inline void mem_fence_local() {
  atomic_thread_fence(mem_flags::mem_threadgroup,
                      memory_order_seq_cst,
                      thread_scope_threadgroup);
}

static inline void mem_fence_global() {
  atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_threadgroup,
                      memory_order_seq_cst,
                      thread_scope_device);
}

// Kernel declaration macros. The generated launch code dispatches
// exact threadgroup sizes, so max_total_threads_per_threadgroup is the
// appropriate occupancy hint for statically-sized kernels.
#define FUTHARK_KERNEL kernel
#define FUTHARK_KERNEL_SIZED(a, b, c) \
  [[max_total_threads_per_threadgroup((a) * (b) * (c))]] kernel

// For the hand-written RTS kernels (transpose.cl, copy.cl): one macro
// providing the dynamic shared memory parameter plus all builtin index
// parameters. Builtin-attributed parameters consume no [[buffer(n)]]
// slots, so the automatic buffer index assignment of the following
// pointer parameters still starts at 0, matching the host launch ABI.
#define SHARED_MEM_PARAM                                        \
  threadgroup uint64_t *shared_mem [[threadgroup(0)]],          \
  uint3 FUTHARK_TID_PARAM,                                      \
  uint3 FUTHARK_TBLOCK_PARAM,                                   \
  uint3 FUTHARK_TBSIZE_PARAM,                                   \
  uint3 FUTHARK_NTBLOCKS_PARAM,

// End of prelude.metal
