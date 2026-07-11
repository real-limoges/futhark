// Start of rts/metal/shim.h.
//
// C API for the metal-cpp shim (rts/metal/shim.cpp).  This is the
// boundary between the generated C99 program (which implements the
// gpu.h contract in rts/c/backends/metal.h) and the single C++17
// translation unit that talks to Metal.  See rts/metal/DESIGN.md.
//
// NOTE: rts/c/backends/metal.h contains a verbatim copy of the
// declarations block below, because the generated program is a single
// concatenated C file that cannot #include other RTS headers.  If you
// change anything between the BEGIN/END markers, change it in both
// places.

#ifndef FUTHARK_METAL_SHIM_H
#define FUTHARK_METAL_SHIM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// -- BEGIN futmtl declarations (keep in sync with rts/c/backends/metal.h) --

// Return codes for futmtl functions returning int.
#define FUTMTL_SUCCESS 0
#define FUTMTL_ERROR 1
#define FUTMTL_OUT_OF_MEMORY 2

// Opaque handles.  futmtl_ctx wraps an MTLDevice + MTLCommandQueue;
// futmtl_program wraps an MTLLibrary; futmtl_kernel wraps an
// MTLComputePipelineState plus a reflection-derived argument-kind
// table; futmtl_mem wraps an MTLBuffer (storageModeShared).
typedef struct futmtl_ctx futmtl_ctx;
typedef struct futmtl_program futmtl_program;
typedef struct futmtl_kernel futmtl_kernel;
typedef struct futmtl_mem futmtl_mem;

// Error convention: unless otherwise noted, functions that can fail
// return NULL on success and a malloc()ed error message on failure,
// which the caller must free().

// Device enumeration, for logging and diagnostics.  The device
// numbering is that of MTLCopyAllDevices() and is consistent with the
// selection done by futmtl_ctx_new().
int futmtl_device_count(void);
// Returns a malloc()ed device name, or NULL if 'i' is out of range.
char* futmtl_get_device_name(int i);

// Create a context on the first device whose name contains
// 'preferred_device' as a substring, skipping the first
// 'preferred_device_num' matches.  An empty (or NULL) string matches
// any device.  Falls back to the system default device if no devices
// are enumerable but a default exists.
char* futmtl_ctx_new(const char *preferred_device,
                     int preferred_device_num,
                     futmtl_ctx **ctx_out);
void futmtl_ctx_free(futmtl_ctx *ctx);
// Returns a malloc()ed copy of the chosen device's name (never fails).
char* futmtl_ctx_get_device_name(futmtl_ctx *ctx);

// Device limit introspection.  These cannot fail.
// maxThreadsPerThreadgroup().width of the device.
size_t futmtl_ctx_max_threads_per_threadgroup(futmtl_ctx *ctx);
// maxThreadgroupMemoryLength of the device.
size_t futmtl_ctx_max_threadgroup_memory(futmtl_ctx *ctx);
// threadExecutionWidth is a per-pipeline property in Metal; this is
// the value from a trivial probe pipeline compiled at context
// creation, or a conservative 32 if the probe failed.
size_t futmtl_ctx_thread_execution_width(futmtl_ctx *ctx);
// recommendedMaxWorkingSetSize of the device.
size_t futmtl_ctx_max_working_set_size(futmtl_ctx *ctx);

// Compile MSL source to a library.  The macros are passed as
// preprocessor defines (MTLCompileOptions.preprocessorMacros).
char* futmtl_build_program(futmtl_ctx *ctx, const char *src,
                           int num_macros,
                           const char *const *macro_names,
                           const int64_t *macro_vals,
                           futmtl_program **program_out);
void futmtl_program_free(futmtl_program *program);

// Create a compute pipeline for the named kernel function, capturing
// reflection info used to bind launch arguments.
char* futmtl_create_kernel(futmtl_ctx *ctx, futmtl_program *program,
                           const char *name,
                           futmtl_kernel **kernel_out);
void futmtl_kernel_free(futmtl_kernel *kernel);

// Launch a kernel: one command buffer per launch, committed
// immediately and not waited for.  'grid' is in thread blocks
// (threadgroups).  For argument i, if buffer index i is a pointer
// binding per reflection, args[i] must point at a futmtl_mem*;
// otherwise args[i]/args_sizes[i] are passed with setBytes.
char* futmtl_launch_kernel(futmtl_ctx *ctx, futmtl_kernel *kernel,
                           const int32_t grid[3],
                           const int32_t block[3],
                           unsigned int shared_mem_bytes,
                           int num_args,
                           void *const *args,
                           const size_t *args_sizes);

// Wait for the last committed command buffer (if any) and report any
// execution error.
char* futmtl_sync(futmtl_ctx *ctx);

// Memory management.  Buffers use storageModeShared (Apple Silicon
// UMA), so host<->device copies are memcpy()s against the buffer
// contents, after synchronising with pending GPU work.
//
// futmtl_alloc returns FUTMTL_SUCCESS, FUTMTL_OUT_OF_MEMORY, or
// FUTMTL_ERROR; on FUTMTL_ERROR, *error_out (if non-NULL) is set to a
// malloc()ed message that the caller must free().
int futmtl_alloc(futmtl_ctx *ctx, size_t size,
                 futmtl_mem **mem_out, char **error_out);
char* futmtl_free(futmtl_ctx *ctx, futmtl_mem *mem);
// Device-to-device copy through a blit command encoder (asynchronous;
// tracked like a kernel launch).
char* futmtl_memcpy_d2d(futmtl_ctx *ctx,
                        futmtl_mem *dst, int64_t dst_offset,
                        futmtl_mem *src, int64_t src_offset,
                        int64_t nbytes);
// Synchronous host<->device copies (sync-then-memcpy).
char* futmtl_memcpy_h2d(futmtl_ctx *ctx,
                        futmtl_mem *dst, int64_t dst_offset,
                        const void *src, int64_t nbytes);
char* futmtl_memcpy_d2h(futmtl_ctx *ctx, void *dst,
                        futmtl_mem *src, int64_t src_offset,
                        int64_t nbytes);

// -- END futmtl declarations --

#ifdef __cplusplus
}
#endif

#endif // FUTHARK_METAL_SHIM_H

// End of rts/metal/shim.h.
