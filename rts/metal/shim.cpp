// Start of rts/metal/shim.cpp.
//
// The metal-cpp shim for the Futhark Metal backend.  This is the only
// translation unit that includes metal-cpp; it exposes the plain C
// futmtl_* API declared in rts/metal/shim.h, which the generated C99
// program (rts/c/backends/metal.h) wraps to implement the gpu.h
// contract.  See rts/metal/DESIGN.md for the conventions, especially
// the kernel ABI that the argument binding in futmtl_launch_kernel()
// relies on.
//
// Compiled as C++17 (no Objective-C syntax):
//   c++ -std=c++17 -c prog.metal.cpp -I <metal-cpp>
//
// Memory management notes: metal-cpp objects follow Cocoa ownership
// rules (methods named new*/create*/Copy* return +1 references; other
// accessors return autoreleased objects).  Since our callers are plain
// C with no ambient autorelease pool, every futmtl entry point that
// touches autoreleased objects creates its own pool (futmtl_pool
// below).  Objects that must outlive the call (command buffers,
// pipelines, buffers, ...) are explicitly retained.
//
// No C++ exceptions may escape the extern "C" boundary; we avoid
// throwing constructs entirely (C allocation, no STL containers).

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// This file is written next to the generated program as
// prog.metal.cpp with no accompanying header, so instead of
// #include "shim.h" it carries a verbatim copy of the declarations
// block from rts/metal/shim.h.  Keep them in sync.
extern "C" {

// -- BEGIN futmtl declarations (keep in sync with rts/metal/shim.h) --

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

} // extern "C"

// Internal definitions of the opaque handles.

struct futmtl_mem {
  MTL::Buffer *buf;
};

// Per-buffer-index argument kinds, captured from pipeline reflection
// at kernel creation and consulted at launch.
enum futmtl_arg_kind {
  FUTMTL_ARG_BYTES = 0,   // Scalar passed with setBytes ('constant T*' in MSL).
  FUTMTL_ARG_POINTER = 1  // futmtl_mem* bound with setBuffer ('device T*' in MSL).
};

struct futmtl_kernel {
  MTL::ComputePipelineState *pipeline;
  unsigned char *arg_kinds; // Indexed by buffer index; FUTMTL_ARG_*.
  int num_arg_kinds;
};

struct futmtl_program {
  MTL::Library *lib;
};

struct futmtl_ctx {
  MTL::Device *device;
  MTL::CommandQueue *queue;
  // The most recently committed command buffer (retained), or NULL.
  // Command buffers on a single queue start execution in commit
  // order, so waiting for the last one suffices for synchronisation.
  MTL::CommandBuffer *last_cb;
  size_t thread_execution_width;
};

// RAII autorelease pool; one per futmtl call that touches autoreleased
// objects, since the plain-C callers provide no ambient pool.
struct futmtl_pool {
  NS::AutoreleasePool *pool;
  futmtl_pool() : pool(NS::AutoreleasePool::alloc()->init()) {}
  ~futmtl_pool() { pool->release(); }
  futmtl_pool(const futmtl_pool&) = delete;
  futmtl_pool& operator=(const futmtl_pool&) = delete;
};

static char* futmtl_strdup(const char *s) {
  size_t n = strlen(s) + 1;
  char *r = (char*)malloc(n);
  memcpy(r, s, n);
  return r;
}

static char* futmtl_msgprintf(const char *fmt, ...) {
  va_list vl;
  va_start(vl, fmt);
  size_t needed = 1 + (size_t)vsnprintf(NULL, 0, fmt, vl);
  va_end(vl);
  char *buffer = (char*)malloc(needed);
  va_start(vl, fmt);
  vsnprintf(buffer, needed, fmt, vl);
  va_end(vl);
  return buffer;
}

static char* futmtl_nserror_msg(const char *what, NS::Error *error) {
  const char *desc = "unknown error";
  if (error != NULL && error->localizedDescription() != NULL) {
    desc = error->localizedDescription()->utf8String();
  }
  return futmtl_msgprintf("%s:\n%s", what, desc);
}

// Device enumeration.

extern "C" int futmtl_device_count(void) {
  futmtl_pool pool;
  NS::Array *devices = MTL::CopyAllDevices();
  int count = devices != NULL ? (int)devices->count() : 0;
  if (devices != NULL) { devices->release(); }
  return count;
}

extern "C" char* futmtl_get_device_name(int i) {
  futmtl_pool pool;
  char *name = NULL;
  NS::Array *devices = MTL::CopyAllDevices();
  if (devices != NULL) {
    if (i >= 0 && (NS::UInteger)i < devices->count()) {
      MTL::Device *dev = devices->object<MTL::Device>((NS::UInteger)i);
      name = futmtl_strdup(dev->name()->utf8String());
    }
    devices->release();
  }
  return name;
}

// Context management.

// threadExecutionWidth is a property of a pipeline, not of the device,
// so we compile a trivial pipeline to obtain it.  Returns a
// conservative 32 if anything fails.
static size_t futmtl_probe_thread_execution_width(MTL::Device *device) {
  static const char *probe_src =
    "kernel void futmtl_probe(device unsigned int *p [[buffer(0)]],"
    "                         uint i [[thread_position_in_grid]]) {"
    "  p[0] = i;"
    "}";
  size_t width = 32;
  NS::Error *error = NULL;
  MTL::Library *lib =
    device->newLibrary(NS::String::string(probe_src, NS::UTF8StringEncoding),
                       (const MTL::CompileOptions*)NULL, &error);
  if (lib == NULL) { return width; }
  MTL::Function *fn =
    lib->newFunction(NS::String::string("futmtl_probe", NS::UTF8StringEncoding));
  if (fn != NULL) {
    MTL::ComputePipelineState *pso = device->newComputePipelineState(fn, &error);
    if (pso != NULL) {
      width = pso->threadExecutionWidth();
      pso->release();
    }
    fn->release();
  }
  lib->release();
  return width;
}

extern "C" char* futmtl_ctx_new(const char *preferred_device,
                                int preferred_device_num,
                                futmtl_ctx **ctx_out) {
  futmtl_pool pool;
  *ctx_out = NULL;
  if (preferred_device == NULL) { preferred_device = ""; }
  int has_preference = preferred_device[0] != '\0' || preferred_device_num > 0;

  MTL::Device *chosen = NULL;
  NS::Array *devices = MTL::CopyAllDevices();
  int num_matches = 0;
  NS::UInteger count = devices != NULL ? devices->count() : 0;
  for (NS::UInteger i = 0; i < count; i++) {
    MTL::Device *dev = devices->object<MTL::Device>(i);
    if (strstr(dev->name()->utf8String(), preferred_device) != NULL &&
        num_matches++ == preferred_device_num) {
      chosen = dev->retain();
      break;
    }
  }
  if (devices != NULL) { devices->release(); }

  if (chosen == NULL) {
    if (has_preference) {
      return futmtl_msgprintf("No Metal device matching \"%s\" (#%d) found.",
                              preferred_device, preferred_device_num);
    }
    // MTLCopyAllDevices() came up empty; try the default device.
    chosen = MTL::CreateSystemDefaultDevice();
    if (chosen == NULL) {
      return futmtl_strdup("No Metal devices found.");
    }
  }

  MTL::CommandQueue *queue = chosen->newCommandQueue();
  if (queue == NULL) {
    chosen->release();
    return futmtl_strdup("Failed to create Metal command queue.");
  }

  futmtl_ctx *ctx = (futmtl_ctx*)malloc(sizeof(futmtl_ctx));
  ctx->device = chosen;
  ctx->queue = queue;
  ctx->last_cb = NULL;
  ctx->thread_execution_width = futmtl_probe_thread_execution_width(chosen);
  *ctx_out = ctx;
  return NULL;
}

extern "C" void futmtl_ctx_free(futmtl_ctx *ctx) {
  if (ctx == NULL) { return; }
  futmtl_pool pool;
  if (ctx->last_cb != NULL) {
    // Let outstanding work drain before tearing down.
    ctx->last_cb->waitUntilCompleted();
    ctx->last_cb->release();
  }
  ctx->queue->release();
  ctx->device->release();
  free(ctx);
}

extern "C" char* futmtl_ctx_get_device_name(futmtl_ctx *ctx) {
  futmtl_pool pool;
  return futmtl_strdup(ctx->device->name()->utf8String());
}

// Device limit introspection.

extern "C" size_t futmtl_ctx_max_threads_per_threadgroup(futmtl_ctx *ctx) {
  return ctx->device->maxThreadsPerThreadgroup().width;
}

extern "C" size_t futmtl_ctx_max_threadgroup_memory(futmtl_ctx *ctx) {
  return ctx->device->maxThreadgroupMemoryLength();
}

extern "C" size_t futmtl_ctx_thread_execution_width(futmtl_ctx *ctx) {
  return ctx->thread_execution_width;
}

extern "C" size_t futmtl_ctx_max_working_set_size(futmtl_ctx *ctx) {
  return ctx->device->recommendedMaxWorkingSetSize();
}

// Program building.

extern "C" char* futmtl_build_program(futmtl_ctx *ctx, const char *src,
                                      int num_macros,
                                      const char *const *macro_names,
                                      const int64_t *macro_vals,
                                      futmtl_program **program_out) {
  futmtl_pool pool;
  *program_out = NULL;

  MTL::CompileOptions *opts = MTL::CompileOptions::alloc()->init();
  // Metal defaults to fast math, which does not preserve NaN/infinity
  // semantics that Futhark programs rely on.  This mirrors the OpenCL
  // backend asking for correctly rounded operations.
  opts->setFastMathEnabled(false);

  if (num_macros > 0) {
    const NS::Object **keys =
      (const NS::Object**)malloc((size_t)num_macros * sizeof(NS::Object*));
    const NS::Object **vals =
      (const NS::Object**)malloc((size_t)num_macros * sizeof(NS::Object*));
    for (int i = 0; i < num_macros; i++) {
      keys[i] = NS::String::string(macro_names[i], NS::UTF8StringEncoding);
      vals[i] = NS::Number::number((long long)macro_vals[i]);
    }
    NS::Dictionary *macros =
      NS::Dictionary::dictionary(vals, keys, (NS::UInteger)num_macros);
    opts->setPreprocessorMacros(macros);
    free(keys);
    free(vals);
  }

  NS::Error *error = NULL;
  MTL::Library *lib =
    ctx->device->newLibrary(NS::String::string(src, NS::UTF8StringEncoding),
                            opts, &error);
  opts->release();
  if (lib == NULL) {
    return futmtl_nserror_msg("Metal library compilation failed", error);
  }

  futmtl_program *program = (futmtl_program*)malloc(sizeof(futmtl_program));
  program->lib = lib;
  *program_out = program;
  return NULL;
}

extern "C" void futmtl_program_free(futmtl_program *program) {
  if (program == NULL) { return; }
  program->lib->release();
  free(program);
}

// Kernel creation.

extern "C" char* futmtl_create_kernel(futmtl_ctx *ctx, futmtl_program *program,
                                      const char *name,
                                      futmtl_kernel **kernel_out) {
  futmtl_pool pool;
  *kernel_out = NULL;

  MTL::Function *fn =
    program->lib->newFunction(NS::String::string(name, NS::UTF8StringEncoding));
  if (fn == NULL) {
    return futmtl_msgprintf("Unknown Metal kernel function: %s", name);
  }

  NS::Error *error = NULL;
  MTL::AutoreleasedComputePipelineReflection reflection = NULL;
  MTL::ComputePipelineState *pipeline =
    ctx->device->newComputePipelineState
    (fn,
     (MTL::PipelineOption)(MTL::PipelineOptionBindingInfo
                           | MTL::PipelineOptionBufferTypeInfo),
     &reflection, &error);
  fn->release();
  if (pipeline == NULL) {
    char *msg0 = futmtl_nserror_msg("Failed to create compute pipeline", error);
    char *msg = futmtl_msgprintf("Kernel %s: %s", name, msg0);
    free(msg0);
    return msg;
  }
  if (reflection == NULL) {
    pipeline->release();
    return futmtl_msgprintf("Kernel %s: no reflection information available.",
                            name);
  }

  // Build the argument-kind table from the buffer bindings.  Scalar
  // arguments are declared 'constant T*' in MSL (read-only address
  // space), while memory arguments (and the failure buffers) are
  // non-const 'device T*'; reflection reports the former as read-only
  // and the latter as read-write, which is our discriminator between
  // setBytes and setBuffer.  (The code generator never emits
  // 'const device' pointers, so read-only implies constant.)
  NS::Array *bindings = reflection->bindings();
  NS::UInteger num_bindings = bindings != NULL ? bindings->count() : 0;
  int num_arg_kinds = 0;
  for (NS::UInteger i = 0; i < num_bindings; i++) {
    MTL::Binding *b = bindings->object<MTL::Binding>(i);
    if (b->type() == MTL::BindingTypeBuffer &&
        (int)b->index() + 1 > num_arg_kinds) {
      num_arg_kinds = (int)b->index() + 1;
    }
  }
  unsigned char *arg_kinds = NULL;
  if (num_arg_kinds > 0) {
    arg_kinds = (unsigned char*)malloc((size_t)num_arg_kinds);
    // Unlisted indexes (which should not occur) default to setBytes.
    memset(arg_kinds, FUTMTL_ARG_BYTES, (size_t)num_arg_kinds);
    for (NS::UInteger i = 0; i < num_bindings; i++) {
      MTL::Binding *b = bindings->object<MTL::Binding>(i);
      if (b->type() == MTL::BindingTypeBuffer) {
        arg_kinds[b->index()] =
          b->access() == MTL::BindingAccessReadOnly
          ? FUTMTL_ARG_BYTES
          : FUTMTL_ARG_POINTER;
      }
    }
  }

  futmtl_kernel *kernel = (futmtl_kernel*)malloc(sizeof(futmtl_kernel));
  kernel->pipeline = pipeline;
  kernel->arg_kinds = arg_kinds;
  kernel->num_arg_kinds = num_arg_kinds;
  *kernel_out = kernel;
  return NULL;
}

extern "C" void futmtl_kernel_free(futmtl_kernel *kernel) {
  if (kernel == NULL) { return; }
  kernel->pipeline->release();
  free(kernel->arg_kinds);
  free(kernel);
}

// Command buffer tracking; 'cb' must be committed already.  Retains
// 'cb' and releases the previously tracked command buffer.
static void futmtl_track_cb(futmtl_ctx *ctx, MTL::CommandBuffer *cb) {
  cb->retain();
  if (ctx->last_cb != NULL) {
    ctx->last_cb->release();
  }
  ctx->last_cb = cb;
}

// Kernel launch.

extern "C" char* futmtl_launch_kernel(futmtl_ctx *ctx, futmtl_kernel *kernel,
                                      const int32_t grid[3],
                                      const int32_t block[3],
                                      unsigned int shared_mem_bytes,
                                      int num_args,
                                      void *const *args,
                                      const size_t *args_sizes) {
  futmtl_pool pool;

  MTL::CommandBuffer *cb = ctx->queue->commandBuffer();
  if (cb == NULL) {
    return futmtl_strdup("Failed to create Metal command buffer.");
  }
  MTL::ComputeCommandEncoder *enc = cb->computeCommandEncoder();
  if (enc == NULL) {
    return futmtl_strdup("Failed to create Metal compute command encoder.");
  }

  enc->setComputePipelineState(kernel->pipeline);

  for (int i = 0; i < num_args; i++) {
    int kind = i < kernel->num_arg_kinds
      ? kernel->arg_kinds[i]
      : FUTMTL_ARG_BYTES;
    if (kind == FUTMTL_ARG_POINTER) {
      // args[i] points at a futmtl_mem* (a gpu_mem in the generated
      // code); buffers are always bound at offset 0 and the kernel
      // receives byte offsets as separate scalar arguments.
      futmtl_mem *mem = *(futmtl_mem *const *)args[i];
      enc->setBuffer(mem != NULL ? mem->buf : NULL, 0, (NS::UInteger)i);
    } else {
      enc->setBytes(args[i], args_sizes[i], (NS::UInteger)i);
    }
  }

  // The dynamic shared memory parameter is threadgroup slot 0 on every
  // kernel (see DESIGN.md); Metal requires the length to be nonzero
  // and a multiple of 16.
  size_t tg_bytes = shared_mem_bytes < 16 ? 16 : shared_mem_bytes;
  tg_bytes = (tg_bytes + 15) & ~(size_t)15;
  enc->setThreadgroupMemoryLength(tg_bytes, 0);

  // 'grid' is in thread blocks, matching dispatchThreadgroups().
  enc->dispatchThreadgroups
    (MTL::Size::Make((NS::UInteger)grid[0],
                     (NS::UInteger)grid[1],
                     (NS::UInteger)grid[2]),
     MTL::Size::Make((NS::UInteger)block[0],
                     (NS::UInteger)block[1],
                     (NS::UInteger)block[2]));
  enc->endEncoding();
  cb->commit(); // Deliberately no waitUntilCompleted(); see futmtl_sync().

  futmtl_track_cb(ctx, cb);
  return NULL;
}

// Synchronisation.

extern "C" char* futmtl_sync(futmtl_ctx *ctx) {
  if (ctx->last_cb == NULL) {
    return NULL;
  }
  futmtl_pool pool;
  MTL::CommandBuffer *cb = ctx->last_cb;
  ctx->last_cb = NULL;
  cb->waitUntilCompleted();
  char *msg = NULL;
  if (cb->status() == MTL::CommandBufferStatusError) {
    msg = futmtl_nserror_msg("Metal command buffer execution failed",
                             cb->error());
  }
  cb->release();
  return msg;
}

// Memory management.

extern "C" int futmtl_alloc(futmtl_ctx *ctx, size_t size,
                            futmtl_mem **mem_out, char **error_out) {
  futmtl_pool pool;
  *mem_out = NULL;
  // storageModeShared: host and device share the allocation (UMA), so
  // host<->device copies can be plain memcpy()s against contents().
  MTL::Buffer *buf =
    ctx->device->newBuffer((NS::UInteger)size, MTL::ResourceStorageModeShared);
  if (buf == NULL) {
    // Metal reports allocation failure by returning nil.
    return FUTMTL_OUT_OF_MEMORY;
  }
  futmtl_mem *mem = (futmtl_mem*)malloc(sizeof(futmtl_mem));
  if (mem == NULL) {
    buf->release();
    if (error_out != NULL) {
      *error_out = futmtl_strdup("Out of host memory.");
    }
    return FUTMTL_ERROR;
  }
  mem->buf = buf;
  *mem_out = mem;
  return FUTMTL_SUCCESS;
}

extern "C" char* futmtl_free(futmtl_ctx *ctx, futmtl_mem *mem) {
  (void)ctx;
  if (mem == NULL) { return NULL; }
  // Metal retains resources referenced by in-flight command buffers,
  // so releasing here is safe even if the GPU is still using the
  // buffer.
  mem->buf->release();
  free(mem);
  return NULL;
}

extern "C" char* futmtl_memcpy_d2d(futmtl_ctx *ctx,
                                   futmtl_mem *dst, int64_t dst_offset,
                                   futmtl_mem *src, int64_t src_offset,
                                   int64_t nbytes) {
  if (nbytes <= 0) {
    return NULL;
  }
  futmtl_pool pool;
  MTL::CommandBuffer *cb = ctx->queue->commandBuffer();
  if (cb == NULL) {
    return futmtl_strdup("Failed to create Metal command buffer.");
  }
  MTL::BlitCommandEncoder *blit = cb->blitCommandEncoder();
  if (blit == NULL) {
    return futmtl_strdup("Failed to create Metal blit command encoder.");
  }
  blit->copyFromBuffer(src->buf, (NS::UInteger)src_offset,
                       dst->buf, (NS::UInteger)dst_offset,
                       (NS::UInteger)nbytes);
  blit->endEncoding();
  cb->commit();
  futmtl_track_cb(ctx, cb);
  return NULL;
}

// Host<->device copies: buffers are storageModeShared, so after
// synchronising with any pending GPU work (which may touch the
// buffer), a plain memcpy() against contents() is coherent in both
// directions.  This is the simple-and-correct scheme from DESIGN.md;
// the 'sync' flag distinction made by other backends is intentionally
// collapsed to always-synchronous.

extern "C" char* futmtl_memcpy_h2d(futmtl_ctx *ctx,
                                   futmtl_mem *dst, int64_t dst_offset,
                                   const void *src, int64_t nbytes) {
  if (nbytes <= 0) {
    return NULL;
  }
  char *msg = futmtl_sync(ctx);
  if (msg != NULL) {
    return msg;
  }
  memcpy((unsigned char*)dst->buf->contents() + dst_offset, src, (size_t)nbytes);
  return NULL;
}

extern "C" char* futmtl_memcpy_d2h(futmtl_ctx *ctx, void *dst,
                                   futmtl_mem *src, int64_t src_offset,
                                   int64_t nbytes) {
  if (nbytes <= 0) {
    return NULL;
  }
  char *msg = futmtl_sync(ctx);
  if (msg != NULL) {
    return msg;
  }
  memcpy(dst, (unsigned char*)src->buf->contents() + src_offset, (size_t)nbytes);
  return NULL;
}

// End of rts/metal/shim.cpp.
