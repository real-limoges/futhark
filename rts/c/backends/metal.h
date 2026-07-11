// Start of backends/metal.h.

// This backend implements the gpu.h contract as thin wrappers over
// the futmtl_* C API implemented by the metal-cpp shim
// (rts/metal/shim.cpp), which is compiled as a separate C++17
// translation unit and linked in.  See rts/metal/DESIGN.md for the
// conventions shared between the code generator, the MSL prelude,
// this file and the shim.

// Forward declarations.
// Invoked by setup_opencl() after the platform and device has been
// found, but before the program is loaded.  Its intended use is to
// tune constants based on the selected platform and device.
static void set_tuning_params(struct futhark_context* ctx);
static char* get_failure_msg(int failure_idx, int64_t args[]);

// The generated program is a single concatenated C file, so we cannot
// #include rts/metal/shim.h here.  The following block is a verbatim
// copy of the declarations block from rts/metal/shim.h and MUST be
// kept in sync with it.
//
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

// Error handling.  The futmtl functions return NULL on success and a
// malloc()ed message on failure, so the macros below take that
// message rather than an error code.

#define METAL_SUCCEED_FATAL(x) metal_api_succeed_fatal(x, #x, __FILE__, __LINE__)
#define METAL_SUCCEED_NONFATAL(x) metal_api_succeed_nonfatal(x, #x, __FILE__, __LINE__)
// Take care not to override an existing error.
#define METAL_SUCCEED_OR_RETURN(e) {            \
    char *serror = METAL_SUCCEED_NONFATAL(e);   \
    if (serror) {                               \
      if (!ctx->error) {                        \
        ctx->error = serror;                    \
      } else {                                  \
        free(serror);                           \
      }                                         \
      return bad;                               \
    }                                           \
  }

// METAL_SUCCEED_OR_RETURN returns the value of the variable 'bad' in
// scope.  By default, it will be this one.  Create a local variable
// of some other type if needed.  This is a bit of a hack, but it
// saves effort in the code generator.
static const int bad = 1;

static inline void metal_api_succeed_fatal(char *msg, const char *call,
                                           const char *file, int line) {
  if (msg != NULL) {
    futhark_panic(-1, "%s:%d: Metal call\n  %s\nfailed with error:\n%s\n",
                  file, line, call, msg);
  }
}

static char* metal_api_succeed_nonfatal(char *msg, const char *call,
                                        const char *file, int line) {
  if (msg != NULL) {
    char *serror = msgprintf("%s:%d: Metal call\n  %s\nfailed with error:\n%s\n",
                             file, line, call, msg);
    free(msg);
    return serror;
  } else {
    return NULL;
  }
}

struct futhark_context_config {
  int in_use;
  int debugging;
  int profiling;
  int logging;
  char *cache_fname; // Accepted but ignored: no binary caching in v1.
  struct tuning_param tuning_params[NUM_TUNING_PARAMS];
  // Uniform fields above.

  char* program;
  int num_build_opts;
  char* *build_opts;

  char* preferred_device;
  int preferred_device_num;

  int unified_memory;

  char* dump_msl_to;
  char* load_msl_from;

  struct gpu_config gpu;
};

static void backend_context_config_setup(struct futhark_context_config *cfg) {
  cfg->num_build_opts = 0;
  cfg->build_opts = (char**) malloc(sizeof(char*));
  cfg->build_opts[0] = NULL;

  cfg->program = strconcat(gpu_program);

  cfg->preferred_device_num = 0;
  cfg->preferred_device = strdup("");

  cfg->dump_msl_to = NULL;
  cfg->load_msl_from = NULL;

  // Metal buffers are always storageModeShared; this field exists only
  // for interface compatibility with the other GPU backends.
  cfg->unified_memory = 1;

  cfg->gpu = gpu_config_initial;
  cfg->gpu.default_block_size = 256;
  cfg->gpu.default_tile_size = 32;
  cfg->gpu.default_reg_tile_size = 2;
  cfg->gpu.default_threshold = 32*1024;
}

static void backend_context_config_teardown(struct futhark_context_config* cfg) {
  for (int i = 0; i < cfg->num_build_opts; i++) {
    free(cfg->build_opts[i]);
  }
  free(cfg->build_opts);
  free(cfg->dump_msl_to);
  free(cfg->load_msl_from);
  free(cfg->preferred_device);
  free(cfg->program);
}

// Metal has no free-form compiler option strings; options of the form
// -Dname=value (or -Dname) are turned into preprocessor macros at
// program build time, and anything else is rejected then.
void futhark_context_config_add_build_option(struct futhark_context_config *cfg, const char *opt) {
  cfg->build_opts[cfg->num_build_opts] = strdup(opt);
  cfg->num_build_opts++;
  cfg->build_opts = (char **) realloc(cfg->build_opts, (cfg->num_build_opts + 1) * sizeof(char *));
  cfg->build_opts[cfg->num_build_opts] = NULL;
}

void futhark_context_config_set_device(struct futhark_context_config *cfg, const char *s) {
  int x = 0;
  if (*s == '#') {
    s++;
    while (isdigit(*s)) {
      x = x * 10 + (*s++)-'0';
    }
    // Skip trailing spaces.
    while (isspace(*s)) {
      s++;
    }
  }
  free(cfg->preferred_device);
  cfg->preferred_device = strdup(s);
  cfg->preferred_device_num = x;
}

const char* futhark_context_config_get_program(struct futhark_context_config *cfg) {
  return cfg->program;
}

void futhark_context_config_set_program(struct futhark_context_config *cfg, const char *s) {
  free(cfg->program);
  cfg->program = strdup(s);
}

// The Metal analogue of dumping/loading PTX is dumping/loading the MSL
// source that is compiled at context setup.
void futhark_context_config_dump_msl_to(struct futhark_context_config *cfg, const char *path) {
  free(cfg->dump_msl_to);
  cfg->dump_msl_to = strdup(path);
}

void futhark_context_config_load_msl_from(struct futhark_context_config *cfg, const char *path) {
  free(cfg->load_msl_from);
  cfg->load_msl_from = strdup(path);
}

void futhark_context_config_set_unified_memory(struct futhark_context_config* cfg, int flag) {
  // Accepted for interface compatibility; Metal memory is always unified.
  (void)flag;
  cfg->unified_memory = 1;
}

// A record of something that happened.  Profiling is stubbed in v1
// (see DESIGN.md): events are recorded so the report structure matches
// the other backends, but all durations are zero.
struct metal_event {
  int unused;
};

struct futhark_context {
  struct futhark_context_config* cfg;
  int detail_memory;
  int debugging;
  int profiling;
  int profiling_paused;
  int logging;
  lock_t lock;
  char *error;
  lock_t error_lock;
  FILE *log;
  struct constants *constants;
  struct free_list free_list;
  struct event_list event_list;
  int64_t peak_mem_usage_default;
  int64_t cur_mem_usage_default;
  struct program* program;
  bool program_initialised;
  // Uniform fields above.

  struct futmtl_mem *global_failure;
  struct futmtl_mem *global_failure_args;
  // True if a potentially failing kernel has been enqueued.
  int32_t failure_is_an_option;
  int total_runs;
  long int total_runtime;
  int64_t peak_mem_usage_device;
  int64_t cur_mem_usage_device;

  struct futmtl_ctx *mtl;
  struct futmtl_program *mtl_program;

  struct free_list gpu_free_list;

  size_t max_thread_block_size;
  size_t max_grid_size;
  size_t max_tile_size;
  size_t max_threshold;
  size_t max_shared_memory;
  size_t max_bespoke;
  size_t max_registers;
  size_t max_cache;

  size_t lockstep_width;

  struct builtin_kernels* kernels;
};

static int metal_device_setup(struct futhark_context *ctx) {
  struct futhark_context_config *cfg = ctx->cfg;

  if (cfg->logging) {
    int count = futmtl_device_count();
    for (int i = 0; i < count; i++) {
      char *name = futmtl_get_device_name(i);
      if (name != NULL) {
        fprintf(ctx->log, "Device #%d: name=\"%s\"\n", i, name);
        free(name);
      }
    }
  }

  char *error = futmtl_ctx_new(cfg->preferred_device,
                               cfg->preferred_device_num,
                               &ctx->mtl);
  if (error != NULL) {
    futhark_panic(-1, "%s\n", error);
  }

  if (cfg->logging) {
    char *name = futmtl_ctx_get_device_name(ctx->mtl);
    fprintf(ctx->log, "Using device: %s\n", name);
    free(name);
  }

  return 0;
}

// Assemble the preprocessor macros passed to the MSL compiler
// (mirrors cuda_nvrtc_mk_build_options(), except that Metal takes
// name/value macro pairs rather than option strings).  All returned
// names are malloc()ed; free with metal_free_macros().
static void metal_mk_macros(struct futhark_context *ctx,
                            char ***names_out, int64_t **vals_out,
                            int *num_out) {
  struct futhark_context_config *cfg = ctx->cfg;

  char** macro_names;
  int64_t* macro_vals;
  int num_macros = gpu_macros(ctx, &macro_names, &macro_vals);

  int n_alloc = 10 + num_macros + cfg->num_build_opts + 2*NUM_TUNING_PARAMS;
  char **names = (char**) malloc(n_alloc * sizeof(char*));
  int64_t *vals = (int64_t*) malloc(n_alloc * sizeof(int64_t));
  int i = 0;

  names[i] = strdup("max_thread_block_size");
  vals[i++] = (int64_t)ctx->max_thread_block_size;
  names[i] = strdup("max_shared_memory");
  vals[i++] = (int64_t)ctx->max_shared_memory;
  names[i] = strdup("max_registers");
  vals[i++] = (int64_t)ctx->max_registers;
  names[i] = strdup("MAX_THREADS_PER_BLOCK");
  vals[i++] = (int64_t)ctx->max_thread_block_size;
  // Note: no LOCKSTEP_WIDTH here, unlike other backends - per
  // DESIGN.md the device-side LOCKSTEP_WIDTH is pinned to a
  // conservative 1 by the MSL prelude (rts/metal/prelude.metal),
  // since Metal makes no implicit intra-simdgroup synchronisation
  // guarantees.  The host-side ctx->lockstep_width (used only for
  // scheduling heuristics) is the probed threadExecutionWidth.

  for (int j = 0; j < num_macros; j++) {
    names[i] = strdup(macro_names[j]);
    vals[i++] = macro_vals[j];
  }

  for (int j = 0; j < NUM_TUNING_PARAMS; j++) {
    names[i] = msgprintf("set_%s", cfg->tuning_params[j].var);
    vals[i++] = (int64_t)cfg->tuning_params[j].set;
    names[i] = msgprintf("val_%s", cfg->tuning_params[j].var);
    vals[i++] = cfg->tuning_params[j].val;
  }

  names[i] = strdup("TR_BLOCK_DIM");
  vals[i++] = TR_BLOCK_DIM;
  names[i] = strdup("TR_TILE_DIM");
  vals[i++] = TR_TILE_DIM;
  names[i] = strdup("TR_ELEMS_PER_THREAD");
  vals[i++] = TR_ELEMS_PER_THREAD;

  // User-provided build options must be -Dname=value (or -Dname,
  // which defines the name as 1), with an integral value.
  for (int j = 0; j < cfg->num_build_opts; j++) {
    const char *opt = cfg->build_opts[j];
    if (strncmp(opt, "-D", 2) == 0) {
      const char *eq = strchr(opt+2, '=');
      if (eq != NULL) {
        names[i] = msgprintf("%.*s", (int)(eq-(opt+2)), opt+2);
        vals[i++] = strtoll(eq+1, NULL, 0);
      } else {
        names[i] = strdup(opt+2);
        vals[i++] = 1;
      }
    } else {
      futhark_panic(1, "Invalid Metal build option: %s\n(only -Dname=value is supported)\n",
                    opt);
    }
  }

  free(macro_names);
  free(macro_vals);

  *names_out = names;
  *vals_out = vals;
  *num_out = i;
}

static void metal_free_macros(int num_macros, char **names, int64_t *vals) {
  for (int i = 0; i < num_macros; i++) {
    free(names[i]);
  }
  free(names);
  free(vals);
}

static void metal_size_setup(struct futhark_context *ctx) {
  struct futhark_context_config *cfg = ctx->cfg;
  if (cfg->gpu.default_block_size > ctx->max_thread_block_size) {
    if (cfg->gpu.default_block_size_changed) {
      fprintf(stderr,
              "Note: Device limits default block size to %zu (down from %zu).\n",
              ctx->max_thread_block_size, cfg->gpu.default_block_size);
    }
    cfg->gpu.default_block_size = ctx->max_thread_block_size;
  }
  if (cfg->gpu.default_grid_size > ctx->max_grid_size) {
    if (cfg->gpu.default_grid_size_changed) {
      fprintf(stderr,
              "Note: Device limits default grid size to %zu (down from %zu).\n",
              ctx->max_grid_size, cfg->gpu.default_grid_size);
    }
    cfg->gpu.default_grid_size = ctx->max_grid_size;
  }
  if (cfg->gpu.default_tile_size > ctx->max_tile_size) {
    if (cfg->gpu.default_tile_size_changed) {
      fprintf(stderr,
              "Note: Device limits default tile size to %zu (down from %zu).\n",
              ctx->max_tile_size, cfg->gpu.default_tile_size);
    }
    cfg->gpu.default_tile_size = ctx->max_tile_size;
  }

  if (!cfg->gpu.default_grid_size_changed) {
    // Unlike CUDA/HIP, Metal does not expose the number of GPU cores
    // or the maximum occupancy, so we cannot compute an
    // occupancy-based default; use a fixed count of thread blocks in
    // the right ballpark for Apple Silicon.
    cfg->gpu.default_grid_size = 256;
  }

  for (int i = 0; i < NUM_TUNING_PARAMS; i++) {
    const char *size_class = cfg->tuning_params[i].class;
    int64_t *size_value = &cfg->tuning_params[i].val;
    const char* size_name = cfg->tuning_params[i].name;
    int64_t max_value = 0, default_value = 0;

    if (strstr(size_class, "thread_block_size") == size_class) {
      max_value = ctx->max_thread_block_size;
      default_value = cfg->gpu.default_block_size;
    } else if (strstr(size_class, "grid_size") == size_class) {
      max_value = ctx->max_grid_size;
      default_value = cfg->gpu.default_grid_size;
      // XXX: as a quick and dirty hack, use twice as many threads for
      // histograms by default.  We really should just be smarter
      // about sizes somehow.
      if (strstr(size_name, ".seghist_") != NULL) {
        default_value *= 2;
      }
    } else if (strstr(size_class, "tile_size") == size_class) {
      max_value = ctx->max_tile_size;
      default_value = cfg->gpu.default_tile_size;
    } else if (strstr(size_class, "reg_tile_size") == size_class) {
      max_value = 0; // No limit.
      default_value = cfg->gpu.default_reg_tile_size;
    } else if (strstr(size_class, "shared_memory") == size_class) {
      max_value = ctx->max_shared_memory;
      default_value = ctx->max_shared_memory;
    } else if (strstr(size_class, "cache") == size_class) {
      max_value = ctx->max_cache;
      default_value = ctx->max_cache;
    } else if (strstr(size_class, "threshold") == size_class) {
      // Threshold can be as large as it takes.
      default_value = cfg->gpu.default_threshold;
    } else {
      // Bespoke sizes have no limit or default.
    }

    if (*size_value == 0) {
      *size_value = default_value;
    } else if (max_value > 0 && *size_value > max_value) {
      fprintf(stderr, "Note: Device limits %s to %zu (down from %zu)\n",
              size_name, max_value, *size_value);
      *size_value = max_value;
    }
  }
}

// Build the embedded (or loaded) MSL program.  Returns NULL on
// success, otherwise a malloc()ed error message.
static char* metal_program_setup(struct futhark_context *ctx) {
  struct futhark_context_config *cfg = ctx->cfg;
  char *src = NULL;

  if (cfg->load_msl_from != NULL) {
    src = slurp_file(cfg->load_msl_from, NULL);
    if (src == NULL) {
      return msgprintf("Failed to read MSL source from %s\n", cfg->load_msl_from);
    }
  } else {
    src = strdup(cfg->program);
  }

  if (cfg->dump_msl_to != NULL) {
    dump_file(cfg->dump_msl_to, src, strlen(src));
  }

  // No binary caching in v1: cfg->cache_fname is accepted but ignored,
  // as the Metal driver maintains its own shader cache.

  char **macro_names;
  int64_t *macro_vals;
  int num_macros;
  metal_mk_macros(ctx, &macro_names, &macro_vals, &num_macros);

  if (cfg->logging) {
    fprintf(stderr, "MSL compile macros:\n");
    for (int j = 0; j < num_macros; j++) {
      fprintf(stderr, "\t%s=%lld\n", macro_names[j], (long long)macro_vals[j]);
    }
    fprintf(stderr, "\n");
  }

  char *problem = futmtl_build_program(ctx->mtl, src,
                                       num_macros,
                                       (const char* const*)macro_names,
                                       macro_vals,
                                       &ctx->mtl_program);

  metal_free_macros(num_macros, macro_names, macro_vals);
  free(src);

  return problem;
}

static struct metal_event* metal_event_new(struct futhark_context* ctx) {
  if (ctx->profiling && !ctx->profiling_paused) {
    struct metal_event* e = malloc(sizeof(struct metal_event));
    return e;
  } else {
    return NULL;
  }
}

static int metal_event_report(struct str_builder* sb, struct metal_event* e) {
  // Stubbed: report a zero duration (see DESIGN.md).
  str_builder(sb, ",\"duration\":%f", 0.0);
  free(e);
  return 0;
}

int futhark_context_sync(struct futhark_context* ctx) {
  METAL_SUCCEED_OR_RETURN(futmtl_sync(ctx->mtl));
  if (ctx->failure_is_an_option) {
    // Check for any delayed error.
    int32_t failure_idx;
    METAL_SUCCEED_OR_RETURN(
                            futmtl_memcpy_d2h(ctx->mtl,
                                              &failure_idx,
                                              ctx->global_failure,
                                              0, sizeof(int32_t)));
    ctx->failure_is_an_option = 0;

    if (failure_idx >= 0) {
      // We have to clear global_failure so that the next entry point
      // is not considered a failure from the start.
      int32_t no_failure = -1;
      METAL_SUCCEED_OR_RETURN(
                              futmtl_memcpy_h2d(ctx->mtl,
                                                ctx->global_failure, 0,
                                                &no_failure,
                                                sizeof(int32_t)));

      int64_t args[max_failure_args+1];
      METAL_SUCCEED_OR_RETURN(
                              futmtl_memcpy_d2h(ctx->mtl,
                                                &args,
                                                ctx->global_failure_args,
                                                0, sizeof(args)));

      ctx->error = get_failure_msg(failure_idx, args);

      return FUTHARK_PROGRAM_ERROR;
    }
  }
  return 0;
}

struct builtin_kernels* init_builtin_kernels(struct futhark_context* ctx);
void free_builtin_kernels(struct futhark_context* ctx, struct builtin_kernels* kernels);

int backend_context_setup(struct futhark_context* ctx) {
  ctx->failure_is_an_option = 0;
  ctx->total_runs = 0;
  ctx->total_runtime = 0;
  ctx->peak_mem_usage_device = 0;
  ctx->cur_mem_usage_device = 0;
  ctx->kernels = NULL;
  ctx->mtl = NULL;
  ctx->mtl_program = NULL;

  if (metal_device_setup(ctx) != 0) {
    futhark_panic(-1, "No suitable Metal device found.\n");
  }

  free_list_init(&ctx->gpu_free_list);

  ctx->max_thread_block_size = futmtl_ctx_max_threads_per_threadgroup(ctx->mtl);
  ctx->max_grid_size = ((size_t)1<<31)-1; // No limit (see DESIGN.md).
  ctx->max_tile_size = sqrt(ctx->max_thread_block_size);
  ctx->max_threshold = 1U<<31; // No limit.
  ctx->max_bespoke = 1U<<31; // No limit.

  if (ctx->cfg->gpu.default_registers != 0) {
    ctx->max_registers = ctx->cfg->gpu.default_registers;
  } else {
    ctx->max_registers = 1<<16; // Metal provides no way to query this.
  }

  if (ctx->cfg->gpu.default_shared_memory != 0) {
    ctx->max_shared_memory = ctx->cfg->gpu.default_shared_memory;
  } else {
    ctx->max_shared_memory = futmtl_ctx_max_threadgroup_memory(ctx->mtl);
  }

  if (ctx->cfg->gpu.default_cache != 0) {
    ctx->max_cache = ctx->cfg->gpu.default_cache;
  } else {
    ctx->max_cache = 1024*1024; // Metal provides no way to query this.
  }

  ctx->lockstep_width = futmtl_ctx_thread_execution_width(ctx->mtl);

  metal_size_setup(ctx);

  gpu_init_log(ctx);

  ctx->error = metal_program_setup(ctx);

  if (ctx->error != NULL) {
    futhark_panic(1, "During Metal initialisation:\n%s\n", ctx->error);
  }

  int32_t no_error = -1;
  char *alloc_error = NULL;
  if (futmtl_alloc(ctx->mtl, sizeof(no_error),
                   &ctx->global_failure, &alloc_error) != FUTMTL_SUCCESS) {
    futhark_panic(-1, "Failed to allocate global_failure buffer:\n%s\n",
                  alloc_error != NULL ? alloc_error : "out of memory");
  }
  METAL_SUCCEED_FATAL(futmtl_memcpy_h2d(ctx->mtl, ctx->global_failure, 0,
                                        &no_error, sizeof(no_error)));
  // The +1 is to avoid zero-byte allocations.
  if (futmtl_alloc(ctx->mtl, sizeof(int64_t)*(max_failure_args+1),
                   &ctx->global_failure_args, &alloc_error) != FUTMTL_SUCCESS) {
    futhark_panic(-1, "Failed to allocate global_failure_args buffer:\n%s\n",
                  alloc_error != NULL ? alloc_error : "out of memory");
  }

  if ((ctx->kernels = init_builtin_kernels(ctx)) == NULL) {
    return 1;
  }

  return 0;
}

void backend_context_teardown(struct futhark_context* ctx) {
  if (ctx->kernels != NULL) {
    free_builtin_kernels(ctx, ctx->kernels);
    METAL_SUCCEED_FATAL(futmtl_free(ctx->mtl, ctx->global_failure));
    METAL_SUCCEED_FATAL(futmtl_free(ctx->mtl, ctx->global_failure_args));
    (void)gpu_free_all(ctx);
    futmtl_program_free(ctx->mtl_program);
    futmtl_ctx_free(ctx->mtl);
  }
  free_list_destroy(&ctx->gpu_free_list);
}

// GPU ABSTRACTION LAYER

// Types.

typedef struct futmtl_kernel* gpu_kernel;
typedef struct futmtl_mem* gpu_mem;

static void gpu_create_kernel(struct futhark_context *ctx,
                              gpu_kernel* kernel,
                              const char* name) {
  if (ctx->debugging) {
    fprintf(ctx->log, "Creating kernel %s.\n", name);
  }
  METAL_SUCCEED_FATAL(futmtl_create_kernel(ctx->mtl, ctx->mtl_program, name, kernel));
}

static void gpu_free_kernel(struct futhark_context *ctx,
                            gpu_kernel kernel) {
  (void)ctx;
  futmtl_kernel_free(kernel);
}

static int gpu_scalar_to_device(struct futhark_context* ctx,
                                const char *provenance,
                                gpu_mem dst, size_t offset, size_t size,
                                void *src) {
  struct metal_event *event = metal_event_new(ctx);
  if (event != NULL) {
    add_event(ctx,
              "copy_scalar_to_dev",
              provenance,
              NULL,
              event,
              (event_report_fn)metal_event_report);
  }
  METAL_SUCCEED_OR_RETURN(futmtl_memcpy_h2d(ctx->mtl, dst, offset, src, size));
  return FUTHARK_SUCCESS;
}

static int gpu_scalar_from_device(struct futhark_context* ctx,
                                  const char *provenance,
                                  void *dst,
                                  gpu_mem src, size_t offset, size_t size) {
  struct metal_event *event = metal_event_new(ctx);
  if (event != NULL) {
    add_event(ctx,
              "copy_scalar_from_dev",
              provenance,
              NULL,
              event,
              (event_report_fn)metal_event_report);
  }
  METAL_SUCCEED_OR_RETURN(futmtl_memcpy_d2h(ctx->mtl, dst, src, offset, size));
  return FUTHARK_SUCCESS;
}

static int gpu_memcpy(struct futhark_context* ctx,
                      const char *provenance,
                      gpu_mem dst, int64_t dst_offset,
                      gpu_mem src, int64_t src_offset,
                      int64_t nbytes) {
  struct metal_event *event = metal_event_new(ctx);
  if (event != NULL) {
    add_event(ctx,
              "copy_dev_to_dev",
              provenance,
              NULL,
              event,
              (event_report_fn)metal_event_report);
  }
  METAL_SUCCEED_OR_RETURN(futmtl_memcpy_d2d(ctx->mtl,
                                            dst, dst_offset,
                                            src, src_offset,
                                            nbytes));
  return FUTHARK_SUCCESS;
}

static int memcpy_host2gpu(struct futhark_context* ctx,
                           const char *provenance,
                           bool sync,
                           gpu_mem dst, int64_t dst_offset,
                           const unsigned char* src, int64_t src_offset,
                           int64_t nbytes) {
  // Buffers are storageModeShared, so all host<->device copies are
  // synchronous memcpy()s (after syncing with pending GPU work); the
  // 'sync' flag makes no difference.
  (void)sync;
  if (nbytes > 0) {
    struct metal_event *event = metal_event_new(ctx);
    if (event != NULL) {
      add_event(ctx,
                "copy_host_to_dev",
                provenance,
                NULL,
                event,
                (event_report_fn)metal_event_report);
    }
    METAL_SUCCEED_OR_RETURN(futmtl_memcpy_h2d(ctx->mtl,
                                              dst, dst_offset,
                                              src + src_offset,
                                              nbytes));
  }
  return FUTHARK_SUCCESS;
}

static int memcpy_gpu2host(struct futhark_context* ctx,
                           const char *provenance,
                           bool sync,
                           unsigned char* dst, int64_t dst_offset,
                           gpu_mem src, int64_t src_offset,
                           int64_t nbytes) {
  if (nbytes > 0) {
    struct metal_event *event = metal_event_new(ctx);
    if (event != NULL) {
      add_event(ctx,
                "copy_dev_to_host",
                provenance,
                NULL,
                event,
                (event_report_fn)metal_event_report);
    }
    METAL_SUCCEED_OR_RETURN(futmtl_memcpy_d2h(ctx->mtl,
                                              dst + dst_offset,
                                              src, src_offset,
                                              nbytes));
    // The copy itself always synchronises with the device, but a
    // delayed kernel failure must still be turned into an error
    // before the caller consumes the data.
    if (sync &&
        ctx->failure_is_an_option &&
        futhark_context_sync(ctx) != 0) {
      return 1;
    }
  }
  return FUTHARK_SUCCESS;
}

static int gpu_launch_kernel(struct futhark_context* ctx,
                             gpu_kernel kernel,
                             const char *name, const char *provenance,
                             const int32_t grid[3],
                             const int32_t block[3],
                             unsigned int shared_mem_bytes,
                             int num_args,
                             void* args[num_args],
                             size_t args_sizes[num_args]) {
  if (shared_mem_bytes > ctx->max_shared_memory) {
    set_error(ctx, msgprintf("Kernel %s with %d bytes of memory exceeds device limit of %d\n",
                             name, shared_mem_bytes, (int)ctx->max_shared_memory));
    return 1;
  }

  int64_t time_start = 0, time_end = 0;
  if (ctx->debugging) {
    time_start = get_wall_time();
  }

  struct metal_event *event = metal_event_new(ctx);

  if (event != NULL) {
    struct kvs *kvs = kvs_new();
    kvs_printf(kvs, "kernel", "\"%s\"", name);
    kvs_printf(kvs, "grid", "[%d,%d,%d]", grid[0], grid[1], grid[2]);
    kvs_printf(kvs, "block", "[%d,%d,%d]", block[0], block[1], block[2]);
    kvs_printf(kvs, "shared memory", "%d", shared_mem_bytes);

    add_event(ctx,
              name,
              provenance,
              kvs,
              event,
              (event_report_fn)metal_event_report);
  }

  METAL_SUCCEED_OR_RETURN
    (futmtl_launch_kernel(ctx->mtl, kernel,
                          grid, block,
                          shared_mem_bytes,
                          num_args, args, args_sizes));

  if (ctx->debugging) {
    METAL_SUCCEED_FATAL(futmtl_sync(ctx->mtl));
    time_end = get_wall_time();
    long int time_diff = time_end - time_start;
    fprintf(ctx->log, "  runtime: %ldus\n", time_diff);
  }
  if (ctx->logging) {
    fprintf(ctx->log, "\n");
  }

  return FUTHARK_SUCCESS;
}

static int gpu_alloc_actual(struct futhark_context *ctx, size_t size, gpu_mem *mem_out) {
  char *error = NULL;
  int res = futmtl_alloc(ctx->mtl, size, mem_out, &error);

  if (res == FUTMTL_OUT_OF_MEMORY) {
    return FUTHARK_OUT_OF_MEMORY;
  }

  if (res != FUTMTL_SUCCESS) {
    if (!ctx->error) {
      ctx->error = error;
    } else {
      free(error);
    }
    return 1;
  }

  return FUTHARK_SUCCESS;
}

static int gpu_free_actual(struct futhark_context *ctx, gpu_mem mem) {
  METAL_SUCCEED_OR_RETURN(futmtl_free(ctx->mtl, mem));
  return FUTHARK_SUCCESS;
}

// End of backends/metal.h.
