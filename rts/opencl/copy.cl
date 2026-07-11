// Start of copy.cl

// MSL does not allow plain by-value scalar kernel arguments: every
// kernel argument must be a resource, e.g. a device/constant pointer
// or reference (MSL spec section 5.2).  The Metal prelude
// (rts/metal/prelude.metal) therefore defines SCALAR_KARG(T) as
// 'constant T&' (one [[buffer(n)]] slot, bound with setBytes by the
// runtime); for the other dialects it is a plain by-value scalar.
#ifndef SCALAR_KARG
#define SCALAR_KARG(T) T
#endif

#define GEN_COPY_KERNEL(NAME, ELEM_TYPE) \
FUTHARK_KERNEL void lmad_copy_##NAME(SHARED_MEM_PARAM                   \
                               __global ELEM_TYPE *dst_mem,             \
                               SCALAR_KARG(int64_t) dst_offset_arg,     \
                               __global ELEM_TYPE *src_mem,             \
                               SCALAR_KARG(int64_t) src_offset_arg,     \
                               SCALAR_KARG(int64_t) n,                  \
                               SCALAR_KARG(int) r,                      \
                               SCALAR_KARG(int64_t) shape0, SCALAR_KARG(int64_t) dst_stride0, SCALAR_KARG(int64_t) src_stride0, \
                               SCALAR_KARG(int64_t) shape1, SCALAR_KARG(int64_t) dst_stride1, SCALAR_KARG(int64_t) src_stride1, \
                               SCALAR_KARG(int64_t) shape2, SCALAR_KARG(int64_t) dst_stride2, SCALAR_KARG(int64_t) src_stride2, \
                               SCALAR_KARG(int64_t) shape3, SCALAR_KARG(int64_t) dst_stride3, SCALAR_KARG(int64_t) src_stride3, \
                               SCALAR_KARG(int64_t) shape4, SCALAR_KARG(int64_t) dst_stride4, SCALAR_KARG(int64_t) src_stride4, \
                               SCALAR_KARG(int64_t) shape5, SCALAR_KARG(int64_t) dst_stride5, SCALAR_KARG(int64_t) src_stride5, \
                               SCALAR_KARG(int64_t) shape6, SCALAR_KARG(int64_t) dst_stride6, SCALAR_KARG(int64_t) src_stride6, \
                               SCALAR_KARG(int64_t) shape7, SCALAR_KARG(int64_t) dst_stride7, SCALAR_KARG(int64_t) src_stride7) { \
  int64_t gtid = get_global_id(0);                                      \
  int64_t remainder = gtid;                                             \
  int64_t dst_offset = dst_offset_arg;                                  \
  int64_t src_offset = src_offset_arg;                                  \
                                                                        \
  if (gtid >= n) {                                                      \
    return;                                                             \
  }                                                                     \
                                                                        \
  if (r > 0) {                                                          \
    int64_t i = remainder % shape0;                                     \
    dst_offset += i * dst_stride0;                                      \
    src_offset += i * src_stride0;                                      \
    remainder /= shape0;                                                \
  }                                                                     \
  if (r > 1) {                                                          \
    int64_t i = remainder % shape1;                                     \
    dst_offset += i * dst_stride1;                                      \
    src_offset += i * src_stride1;                                      \
    remainder /= shape1;                                                \
  }                                                                     \
  if (r > 2) {                                                          \
    int64_t i = remainder % shape2;                                     \
    dst_offset += i * dst_stride2;                                      \
    src_offset += i * src_stride2;                                      \
    remainder /= shape2;                                                \
  }                                                                     \
  if (r > 3) {                                                          \
    int64_t i = remainder % shape3;                                     \
    dst_offset += i * dst_stride3;                                      \
    src_offset += i * src_stride3;                                      \
    remainder /= shape3;                                                \
  }                                                                     \
  if (r > 4) {                                                          \
    int64_t i = remainder % shape4;                                     \
    dst_offset += i * dst_stride4;                                      \
    src_offset += i * src_stride4;                                      \
    remainder /= shape4;                                                \
  }                                                                     \
  if (r > 5) {                                                          \
    int64_t i = remainder % shape5;                                     \
    dst_offset += i * dst_stride5;                                      \
    src_offset += i * src_stride5;                                      \
    remainder /= shape5;                                                \
  }                                                                     \
  if (r > 6) {                                                          \
    int64_t i = remainder % shape6;                                     \
    dst_offset += i * dst_stride6;                                      \
    src_offset += i * src_stride6;                                      \
    remainder /= shape6;                                                \
  }                                                                     \
  if (r > 7) {                                                          \
    int64_t i = remainder % shape7;                                     \
    dst_offset += i * dst_stride7;                                      \
    src_offset += i * src_stride7;                                      \
    remainder /= shape7;                                                \
  }                                                                     \
                                                                        \
  dst_mem[dst_offset] = src_mem[src_offset];                            \
}

GEN_COPY_KERNEL(1b, uint8_t)
GEN_COPY_KERNEL(2b, uint16_t)
GEN_COPY_KERNEL(4b, uint32_t)
GEN_COPY_KERNEL(8b, uint64_t)

// End of copy.cl
