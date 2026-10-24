#include "morsel/kernels.h"

#include <atomic>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define MORSEL_HAVE_AVX2 1
#endif

namespace morsel {

CmpOp flip(CmpOp op) {
  switch (op) {
    case CmpOp::Lt:
      return CmpOp::Gt;
    case CmpOp::Le:
      return CmpOp::Ge;
    case CmpOp::Gt:
      return CmpOp::Lt;
    case CmpOp::Ge:
      return CmpOp::Le;
    default:
      return op;
  }
}

namespace {

// Instantiates F<op>(args...) for the runtime op, so the op is a compile-time
// constant inside every loop.
#define MORSEL_DISPATCH_OP(op, F, ...)        \
  switch (op) {                               \
    case CmpOp::Eq:                           \
      return F<CmpOp::Eq>(__VA_ARGS__);       \
    case CmpOp::Ne:                           \
      return F<CmpOp::Ne>(__VA_ARGS__);       \
    case CmpOp::Lt:                           \
      return F<CmpOp::Lt>(__VA_ARGS__);       \
    case CmpOp::Le:                           \
      return F<CmpOp::Le>(__VA_ARGS__);       \
    case CmpOp::Gt:                           \
      return F<CmpOp::Gt>(__VA_ARGS__);       \
    default:                                  \
      return F<CmpOp::Ge>(__VA_ARGS__);       \
  }

template <CmpOp OP, class T>
inline bool compare(T a, T b) {
  switch (OP) {
    case CmpOp::Eq:
      return a == b;
    case CmpOp::Ne:
      return a != b;
    case CmpOp::Lt:
      return a < b;
    case CmpOp::Le:
      return a <= b;
    case CmpOp::Gt:
      return a > b;
    default:
      return a >= b;
  }
}

// ---------------------------------------------------------------------------
// Scalar kernels. Selection is written branch-free: every position is stored
// and the write index advances only when the row passes, so the loop has no
// data-dependent branch to mispredict.

template <CmpOp OP, class T>
std::uint32_t select_scalar(const T* x, std::uint32_t n, T c, std::uint32_t* out) {
  std::uint32_t k = 0;
  for (std::uint32_t i = 0; i < n; ++i) {
    out[k] = i;
    k += compare<OP>(x[i], c);
  }
  return k;
}

template <CmpOp OP, class T>
std::uint32_t refine_scalar(const T* x, const std::uint32_t* sel, std::uint32_t n, T c,
                            std::uint32_t* out) {
  std::uint32_t k = 0;
  for (std::uint32_t j = 0; j < n; ++j) {
    const std::uint32_t i = sel[j];
    out[k] = i;
    k += compare<OP>(x[i], c);
  }
  return k;
}

template <class T>
std::uint32_t between_scalar(const T* x, std::uint32_t n, T lo, T hi, std::uint32_t* out) {
  std::uint32_t k = 0;
  for (std::uint32_t i = 0; i < n; ++i) {
    out[k] = i;
    k += (x[i] >= lo) & (x[i] <= hi);
  }
  return k;
}

template <class T>
std::uint32_t refine_between_scalar(const T* x, const std::uint32_t* sel, std::uint32_t n, T lo,
                                    T hi, std::uint32_t* out) {
  std::uint32_t k = 0;
  for (std::uint32_t j = 0; j < n; ++j) {
    const std::uint32_t i = sel[j];
    out[k] = i;
    k += (x[i] >= lo) & (x[i] <= hi);
  }
  return k;
}

template <CmpOp OP>
std::uint32_t select_cols_scalar(const std::int32_t* a, const std::int32_t* b, std::uint32_t n,
                                 std::uint32_t* out) {
  std::uint32_t k = 0;
  for (std::uint32_t i = 0; i < n; ++i) {
    out[k] = i;
    k += compare<OP>(a[i], b[i]);
  }
  return k;
}

template <CmpOp OP>
std::uint32_t refine_cols_scalar(const std::int32_t* a, const std::int32_t* b,
                                 const std::uint32_t* sel, std::uint32_t n, std::uint32_t* out) {
  std::uint32_t k = 0;
  for (std::uint32_t j = 0; j < n; ++j) {
    const std::uint32_t i = sel[j];
    out[k] = i;
    k += compare<OP>(a[i], b[i]);
  }
  return k;
}

std::uint32_t select_i32_scalar(const std::int32_t* x, std::uint32_t n, CmpOp op, std::int32_t c,
                                std::uint32_t* out) {
  MORSEL_DISPATCH_OP(op, select_scalar, x, n, c, out)
}
std::uint32_t select_i64_scalar(const std::int64_t* x, std::uint32_t n, CmpOp op, std::int64_t c,
                                std::uint32_t* out) {
  MORSEL_DISPATCH_OP(op, select_scalar, x, n, c, out)
}
std::uint32_t refine_i32_scalar(const std::int32_t* x, const std::uint32_t* sel, std::uint32_t n,
                                CmpOp op, std::int32_t c, std::uint32_t* out) {
  MORSEL_DISPATCH_OP(op, refine_scalar, x, sel, n, c, out)
}
std::uint32_t refine_i64_scalar(const std::int64_t* x, const std::uint32_t* sel, std::uint32_t n,
                                CmpOp op, std::int64_t c, std::uint32_t* out) {
  MORSEL_DISPATCH_OP(op, refine_scalar, x, sel, n, c, out)
}
std::uint32_t between_i32_scalar(const std::int32_t* x, std::uint32_t n, std::int32_t lo,
                                 std::int32_t hi, std::uint32_t* out) {
  return between_scalar(x, n, lo, hi, out);
}
std::uint32_t between_i64_scalar(const std::int64_t* x, std::uint32_t n, std::int64_t lo,
                                 std::int64_t hi, std::uint32_t* out) {
  return between_scalar(x, n, lo, hi, out);
}
std::uint32_t refine_between_i32_scalar(const std::int32_t* x, const std::uint32_t* sel,
                                        std::uint32_t n, std::int32_t lo, std::int32_t hi,
                                        std::uint32_t* out) {
  return refine_between_scalar(x, sel, n, lo, hi, out);
}
std::uint32_t refine_between_i64_scalar(const std::int64_t* x, const std::uint32_t* sel,
                                        std::uint32_t n, std::int64_t lo, std::int64_t hi,
                                        std::uint32_t* out) {
  return refine_between_scalar(x, sel, n, lo, hi, out);
}
std::uint32_t select_cols_i32_scalar(const std::int32_t* a, const std::int32_t* b, std::uint32_t n,
                                     CmpOp op, std::uint32_t* out) {
  MORSEL_DISPATCH_OP(op, select_cols_scalar, a, b, n, out)
}
std::uint32_t refine_cols_i32_scalar(const std::int32_t* a, const std::int32_t* b,
                                     const std::uint32_t* sel, std::uint32_t n, CmpOp op,
                                     std::uint32_t* out) {
  MORSEL_DISPATCH_OP(op, refine_cols_scalar, a, b, sel, n, out)
}

// Unsigned arithmetic: wraps instead of being undefined behaviour on overflow.
inline std::int64_t wrap_add(std::int64_t a, std::int64_t b) {
  return static_cast<std::int64_t>(static_cast<std::uint64_t>(a) + static_cast<std::uint64_t>(b));
}
inline std::int64_t wrap_sub(std::int64_t a, std::int64_t b) {
  return static_cast<std::int64_t>(static_cast<std::uint64_t>(a) - static_cast<std::uint64_t>(b));
}
inline std::int64_t wrap_mul(std::int64_t a, std::int64_t b) {
  return static_cast<std::int64_t>(static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(b));
}

void add_i64_scalar(const std::int64_t* a, const std::int64_t* b, std::int64_t* out,
                    std::uint32_t n) {
  for (std::uint32_t i = 0; i < n; ++i) out[i] = wrap_add(a[i], b[i]);
}
void sub_i64_scalar(const std::int64_t* a, const std::int64_t* b, std::int64_t* out,
                    std::uint32_t n) {
  for (std::uint32_t i = 0; i < n; ++i) out[i] = wrap_sub(a[i], b[i]);
}
void mul_i64_scalar(const std::int64_t* a, const std::int64_t* b, std::int64_t* out,
                    std::uint32_t n) {
  for (std::uint32_t i = 0; i < n; ++i) out[i] = wrap_mul(a[i], b[i]);
}
void add_const_i64_scalar(const std::int64_t* a, std::int64_t c, std::int64_t* out,
                          std::uint32_t n) {
  for (std::uint32_t i = 0; i < n; ++i) out[i] = wrap_add(a[i], c);
}
void rsub_const_i64_scalar(const std::int64_t* a, std::int64_t c, std::int64_t* out,
                           std::uint32_t n) {
  for (std::uint32_t i = 0; i < n; ++i) out[i] = wrap_sub(c, a[i]);
}
void mul_const_i64_scalar(const std::int64_t* a, std::int64_t c, std::int64_t* out,
                          std::uint32_t n) {
  for (std::uint32_t i = 0; i < n; ++i) out[i] = wrap_mul(a[i], c);
}

#if MORSEL_HAVE_AVX2
// ---------------------------------------------------------------------------
// AVX2 kernels, compiled through the target attribute so the rest of the
// library builds for baseline x86-64 and kernels() only hands these out after
// checking the CPU.
#define MORSEL_AVX2_FN __attribute__((target("avx2,bmi,popcnt")))

// kCompact[m] lists the set bits of the 8-bit mask m, in order. Permuting a
// register of 8 positions by it packs the passing positions to the front:
// the AVX2 stand-in for AVX-512's vpcompressd.
struct CompactTable {
  alignas(32) std::uint32_t lanes[256][8];
  CompactTable() {
    for (int m = 0; m < 256; ++m) {
      int k = 0;
      for (int b = 0; b < 8; ++b) {
        if (m & (1 << b)) lanes[m][k++] = static_cast<std::uint32_t>(b);
      }
      while (k < 8) lanes[m][k++] = 0;
    }
  }
};
const CompactTable kCompact;

template <CmpOp OP>
MORSEL_AVX2_FN inline unsigned mask_i32(__m256i a, __m256i b) {
  __m256i m;
  bool invert = false;
  switch (OP) {
    case CmpOp::Eq:
      m = _mm256_cmpeq_epi32(a, b);
      break;
    case CmpOp::Ne:
      m = _mm256_cmpeq_epi32(a, b);
      invert = true;
      break;
    case CmpOp::Gt:
      m = _mm256_cmpgt_epi32(a, b);
      break;
    case CmpOp::Lt:
      m = _mm256_cmpgt_epi32(b, a);
      break;
    case CmpOp::Ge:
      m = _mm256_cmpgt_epi32(b, a);
      invert = true;
      break;
    default:  // Le
      m = _mm256_cmpgt_epi32(a, b);
      invert = true;
      break;
  }
  const unsigned bits = static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(m)));
  return invert ? bits ^ 0xFFu : bits;
}

template <CmpOp OP>
MORSEL_AVX2_FN inline unsigned mask_i64x4(__m256i a, __m256i b) {
  __m256i m;
  bool invert = false;
  switch (OP) {
    case CmpOp::Eq:
      m = _mm256_cmpeq_epi64(a, b);
      break;
    case CmpOp::Ne:
      m = _mm256_cmpeq_epi64(a, b);
      invert = true;
      break;
    case CmpOp::Gt:
      m = _mm256_cmpgt_epi64(a, b);
      break;
    case CmpOp::Lt:
      m = _mm256_cmpgt_epi64(b, a);
      break;
    case CmpOp::Ge:
      m = _mm256_cmpgt_epi64(b, a);
      invert = true;
      break;
    default:  // Le
      m = _mm256_cmpgt_epi64(a, b);
      invert = true;
      break;
  }
  const unsigned bits = static_cast<unsigned>(_mm256_movemask_pd(_mm256_castsi256_pd(m)));
  return invert ? bits ^ 0xFu : bits;
}

// Appends the positions in `positions` whose bit is set in mask to out[k..].
MORSEL_AVX2_FN inline std::uint32_t compact_store(__m256i positions, unsigned mask,
                                                  std::uint32_t* out, std::uint32_t k) {
  const __m256i perm = _mm256_load_si256(reinterpret_cast<const __m256i*>(kCompact.lanes[mask]));
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + k),
                      _mm256_permutevar8x32_epi32(positions, perm));
  return k + static_cast<std::uint32_t>(_mm_popcnt_u32(mask));
}

template <CmpOp OP>
MORSEL_AVX2_FN std::uint32_t select_i32_avx2_op(const std::int32_t* x, std::uint32_t n,
                                                std::int32_t c, std::uint32_t* out) {
  const __m256i cv = _mm256_set1_epi32(c);
  const __m256i step = _mm256_set1_epi32(8);
  __m256i pos = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  std::uint32_t k = 0, i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i));
    k = compact_store(pos, mask_i32<OP>(v, cv), out, k);
    pos = _mm256_add_epi32(pos, step);
  }
  for (; i < n; ++i) {
    out[k] = i;
    k += compare<OP>(x[i], c);
  }
  return k;
}

template <CmpOp OP>
MORSEL_AVX2_FN std::uint32_t select_i64_avx2_op(const std::int64_t* x, std::uint32_t n,
                                                std::int64_t c, std::uint32_t* out) {
  const __m256i cv = _mm256_set1_epi64x(c);
  const __m256i step = _mm256_set1_epi32(8);
  __m256i pos = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  std::uint32_t k = 0, i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i));
    const __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i + 4));
    const unsigned mask = mask_i64x4<OP>(v0, cv) | (mask_i64x4<OP>(v1, cv) << 4);
    k = compact_store(pos, mask, out, k);
    pos = _mm256_add_epi32(pos, step);
  }
  for (; i < n; ++i) {
    out[k] = i;
    k += compare<OP>(x[i], c);
  }
  return k;
}

// Refinement loads only the selected values, with a gather.
template <CmpOp OP>
MORSEL_AVX2_FN std::uint32_t refine_i32_avx2_op(const std::int32_t* x, const std::uint32_t* sel,
                                                std::uint32_t n, std::int32_t c,
                                                std::uint32_t* out) {
  const __m256i cv = _mm256_set1_epi32(c);
  std::uint32_t k = 0, j = 0;
  for (; j + 8 <= n; j += 8) {
    const __m256i s = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sel + j));
    const __m256i v = _mm256_i32gather_epi32(reinterpret_cast<const int*>(x), s, 4);
    k = compact_store(s, mask_i32<OP>(v, cv), out, k);
  }
  for (; j < n; ++j) {
    const std::uint32_t i = sel[j];
    out[k] = i;
    k += compare<OP>(x[i], c);
  }
  return k;
}

MORSEL_AVX2_FN inline void gather_i64x8(const std::int64_t* x, __m256i s, __m256i* v0,
                                        __m256i* v1) {
  const auto* base = reinterpret_cast<const long long*>(x);
  *v0 = _mm256_i32gather_epi64(base, _mm256_castsi256_si128(s), 8);
  *v1 = _mm256_i32gather_epi64(base, _mm256_extracti128_si256(s, 1), 8);
}

template <CmpOp OP>
MORSEL_AVX2_FN std::uint32_t refine_i64_avx2_op(const std::int64_t* x, const std::uint32_t* sel,
                                                std::uint32_t n, std::int64_t c,
                                                std::uint32_t* out) {
  const __m256i cv = _mm256_set1_epi64x(c);
  std::uint32_t k = 0, j = 0;
  for (; j + 8 <= n; j += 8) {
    const __m256i s = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sel + j));
    __m256i v0, v1;
    gather_i64x8(x, s, &v0, &v1);
    const unsigned mask = mask_i64x4<OP>(v0, cv) | (mask_i64x4<OP>(v1, cv) << 4);
    k = compact_store(s, mask, out, k);
  }
  for (; j < n; ++j) {
    const std::uint32_t i = sel[j];
    out[k] = i;
    k += compare<OP>(x[i], c);
  }
  return k;
}

// lo <= v <= hi  ==  !(lo > v) && !(v > hi)
MORSEL_AVX2_FN inline unsigned between_mask_i32(__m256i v, __m256i lo, __m256i hi) {
  const __m256i out = _mm256_or_si256(_mm256_cmpgt_epi32(lo, v), _mm256_cmpgt_epi32(v, hi));
  return static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(out))) ^ 0xFFu;
}

MORSEL_AVX2_FN inline unsigned between_mask_i64x4(__m256i v, __m256i lo, __m256i hi) {
  const __m256i out = _mm256_or_si256(_mm256_cmpgt_epi64(lo, v), _mm256_cmpgt_epi64(v, hi));
  return static_cast<unsigned>(_mm256_movemask_pd(_mm256_castsi256_pd(out))) ^ 0xFu;
}

MORSEL_AVX2_FN std::uint32_t between_i32_avx2(const std::int32_t* x, std::uint32_t n,
                                              std::int32_t lo, std::int32_t hi,
                                              std::uint32_t* out) {
  const __m256i lov = _mm256_set1_epi32(lo), hiv = _mm256_set1_epi32(hi);
  const __m256i step = _mm256_set1_epi32(8);
  __m256i pos = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  std::uint32_t k = 0, i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i));
    k = compact_store(pos, between_mask_i32(v, lov, hiv), out, k);
    pos = _mm256_add_epi32(pos, step);
  }
  for (; i < n; ++i) {
    out[k] = i;
    k += (x[i] >= lo) & (x[i] <= hi);
  }
  return k;
}

MORSEL_AVX2_FN std::uint32_t between_i64_avx2(const std::int64_t* x, std::uint32_t n,
                                              std::int64_t lo, std::int64_t hi,
                                              std::uint32_t* out) {
  const __m256i lov = _mm256_set1_epi64x(lo), hiv = _mm256_set1_epi64x(hi);
  const __m256i step = _mm256_set1_epi32(8);
  __m256i pos = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  std::uint32_t k = 0, i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i));
    const __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(x + i + 4));
    const unsigned mask = between_mask_i64x4(v0, lov, hiv) | (between_mask_i64x4(v1, lov, hiv) << 4);
    k = compact_store(pos, mask, out, k);
    pos = _mm256_add_epi32(pos, step);
  }
  for (; i < n; ++i) {
    out[k] = i;
    k += (x[i] >= lo) & (x[i] <= hi);
  }
  return k;
}

MORSEL_AVX2_FN std::uint32_t refine_between_i32_avx2(const std::int32_t* x,
                                                     const std::uint32_t* sel, std::uint32_t n,
                                                     std::int32_t lo, std::int32_t hi,
                                                     std::uint32_t* out) {
  const __m256i lov = _mm256_set1_epi32(lo), hiv = _mm256_set1_epi32(hi);
  std::uint32_t k = 0, j = 0;
  for (; j + 8 <= n; j += 8) {
    const __m256i s = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sel + j));
    const __m256i v = _mm256_i32gather_epi32(reinterpret_cast<const int*>(x), s, 4);
    k = compact_store(s, between_mask_i32(v, lov, hiv), out, k);
  }
  for (; j < n; ++j) {
    const std::uint32_t i = sel[j];
    out[k] = i;
    k += (x[i] >= lo) & (x[i] <= hi);
  }
  return k;
}

MORSEL_AVX2_FN std::uint32_t refine_between_i64_avx2(const std::int64_t* x,
                                                     const std::uint32_t* sel, std::uint32_t n,
                                                     std::int64_t lo, std::int64_t hi,
                                                     std::uint32_t* out) {
  const __m256i lov = _mm256_set1_epi64x(lo), hiv = _mm256_set1_epi64x(hi);
  std::uint32_t k = 0, j = 0;
  for (; j + 8 <= n; j += 8) {
    const __m256i s = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sel + j));
    __m256i v0, v1;
    gather_i64x8(x, s, &v0, &v1);
    const unsigned mask = between_mask_i64x4(v0, lov, hiv) | (between_mask_i64x4(v1, lov, hiv) << 4);
    k = compact_store(s, mask, out, k);
  }
  for (; j < n; ++j) {
    const std::uint32_t i = sel[j];
    out[k] = i;
    k += (x[i] >= lo) & (x[i] <= hi);
  }
  return k;
}

template <CmpOp OP>
MORSEL_AVX2_FN std::uint32_t select_cols_i32_avx2_op(const std::int32_t* a, const std::int32_t* b,
                                                     std::uint32_t n, std::uint32_t* out) {
  const __m256i step = _mm256_set1_epi32(8);
  __m256i pos = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  std::uint32_t k = 0, i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
    const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
    k = compact_store(pos, mask_i32<OP>(va, vb), out, k);
    pos = _mm256_add_epi32(pos, step);
  }
  for (; i < n; ++i) {
    out[k] = i;
    k += compare<OP>(a[i], b[i]);
  }
  return k;
}

template <CmpOp OP>
MORSEL_AVX2_FN std::uint32_t refine_cols_i32_avx2_op(const std::int32_t* a, const std::int32_t* b,
                                                     const std::uint32_t* sel, std::uint32_t n,
                                                     std::uint32_t* out) {
  std::uint32_t k = 0, j = 0;
  for (; j + 8 <= n; j += 8) {
    const __m256i s = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sel + j));
    const __m256i va = _mm256_i32gather_epi32(reinterpret_cast<const int*>(a), s, 4);
    const __m256i vb = _mm256_i32gather_epi32(reinterpret_cast<const int*>(b), s, 4);
    k = compact_store(s, mask_i32<OP>(va, vb), out, k);
  }
  for (; j < n; ++j) {
    const std::uint32_t i = sel[j];
    out[k] = i;
    k += compare<OP>(a[i], b[i]);
  }
  return k;
}

std::uint32_t select_i32_avx2(const std::int32_t* x, std::uint32_t n, CmpOp op, std::int32_t c,
                              std::uint32_t* out) {
  MORSEL_DISPATCH_OP(op, select_i32_avx2_op, x, n, c, out)
}
std::uint32_t select_i64_avx2(const std::int64_t* x, std::uint32_t n, CmpOp op, std::int64_t c,
                              std::uint32_t* out) {
  MORSEL_DISPATCH_OP(op, select_i64_avx2_op, x, n, c, out)
}
std::uint32_t refine_i32_avx2(const std::int32_t* x, const std::uint32_t* sel, std::uint32_t n,
                              CmpOp op, std::int32_t c, std::uint32_t* out) {
  MORSEL_DISPATCH_OP(op, refine_i32_avx2_op, x, sel, n, c, out)
}
std::uint32_t refine_i64_avx2(const std::int64_t* x, const std::uint32_t* sel, std::uint32_t n,
                              CmpOp op, std::int64_t c, std::uint32_t* out) {
  MORSEL_DISPATCH_OP(op, refine_i64_avx2_op, x, sel, n, c, out)
}
std::uint32_t select_cols_i32_avx2(const std::int32_t* a, const std::int32_t* b, std::uint32_t n,
                                   CmpOp op, std::uint32_t* out) {
  MORSEL_DISPATCH_OP(op, select_cols_i32_avx2_op, a, b, n, out)
}
std::uint32_t refine_cols_i32_avx2(const std::int32_t* a, const std::int32_t* b,
                                   const std::uint32_t* sel, std::uint32_t n, CmpOp op,
                                   std::uint32_t* out) {
  MORSEL_DISPATCH_OP(op, refine_cols_i32_avx2_op, a, b, sel, n, out)
}

// AVX2 has no 64-bit multiply (vpmullq is AVX-512), so build the low 64 bits
// of the product from three 32x32->64 multiplies:
//   a*b mod 2^64 = lo(a)lo(b) + ((hi(a)lo(b) + lo(a)hi(b)) << 32)
MORSEL_AVX2_FN inline __m256i mul64(__m256i a, __m256i b) {
  const __m256i lo = _mm256_mul_epu32(a, b);
  const __m256i cross = _mm256_add_epi64(_mm256_mul_epu32(_mm256_srli_epi64(a, 32), b),
                                         _mm256_mul_epu32(a, _mm256_srli_epi64(b, 32)));
  return _mm256_add_epi64(lo, _mm256_slli_epi64(cross, 32));
}

#define MORSEL_AVX2_BINARY(name, vec_expr, scalar_fn)                                         \
  MORSEL_AVX2_FN void name(const std::int64_t* a, const std::int64_t* b, std::int64_t* out,  \
                           std::uint32_t n) {                                                \
    std::uint32_t i = 0;                                                                     \
    for (; i + 4 <= n; i += 4) {                                                             \
      const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));        \
      const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));        \
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), vec_expr);                    \
    }                                                                                        \
    for (; i < n; ++i) out[i] = scalar_fn(a[i], b[i]);                                       \
  }

MORSEL_AVX2_BINARY(add_i64_avx2, _mm256_add_epi64(va, vb), wrap_add)
MORSEL_AVX2_BINARY(sub_i64_avx2, _mm256_sub_epi64(va, vb), wrap_sub)
MORSEL_AVX2_BINARY(mul_i64_avx2, mul64(va, vb), wrap_mul)

#define MORSEL_AVX2_CONST(name, vec_expr, scalar_expr)                                     \
  MORSEL_AVX2_FN void name(const std::int64_t* a, std::int64_t c, std::int64_t* out,      \
                           std::uint32_t n) {                                             \
    const __m256i vc = _mm256_set1_epi64x(c);                                             \
    std::uint32_t i = 0;                                                                  \
    for (; i + 4 <= n; i += 4) {                                                          \
      const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));     \
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), vec_expr);                 \
    }                                                                                     \
    for (; i < n; ++i) out[i] = scalar_expr;                                              \
  }

MORSEL_AVX2_CONST(add_const_i64_avx2, _mm256_add_epi64(va, vc), wrap_add(a[i], c))
MORSEL_AVX2_CONST(rsub_const_i64_avx2, _mm256_sub_epi64(vc, va), wrap_sub(c, a[i]))
MORSEL_AVX2_CONST(mul_const_i64_avx2, mul64(va, vc), wrap_mul(a[i], c))

bool cpu_has_avx2() {
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("bmi") &&
         __builtin_cpu_supports("popcnt");
}

const Kernels kAvx2Kernels = {
    select_i32_avx2,         select_i64_avx2,         refine_i32_avx2,
    refine_i64_avx2,         between_i32_avx2,        between_i64_avx2,
    refine_between_i32_avx2, refine_between_i64_avx2, select_cols_i32_avx2,
    refine_cols_i32_avx2,    add_i64_avx2,            sub_i64_avx2,
    mul_i64_avx2,            add_const_i64_avx2,      rsub_const_i64_avx2,
    mul_const_i64_avx2,
};
#endif  // MORSEL_HAVE_AVX2

const Kernels kScalarKernels = {
    select_i32_scalar,         select_i64_scalar,         refine_i32_scalar,
    refine_i64_scalar,         between_i32_scalar,        between_i64_scalar,
    refine_between_i32_scalar, refine_between_i64_scalar, select_cols_i32_scalar,
    refine_cols_i32_scalar,    add_i64_scalar,            sub_i64_scalar,
    mul_i64_scalar,            add_const_i64_scalar,      rsub_const_i64_scalar,
    mul_const_i64_scalar,
};

const Kernels& best_kernels() {
#if MORSEL_HAVE_AVX2
  static const bool avx2 = cpu_has_avx2();
  if (avx2) return kAvx2Kernels;
#endif
  return kScalarKernels;
}

std::atomic<bool> g_simd_enabled{true};

}  // namespace

const Kernels& kernels(bool allow_simd) { return allow_simd ? best_kernels() : kScalarKernels; }

const char* simd_name() { return &best_kernels() == &kScalarKernels ? "scalar" : "avx2"; }

void set_simd_enabled(bool enabled) { g_simd_enabled.store(enabled, std::memory_order_relaxed); }
bool simd_enabled() { return g_simd_enabled.load(std::memory_order_relaxed); }

}  // namespace morsel
