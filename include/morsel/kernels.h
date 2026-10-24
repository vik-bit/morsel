#pragma once

// The hot loops of the engine: building selection vectors from comparisons
// and int64 arithmetic over one vector of values. Each kernel has a scalar
// version and an AVX2 version; kernels() picks one at runtime from the CPU,
// the same scheme vecdex uses for its distance functions.
//
// A selection vector is an ascending list of positions (0 .. n-1) of the rows
// that are still alive. "select" kernels read every position of a dense
// vector; "refine" kernels read only the positions in an input selection.
// Outputs may alias the input selection (the write index never passes the
// read index). Output arrays need room for n + 8 entries: the AVX2 kernels
// store a full register past the last selected position.

#include <cstdint>

namespace morsel {

enum class CmpOp : std::uint8_t { Eq, Ne, Lt, Le, Gt, Ge };

// Flips the operands: (c op x) == (x flip(op) c).
CmpOp flip(CmpOp op);

struct Kernels {
  // x op c
  std::uint32_t (*select_i32)(const std::int32_t* x, std::uint32_t n, CmpOp op, std::int32_t c,
                              std::uint32_t* out);
  std::uint32_t (*select_i64)(const std::int64_t* x, std::uint32_t n, CmpOp op, std::int64_t c,
                              std::uint32_t* out);
  std::uint32_t (*refine_i32)(const std::int32_t* x, const std::uint32_t* sel, std::uint32_t n,
                              CmpOp op, std::int32_t c, std::uint32_t* out);
  std::uint32_t (*refine_i64)(const std::int64_t* x, const std::uint32_t* sel, std::uint32_t n,
                              CmpOp op, std::int64_t c, std::uint32_t* out);

  // lo <= x <= hi
  std::uint32_t (*between_i32)(const std::int32_t* x, std::uint32_t n, std::int32_t lo,
                               std::int32_t hi, std::uint32_t* out);
  std::uint32_t (*between_i64)(const std::int64_t* x, std::uint32_t n, std::int64_t lo,
                               std::int64_t hi, std::uint32_t* out);
  std::uint32_t (*refine_between_i32)(const std::int32_t* x, const std::uint32_t* sel,
                                      std::uint32_t n, std::int32_t lo, std::int32_t hi,
                                      std::uint32_t* out);
  std::uint32_t (*refine_between_i64)(const std::int64_t* x, const std::uint32_t* sel,
                                      std::uint32_t n, std::int64_t lo, std::int64_t hi,
                                      std::uint32_t* out);

  // a op b, two columns
  std::uint32_t (*select_cols_i32)(const std::int32_t* a, const std::int32_t* b, std::uint32_t n,
                                   CmpOp op, std::uint32_t* out);
  std::uint32_t (*refine_cols_i32)(const std::int32_t* a, const std::int32_t* b,
                                   const std::uint32_t* sel, std::uint32_t n, CmpOp op,
                                   std::uint32_t* out);

  // Dense int64 arithmetic with wrap-around (two's complement) semantics.
  void (*add_i64)(const std::int64_t* a, const std::int64_t* b, std::int64_t* out, std::uint32_t n);
  void (*sub_i64)(const std::int64_t* a, const std::int64_t* b, std::int64_t* out, std::uint32_t n);
  void (*mul_i64)(const std::int64_t* a, const std::int64_t* b, std::int64_t* out, std::uint32_t n);
  void (*add_const_i64)(const std::int64_t* a, std::int64_t c, std::int64_t* out, std::uint32_t n);
  void (*rsub_const_i64)(const std::int64_t* a, std::int64_t c, std::int64_t* out,
                         std::uint32_t n);  // c - a
  void (*mul_const_i64)(const std::int64_t* a, std::int64_t c, std::int64_t* out, std::uint32_t n);
};

const Kernels& kernels(bool allow_simd = true);
const char* simd_name();

// Process-wide switch used by the CLI's --scalar flag and the tests.
void set_simd_enabled(bool enabled);
bool simd_enabled();
inline const Kernels& active_kernels() { return kernels(simd_enabled()); }

}  // namespace morsel
