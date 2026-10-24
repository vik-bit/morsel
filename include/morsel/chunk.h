#pragma once

// The unit of vectorized execution: up to kVectorSize rows of a few columns,
// plus a selection vector naming the rows that are still alive. Filters only
// shrink the selection; values stay where they are until an operator (a join
// probe, a sink) gathers them.

#include <cstdint>
#include <vector>

#include "morsel/types.h"

namespace morsel {

// 2048 rows of an int64 column is 16 KiB, so the handful of columns a query
// touches at once stays in L1/L2 while a vector moves through the pipeline.
constexpr std::uint32_t kVectorSize = 2048;

// Room for a full vector of selection indices plus the 8 entries of slack the
// AVX2 compaction stores past the end.
constexpr std::uint32_t kSelCapacity = kVectorSize + 8;

// Active rows. idx == nullptr means all of 0 .. size-1 (a "dense" chunk).
struct Sel {
  const std::uint32_t* idx = nullptr;
  std::uint32_t count = 0;
};

struct Chunk {
  std::vector<const void*> cols;  // one pointer per schema column, valid for [0, size)
  std::uint32_t size = 0;
  Sel sel;

  Sel all() const { return {nullptr, size}; }
};

// Expressions compute over every position when most rows are alive (the SIMD
// kernels run over the whole vector) and over just the selected positions
// when few are.
inline bool compute_dense(const Chunk& chunk, const Sel& sel) {
  return sel.idx == nullptr || sel.count * 4 >= chunk.size;
}

template <class F>
inline void for_each_selected(const Sel& sel, F&& f) {
  if (sel.idx) {
    for (std::uint32_t j = 0; j < sel.count; ++j) f(sel.idx[j]);
  } else {
    for (std::uint32_t i = 0; i < sel.count; ++i) f(i);
  }
}

// Positions an expression must produce for this chunk and active set.
template <class F>
inline void for_each_computed(const Chunk& chunk, const Sel& sel, F&& f) {
  if (compute_dense(chunk, sel)) {
    for (std::uint32_t i = 0; i < chunk.size; ++i) f(i);
  } else {
    for (std::uint32_t j = 0; j < sel.count; ++j) f(sel.idx[j]);
  }
}

}  // namespace morsel
