#pragma once

// Execution internals shared by the operator implementations.

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "morsel/chunk.h"
#include "morsel/executor.h"
#include "morsel/plan.h"

namespace morsel::detail {

// ---------------------------------------------------------------------------
// Push-based pipeline plumbing. Each worker builds its own chain of stages;
// an operator hands each output chunk to the next stage, so a join probe that
// produces several vectors from one input vector simply calls emit several
// times.

class Emitter {
 public:
  virtual ~Emitter() = default;
  virtual void emit(Chunk& chunk) = 0;
};

struct OpState {
  virtual ~OpState() = default;
};

class Operator {
 public:
  virtual ~Operator() = default;
  virtual std::unique_ptr<OpState> make_state() const = 0;
  virtual void execute(Chunk& in, OpState& state, Emitter& next) const = 0;
};

struct SinkState {
  virtual ~SinkState() = default;
};

class Sink {
 public:
  virtual ~Sink() = default;
  virtual std::unique_ptr<SinkState> make_local() const = 0;
  virtual void consume(Chunk& chunk, SinkState& local) const = 0;
  // Merges the workers' local states; may use the pool to do it in parallel.
  virtual void finalize(std::vector<std::unique_ptr<SinkState>>& locals, ThreadPool& pool) = 0;

  MaterializedTable result;  // for sinks whose output feeds a later pipeline
};

std::unique_ptr<Operator> make_filter(const PlanNode& node);
std::unique_ptr<Operator> make_project(const PlanNode& node);
std::shared_ptr<Sink> make_join_build(const PlanNode& join);
std::unique_ptr<Operator> make_join_probe(const PlanNode& join, const std::shared_ptr<Sink>& build);
std::shared_ptr<Sink> make_aggregate(const PlanNode& node);
std::shared_ptr<Sink> make_collect(const Schema& schema);
std::shared_ptr<Sink> make_sort(const PlanNode& node);

// ---------------------------------------------------------------------------
// Keys for hash joins and hash aggregation: up to 16 bytes of key columns
// packed into one or two 64-bit words, so comparing keys is one or two
// integer compares whatever the column types.

struct Key128 {
  std::uint64_t lo = 0, hi = 0;
  bool operator==(const Key128& o) const { return lo == o.lo && hi == o.hi; }
};

// The 64-bit finalizer of MurmurHash3: cheap, and every input bit affects
// every output bit, so both the low bits (bucket) and the high bits
// (partition) are well mixed.
inline std::uint64_t mix64(std::uint64_t k) {
  k ^= k >> 33;
  k *= 0xff51afd7ed558ccdULL;
  k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53ULL;
  k ^= k >> 33;
  return k;
}
inline std::uint64_t hash_key(std::uint64_t k) { return mix64(k); }
inline std::uint64_t hash_key(const Key128& k) { return mix64(k.lo ^ mix64(k.hi + 0x9e3779b97f4a7c15ULL)); }

inline std::uint64_t& key_word(std::uint64_t& k, int) { return k; }
inline std::uint64_t& key_word(Key128& k, int w) { return w ? k.hi : k.lo; }
inline std::uint64_t key_word(const std::uint64_t& k, int) { return k; }
inline std::uint64_t key_word(const Key128& k, int w) { return w ? k.hi : k.lo; }

struct KeyLayout {
  struct Part {
    int column;  // in the input schema
    TypeId type;
    int word;    // 0 or 1
    int shift;   // bit offset in the word
  };
  std::vector<Part> parts;  // in key order
  bool wide = false;        // needs Key128

  // Widest columns first, so no column straddles the two words.
  static KeyLayout make(const Schema& schema, const std::vector<int>& columns);
};

template <class K>
void pack_keys(const KeyLayout& layout, const Chunk& chunk, const Sel& sel, K* keys) {
  for_each_selected(sel, [&](std::uint32_t i) { keys[i] = K{}; });
  for (const auto& part : layout.parts) {
    const void* col = chunk.cols[static_cast<std::size_t>(part.column)];
    const int w = part.word, s = part.shift;
    switch (part.type) {
      case TypeId::Int64: {
        const auto* v = static_cast<const std::int64_t*>(col);
        for_each_selected(sel, [&](std::uint32_t i) { key_word(keys[i], w) |= static_cast<std::uint64_t>(v[i]); });
        break;
      }
      case TypeId::Dict: {
        const auto* v = static_cast<const std::uint8_t*>(col);
        for_each_selected(sel, [&](std::uint32_t i) { key_word(keys[i], w) |= static_cast<std::uint64_t>(v[i]) << s; });
        break;
      }
      default: {  // Int32, Date
        const auto* v = static_cast<const std::uint32_t*>(col);
        for_each_selected(sel, [&](std::uint32_t i) { key_word(keys[i], w) |= static_cast<std::uint64_t>(v[i]) << s; });
        break;
      }
    }
  }
}

// Writes key part p of `key` as row `row` of an output column.
template <class K>
void unpack_key_part(const KeyLayout::Part& part, const K& key, void* column, std::size_t row) {
  const std::uint64_t word = key_word(key, part.word) >> part.shift;
  switch (part.type) {
    case TypeId::Int64:
      static_cast<std::int64_t*>(column)[row] = static_cast<std::int64_t>(word);
      break;
    case TypeId::Dict:
      static_cast<std::uint8_t*>(column)[row] = static_cast<std::uint8_t>(word);
      break;
    default:
      static_cast<std::uint32_t*>(column)[row] = static_cast<std::uint32_t>(word);
  }
}

// dst[j] = src[idx[j]] for elements of `width` bytes.
void gather(const void* src, std::size_t width, const std::uint32_t* idx, std::uint32_t n, void* dst);

// Appends the selected rows of `src` to dst, which holds `rows` rows already.
void append_selected(Buffer& dst, std::size_t rows, const void* src, std::size_t width, const Sel& sel);

inline std::size_t next_pow2(std::size_t v) {
  std::size_t p = 1;
  while (p < v) p <<= 1;
  return p;
}

}  // namespace morsel::detail
