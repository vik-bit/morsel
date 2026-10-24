#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

#include "check.h"
#include "morsel/chunk.h"
#include "morsel/kernels.h"

using morsel::CmpOp;
using morsel::kernels;
using morsel::Kernels;

namespace {

const CmpOp kOps[] = {CmpOp::Eq, CmpOp::Ne, CmpOp::Lt, CmpOp::Le, CmpOp::Gt, CmpOp::Ge};

template <class T>
bool reference_cmp(T a, CmpOp op, T b) {
  switch (op) {
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

// Random ascending subset of 0..n-1.
std::vector<std::uint32_t> random_sel(std::uint32_t n, double density, std::mt19937& rng) {
  std::bernoulli_distribution keep(density);
  std::vector<std::uint32_t> sel;
  for (std::uint32_t i = 0; i < n; ++i) {
    if (keep(rng)) sel.push_back(i);
  }
  return sel;
}

// Small value range so that every comparison sees ties; includes extremes.
template <class T>
std::vector<T> random_values(std::uint32_t n, std::mt19937& rng) {
  std::uniform_int_distribution<int> small(-4, 4);
  std::vector<T> v(n);
  for (auto& x : v) x = static_cast<T>(small(rng));
  if (n > 3) {
    v[0] = std::numeric_limits<T>::min();
    v[n - 1] = std::numeric_limits<T>::max();
  }
  return v;
}

std::vector<std::uint32_t> lengths() {
  std::vector<std::uint32_t> ns;
  for (std::uint32_t n = 0; n <= 70; ++n) ns.push_back(n);
  ns.push_back(1000);
  ns.push_back(morsel::kVectorSize);
  return ns;
}

}  // namespace

// The AVX2 and scalar kernels must give exactly the reference selection for
// every op, every length (full registers plus every tail) and every density.
TEST(select_kernels_match_reference) {
  std::mt19937 rng(1);
  const Kernels& fast = kernels(true);
  const Kernels& slow = kernels(false);
  for (std::uint32_t n : lengths()) {
    const auto v32 = random_values<std::int32_t>(n, rng);
    const auto v64 = random_values<std::int64_t>(n, rng);
    for (CmpOp op : kOps) {
      for (std::int64_t c : {-1LL, 0LL, 3LL}) {
        std::vector<std::uint32_t> expect32, expect64;
        for (std::uint32_t i = 0; i < n; ++i) {
          if (reference_cmp<std::int32_t>(v32[i], op, static_cast<std::int32_t>(c))) expect32.push_back(i);
          if (reference_cmp<std::int64_t>(v64[i], op, c)) expect64.push_back(i);
        }
        for (const Kernels* k : {&fast, &slow}) {
          std::vector<std::uint32_t> out(n + 8);
          std::uint32_t m = k->select_i32(v32.data(), n, op, static_cast<std::int32_t>(c), out.data());
          CHECK(std::vector<std::uint32_t>(out.begin(), out.begin() + m) == expect32);
          m = k->select_i64(v64.data(), n, op, c, out.data());
          CHECK(std::vector<std::uint32_t>(out.begin(), out.begin() + m) == expect64);
        }
      }
    }
  }
}

TEST(refine_kernels_match_reference_and_work_in_place) {
  std::mt19937 rng(2);
  for (std::uint32_t n : lengths()) {
    const auto v32 = random_values<std::int32_t>(n, rng);
    const auto v64 = random_values<std::int64_t>(n, rng);
    for (double density : {0.1, 0.5, 1.0}) {
      const auto sel = random_sel(n, density, rng);
      const auto s = static_cast<std::uint32_t>(sel.size());
      for (CmpOp op : kOps) {
        std::vector<std::uint32_t> expect32, expect64;
        for (std::uint32_t i : sel) {
          if (reference_cmp<std::int32_t>(v32[i], op, 1)) expect32.push_back(i);
          if (reference_cmp<std::int64_t>(v64[i], op, 1)) expect64.push_back(i);
        }
        for (bool simd : {true, false}) {
          const Kernels& k = kernels(simd);
          std::vector<std::uint32_t> buf(sel);
          buf.resize(s + 8);
          // Output aliases the input selection, as AND chains use it.
          std::uint32_t m = k.refine_i32(v32.data(), buf.data(), s, op, 1, buf.data());
          CHECK(std::vector<std::uint32_t>(buf.begin(), buf.begin() + m) == expect32);
          buf.assign(sel.begin(), sel.end());
          buf.resize(s + 8);
          m = k.refine_i64(v64.data(), buf.data(), s, op, 1, buf.data());
          CHECK(std::vector<std::uint32_t>(buf.begin(), buf.begin() + m) == expect64);
        }
      }
    }
  }
}

TEST(between_kernels_are_inclusive) {
  std::mt19937 rng(3);
  for (std::uint32_t n : lengths()) {
    const auto v32 = random_values<std::int32_t>(n, rng);
    const auto v64 = random_values<std::int64_t>(n, rng);
    const auto sel = random_sel(n, 0.6, rng);
    std::vector<std::uint32_t> dense32, dense64, sparse32, sparse64;
    for (std::uint32_t i = 0; i < n; ++i) {
      if (v32[i] >= -1 && v32[i] <= 2) dense32.push_back(i);
      if (v64[i] >= -1 && v64[i] <= 2) dense64.push_back(i);
    }
    for (std::uint32_t i : sel) {
      if (v32[i] >= -1 && v32[i] <= 2) sparse32.push_back(i);
      if (v64[i] >= -1 && v64[i] <= 2) sparse64.push_back(i);
    }
    for (bool simd : {true, false}) {
      const Kernels& k = kernels(simd);
      std::vector<std::uint32_t> out(n + 8);
      auto got = [&](std::uint32_t m) { return std::vector<std::uint32_t>(out.begin(), out.begin() + m); };
      CHECK(got(k.between_i32(v32.data(), n, -1, 2, out.data())) == dense32);
      CHECK(got(k.between_i64(v64.data(), n, -1, 2, out.data())) == dense64);
      const auto s = static_cast<std::uint32_t>(sel.size());
      CHECK(got(k.refine_between_i32(v32.data(), sel.data(), s, -1, 2, out.data())) == sparse32);
      CHECK(got(k.refine_between_i64(v64.data(), sel.data(), s, -1, 2, out.data())) == sparse64);
    }
  }
}

TEST(column_compare_kernels_match_reference) {
  std::mt19937 rng(4);
  for (std::uint32_t n : lengths()) {
    const auto a = random_values<std::int32_t>(n, rng);
    const auto b = random_values<std::int32_t>(n, rng);
    const auto sel = random_sel(n, 0.5, rng);
    for (CmpOp op : kOps) {
      std::vector<std::uint32_t> dense, sparse;
      for (std::uint32_t i = 0; i < n; ++i) {
        if (reference_cmp(a[i], op, b[i])) dense.push_back(i);
      }
      for (std::uint32_t i : sel) {
        if (reference_cmp(a[i], op, b[i])) sparse.push_back(i);
      }
      for (bool simd : {true, false}) {
        const Kernels& k = kernels(simd);
        std::vector<std::uint32_t> out(n + 8);
        std::uint32_t m = k.select_cols_i32(a.data(), b.data(), n, op, out.data());
        CHECK(std::vector<std::uint32_t>(out.begin(), out.begin() + m) == dense);
        m = k.refine_cols_i32(a.data(), b.data(), sel.data(), static_cast<std::uint32_t>(sel.size()), op, out.data());
        CHECK(std::vector<std::uint32_t>(out.begin(), out.begin() + m) == sparse);
      }
    }
  }
}

// Full-range operands: the emulated AVX2 64-bit multiply must wrap exactly
// like unsigned scalar arithmetic, including on overflow.
TEST(int64_arithmetic_kernels_wrap_like_scalar) {
  std::mt19937_64 rng(5);
  for (std::uint32_t n : lengths()) {
    std::vector<std::int64_t> a(n), b(n);
    for (std::uint32_t i = 0; i < n; ++i) {
      a[i] = static_cast<std::int64_t>(rng());
      b[i] = i % 3 ? static_cast<std::int64_t>(rng()) : static_cast<std::int64_t>(rng() % 2000) - 1000;
    }
    const std::int64_t c = -123456789012LL;
    for (int op = 0; op < 6; ++op) {
      std::vector<std::int64_t> fast(n), slow(n), expect(n);
      for (std::uint32_t i = 0; i < n; ++i) {
        const auto x = static_cast<std::uint64_t>(a[i]), y = static_cast<std::uint64_t>(b[i]);
        const auto uc = static_cast<std::uint64_t>(c);
        const std::uint64_t r = op == 0 ? x + y : op == 1 ? x - y : op == 2 ? x * y : op == 3 ? x + uc : op == 4 ? uc - x : x * uc;
        expect[i] = static_cast<std::int64_t>(r);
      }
      for (bool simd : {true, false}) {
        const Kernels& k = kernels(simd);
        std::vector<std::int64_t>& out = simd ? fast : slow;
        switch (op) {
          case 0:
            k.add_i64(a.data(), b.data(), out.data(), n);
            break;
          case 1:
            k.sub_i64(a.data(), b.data(), out.data(), n);
            break;
          case 2:
            k.mul_i64(a.data(), b.data(), out.data(), n);
            break;
          case 3:
            k.add_const_i64(a.data(), c, out.data(), n);
            break;
          case 4:
            k.rsub_const_i64(a.data(), c, out.data(), n);
            break;
          default:
            k.mul_const_i64(a.data(), c, out.data(), n);
        }
      }
      CHECK(fast == expect);
      CHECK(slow == expect);
    }
  }
}

TEST(flip_swaps_operand_order) {
  CHECK(morsel::flip(CmpOp::Lt) == CmpOp::Gt);
  CHECK(morsel::flip(CmpOp::Ge) == CmpOp::Le);
  CHECK(morsel::flip(CmpOp::Eq) == CmpOp::Eq);
  CHECK(morsel::flip(CmpOp::Ne) == CmpOp::Ne);
}
