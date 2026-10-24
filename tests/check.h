#pragma once

// A minimal test harness: TEST(name) registers a function, CHECK records a
// failure without stopping the test. Keeps the project free of third-party
// dependencies.

#include <cstdio>
#include <functional>
#include <string>
#include <vector>

namespace morsel_test {

struct TestCase {
  const char* name;
  std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> tests;
  return tests;
}

inline int& failures() {
  static int count = 0;
  return count;
}

struct Registrar {
  Registrar(const char* name, std::function<void()> fn) { registry().push_back({name, std::move(fn)}); }
};

}  // namespace morsel_test

#define MORSEL_CONCAT_(a, b) a##b
#define MORSEL_CONCAT(a, b) MORSEL_CONCAT_(a, b)

#define TEST(name)                                                                     \
  static void name();                                                                  \
  static const morsel_test::Registrar MORSEL_CONCAT(registrar_, name)(#name, &name);   \
  static void name()

#define CHECK(cond)                                                                    \
  do {                                                                                 \
    if (!(cond)) {                                                                     \
      std::fprintf(stderr, "  %s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);  \
      ++morsel_test::failures();                                                       \
    }                                                                                  \
  } while (0)

#define CHECK_THROWS(expr, exception_type)                                             \
  do {                                                                                 \
    bool thrown_ = false;                                                              \
    try {                                                                              \
      expr;                                                                            \
    } catch (const exception_type&) {                                                  \
      thrown_ = true;                                                                  \
    }                                                                                  \
    if (!thrown_) {                                                                    \
      std::fprintf(stderr, "  %s:%d: expected %s from: %s\n", __FILE__, __LINE__,      \
                   #exception_type, #expr);                                            \
      ++morsel_test::failures();                                                       \
    }                                                                                  \
  } while (0)
