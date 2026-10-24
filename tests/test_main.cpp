#include <cstdio>
#include <cstring>

#include "check.h"
#include "morsel/kernels.h"

// Usage: morsel_tests [substring]  runs only tests whose name contains it.
int main(int argc, char** argv) {
  const char* filter = argc > 1 ? argv[1] : nullptr;
  std::printf("morsel tests (kernels: %s)\n", morsel::simd_name());
  int run = 0;
  int failed_tests = 0;
  for (const auto& test : morsel_test::registry()) {
    if (filter && !std::strstr(test.name, filter)) continue;
    const int before = morsel_test::failures();
    test.fn();
    ++run;
    const bool ok = morsel_test::failures() == before;
    if (!ok) ++failed_tests;
    std::printf("[%s] %s\n", ok ? " OK " : "FAIL", test.name);
  }
  std::printf("%d tests, %d failed\n", run, failed_tests);
  return failed_tests == 0 ? 0 : 1;
}
