cmake_minimum_required(VERSION 3.18)
project(morsel VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(MORSEL_BUILD_TESTS "Build the C++ tests" ON)
option(MORSEL_SANITIZE "Build with AddressSanitizer and UBSan" OFF)
option(MORSEL_TSAN "Build with ThreadSanitizer" OFF)

find_package(Threads REQUIRED)

# Built for baseline x86-64 on purpose (no -march=native): the AVX2 kernels
# are compiled per function and chosen at runtime.
add_library(morsel STATIC
  src/types.cpp
  src/storage.cpp
  src/kernels.cpp
  src/expr.cpp
  src/plan.cpp
  src/thread_pool.cpp
  src/executor.cpp
  src/join.cpp
  src/aggregate.cpp
  src/tpch.cpp)
target_include_directories(morsel PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(morsel PUBLIC Threads::Threads)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(morsel PRIVATE -Wall -Wextra -Wshadow)
  if(MORSEL_SANITIZE)
    target_compile_options(morsel PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(morsel PUBLIC -fsanitize=address,undefined)
  endif()
  if(MORSEL_TSAN)
    target_compile_options(morsel PUBLIC -fsanitize=thread)
    target_link_options(morsel PUBLIC -fsanitize=thread)
  endif()
endif()

add_executable(morsel_cli tools/morsel_cli.cpp)
set_target_properties(morsel_cli PROPERTIES OUTPUT_NAME morsel)
target_link_libraries(morsel_cli PRIVATE morsel)

if(MORSEL_BUILD_TESTS)
  enable_testing()
  add_executable(morsel_tests
    tests/test_main.cpp
    tests/test_kernels.cpp
    tests/test_expr.cpp
    tests/test_operators.cpp)
  target_link_libraries(morsel_tests PRIVATE morsel)
  add_test(NAME morsel_tests COMMAND morsel_tests)
endif()
