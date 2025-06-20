// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_TEST_FRAMEWORK_UTILS_H_
#define KLEIDICV_TEST_FRAMEWORK_UTILS_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <tuple>
#include <type_traits>

#include "framework/abstract.h"
#include "framework/types.h"
#include "kleidicv/ctypes.h"

#define KLEIDICV_API(name, impl, type)                                        \
  template <typename ElementType,                                             \
            std::enable_if_t<std::is_same_v<ElementType, type>, bool> = true> \
  static decltype(auto) name() {                                              \
    return (impl);                                                            \
  }

#define KLEIDICV_API_DIFFERENT_IO_TYPES(name, impl, itype, otype)             \
  template <typename InputType, typename OutputType,                          \
            std::enable_if_t<std::is_same_v<InputType, itype>, bool> = true,  \
            std::enable_if_t<std::is_same_v<OutputType, otype>, bool> = true> \
  static decltype(auto) name() {                                              \
    return impl;                                                              \
  }

// Generates a fatal failure with a generic message, and returns with a given
// value.
#define TEST_FAIL_WITH(return_value, message)                          \
  do {                                                                 \
    GTEST_MESSAGE_("Failed", ::testing::TestPartResult::kFatalFailure) \
        << message;                                                    \
    return (return_value);                                             \
  } while (0 != 0)

// Returns if the test has any failures.
#define ASSERT_NO_FAILURES()           \
  if (::testing::Test::HasFailure()) { \
    return;                            \
  }

class MockMallocToFail {
 public:
  static void enable() { enabled = true; }
  static void disable() { enabled = false; }
  static bool is_enabled() { return enabled; }

 private:
  static bool enabled;
};

namespace test {

class Options {
 public:
  // Returns the vector length being tested. This is in bytes.
  static size_t vector_length() { return vector_length_; }

  // Returns seed to use.
  static uint64_t seed() { return seed_; }

  // Whether long running tests should be skipped.
  static bool are_long_running_tests_skipped() {
    return are_long_running_tests_skipped_;
  }

  // Returns the number of lanes in a vector for a given arithmetic type.
  template <typename ElementType,
            std::enable_if_t<std::is_arithmetic_v<ElementType>, bool> = true>
  static size_t vector_lanes() {
    return vector_length_ / sizeof(ElementType);
  }

  // Sets the vector length in bytes.
  static void set_vector_length(size_t value) {
    // Check for power of two.
    if ((value == 0) || ((value & (value - 1)) != 0)) {
      std::cerr << "Vector length must be a power of two: " << value
                << std::endl;
      std::exit(1);
    }

    vector_length_ = value;
  }

  // Sets the seed.
  static void set_seed(uint64_t value) { seed_ = value; }

  // Turns on long running tests.
  static void turn_on_long_running_tests() {
    are_long_running_tests_skipped_ = false;
  }

 private:
  // Vector length being tested.
  static size_t vector_length_;
  // Seed to use.
  static uint64_t seed_;
  // Whether long running tests should be skipped.
  static bool are_long_running_tests_skipped_;
};  // end of class Options

// Prints all the elements in a two-dimensional space.
template <typename ElementType>
void dump(const TwoDimensional<ElementType> *elements);

// Returns default border values.
const std::array<kleidicv_border_values_t, 1> &default_border_values();

// Returns an array of just a few small layouts.
std::array<test::ArrayLayout, 7> small_array_layouts(size_t min_width,
                                                     size_t min_height);
// Returns an array of default tested layouts.
std::array<test::ArrayLayout, 14> default_array_layouts(size_t min_width,
                                                        size_t min_height);

std::array<test::ArrayLayout, 6> default_1channel_array_layouts(
    size_t min_width, size_t min_height);

namespace internal {
template <typename Function, typename Tuple>
class NullPointerTester {
  // Set the given argument to null and test that the function diagnoses the
  // error correctly.
  template <typename ArgType, size_t ArgIndex>
  static typename std::enable_if<std::is_pointer_v<ArgType>>::type
  test_with_null_arg(Function f, Tuple t) {
    std::get<ArgIndex>(t) = nullptr;
    EXPECT_EQ(KLEIDICV_ERROR_NULL_POINTER, std::apply(f, t));
  }

  // Skip arguments that aren't pointers.
  template <typename ArgType, size_t ArgIndex>
  static typename std::enable_if<!std::is_pointer_v<ArgType>>::type
  test_with_null_arg(Function, const Tuple &) {}

 public:
  template <int ArgIndex>
  static void test(Function f, const Tuple &t) {
    // Recurse to test earlier arguments first
    if constexpr (ArgIndex > 0) {
      test<ArgIndex - 1>(f, t);
    }
    using ArgType = typename std::tuple_element_t<ArgIndex, Tuple>;
    test_with_null_arg<ArgType, ArgIndex>(f, t);
  }
};

template <typename Func>
class ParamsExtractor;
template <typename Ret, typename... Params>
class ParamsExtractor<Ret (*)(Params...)> {
 public:
  using Tuple = std::tuple<Params...>;
};
}  // namespace internal

// Tests that the function returns KLEIDICV_ERROR_NULL_POINTER if any of its
// pointer arguments are null.
template <typename Function, typename... Args>
void test_null_args(Function f, Args... args) {
  // Ensure that the function parameter types are used otherwise arguments may
  // not be recognised as pointers.
  using Tuple = typename internal::ParamsExtractor<Function>::Tuple;
  constexpr int LastArgIndex = std::tuple_size_v<Tuple> - 1;
  using Tester = internal::NullPointerTester<Function, Tuple>;
  Tester::template test<LastArgIndex>(f, Tuple(args...));
}

template <typename T>
T saturating_add(T a, T b) {
  T result;
  if (__builtin_add_overflow(a, b, &result)) {
    if constexpr (std::is_unsigned_v<T>) {
      result = std::numeric_limits<T>::max();
    } else {
      result = b < 0 ? std::numeric_limits<T>::lowest()
                     : std::numeric_limits<T>::max();
    }
  }

  return result;
}

template <typename T>
T saturating_mul(T a, T b) {
  T result;
  if (__builtin_mul_overflow(a, b, &result)) {
    if constexpr (std::is_unsigned_v<T>) {
      result = std::numeric_limits<T>::max();
    } else {
      if ((a < 0 && b < 0) || (a > 0 && b > 0)) {
        result = std::numeric_limits<T>::max();
      } else {
        result = std::numeric_limits<T>::lowest();
      }
    }
  }

  return result;
}

}  // namespace test

#endif  // KLEIDICV_TEST_FRAMEWORK_UTILS_H_
