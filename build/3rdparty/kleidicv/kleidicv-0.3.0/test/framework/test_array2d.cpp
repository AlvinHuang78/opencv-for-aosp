// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <array>
#include <utility>

#include "framework/array.h"
#include "framework/generator.h"
#include "framework/utils.h"

// Tests that the default constructor of test::Array2D<T> default constructor
// always creates an empty object.
TEST(Array2D, DefaultConstructor) {
  test::Array2D<uint32_t> array;
  EXPECT_EQ(array.width(), 0);
  EXPECT_EQ(array.height(), 0);
  EXPECT_EQ(array.channels(), 0);
  EXPECT_EQ(array.stride(), 0);
  EXPECT_FALSE(array.valid());
}

// Tests that test::Array2D<T> constructor always creates an object with the
// same contents.
TEST(Array2D, Constructor) {
  size_t width = 5, height = 5;
  test::Array2D<uint32_t> array_1{width, height};
  array_1.fill(0);
  EXPECT_EQ(array_1.width(), width);
  EXPECT_EQ(array_1.height(), height);
  EXPECT_EQ(array_1.channels(), 1);
  EXPECT_EQ(array_1.stride(), width * sizeof(uint32_t));
  EXPECT_TRUE(array_1.valid());

  test::Array2D<uint32_t> array_2{width, height};
  array_2.fill(0);
  EXPECT_EQ_ARRAY2D(array_1, array_2);
}

// Tests that test::Array2D<T> is constructible with test::ArrayLayout.
TEST(Array2D, ConstructFromArrayLayout) {
  using ElementType = uint32_t;

  size_t width = 1, height = 2, padding = 3, channels = 4;
  test::ArrayLayout layout{width, height, padding, channels};
  test::Array2D<ElementType> array{layout};
  EXPECT_EQ(array.width(), width);
  EXPECT_EQ(array.height(), height);
  EXPECT_EQ(array.channels(), channels);
  EXPECT_EQ(array.stride(), (width + padding) * sizeof(ElementType));
  EXPECT_TRUE(array.valid());
}

// Tests that the copy assignment operator of test::Array2D<T> works.
TEST(Array2D, CopyAssignment) {
  size_t width = 5, height = 5;
  test::Array2D<uint32_t> array_1{width, height};

  test::Array2D<uint32_t> array_2 = array_1;
  EXPECT_EQ_ARRAY2D(array_1, array_2);
  EXPECT_NE(array_1.data(), array_2.data());
}

// Tests that the move assignment operator of test::Array2D<T> works.
TEST(Array2D, MoveAssignment) {
  size_t width = 5, height = 5, channels = 3;
  test::Array2D<uint32_t> array_1{width, height, 0, channels};
  test::Array2D<uint32_t> array_2;

  array_2 = std::move(array_1);
  EXPECT_EQ(array_2.width(), width);
  EXPECT_EQ(array_2.height(), height);
  EXPECT_EQ(array_2.channels(), channels);
  EXPECT_EQ(array_2.stride(), width * sizeof(uint32_t));
  EXPECT_TRUE(array_2.valid());

  // This test is specifically to find out what happens to an object after it is
  // moved so disable this static analysis check.
  // NOLINTBEGIN(clang-analyzer-cplusplus.Move, bugprone-use-after-move,
  // hicpp-invalid-access-moved)
  EXPECT_EQ(array_1.width(), 0);
  EXPECT_EQ(array_1.height(), 0);
  EXPECT_EQ(array_1.channels(), 0);
  EXPECT_EQ(array_1.stride(), 0);
  EXPECT_FALSE(array_1.valid());
  // NOLINTEND(clang-analyzer-cplusplus.Move, bugprone-use-after-move,
  // hicpp-invalid-access-moved)
}

// Tests that test::Array2D<T>.at() works for set/get.
TEST(Array2D, At) {
  size_t width = 1, height = 1;
  test::Array2D<uint32_t> array{width, height};

  array.at(0, 0)[0] = 1;
  EXPECT_EQ(array.at(0, 0)[0], 1);

  array.at(0, 0)[0] = 2;
  EXPECT_EQ(array.at(0, 0)[0], 2);

  const test::Array2D<uint32_t> const_array{width, height};
  EXPECT_EQ(const_array.at(0, 0)[0], 0);
}

// Tests that test::Array2D<T>.set() works.
TEST(Array2D, Set) {
  size_t width = 5, height = 2;
  test::Array2D<uint32_t> array_1{width, height};

  array_1.set(0, 0, {1, 2, 3, 4, 5});
  EXPECT_EQ(array_1.at(0, 0)[0], 1);
  EXPECT_EQ(array_1.at(0, 1)[0], 2);
  EXPECT_EQ(array_1.at(0, 2)[0], 3);
  EXPECT_EQ(array_1.at(0, 3)[0], 4);
  EXPECT_EQ(array_1.at(0, 4)[0], 5);

  array_1.set(0, 4, {42});
  EXPECT_EQ(array_1.at(0, 4)[0], 42);

  array_1.set(1, 0, {11});
  EXPECT_EQ(array_1.at(1, 0)[0], 11);
}

// Tests that test::Array2D<T>.set() works for TwoDimensional instances.
TEST(Array2D, SetWithTwoDimensional) {
  using ElementType = uint32_t;

  // array_1:
  //  0 1 2
  //  3 4 5
  //  6 7 8
  test::Array2D<ElementType> array_1{3, 3};
  array_1.set(0, 0, {0, 1, 2});
  array_1.set(1, 0, {3, 4, 5});
  array_1.set(2, 0, {6, 7, 8});

  // array_2:
  //  9 9 9 9 9
  //  9 9 9 9 9
  //  9 9 9 9 9
  //  9 9 9 9 9
  //  9 9 9 9 9
  test::Array2D<ElementType> array_2{5, 5};
  array_2.fill(9);

  // Expected array_2:
  //  9 9 9 9 9
  //  9 9 0 1 2
  //  9 9 3 4 5
  //  9 9 6 7 8
  //  9 9 9 9 9
  array_2.set(1, 2, &array_1);

  EXPECT_EQ(array_2.at(0, 0)[0], 9);
  EXPECT_EQ(array_2.at(0, 1)[0], 9);
  EXPECT_EQ(array_2.at(0, 2)[0], 9);
  EXPECT_EQ(array_2.at(0, 3)[0], 9);
  EXPECT_EQ(array_2.at(0, 4)[0], 9);

  EXPECT_EQ(array_2.at(1, 0)[0], 9);
  EXPECT_EQ(array_2.at(1, 1)[0], 9);
  EXPECT_EQ(array_2.at(1, 2)[0], 0);
  EXPECT_EQ(array_2.at(1, 3)[0], 1);
  EXPECT_EQ(array_2.at(1, 4)[0], 2);

  EXPECT_EQ(array_2.at(2, 0)[0], 9);
  EXPECT_EQ(array_2.at(2, 1)[0], 9);
  EXPECT_EQ(array_2.at(2, 2)[0], 3);
  EXPECT_EQ(array_2.at(2, 3)[0], 4);
  EXPECT_EQ(array_2.at(2, 4)[0], 5);

  EXPECT_EQ(array_2.at(3, 0)[0], 9);
  EXPECT_EQ(array_2.at(3, 1)[0], 9);
  EXPECT_EQ(array_2.at(3, 2)[0], 6);
  EXPECT_EQ(array_2.at(3, 3)[0], 7);
  EXPECT_EQ(array_2.at(3, 4)[0], 8);

  EXPECT_EQ(array_2.at(4, 0)[0], 9);
  EXPECT_EQ(array_2.at(4, 1)[0], 9);
  EXPECT_EQ(array_2.at(4, 2)[0], 9);
  EXPECT_EQ(array_2.at(4, 3)[0], 9);
  EXPECT_EQ(array_2.at(4, 4)[0], 9);
}

// Tests that test::Array2D<T>.fill() works.
TEST(Array2D, Fill) {
  size_t width = 5, height = 2;
  test::Array2D<uint32_t> array_1{width, height};
  test::Array2D<uint32_t> array_2{width, height};

  array_1.fill(uint32_t{0});
  for (size_t row = 0; row < height; ++row) {
    for (size_t column = 0; column < width; ++column) {
      EXPECT_EQ(array_1.at(row, column)[0], 0);
    }
  }

  array_1.fill(42);
  for (size_t row = 0; row < height; ++row) {
    for (size_t column = 0; column < width; ++column) {
      EXPECT_EQ(array_1.at(row, column)[0], 42);
    }
  }
}

// Tests that test::Array2D<T>.fill(Generator<T> *) works.
TEST(Array2D, FillWithGenerator) {
  using ElementType = uint32_t;

  const size_t width = 3, height = 2;
  test::Array2D<ElementType> array{width, height};

  const size_t num_elements = width * height;
  std::array<ElementType, num_elements> elements = {11, 12, 13, 14, 15, 16};
  test::SequenceGenerator generator{elements};
  array.fill(generator);
  EXPECT_EQ(array.at(0, 0)[0], 11);
  EXPECT_EQ(array.at(0, 1)[0], 12);
  EXPECT_EQ(array.at(0, 2)[0], 13);
  EXPECT_EQ(array.at(1, 0)[0], 14);
  EXPECT_EQ(array.at(1, 1)[0], 15);
  EXPECT_EQ(array.at(1, 2)[0], 16);
}

// Tests that EXPECT_EQ_ARRAY2D() macro works for fully-equal objects.
TEST(Array2D, ExpectEq_Equal) {
  size_t width = 5, height = 2;
  test::Array2D<uint32_t> array_1{width, height};
  test::Array2D<uint32_t> array_2{width, height};
  array_1.fill(0);
  array_2.fill(0);
  EXPECT_EQ_ARRAY2D(array_1, array_2);
}

// Tests that EXPECT_EQ_ARRAY2D() macro works for equal objects where stride is
// different.
TEST(Array2D, ExpectEq_Equal_StrideInvariant) {
  size_t width = 5, height = 2;
  test::Array2D<uint32_t> array_1{width, height};
  test::Array2D<uint32_t> array_2{width, height, 1};
  array_1.fill(0);
  array_2.fill(0);
  EXPECT_EQ_ARRAY2D(array_1, array_2);
}

// Tests that EXPECT_EQ_ARRAY2D() macro works for non-equal objects where width
// is different.
TEST(Array2D, ExpectEq_NotEqual_Width) {
  struct Test {
    static void test() {
      size_t width = 5, height = 2;
      test::Array2D<uint32_t> array_1{width, height};
      test::Array2D<uint32_t> array_2{width + 1, height};
      EXPECT_EQ_ARRAY2D(array_1, array_2);
    }
  };

  EXPECT_FATAL_FAILURE(Test::test(), "Mismatch in width.");
}

// Tests that EXPECT_EQ_ARRAY2D() macro works for non-equal objects where
// height is different.
TEST(Array2D, ExpectEq_NotEqual_Height) {
  struct Test {
    static void test() {
      size_t width = 5, height = 2;
      test::Array2D<uint32_t> array_1{width, height};
      test::Array2D<uint32_t> array_2{width, height + 1};
      EXPECT_EQ_ARRAY2D(array_1, array_2);
    }
  };

  EXPECT_FATAL_FAILURE(Test::test(), "Mismatch in height.");
}

// Tests that EXPECT_EQ_ARRAY2D() macro works for non-equal objects where
// channels is different.
TEST(Array2D, ExpectEq_NotEqual_Channels) {
  struct Test {
    static void test() {
      size_t width = 5, height = 2;
      test::Array2D<uint32_t> array_1{width, height, 0, 1};
      test::Array2D<uint32_t> array_2{width, height, 0, 2};
      EXPECT_EQ_ARRAY2D(array_1, array_2);
    }
  };

  EXPECT_FATAL_FAILURE(Test::test(), "Mismatch in channels.");
}

// Tests that EXPECT_EQ_ARRAY2D() macro works for non-equal objects where data
// is different.
TEST(Array2D, ExpectEq_NotEqual_Data) {
  struct Test {
    static void test() {
      size_t width = 5, height = 2;
      test::Array2D<uint32_t> array_1{width, height};
      test::Array2D<uint32_t> array_2{width, height};
      array_1.fill(0);
      array_2.fill(0);
      array_2.at(0, 0)[0] = 42;
      EXPECT_EQ_ARRAY2D(array_1, array_2);
    }
  };

  EXPECT_FATAL_FAILURE(Test::test(), "Mismatch at (row=0, col=0): 0 vs 0x2a.");
}

// Tests that EXPECT_NE_ARRAY2D() macro works for equal objects.
TEST(Array2D, ExpectNe_Equal) {
  struct Test {
    static void test() {
      size_t width = 5, height = 2;
      test::Array2D<uint32_t> array_1{width, height};
      test::Array2D<uint32_t> array_2{width, height};
      array_1.fill(0);
      array_2.fill(0);
      EXPECT_NE_ARRAY2D(array_1, array_2);
    }
  };

  EXPECT_FATAL_FAILURE(Test::test(),
                       "Objects are equal, but expected to differ.");
}

// Tests that EXPECT_NE_ARRAY2D() macro works for non-equal objects where width
// is different.
TEST(Array2D, ExpectNe_NotEqual_Width) {
  struct Test {
    static void test() {
      size_t width = 5, height = 2;
      test::Array2D<uint32_t> array_1{width, height};
      test::Array2D<uint32_t> array_2{width + 1, height};
      EXPECT_NE_ARRAY2D(array_1, array_2);
    }
  };

  EXPECT_FATAL_FAILURE(Test::test(), "Mismatch in width.");
}

// Tests that EXPECT_NE_ARRAY2D() macro works for non-equal objects where
// height is different.
TEST(Array2D, ExpectNe_NotEqual_Height) {
  struct Test {
    static void test() {
      size_t width = 5, height = 2;
      test::Array2D<uint32_t> array_1{width, height};
      test::Array2D<uint32_t> array_2{width, height + 1};
      EXPECT_NE_ARRAY2D(array_1, array_2);
    }
  };

  EXPECT_FATAL_FAILURE(Test::test(), "Mismatch in height.");
}

// Tests that EXPECT_NE_ARRAY2D() macro works for non-equal objects where
// channels is different.
TEST(Array2D, ExpectNe_NotEqual_Channels) {
  struct Test {
    static void test() {
      size_t width = 5, height = 2;
      test::Array2D<uint32_t> array_1{width, height, 0, 1};
      test::Array2D<uint32_t> array_2{width, height, 0, 2};
      EXPECT_NE_ARRAY2D(array_1, array_2);
    }
  };

  EXPECT_FATAL_FAILURE(Test::test(), "Mismatch in channels.");
}

// Tests that EXPECT_NE_ARRAY2D() macro works for non-equal objects where there
// is a difference in data.
TEST(Array2D, ExpectNe_NotEqual_Data) {
  size_t width = 5, height = 2;
  test::Array2D<uint32_t> array_1{width, height};
  test::Array2D<uint32_t> array_2{width, height};
  array_2.at(1, 0)[0] = 42;
  EXPECT_NE_ARRAY2D(array_1, array_2);
}

static void PaddingClobbered(size_t row, size_t offset) {
  using ElementType = uint32_t;

  size_t width = 5, height = 2, padding = 10;
  size_t stride = (width + padding) * sizeof(ElementType);
  test::Array2D<ElementType> array(width, height, padding);

  uint8_t* ptr = reinterpret_cast<uint8_t*>(array.data());
  ptr[row * stride + width * sizeof(ElementType) + offset] = 1;
}

// Tests that clobbering the first padding byte in the first row is caught.
TEST(Array2D, PaddingClobbered_FirstRow_FirstByte) {
  EXPECT_FATAL_FAILURE(PaddingClobbered(0, 0),
                       "Padding byte was overwritten at (row=0, offset=20)");
}

// Tests that clobbering the last padding byte in the first row is caught.
TEST(Array2D, PaddingClobbered_FirstRow_LastByte) {
  EXPECT_FATAL_FAILURE(PaddingClobbered(0, 9),
                       "Padding byte was overwritten at (row=0, offset=29)");
}

// Tests that clobbering the first padding byte in the last row is caught.
TEST(Array2D, PaddingClobbered_LastRow_FirstByte) {
  EXPECT_FATAL_FAILURE(PaddingClobbered(1, 0),
                       "Padding byte was overwritten at (row=1, offset=20)");
}

// Tests that clobbering the last padding byte in the last row is caught.
TEST(Array2D, PaddingClobbered_LastRow_LastByte) {
  EXPECT_FATAL_FAILURE(PaddingClobbered(1, 9),
                       "Padding byte was overwritten at (row=1, offset=29)");
}

static void Array2DCoverageBadAlloc() {
  size_t big = std::numeric_limits<size_t>::max() / sizeof(uint64_t);
  // Allow a little extra room for Array2D internals, to avoid overflow.
  big -= 0x1000;
  test::Array2D<uint64_t> array{big, 1};
  EXPECT_FALSE(array.valid());
}

static void Array2DCoverageSetInvalidArray() {
  test::Array2D<uint64_t> array;
  array.set(0, 0, {});
}

static void Array2DCoverageAtInvalidArray() {
  test::Array2D<uint64_t> array;
  array.at(0, 0);
}

// Additional tests for coverage purposes.
TEST(Array2D, Coverage) {
  EXPECT_FATAL_FAILURE(Array2DCoverageBadAlloc(),
                       "Failed to allocate memory of");
  EXPECT_FATAL_FAILURE(Array2DCoverageSetInvalidArray(), "Array is invalid.");
  EXPECT_FATAL_FAILURE(
      Array2DCoverageAtInvalidArray(),
      "Access is either out-of-bounds or the array is invalid.");
}
