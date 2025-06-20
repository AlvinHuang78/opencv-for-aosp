// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <limits>

#include "framework/array.h"
#include "framework/utils.h"
#include "kleidicv/kleidicv.h"
#include "test_config.h"

#define KLEIDICV_COUNT_NONZEROS(type, suffix) \
  KLEIDICV_API(count_nonzeros, kleidicv_count_nonzeros_##suffix, type)

KLEIDICV_COUNT_NONZEROS(uint8_t, u8);

template <typename ElementType>
class TestDataAllZeros {
 public:
  ElementType generateInput(size_t, size_t) { return 0; }
  size_t calculateExpected(size_t, size_t) { return 0; }
};

template <typename ElementType>
class TestDataSomeZeros {
 public:
  ElementType generateInput(size_t x, size_t y) {
    return x == y ? 0
                  : ((y % (std::numeric_limits<ElementType>::max() - 1)) + 1);
  }
  size_t calculateExpected(size_t width, size_t height) {
    return height * (width - 1);
  }
};

template <typename ElementType>
class TestDataAllNonZeros {
 public:
  ElementType generateInput(size_t x, size_t y) {
    return ((x + y) % std::numeric_limits<ElementType>::max()) + 1;
  }
  size_t calculateExpected(size_t width, size_t height) {
    return height * width;
  }
};

template <typename ElementType, template <class> class TestDataType>
class CountNonZerosTest {
 public:
  static void test(size_t src_padding) {
    // Add enough elements to test against overflow
    // to make sure block_finished() works as expected
    // That means more than std::max<ElementType> * vector_length() elements
    // Our generator (for some_zeros case) will add one 0 per line, so we will
    // need std::max<ElementType> + 2 elements
    const size_t width = (std::numeric_limits<ElementType>::max() + 2) *
                         test::Options::vector_length();
    const size_t height = test::Options::vector_length();

    test::Array2D<ElementType> source{width, height, src_padding};
    ASSERT_TRUE(source.valid());

    TestDataType<ElementType> data_;

    for (size_t i = 0; i < height; ++i) {
      for (size_t j = 0; j < width; ++j) {
        *source.at(i, j) = data_.generateInput(i, j);
      }
    }

    size_t expected = data_.calculateExpected(width, height);
    size_t actual = SIZE_MAX;
    EXPECT_EQ(KLEIDICV_OK,
              count_nonzeros<ElementType>()(source.data(), source.stride(),
                                            width, height, &actual));
    EXPECT_EQ(expected, actual);
  }
};

using ElementTypes = ::testing::Types<uint8_t>;

template <typename ElementType>
class CountNonZeros : public testing::Test {};

TYPED_TEST_SUITE(CountNonZeros, ElementTypes);

TYPED_TEST(CountNonZeros, AllZeros) {
  CountNonZerosTest<TypeParam, TestDataAllZeros>{}.test(0);
}

TYPED_TEST(CountNonZeros, AllZerosPadded) {
  CountNonZerosTest<TypeParam, TestDataAllZeros>{}.test(
      test::Options::vector_length());
}

TYPED_TEST(CountNonZeros, SomeZeros) {
  CountNonZerosTest<TypeParam, TestDataSomeZeros>{}.test(0);
}

TYPED_TEST(CountNonZeros, SomeZerosPadded) {
  CountNonZerosTest<TypeParam, TestDataSomeZeros>{}.test(
      test::Options::vector_length());
}

TYPED_TEST(CountNonZeros, AllNonZeros) {
  CountNonZerosTest<TypeParam, TestDataAllNonZeros>{}.test(0);
}

TYPED_TEST(CountNonZeros, AllNonZerosPadded) {
  CountNonZerosTest<TypeParam, TestDataAllNonZeros>{}.test(
      test::Options::vector_length());
}

TYPED_TEST(CountNonZeros, NullPointer) {
  TypeParam src[1];
  size_t count = 0;
  test::test_null_args(count_nonzeros<TypeParam>(), src, sizeof(TypeParam), 1,
                       1, &count);
}

TYPED_TEST(CountNonZeros, Misalignment) {
  if (sizeof(TypeParam) == 1) {
    // misalignment impossible
    return;
  }
  TypeParam src[2];
  size_t count = 0;
  EXPECT_EQ(
      KLEIDICV_ERROR_ALIGNMENT,
      count_nonzeros<TypeParam>()(src, sizeof(TypeParam) + 1, 1, 2, &count));
}

TYPED_TEST(CountNonZeros, ZeroImageSize) {
  TypeParam src[1] = {};
  size_t count = 123;
  EXPECT_EQ(KLEIDICV_OK,
            count_nonzeros<TypeParam>()(src, sizeof(TypeParam), 0, 1, &count));
  EXPECT_EQ(KLEIDICV_OK,
            count_nonzeros<TypeParam>()(src, sizeof(TypeParam), 1, 0, &count));
  EXPECT_EQ(0, count);
}

TYPED_TEST(CountNonZeros, OversizeImage) {
  TypeParam src[1] = {};
  size_t count = 0;
  EXPECT_EQ(
      KLEIDICV_ERROR_RANGE,
      count_nonzeros<TypeParam>()(src, sizeof(TypeParam),
                                  KLEIDICV_MAX_IMAGE_PIXELS + 1, 1, &count));
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            count_nonzeros<TypeParam>()(src, sizeof(TypeParam),
                                        KLEIDICV_MAX_IMAGE_PIXELS,
                                        KLEIDICV_MAX_IMAGE_PIXELS, &count));
}
