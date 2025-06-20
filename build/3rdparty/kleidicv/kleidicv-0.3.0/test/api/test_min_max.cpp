// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "framework/array.h"
#include "framework/utils.h"
#include "kleidicv/kleidicv.h"
#include "test_config.h"

#define KLEIDICV_MIN_MAX(type, suffix) \
  KLEIDICV_API(min_max, kleidicv_min_max_##suffix, type)

KLEIDICV_MIN_MAX(int8_t, s8);
KLEIDICV_MIN_MAX(uint8_t, u8);
KLEIDICV_MIN_MAX(int16_t, s16);
KLEIDICV_MIN_MAX(uint16_t, u16);
KLEIDICV_MIN_MAX(int32_t, s32);
KLEIDICV_MIN_MAX(float, f32);

#define KLEIDICV_MIN_MAX_LOC(type, suffix) \
  KLEIDICV_API(min_max_loc, kleidicv_min_max_loc_##suffix, type)

KLEIDICV_MIN_MAX_LOC(uint8_t, u8);

template <typename ElementType, size_t Padding>
class MinMaxTest {
  using ArrayType = test::Array2D<ElementType>;

 protected:
  // Returns the lowest value for ElementType.
  static constexpr ElementType lowest() {
    return std::numeric_limits<ElementType>::lowest();
  }

  // Returns the maximum value for ElementType.
  static constexpr ElementType max() {
    return std::numeric_limits<ElementType>::max();
  }

  // Returns the number of vector lanes.
  static size_t lanes() { return test::Options::vector_lanes<ElementType>(); }

  // We have 2 rows, source_row0 and source_row1
  static size_t height() { return 2; }

  // Tested number of elements in a row.
  static size_t width() {
    // Sufficient number of elements to exercise both vector and scalar paths.
    return 3 * lanes() - 1;
  }

  // Number of bytes in a row where the scalar path begins
  static size_t scalar_offset() { return 2 * test::Options::vector_length(); }

  // Total number of bytes per row, padding included
  static size_t stride() { return (width() + Padding) * sizeof(ElementType); }

  struct Elements {
    std::initializer_list<ElementType> source_row0_vector;
    std::initializer_list<ElementType> source_row0_scalar;
    std::initializer_list<ElementType> source_row1_vector;
    std::initializer_list<ElementType> source_row1_scalar;
    ElementType filler_value;
    ElementType expected_min;
    ElementType expected_max;
    size_t expected_min_offset;
    size_t expected_max_offset;
  };

  const std::array<Elements, 15>& test_elements() const {
    static const std::array<Elements, 15> elements = {{
        // clang-format off
      {
        {},  {},
        {},  {},
        10,
        10, 10,
        0, 0
      },
      {
        {},  {},
        {},  {},
        lowest(),
        lowest(), lowest(),
        0, 0
      },
      {
        {},  {},
        {},  {},
        0,
        0, 0,
        0, 0
      },
      {
        {},  {},
        {},  {},
        max(),
        max(), max(),
        0, 0
      },
      {
        {lowest()+10},    {},
        {max()}, {},
        lowest()+20,
        lowest()+10, max(),
        0, stride()
      },
      {
        {},    {lowest()},
        {},    {max() - 10},
        lowest()+20,
        lowest(), max()-10,
        scalar_offset(), stride() + scalar_offset()
      },
      {
        {9},  {},
        {},  {11},
        10,
        9, 11,
        0, stride() + scalar_offset()
      },
      {
        {},   {9},
        {11}, {},
        10,
        9, 11,
        scalar_offset(), stride()
      },
      {
        {11}, {},
        {},  {9},
        10,
        9, 11,
        stride() + scalar_offset(), 0
      },
      {
        {10, 9, 11},  {},
        {},           {},
        10,
        9, 11,
        1, 2
      },
      {
        {9, 10, 11},  {},
        {},           {},
        10,
        9, 11,
        0, 2
      },
      {
        { 3,  4,  5,  6},  {15, 16, 17},
        {20, 21, 22, 23},  {35,  2, 33},
        10,
        2, 35,
        stride() + scalar_offset() + sizeof(ElementType), stride() + scalar_offset()
      },
      {
        { 3,  4,  5,  6},  {15, 16, 17},
        {20,  2, 22, 23},  {35, 36, 33},
        10,
        2, 36,
        stride() + sizeof(ElementType), stride() + scalar_offset() + sizeof(ElementType)
      },
      {
        { 1,  2,  3,  4},  {15, 16, 42},
        { 1,  2,  3,  4},  {15, 16, 42},
        10,
        1, 42,
        0, (scalar_offset() + 2 * sizeof(ElementType))
      },
      {
        { 5,  6,  7,  8},  {1, 16, 42},
        { 1,  2,  3, 42},  {1, 16, 42},
        10,
        1, 42,
        scalar_offset(), (scalar_offset() + 2 * sizeof(ElementType))
      }
        // clang-format on
    }};
    return elements;
  }

  static void setup(ArrayType& source, const Elements& testData) {
    source.fill(testData.filler_value);
    source.set(0, 0, testData.source_row0_vector);
    source.set(1, 0, testData.source_row1_vector);
    source.set(0, scalar_offset() / sizeof(ElementType),
               testData.source_row0_scalar);
    source.set(1, scalar_offset() / sizeof(ElementType),
               testData.source_row1_scalar);
  }

  void one_test_call(const ArrayType& source, ElementType* p_min,
                     ElementType* p_max, ElementType expected_min,
                     ElementType expected_max) {
    if (p_min) {
      *p_min = std::numeric_limits<ElementType>::max();
    }
    if (p_max) {
      *p_max = std::numeric_limits<ElementType>::lowest();
    }
    EXPECT_EQ(KLEIDICV_OK,
              min_max<ElementType>()(source.data(), source.stride(), width(),
                                     height(), p_min, p_max));
    if (p_min) {
      EXPECT_EQ(*p_min, expected_min);
    }
    if (p_max) {
      EXPECT_EQ(*p_max, expected_max);
    }
  }

 public:
  void test() {
    for (auto& testData : test_elements()) {
      ArrayType source{width(), height(), Padding};
      ASSERT_TRUE(source.valid());

      setup(source, testData);

      ElementType actual_min = 2, actual_max = 1;
      EXPECT_EQ(KLEIDICV_ERROR_NULL_POINTER,
                min_max<ElementType>()(nullptr, source.stride(), width(),
                                       height(), &actual_min, &actual_max));
      EXPECT_EQ(2, actual_min);
      EXPECT_EQ(1, actual_max);

      one_test_call(source, nullptr, nullptr, 0, 0);
      one_test_call(source, &actual_min, nullptr, testData.expected_min, 0);
      one_test_call(source, nullptr, &actual_max, 0, testData.expected_max);
      one_test_call(source, &actual_min, &actual_max, testData.expected_min,
                    testData.expected_max);
    }
  }
};  // end of class MinMaxTest<ElementType>

template <typename ElementType, size_t Padding>
class MinMaxLocTest : public MinMaxTest<ElementType, Padding> {
  using ArrayType = test::Array2D<ElementType>;
  using Super = MinMaxTest<ElementType, Padding>;
  using Super::height;
  using Super::setup;
  using Super::test_elements;
  using Super::width;

  void one_test_call(const ArrayType& source, size_t* p_min_offset,
                     size_t* p_max_offset, size_t expected_min_offset,
                     size_t expected_max_offset) {
    if (p_min_offset) {
      *p_min_offset = std::numeric_limits<size_t>::max();
    }
    if (p_max_offset) {
      *p_max_offset = std::numeric_limits<size_t>::max();
    }
    EXPECT_EQ(KLEIDICV_OK, min_max_loc<ElementType>()(
                               source.data(), source.stride(), width(),
                               height(), p_min_offset, p_max_offset));
    if (p_min_offset) {
      EXPECT_EQ(*p_min_offset, expected_min_offset);
    }
    if (p_max_offset) {
      EXPECT_EQ(*p_max_offset, expected_max_offset);
    }
  }

 public:
  void test() {
    for (auto testData : test_elements()) {
      ArrayType source{width(), height(), Padding};
      ASSERT_TRUE(source.valid());

      setup(source, testData);

      size_t min_offset = 2, max_offset = 1;

      EXPECT_EQ(KLEIDICV_ERROR_NULL_POINTER,
                min_max_loc<ElementType>()(nullptr, source.stride(), width(),
                                           height(), &min_offset, &max_offset));
      EXPECT_EQ(2, min_offset);
      EXPECT_EQ(1, max_offset);

      one_test_call(source, nullptr, nullptr, 0, 0);
      one_test_call(source, &min_offset, nullptr, testData.expected_min_offset,
                    0);
      one_test_call(source, nullptr, &max_offset, 0,
                    testData.expected_max_offset);
      one_test_call(source, &min_offset, &max_offset,
                    testData.expected_min_offset, testData.expected_max_offset);
    }
  }
};  // end of class MinMaxLocTest<ElementType>

template <typename ElementType>
class MinMax : public testing::Test {};

using MinMaxElementTypes =
    ::testing::Types<int8_t, uint8_t, int16_t, uint16_t, int32_t, float>;
TYPED_TEST_SUITE(MinMax, MinMaxElementTypes);

TYPED_TEST(MinMax, API) {
  MinMaxTest<TypeParam, 0>{}.test();
  MinMaxTest<TypeParam, 100>{}.test();
}

TYPED_TEST(MinMax, Misalignment) {
  if (sizeof(TypeParam) == 1) {
    // misalignment impossible
    return;
  }
  TypeParam src[2] = {}, min_value, max_value;
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            min_max<TypeParam>()(src, sizeof(TypeParam) + 1, 1, 2, &min_value,
                                 &max_value));
}

TYPED_TEST(MinMax, ZeroImageSize) {
  TypeParam src[1] = {}, min_value, max_value;
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            min_max<TypeParam>()(src, sizeof(TypeParam), 0, 1, &min_value,
                                 &max_value));
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            min_max<TypeParam>()(src, sizeof(TypeParam), 1, 0, &min_value,
                                 &max_value));
}

TYPED_TEST(MinMax, OversizeImage) {
  TypeParam src[1] = {}, min_value, max_value;
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            min_max<TypeParam>()(src, sizeof(TypeParam),
                                 KLEIDICV_MAX_IMAGE_PIXELS + 1, 1, &min_value,
                                 &max_value));
  EXPECT_EQ(
      KLEIDICV_ERROR_RANGE,
      min_max<TypeParam>()(src, sizeof(TypeParam), KLEIDICV_MAX_IMAGE_PIXELS,
                           KLEIDICV_MAX_IMAGE_PIXELS, &min_value, &max_value));
}

template <typename ElementType>
class MinMaxLoc : public testing::Test {};

using MinMaxLocElementTypes = ::testing::Types<uint8_t>;
TYPED_TEST_SUITE(MinMaxLoc, MinMaxLocElementTypes);

TYPED_TEST(MinMaxLoc, API) {
  MinMaxLocTest<TypeParam, 0>{}.test();
  MinMaxLocTest<TypeParam, 100>{}.test();
}

TYPED_TEST(MinMaxLoc, ZeroImageSize) {
  TypeParam src[1] = {};
  size_t min_offset = 0, max_offset = 0;

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            min_max_loc<TypeParam>()(src, sizeof(TypeParam), 0, 1, &min_offset,
                                     &max_offset));
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            min_max_loc<TypeParam>()(src, sizeof(TypeParam), 1, 0, &min_offset,
                                     &max_offset));
}

TYPED_TEST(MinMaxLoc, OversizeImage) {
  TypeParam src[1] = {};
  size_t min_offset = 0, max_offset = 8;

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            min_max_loc<TypeParam>()(src, sizeof(TypeParam),
                                     KLEIDICV_MAX_IMAGE_PIXELS + 1, 1,
                                     &min_offset, &max_offset));
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            min_max_loc<TypeParam>()(
                src, sizeof(TypeParam), KLEIDICV_MAX_IMAGE_PIXELS,
                KLEIDICV_MAX_IMAGE_PIXELS, &min_offset, &max_offset));
}
