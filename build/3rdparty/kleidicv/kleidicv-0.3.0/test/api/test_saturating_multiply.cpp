// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <type_traits>

#include "framework/operation.h"
#include "kleidicv/kleidicv.h"
#include "test_config.h"

#define KLEIDICV_SATURATING_MULTIPLY(type, suffix) \
  KLEIDICV_API(saturating_multiply, kleidicv_saturating_multiply_##suffix, type)

KLEIDICV_SATURATING_MULTIPLY(uint8_t, u8);
KLEIDICV_SATURATING_MULTIPLY(int8_t, s8);
KLEIDICV_SATURATING_MULTIPLY(uint16_t, u16);
KLEIDICV_SATURATING_MULTIPLY(int16_t, s16);
KLEIDICV_SATURATING_MULTIPLY(int32_t, s32);

template <typename ElementType>
class SaturatingMultiplyTest final : public BinaryOperationTest<ElementType> {
  // Expose constructor of base class.
  using BinaryOperationTest<ElementType>::BinaryOperationTest;

 protected:
  using Elements = typename BinaryOperationTest<ElementType>::Elements;
  using BinaryOperationTest<ElementType>::lowest;
  using BinaryOperationTest<ElementType>::max;

  // Calls the API-under-test in the appropriate way.
  kleidicv_error_t call_api() override {
    return saturating_multiply<ElementType>()(
        this->inputs_[0].data(), this->inputs_[0].stride(),
        this->inputs_[1].data(), this->inputs_[1].stride(),
        this->actual_[0].data(), this->actual_[0].stride(), this->width(),
        this->height(), this->scale());
  }
  double scale() {
    return 1.0;  // This is the only parameter KleidiCV supports.
  }
  // Returns different test data for signed and unsigned element types.
  const std::vector<Elements>& test_elements() override {
    if constexpr (std::is_unsigned_v<ElementType>) {
      static const std::vector<Elements> kTestElements = {
          // clang-format off
          {        0,     0,     0},
          {        1,     1,     1},
          {        2,     2,     4},
          {        6,     8,    48},
          {    max(),     1, max()},
          {    max(),     2, max()},
          {    max(), max(), max()},
          // clang-format on
      };

      return kTestElements;
    } else {
      static const std::vector<Elements> kTestElements = {
          // clang-format off
          {    lowest(), lowest(),     max()},
          {    lowest(), max(),     lowest()},
          {    lowest(),    -1,     max()},
          {lowest() + 2,    -1, max() - 1},
          {        6,    -8,       -48},
          {       -2,     2,        -4},
          {       -1,    -1,         1},
          {        0,     0,         0},
          {        1,     1,         1},
          {        2,     2,         4},
          {        6,     8,        48},
          {max() - 2,    -1, lowest() + 3},
          {    max(),     1,     max()},
          {    max(),     2,     max()},
          {    max(), max(),     max()},
          // clang-format on
      };

      return kTestElements;
    }
  }
};  // end of class SaturatingMultiplyTest<ElementType>

template <typename ElementType>
class SaturatingMultiply : public testing::Test {};

using ElementTypes =
    ::testing::Types<uint8_t, int8_t, uint16_t, int16_t, int32_t>;
TYPED_TEST_SUITE(SaturatingMultiply, ElementTypes);

// Tests kleidicv_saturating_multiply_<type> API.
TYPED_TEST(SaturatingMultiply, API) {
  // Test without padding.
  SaturatingMultiplyTest<TypeParam>{}.test();
  // Test with padding.
  SaturatingMultiplyTest<TypeParam>{}
      .with_padding(test::Options::vector_length())
      .test();

  TypeParam src[1], dst[1];
  test::test_null_args(saturating_multiply<TypeParam>(), src, sizeof(TypeParam),
                       src, sizeof(TypeParam), dst, sizeof(TypeParam), 1, 1, 1);
}

TYPED_TEST(SaturatingMultiply, Misalignment) {
  if (sizeof(TypeParam) == 1) {
    // misalignment impossible
    return;
  }
  TypeParam src[2] = {}, dst[2];
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            saturating_multiply<TypeParam>()(src, sizeof(TypeParam) + 1, src,
                                             sizeof(TypeParam), dst,
                                             sizeof(TypeParam), 1, 2, 1));
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            saturating_multiply<TypeParam>()(src, sizeof(TypeParam), src,
                                             sizeof(TypeParam) + 1, dst,
                                             sizeof(TypeParam), 1, 2, 1));
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            saturating_multiply<TypeParam>()(src, sizeof(TypeParam), src,
                                             sizeof(TypeParam), dst,
                                             sizeof(TypeParam) + 1, 1, 2, 1));
}

TYPED_TEST(SaturatingMultiply, ZeroImageSize) {
  TypeParam src[1] = {}, dst[1];
  EXPECT_EQ(KLEIDICV_OK, saturating_multiply<TypeParam>()(
                             src, sizeof(TypeParam), src, sizeof(TypeParam),
                             dst, sizeof(TypeParam), 0, 1, 1));
  EXPECT_EQ(KLEIDICV_OK, saturating_multiply<TypeParam>()(
                             src, sizeof(TypeParam), src, sizeof(TypeParam),
                             dst, sizeof(TypeParam), 1, 0, 1));
}

TYPED_TEST(SaturatingMultiply, OversizeImage) {
  TypeParam src[1] = {}, dst[1];
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            saturating_multiply<TypeParam>()(
                src, sizeof(TypeParam), src, sizeof(TypeParam), dst,
                sizeof(TypeParam), KLEIDICV_MAX_IMAGE_PIXELS + 1, 1, 1));
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            saturating_multiply<TypeParam>()(
                src, sizeof(TypeParam), src, sizeof(TypeParam), dst,
                sizeof(TypeParam), KLEIDICV_MAX_IMAGE_PIXELS,
                KLEIDICV_MAX_IMAGE_PIXELS, 1));
}
