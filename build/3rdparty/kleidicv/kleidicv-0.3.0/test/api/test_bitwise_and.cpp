// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <type_traits>

#include "framework/operation.h"
#include "kleidicv/kleidicv.h"
#include "test_config.h"

KLEIDICV_API(bitwise_and, kleidicv_bitwise_and, uint8_t)

template <typename ElementType>
class BitwiseAndTest final : public BinaryOperationTest<ElementType> {
  // Expose constructor of base class.
  using BinaryOperationTest<ElementType>::BinaryOperationTest;

 protected:
  using Elements = typename BinaryOperationTest<ElementType>::Elements;
  using BinaryOperationTest<ElementType>::max;

  // Calls the API-under-test in the appropriate way.
  kleidicv_error_t call_api() override {
    return bitwise_and<ElementType>()(
        this->inputs_[0].data(), this->inputs_[0].stride(),
        this->inputs_[1].data(), this->inputs_[1].stride(),
        this->actual_[0].data(), this->actual_[0].stride(), this->width(),
        this->height());
  }

  const std::vector<Elements>& test_elements() override {
    static const std::vector<Elements> kTestElements = {
        // clang-format off
        {    0,         0,     0},
        {    1,         1,     1},
        {    2,         1,     0},
        {    6,        12,     4},
        {    0,     max(),     0},
        {max(),         0,     0},
        {    2,     max(),     2},
        {max(),     max(), max()},
        {max(), max() - 1, max() - 1},
        // clang-format on
    };

    return kTestElements;
  }
};  // end of class BitwiseAndTest<ElementType>

template <typename ElementType>
class BitwiseAnd : public testing::Test {};

using ElementTypes = ::testing::Types<uint8_t>;
TYPED_TEST_SUITE(BitwiseAnd, ElementTypes);

// Tests kleidicv_bitwise_and API.
TYPED_TEST(BitwiseAnd, API) {
  // Test without padding.
  BitwiseAndTest<TypeParam>{}.test();
  // Test with padding.
  BitwiseAndTest<TypeParam>{}
      .with_padding(test::Options::vector_length())
      .test();

  TypeParam src[1], dst[1];
  test::test_null_args(bitwise_and<TypeParam>(), src, sizeof(TypeParam), src,
                       sizeof(TypeParam), dst, sizeof(TypeParam), 1, 1);
}

TYPED_TEST(BitwiseAnd, Misalignment) {
  if (sizeof(TypeParam) == 1) {
    // misalignment impossible
    return;
  }
  TypeParam src[2] = {}, dst[2];
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            bitwise_and<TypeParam>()(src, sizeof(TypeParam) + 1, src,
                                     sizeof(TypeParam), dst, sizeof(TypeParam),
                                     1, 2));
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            bitwise_and<TypeParam>()(src, sizeof(TypeParam), src,
                                     sizeof(TypeParam) + 1, dst,
                                     sizeof(TypeParam), 1, 2));
  EXPECT_EQ(
      KLEIDICV_ERROR_ALIGNMENT,
      bitwise_and<TypeParam>()(src, sizeof(TypeParam), src, sizeof(TypeParam),
                               dst, sizeof(TypeParam) + 1, 1, 2));
}

TYPED_TEST(BitwiseAnd, ZeroImageSize) {
  TypeParam src[1] = {}, dst[1];
  EXPECT_EQ(KLEIDICV_OK, bitwise_and<TypeParam>()(src, sizeof(TypeParam), src,
                                                  sizeof(TypeParam), dst,
                                                  sizeof(TypeParam), 0, 1));
  EXPECT_EQ(KLEIDICV_OK, bitwise_and<TypeParam>()(src, sizeof(TypeParam), src,
                                                  sizeof(TypeParam), dst,
                                                  sizeof(TypeParam), 1, 0));
}

TYPED_TEST(BitwiseAnd, OversizeImage) {
  TypeParam src[1] = {}, dst[1];
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            bitwise_and<TypeParam>()(src, sizeof(TypeParam), src,
                                     sizeof(TypeParam), dst, sizeof(TypeParam),
                                     KLEIDICV_MAX_IMAGE_PIXELS + 1, 1));
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            bitwise_and<TypeParam>()(src, sizeof(TypeParam), src,
                                     sizeof(TypeParam), dst, sizeof(TypeParam),
                                     KLEIDICV_MAX_IMAGE_PIXELS,
                                     KLEIDICV_MAX_IMAGE_PIXELS));
}
