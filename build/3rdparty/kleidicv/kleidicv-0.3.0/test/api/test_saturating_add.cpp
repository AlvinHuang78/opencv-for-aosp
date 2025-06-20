// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <type_traits>

#include "framework/operation.h"
#include "kleidicv/kleidicv.h"
#include "test_config.h"

#define KLEIDICV_SATURATING_ADD(type, suffix) \
  KLEIDICV_API(saturating_add, kleidicv_saturating_add_##suffix, type)

KLEIDICV_SATURATING_ADD(int8_t, s8);
KLEIDICV_SATURATING_ADD(uint8_t, u8);
KLEIDICV_SATURATING_ADD(int16_t, s16);
KLEIDICV_SATURATING_ADD(uint16_t, u16);
KLEIDICV_SATURATING_ADD(int32_t, s32);
KLEIDICV_SATURATING_ADD(uint32_t, u32);
KLEIDICV_SATURATING_ADD(int64_t, s64);
KLEIDICV_SATURATING_ADD(uint64_t, u64);

template <typename ElementType>
class SaturatingAddTest final : public BinaryOperationTest<ElementType> {
  // Expose constructor of base class.
  using BinaryOperationTest<ElementType>::BinaryOperationTest;

 protected:
  using Elements = typename BinaryOperationTest<ElementType>::Elements;
  using BinaryOperationTest<ElementType>::lowest;
  using BinaryOperationTest<ElementType>::max;

  // Calls the API-under-test in the appropriate way.
  kleidicv_error_t call_api() override {
    return saturating_add<ElementType>()(
        this->inputs_[0].data(), this->inputs_[0].stride(),
        this->inputs_[1].data(), this->inputs_[1].stride(),
        this->actual_[0].data(), this->actual_[0].stride(), this->width(),
        this->height());
  }

  // Returns different test data for signed and unsigned element types.
  const std::vector<Elements>& test_elements() override {
    if constexpr (std::is_unsigned_v<ElementType>) {
      static const std::vector<Elements> kTestElements = {
          // clang-format off
          {        0,     0,     0},
          {        1,     1,     2},
          {max() - 1,     1, max()},
          {max() - 1,     2, max()},
          {    max(), max(), max()},
          // clang-format on
      };

      return kTestElements;
    } else {
      static const std::vector<Elements> kTestElements = {
          // clang-format off
          {    lowest(), lowest(), lowest()},
          {lowest() + 1,       -2, lowest()},
          {lowest() + 1,       -1, lowest()},
          {          -1,       -1,       -2},
          {           0,        0,        0},
          {           1,        1,        2},
          {   max() - 1,        1,    max()},
          {   max() - 1,        2,    max()},
          {       max(),    max(),    max()},
          // clang-format on
      };

      return kTestElements;
    }
  }
};  // end of class SaturatingAddTest<ElementType>

template <typename ElementType>
class SaturatingAdd : public testing::Test {};

using ElementTypes = ::testing::Types<int8_t, uint8_t, int16_t, uint16_t,
                                      int32_t, uint32_t, int64_t, uint64_t>;
TYPED_TEST_SUITE(SaturatingAdd, ElementTypes);

// Tests kleidicv_saturating_add_<type> API.
TYPED_TEST(SaturatingAdd, API) {
  // Test without padding.
  SaturatingAddTest<TypeParam>{}.test();
  // Test with padding.
  SaturatingAddTest<TypeParam>{}
      .with_padding(test::Options::vector_length())
      .test();

  TypeParam src[1], dst[1];
  test::test_null_args(saturating_add<TypeParam>(), src, sizeof(TypeParam), src,
                       sizeof(TypeParam), dst, sizeof(TypeParam), 1, 1);
}

// Tests various padding combinations.
TYPED_TEST(SaturatingAdd, Padding) {
  SaturatingAddTest<TypeParam>{}.with_paddings({0, 0}, {0}).test();
  SaturatingAddTest<TypeParam>{}.with_paddings({0, 0}, {1}).test();
  SaturatingAddTest<TypeParam>{}.with_paddings({0, 1}, {0}).test();
  SaturatingAddTest<TypeParam>{}.with_paddings({0, 1}, {1}).test();
  SaturatingAddTest<TypeParam>{}.with_paddings({1, 0}, {0}).test();
  SaturatingAddTest<TypeParam>{}.with_paddings({1, 0}, {1}).test();
  SaturatingAddTest<TypeParam>{}.with_paddings({1, 1}, {0}).test();
  SaturatingAddTest<TypeParam>{}.with_paddings({1, 1}, {1}).test();
}

TYPED_TEST(SaturatingAdd, Misalignment) {
  if (sizeof(TypeParam) == 1) {
    // misalignment impossible
    return;
  }
  TypeParam src[2] = {}, dst[2];
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            saturating_add<TypeParam>()(src, sizeof(TypeParam) + 1, src,
                                        sizeof(TypeParam), dst,
                                        sizeof(TypeParam), 1, 2));
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            saturating_add<TypeParam>()(src, sizeof(TypeParam), src,
                                        sizeof(TypeParam) + 1, dst,
                                        sizeof(TypeParam), 1, 2));
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            saturating_add<TypeParam>()(src, sizeof(TypeParam), src,
                                        sizeof(TypeParam), dst,
                                        sizeof(TypeParam) + 1, 1, 2));
}

TYPED_TEST(SaturatingAdd, ZeroImageSize) {
  TypeParam src[1] = {}, dst[1];
  EXPECT_EQ(KLEIDICV_OK, saturating_add<TypeParam>()(
                             src, sizeof(TypeParam), src, sizeof(TypeParam),
                             dst, sizeof(TypeParam), 0, 1));
  EXPECT_EQ(KLEIDICV_OK, saturating_add<TypeParam>()(
                             src, sizeof(TypeParam), src, sizeof(TypeParam),
                             dst, sizeof(TypeParam), 1, 0));
}

TYPED_TEST(SaturatingAdd, OversizeImage) {
  TypeParam src[1] = {}, dst[1];
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            saturating_add<TypeParam>()(
                src, sizeof(TypeParam), src, sizeof(TypeParam), dst,
                sizeof(TypeParam), KLEIDICV_MAX_IMAGE_PIXELS + 1, 1));
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            saturating_add<TypeParam>()(
                src, sizeof(TypeParam), src, sizeof(TypeParam), dst,
                sizeof(TypeParam), KLEIDICV_MAX_IMAGE_PIXELS,
                KLEIDICV_MAX_IMAGE_PIXELS));
}
