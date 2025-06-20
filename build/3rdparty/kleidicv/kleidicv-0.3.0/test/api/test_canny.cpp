// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "framework/utils.h"
#include "kleidicv/kleidicv.h"
#include "test_config.h"

#if KLEIDICV_EXPERIMENTAL_FEATURE_CANNY

#define KLEIDICV_CANNY(type, suffix) \
  KLEIDICV_API(canny, kleidicv_canny_##suffix, type)

KLEIDICV_CANNY(uint8_t, u8);

using ElementTypes = ::testing::Types<uint8_t>;

template <typename ElementType>
class CannyTest : public testing::Test {};

TYPED_TEST_SUITE(CannyTest, ElementTypes);

TYPED_TEST(CannyTest, NullPointer) {
  TypeParam src[1], dst[1];
  test::test_null_args(canny<TypeParam>(), src, sizeof(TypeParam), dst,
                       sizeof(TypeParam), 1, 1, 0.0, 1.0);
}

TYPED_TEST(CannyTest, Misalignment) {
  if (sizeof(TypeParam) == 1) {
    // misalignment impossible
    return;
  }
  TypeParam src[2], dst[2];
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            canny<TypeParam>()(src, sizeof(TypeParam) + 1, dst,
                               sizeof(TypeParam), 1, 2, 0.0, 1.0));
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            canny<TypeParam>()(src, sizeof(TypeParam), dst,
                               sizeof(TypeParam) + 1, 1, 2, 0.0, 1.0));
}

TYPED_TEST(CannyTest, ZeroImageSize) {
  TypeParam src[1] = {}, dst[1];
  EXPECT_EQ(KLEIDICV_OK, canny<TypeParam>()(src, sizeof(TypeParam), dst,
                                            sizeof(TypeParam), 0, 1, 0.0, 1.0));
  EXPECT_EQ(KLEIDICV_OK, canny<TypeParam>()(src, sizeof(TypeParam), dst,
                                            sizeof(TypeParam), 1, 0, 0.0, 1.0));
}

TYPED_TEST(CannyTest, OversizeImage) {
  TypeParam src[1], dst[1];
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            canny<TypeParam>()(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                               KLEIDICV_MAX_IMAGE_PIXELS + 1, 1, 0.0, 1.0));
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            canny<TypeParam>()(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                               KLEIDICV_MAX_IMAGE_PIXELS,
                               KLEIDICV_MAX_IMAGE_PIXELS, 0.0, 1.0));
}

#endif  // KLEIDICV_EXPERIMENTAL_FEATURE_CANNY
