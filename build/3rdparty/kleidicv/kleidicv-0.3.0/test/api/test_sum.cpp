// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <limits>

#include "kleidicv/kleidicv.h"

TEST(Sum, Ones) {
  // clang-format off
  float src[] = {
    1, 1, 1, 1,
    1, 1, 1, 1,
    1, 1, 1, 1,
    1, 1, 1, 1,
  };
  // clang-format on

  float sum = std::numeric_limits<float>::max();

  EXPECT_EQ(KLEIDICV_OK, kleidicv_sum_f32(src, 4 * sizeof(float), 4, 4, &sum));
  EXPECT_FLOAT_EQ(16, sum);
}

TEST(Sum, NegativeOnes) {
  // clang-format off
  float src[] = {
    -1, -1, -1, -1,
    -1, -1, -1, -1,
    -1, -1, -1, -1,
    -1, -1, -1, -1,
  };
  // clang-format on

  float sum = std::numeric_limits<float>::max();

  EXPECT_EQ(KLEIDICV_OK, kleidicv_sum_f32(src, 4 * sizeof(float), 4, 4, &sum));
  EXPECT_FLOAT_EQ(-16, sum);
}

TEST(Sum, Zeroes) {
  // clang-format off
  float src[] = {
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
  };
  // clang-format on

  float sum = std::numeric_limits<float>::max();

  EXPECT_EQ(KLEIDICV_OK, kleidicv_sum_f32(src, 4 * sizeof(float), 4, 4, &sum));
  EXPECT_FLOAT_EQ(0, sum);
}

TEST(Sum, Single) {
  // clang-format off
  float src[] = {
    1
  };
  // clang-format on

  float sum = std::numeric_limits<float>::max();

  EXPECT_EQ(KLEIDICV_OK, kleidicv_sum_f32(src, 1 * sizeof(float), 1, 1, &sum));
  EXPECT_FLOAT_EQ(1, sum);
}

TEST(Sum, Empty) {
  // clang-format off
  float src[] = {};
  // clang-format on

  float sum = std::numeric_limits<float>::max();

  EXPECT_EQ(KLEIDICV_OK, kleidicv_sum_f32(src, 0, 0, 0, &sum));
  EXPECT_FLOAT_EQ(0, sum);
}

TEST(Sum, WidthGreaterThanLanes) {
  // clang-format off
  float src[] = {
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1
  };
  // clang-format on

  float sum = std::numeric_limits<float>::max();

  EXPECT_EQ(KLEIDICV_OK, kleidicv_sum_f32(src, 5 * sizeof(float), 5, 5, &sum));
  EXPECT_FLOAT_EQ(25, sum);
}

TEST(Sum, WidthGreaterThanLanesPadded) {
  // clang-format off
  float src[] = {
    1, 1, 1, 1, 1, 9, 9,
    1, 1, 1, 1, 1, 9, 9,
    1, 1, 1, 1, 1, 9, 9,
    1, 1, 1, 1, 1, 9, 9,
    1, 1, 1, 1, 1, 9, 9
  };
  // clang-format on

  float sum = std::numeric_limits<float>::max();

  EXPECT_EQ(KLEIDICV_OK, kleidicv_sum_f32(src, 7 * sizeof(float), 5, 5, &sum));
  EXPECT_FLOAT_EQ(25, sum);
}

TEST(Sum, WidthLessThanLanes) {
  // clang-format off
  float src[] = {
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1,
    1, 1, 1
  };
  // clang-format on

  float sum = std::numeric_limits<float>::max();

  EXPECT_EQ(KLEIDICV_OK, kleidicv_sum_f32(src, 3 * sizeof(float), 3, 5, &sum));
  EXPECT_FLOAT_EQ(15, sum);
}

TEST(Sum, WidthLessThanLanesPadded) {
  // clang-format off
  float src[] = {
    1, 1, 1, 9, 9,
    1, 1, 1, 9, 9,
    1, 1, 1, 9, 9,
    1, 1, 1, 9, 9,
    1, 1, 1, 9, 9
  };
  // clang-format on

  float sum = std::numeric_limits<float>::max();

  EXPECT_EQ(KLEIDICV_OK, kleidicv_sum_f32(src, 5 * sizeof(float), 3, 5, &sum));
  EXPECT_FLOAT_EQ(15, sum);
}

TEST(Sum, FloatMax) {
  // clang-format off
  float src[] = {
    std::numeric_limits<float>::max(), 1
  };
  // clang-format on

  float sum = -1;

  EXPECT_EQ(KLEIDICV_OK, kleidicv_sum_f32(src, 2 * sizeof(float), 2, 1, &sum));
  EXPECT_EQ(sum, std::numeric_limits<float>::max());
}

TEST(Sum, FloatLowest) {
  // clang-format off
  float src[] = {
    std::numeric_limits<float>::lowest() , -1
  };
  // clang-format on

  float sum = -1;

  EXPECT_EQ(KLEIDICV_OK, kleidicv_sum_f32(src, 2 * sizeof(float), 2, 1, &sum));
  EXPECT_EQ(sum, std::numeric_limits<float>::lowest());
}

TEST(Sum, FloatInfinity) {
  // clang-format off
  float src[] = {
    std::numeric_limits<float>::max(), std::numeric_limits<float>::max()
  };
  // clang-format on

  float sum = -1;

  EXPECT_EQ(KLEIDICV_OK, kleidicv_sum_f32(src, 2 * sizeof(float), 2, 1, &sum));
  EXPECT_EQ(sum, std::numeric_limits<float>::infinity());
}

TEST(Sum, FloatNegativeInfinity) {
  // clang-format off
  float src[] = {
    std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest()
  };
  // clang-format on

  float sum = -1;

  EXPECT_EQ(KLEIDICV_OK, kleidicv_sum_f32(src, 2 * sizeof(float), 2, 1, &sum));
  EXPECT_EQ(sum, -std::numeric_limits<float>::infinity());
}

TEST(Sum, OversizeImage) {
  // clang-format off
  float src[] = {};
  // clang-format on

  float sum = std::numeric_limits<float>::max();

  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_sum_f32(src, 1 * sizeof(float),
                             KLEIDICV_MAX_IMAGE_PIXELS + 1, 1, &sum));
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_sum_f32(src, 1 * sizeof(float), 1,
                             KLEIDICV_MAX_IMAGE_PIXELS + 1, &sum));
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_sum_f32(src, 1 * sizeof(float), KLEIDICV_MAX_IMAGE_PIXELS,
                             KLEIDICV_MAX_IMAGE_PIXELS, &sum));
}

TEST(Sum, NullPointers) {
  // clang-format off
  float src[] = {};
  // clang-format on

  float sum = std::numeric_limits<float>::max();

  EXPECT_EQ(KLEIDICV_ERROR_NULL_POINTER,
            kleidicv_sum_f32(nullptr, 0, 0, 0, &sum));
  EXPECT_EQ(KLEIDICV_ERROR_NULL_POINTER,
            kleidicv_sum_f32(src, 0, 0, 0, nullptr));
}

TEST(Sum, Misalignment) {
  // clang-format off
  float src[] = {
    1, 1,
    1, 1
  };
  // clang-format on

  float sum = std::numeric_limits<float>::max();

  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            kleidicv_sum_f32(src, sizeof(float) + 1, 2, 2, &sum));
}
