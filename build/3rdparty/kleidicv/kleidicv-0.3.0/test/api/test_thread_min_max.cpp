// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <thread>

#include "framework/array.h"
#include "framework/generator.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv_thread/kleidicv_thread.h"
#include "multithreading_fake.h"

#define KLEIDICV_MIN_MAX(type, suffix) \
  KLEIDICV_API(min_max, kleidicv_min_max_##suffix, type)

KLEIDICV_MIN_MAX(int8_t, s8);
KLEIDICV_MIN_MAX(uint8_t, u8);
KLEIDICV_MIN_MAX(int16_t, s16);
KLEIDICV_MIN_MAX(uint16_t, u16);
KLEIDICV_MIN_MAX(int32_t, s32);
KLEIDICV_MIN_MAX(float, f32);

#define KLEIDICV_THREAD_MIN_MAX(type, suffix) \
  KLEIDICV_API(thread_min_max, kleidicv_thread_min_max_##suffix, type)

KLEIDICV_THREAD_MIN_MAX(int8_t, s8);
KLEIDICV_THREAD_MIN_MAX(uint8_t, u8);
KLEIDICV_THREAD_MIN_MAX(int16_t, s16);
KLEIDICV_THREAD_MIN_MAX(uint16_t, u16);
KLEIDICV_THREAD_MIN_MAX(int32_t, s32);
KLEIDICV_THREAD_MIN_MAX(float, f32);

TEST(MinMaxThread, SimpleFloat) {
  size_t width = 10, height = 10;
  test::Array2D<float> src(width, height);

  for (auto value : {1.0F, -1.0F}) {
    src.fill(value);

    float minval = 1, maxval = -1;

    kleidicv_error_t result =
        thread_min_max<float>()(src.data(), src.stride(), width, height,
                                &minval, &maxval, get_multithreading_fake(3));

    EXPECT_EQ(KLEIDICV_OK, result);
    EXPECT_EQ(value, minval);
    EXPECT_EQ(value, maxval);
  }
}

template <typename ElementType>
class MinMaxThread : public testing::Test {};

TYPED_TEST_SUITE_P(MinMaxThread);

// Tuple of width, height, thread count.
typedef std::tuple<size_t, size_t, size_t> P;

static const auto test_params = {
    P{1, 1, 1}, P{1, 2, 1}, P{1, 2, 2}, P{2, 1, 2}, P{2, 2, 1}, P{1, 3, 2},
    P{2, 3, 1}, P{6, 4, 1}, P{4, 5, 2}, P{2, 6, 3}, P{1, 7, 4}, P{12, 34, 5}};

TYPED_TEST_P(MinMaxThread, CompareWithSingle) {
  test::PseudoRandomNumberGenerator<TypeParam> generator;
  size_t width = 0, height = 0, thread_count = 0;
  for (auto params : test_params) {
    std::tie(width, height, thread_count) = params;
    test::Array2D<TypeParam> src(width, height);
    TypeParam min_single, max_single, min_multi, max_multi;

    src.fill(generator);

    kleidicv_error_t single_result = min_max<TypeParam>()(
        src.data(), src.stride(), width, height, &min_single, &max_single);

    kleidicv_error_t multi_result = thread_min_max<TypeParam>()(
        src.data(), src.stride(), width, height, &min_multi, &max_multi,
        get_multithreading_fake(thread_count));

    EXPECT_EQ(KLEIDICV_OK, single_result);
    EXPECT_EQ(KLEIDICV_OK, multi_result);
    EXPECT_EQ(min_multi, min_single);
    EXPECT_EQ(max_multi, max_single);
  }
}

TYPED_TEST_P(MinMaxThread, NullArguments) {
  size_t width = 1, height = 2, thread_count = 2;
  TypeParam src[2] = {1, 2}, min_value, max_value;

  min_value = max_value = 0;
  kleidicv_error_t res = thread_min_max<TypeParam>()(
      src, width * sizeof(TypeParam), width, height, nullptr, &max_value,
      get_multithreading_fake(thread_count));

  EXPECT_EQ(KLEIDICV_OK, res);
  EXPECT_EQ(0, min_value);
  EXPECT_EQ(2, max_value);

  min_value = max_value = 0;
  res = thread_min_max<TypeParam>()(src, width * sizeof(TypeParam), width,
                                    height, &min_value, nullptr,
                                    get_multithreading_fake(thread_count));

  EXPECT_EQ(KLEIDICV_OK, res);
  EXPECT_EQ(1, min_value);
  EXPECT_EQ(0, max_value);

  min_value = max_value = 0;
  res = thread_min_max<TypeParam>()(src, width * sizeof(TypeParam), width,
                                    height, nullptr, nullptr,
                                    get_multithreading_fake(thread_count));

  EXPECT_EQ(KLEIDICV_OK, res);
  EXPECT_EQ(0, min_value);
  EXPECT_EQ(0, max_value);
}

REGISTER_TYPED_TEST_SUITE_P(MinMaxThread, CompareWithSingle, NullArguments);

using MinMaxElementTypes =
    ::testing::Types<int8_t, uint8_t, int16_t, uint16_t, int32_t, float>;

INSTANTIATE_TYPED_TEST_SUITE_P(MinMax, MinMaxThread, MinMaxElementTypes);
