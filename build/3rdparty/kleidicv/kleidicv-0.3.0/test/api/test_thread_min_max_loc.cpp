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

#define KLEIDICV_MIN_MAX_LOC(type, suffix) \
  KLEIDICV_API(min_max_loc, kleidicv_min_max_loc_##suffix, type)

KLEIDICV_MIN_MAX_LOC(uint8_t, u8);

#define KLEIDICV_THREAD_MIN_MAX_LOC(type, suffix) \
  KLEIDICV_API(thread_min_max_loc, kleidicv_thread_min_max_loc_##suffix, type)

KLEIDICV_THREAD_MIN_MAX_LOC(uint8_t, u8);

template <typename ElementType>
class MinMaxLocThread : public testing::Test {};

TYPED_TEST_SUITE_P(MinMaxLocThread);

// Tuple of width, height, thread count.
typedef std::tuple<size_t, size_t, size_t> P;

static const auto test_params = {
    P{1, 1, 1}, P{1, 2, 1}, P{1, 2, 2}, P{2, 1, 2}, P{2, 2, 1}, P{1, 3, 2},
    P{2, 3, 1}, P{6, 4, 1}, P{4, 5, 2}, P{2, 6, 3}, P{1, 7, 4}, P{12, 34, 5}};

TYPED_TEST_P(MinMaxLocThread, CompareWithSingle) {
  size_t width = 0, height = 0, thread_count = 0;
  for (auto params : test_params) {
    std::tie(width, height, thread_count) = params;
    test::Array2D<TypeParam> src(width, height);
    size_t min_offset_single = 0, max_offset_single = 0, min_offset_multi = 0,
           max_offset_multi = 0;

    test::PseudoRandomNumberGenerator<TypeParam> generator;
    src.fill(generator);

    kleidicv_error_t single_result =
        min_max_loc<TypeParam>()(src.data(), src.stride(), width, height,
                                 &min_offset_single, &max_offset_single);

    kleidicv_error_t multi_result = thread_min_max_loc<TypeParam>()(
        src.data(), src.stride(), width, height, &min_offset_multi,
        &max_offset_multi, get_multithreading_fake(thread_count));

    EXPECT_EQ(KLEIDICV_OK, single_result);
    EXPECT_EQ(KLEIDICV_OK, multi_result);
    EXPECT_EQ(min_offset_multi, min_offset_single);
    EXPECT_EQ(max_offset_multi, max_offset_single);
  }
}

TYPED_TEST_P(MinMaxLocThread, NullArguments) {
  size_t width = 1, height = 2, thread_count = 2;
  TypeParam src[2] = {1, 2};
  // Let it be different from 0, 1 or 2
  const size_t kUnchanged = 99;
  size_t min_offset = kUnchanged, max_offset = kUnchanged;

  kleidicv_error_t res = thread_min_max_loc<TypeParam>()(
      src, width * sizeof(TypeParam), width, height, nullptr, &max_offset,
      get_multithreading_fake(thread_count));

  EXPECT_EQ(KLEIDICV_OK, res);
  EXPECT_EQ(kUnchanged, min_offset);
  EXPECT_EQ(1, max_offset);

  min_offset = max_offset = 99;
  res = thread_min_max_loc<TypeParam>()(src, width * sizeof(TypeParam), width,
                                        height, &min_offset, nullptr,
                                        get_multithreading_fake(thread_count));

  EXPECT_EQ(KLEIDICV_OK, res);
  EXPECT_EQ(0, min_offset);
  EXPECT_EQ(kUnchanged, max_offset);

  min_offset = max_offset = 99;
  res = thread_min_max_loc<TypeParam>()(src, width * sizeof(TypeParam), width,
                                        height, nullptr, nullptr,
                                        get_multithreading_fake(thread_count));

  EXPECT_EQ(KLEIDICV_OK, res);
  EXPECT_EQ(kUnchanged, min_offset);
  EXPECT_EQ(kUnchanged, max_offset);
}

REGISTER_TYPED_TEST_SUITE_P(MinMaxLocThread, CompareWithSingle, NullArguments);

using MinMaxLocElementTypes = ::testing::Types<uint8_t>;

INSTANTIATE_TYPED_TEST_SUITE_P(MinMaxLoc, MinMaxLocThread,
                               MinMaxLocElementTypes);
