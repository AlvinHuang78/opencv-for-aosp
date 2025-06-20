// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "framework/abstract.h"
#include "framework/array.h"
#include "framework/border.h"

class ImplementsBorder : public test::Bordered {
 public:
  explicit ImplementsBorder(size_t left, size_t top, size_t right,
                            size_t bottom)
      : left_{left}, top_{top}, right_{right}, bottom_{bottom} {}

  size_t left() const override { return left_; };
  size_t top() const override { return top_; };
  size_t right() const override { return right_; };
  size_t bottom() const override { return bottom_; };

 private:
  size_t left_;
  size_t top_;
  size_t right_;
  size_t bottom_;
};  // end of class ImplementsBorder

// Tests that replicate border strategy works for one channel and one bordering
// element in every direction.
TEST(Border, Replicate_1Ch_1Element) {
  using ElementType = uint8_t;

  const size_t width = 6, height = 5;

  test::Array2D<ElementType> actual{width, height};
  actual.set(0, 0, {0, 0, 0, 0, 0, 0});
  actual.set(1, 0, {0, 1, 2, 3, 4, 0});
  actual.set(2, 0, {0, 5, 6, 7, 8, 0});
  actual.set(3, 0, {0, 9, 0, 1, 2, 0});
  actual.set(4, 0, {0, 0, 0, 0, 0, 0});

  test::Array2D<ElementType> expected{width, height};
  expected.set(0, 0, {1, 1, 2, 3, 4, 4});
  expected.set(1, 0, {1, 1, 2, 3, 4, 4});
  expected.set(2, 0, {5, 5, 6, 7, 8, 8});
  expected.set(3, 0, {9, 9, 0, 1, 2, 2});
  expected.set(4, 0, {9, 9, 0, 1, 2, 2});

  ImplementsBorder bordered{1, 1, 1, 1};
  test::prepare_borders<ElementType>(KLEIDICV_BORDER_TYPE_REPLICATE,
                                     {0, 0, 0, 0}, &bordered, &actual);
  EXPECT_EQ_ARRAY2D(expected, actual);
}

// Tests that replicate border strategy works for one channel and two bordering
// elements in every direction.
TEST(Border, Replicate_1Ch_2Elements) {
  using ElementType = uint8_t;

  const size_t width = 6, height = 6;

  test::Array2D<ElementType> actual{width, height};
  actual.set(0, 0, {0, 0, 0, 0, 0, 0});
  actual.set(1, 0, {0, 0, 0, 0, 0, 0});
  actual.set(2, 0, {0, 0, 1, 2, 0, 0});
  actual.set(3, 0, {0, 0, 3, 4, 0, 0});
  actual.set(4, 0, {0, 0, 0, 0, 0, 0});
  actual.set(5, 0, {0, 0, 0, 0, 0, 0});

  test::Array2D<ElementType> expected{width, height};
  expected.set(0, 0, {1, 1, 1, 2, 2, 2});
  expected.set(1, 0, {1, 1, 1, 2, 2, 2});
  expected.set(2, 0, {1, 1, 1, 2, 2, 2});
  expected.set(3, 0, {3, 3, 3, 4, 4, 4});
  expected.set(4, 0, {3, 3, 3, 4, 4, 4});
  expected.set(5, 0, {3, 3, 3, 4, 4, 4});

  ImplementsBorder bordered{2, 2, 2, 2};
  test::prepare_borders<ElementType>(KLEIDICV_BORDER_TYPE_REPLICATE,
                                     {0, 0, 0, 0}, &bordered, &actual);
  EXPECT_EQ_ARRAY2D(expected, actual);
}

// Tests that replicate border strategy works for two channels and one
// bordering element in every direction.
TEST(Border, Replicate_2Ch_1Element) {
  using ElementType = uint8_t;

  const size_t width = 8, height = 5, channels = 2;

  test::Array2D<ElementType> actual{width, height, 0, channels};
  actual.set(0, 0, {0, 0, 0, 0, 0, 0, 0, 0});
  actual.set(1, 0, {0, 0, 1, 2, 3, 4, 0, 0});
  actual.set(2, 0, {0, 0, 5, 6, 7, 8, 0, 0});
  actual.set(3, 0, {0, 0, 9, 1, 2, 3, 0, 0});
  actual.set(4, 0, {0, 0, 0, 0, 0, 0, 0, 0});

  test::Array2D<ElementType> expected{width, height, 0, channels};
  expected.set(0, 0, {1, 2, 1, 2, 3, 4, 3, 4});
  expected.set(1, 0, {1, 2, 1, 2, 3, 4, 3, 4});
  expected.set(2, 0, {5, 6, 5, 6, 7, 8, 7, 8});
  expected.set(3, 0, {9, 1, 9, 1, 2, 3, 2, 3});
  expected.set(4, 0, {9, 1, 9, 1, 2, 3, 2, 3});

  ImplementsBorder bordered{1, 1, 1, 1};
  test::prepare_borders<ElementType>(KLEIDICV_BORDER_TYPE_REPLICATE,
                                     {0, 0, 0, 0}, &bordered, &actual);
  EXPECT_EQ_ARRAY2D(expected, actual);
}

// Tests that replicate border strategy works for two channels and two
// bordering elements in every direction.
TEST(Border, Replicate_2Ch_2Elements) {
  using ElementType = uint8_t;

  const size_t width = 12, height = 6, channels = 2;

  test::Array2D<ElementType> actual{width, height, 0, channels};
  actual.set(0, 0, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  actual.set(1, 0, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  actual.set(2, 0, {0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0});
  actual.set(3, 0, {0, 0, 0, 0, 5, 6, 7, 8, 0, 0, 0, 0});
  actual.set(4, 0, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  actual.set(5, 0, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});

  test::Array2D<ElementType> expected{width, height, 0, channels};
  expected.set(0, 0, {1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4});
  expected.set(1, 0, {1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4});
  expected.set(2, 0, {1, 2, 1, 2, 1, 2, 3, 4, 3, 4, 3, 4});
  expected.set(3, 0, {5, 6, 5, 6, 5, 6, 7, 8, 7, 8, 7, 8});
  expected.set(4, 0, {5, 6, 5, 6, 5, 6, 7, 8, 7, 8, 7, 8});
  expected.set(5, 0, {5, 6, 5, 6, 5, 6, 7, 8, 7, 8, 7, 8});

  ImplementsBorder bordered{2, 2, 2, 2};
  test::prepare_borders<ElementType>(KLEIDICV_BORDER_TYPE_REPLICATE,
                                     {0, 0, 0, 0}, &bordered, &actual);
  EXPECT_EQ_ARRAY2D(expected, actual);
}
