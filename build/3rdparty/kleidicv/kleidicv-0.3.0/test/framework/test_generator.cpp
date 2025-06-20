// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <array>
#include <optional>

#include "framework/generator.h"

// Tests test::PseudoRandomNumberGenerator::reset() works.
TEST(PseudoRandomNumberGenerator, Reset) {
  using ElementType = uint8_t;

  test::PseudoRandomNumberGenerator<ElementType> generator;
  ElementType initial_value = generator.next().value_or(123);
  generator.next();
  generator.reset();
  ElementType value = generator.next().value_or(234);
  EXPECT_EQ(initial_value, value);
}

// Tests test::SequenceGenerator::reset() works.
TEST(SequenceGenerator, Reset) {
  using ElementType = uint8_t;

  std::array<ElementType, 3> array{1, 2, 3};
  test::SequenceGenerator generator{array};
  EXPECT_EQ(generator.next().value_or(0), 1);
  EXPECT_EQ(generator.next().value_or(0), 2);
  EXPECT_EQ(generator.next().value_or(0), 3);
  EXPECT_EQ(generator.next(), std::nullopt);

  generator.reset();
  EXPECT_EQ(generator.next().value_or(0), 1);
}
