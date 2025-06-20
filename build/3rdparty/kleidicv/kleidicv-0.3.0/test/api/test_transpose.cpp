// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "framework/array.h"
#include "framework/generator.h"
#include "kleidicv/kleidicv.h"
#include "test_config.h"

template <typename ElementType, bool inplace>
class TestTranspose final {
 public:
  explicit TestTranspose(size_t padding) : padding_(padding) {}

  void scalar_test() {
    size_t first_dim = test::Options::vector_lanes<ElementType>() - 1;
    size_t second_dim =
        inplace ? first_dim : test::Options::vector_lanes<ElementType>() + 1;
    // Exercise horizontal scalar path
    test(first_dim, second_dim);
    if (!inplace) {
      // Exercise vertical scalar path
      test(second_dim, first_dim);
    }
  }

  void vector_test() {
    // Make at least two full vector passes
    size_t src_width = 2 * test::Options::vector_lanes<ElementType>();
    // Set height to be different from width but still bigger than vector_lanes
    size_t src_height =
        inplace ? src_width : 3 * test::Options::vector_lanes<ElementType>();
    test(src_width, src_height);
  }

  // For the 'remaining' part of the outer vector loop of inplace transpose
  void vector_plus_scalar_test() {
    // Two full vector passes, plus some scalar
    size_t first_dim = 3 * test::Options::vector_lanes<ElementType>() - 1;
    size_t second_dim =
        inplace ? first_dim
                : 3 * test::Options::vector_lanes<ElementType>() + 1;
    test(first_dim, second_dim);
    if (!inplace) {
      test(second_dim, first_dim);
    }
  }

 protected:
  void test(size_t src_width, size_t src_height) const {
    const size_t dst_width = src_height;
    const size_t dst_height = src_width;

    test::Array2D<ElementType> source(src_width, src_height, padding_, 1);
    test::Array2D<ElementType> expected(dst_width, dst_height, padding_, 1);
    test::Array2D<ElementType> actual;
    test::Array2D<ElementType> *p_actual = &source;

    // Only allocate actual array if needed
    if constexpr (!inplace) {
      actual = test::Array2D<ElementType>(dst_width, dst_height, padding_, 1);
      p_actual = &actual;
    }

    // No specific values to test
    test::PseudoRandomNumberGenerator<ElementType> generator;
    source.fill(generator);

    calculate_expected(source, expected);

    ASSERT_EQ(KLEIDICV_OK,
              kleidicv_transpose(source.data(), source.stride(),
                                 p_actual->data(), p_actual->stride(),
                                 src_width, src_height, sizeof(ElementType)));

    EXPECT_EQ_ARRAY2D(expected, *p_actual);
  }

  void calculate_expected(const test::Array2D<ElementType> &source,
                          test::Array2D<ElementType> &expected) const {
    for (size_t hindex = 0; hindex < source.height(); ++hindex) {
      for (size_t vindex = 0; vindex < source.width(); ++vindex) {
        // NOLINTBEGIN(clang-analyzer-core.uninitialized.Assign)
        *expected.at(vindex, hindex) = *source.at(hindex, vindex);
        // NOLINTEND(clang-analyzer-core.uninitialized.Assign)
      }
    }
  }

  size_t padding_;
};

class TestNotImplemented {
 public:
  template <typename ElementType>
  static void wrong_dimensions() {
    const size_t width = 1;
    const size_t height = 2;
    size_t stride = width * sizeof(ElementType);

    ElementType source[width * height] = {0xFF};

    ASSERT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              kleidicv_transpose(source, stride, source, stride, width, height,
                                 sizeof(ElementType)));
  }

  template <typename WrongType>
  static void wrong_type() {
    const size_t width = 1, height = 1;
    size_t stride = width * sizeof(WrongType);

    WrongType source[width * height] = {0xFF};

    ASSERT_EQ(KLEIDICV_ERROR_NOT_IMPLEMENTED,
              kleidicv_transpose(source, stride, source, stride, width, height,
                                 sizeof(WrongType)));
  }
};

template <typename ElementType>
class Transpose : public testing::Test {};

using ElementTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;
TYPED_TEST_SUITE(Transpose, ElementTypes);

TYPED_TEST(Transpose, ScalarToNewBufferNoPadding) {
  TestTranspose<TypeParam, false> test(0);
  test.scalar_test();
}

TYPED_TEST(Transpose, VectorToNewBufferNoPadding) {
  TestTranspose<TypeParam, false> test(0);
  test.vector_test();
}

TYPED_TEST(Transpose, ScalarToNewBufferWithPadding) {
  TestTranspose<TypeParam, false> test(1);
  test.scalar_test();
}

TYPED_TEST(Transpose, VectorToNewBufferWithPadding) {
  TestTranspose<TypeParam, false> test(1);
  test.vector_test();
}

TYPED_TEST(Transpose, ScalarInplaceNoPadding) {
  TestTranspose<TypeParam, true> test(0);
  test.scalar_test();
}

TYPED_TEST(Transpose, VectorInplaceNoPadding) {
  TestTranspose<TypeParam, true> test(0);
  test.vector_test();
}

TYPED_TEST(Transpose, ScalarInplaceWithPadding) {
  TestTranspose<TypeParam, true> test(1);
  test.scalar_test();
}

TYPED_TEST(Transpose, VectorInplaceWithPadding) {
  TestTranspose<TypeParam, true> test(1);
  test.vector_test();
}

TYPED_TEST(Transpose, VectorPlusScalarNoPadding) {
  TestTranspose<TypeParam, false> test(0);
  test.vector_plus_scalar_test();
}

TYPED_TEST(Transpose, VectorPlusScalarWithPadding) {
  TestTranspose<TypeParam, false> test(1);
  test.vector_plus_scalar_test();
}

TYPED_TEST(Transpose, VectorPlusScalarInplaceNoPadding) {
  TestTranspose<TypeParam, true> test(0);
  test.vector_plus_scalar_test();
}

TYPED_TEST(Transpose, VectorPlusScalarInplaceWithPadding) {
  TestTranspose<TypeParam, true> test(1);
  test.vector_plus_scalar_test();
}

TYPED_TEST(Transpose, NullPointer) {
  TypeParam src[1] = {}, dst[1];
  test::test_null_args(kleidicv_transpose, src, sizeof(TypeParam), dst,
                       sizeof(TypeParam), 1, 1, sizeof(TypeParam));
}

TYPED_TEST(Transpose, NotImplementedDims) {
  TestNotImplemented::wrong_dimensions<TypeParam>();
}

TEST(Transpose, NotImplementedType) {
  TestNotImplemented::wrong_type<__uint128_t>();
}

TYPED_TEST(Transpose, Misalignment) {
  if (sizeof(TypeParam) == 1) {
    // misalignment impossible
    return;
  }

  const size_t kBufSize = sizeof(TypeParam) * 10;
  char src[kBufSize] = {}, dst[kBufSize] = {};

  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            kleidicv_transpose(src + 1, sizeof(TypeParam), dst,
                               sizeof(TypeParam), 1, 1, sizeof(TypeParam)));
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            kleidicv_transpose(src, sizeof(TypeParam) + 1, dst,
                               sizeof(TypeParam), 1, 2, sizeof(TypeParam)));
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            kleidicv_transpose(src, sizeof(TypeParam), dst + 1,
                               sizeof(TypeParam), 1, 1, sizeof(TypeParam)));
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            kleidicv_transpose(src, sizeof(TypeParam), dst,
                               sizeof(TypeParam) + 1, 2, 1, sizeof(TypeParam)));
  // Ignore stride if there's only one row
  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_transpose(src, sizeof(TypeParam), dst,
                               sizeof(TypeParam) + 1, 1, 1, sizeof(TypeParam)));
}

TYPED_TEST(Transpose, ZeroImageSize) {
  TypeParam src[1] = {}, dst[1];
  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_transpose(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                               0, 1, sizeof(TypeParam)));
  EXPECT_EQ(KLEIDICV_OK,
            kleidicv_transpose(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                               1, 0, sizeof(TypeParam)));
}

TYPED_TEST(Transpose, OversizeImage) {
  TypeParam src[1] = {}, dst[1];
  EXPECT_EQ(
      KLEIDICV_ERROR_RANGE,
      kleidicv_transpose(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                         KLEIDICV_MAX_IMAGE_PIXELS + 1, 1, sizeof(TypeParam)));
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            kleidicv_transpose(src, sizeof(TypeParam), dst, sizeof(TypeParam),
                               KLEIDICV_MAX_IMAGE_PIXELS,
                               KLEIDICV_MAX_IMAGE_PIXELS, sizeof(TypeParam)));
}
