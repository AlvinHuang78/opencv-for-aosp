// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "framework/array.h"
#include "framework/generator.h"
#include "framework/utils.h"
#include "kleidicv/kleidicv.h"

#define KLEIDICV_IN_RANGE(type, suffix) \
  KLEIDICV_API(in_range, kleidicv_in_range_##suffix, type)

KLEIDICV_IN_RANGE(uint8_t, u8);
KLEIDICV_IN_RANGE(float, f32);

template <typename ElementType>
class InRangeTest final {
 private:
  template <typename T>
  static constexpr T lowest() {
    return std::numeric_limits<T>::lowest();
  }

  template <typename T>
  static constexpr T max() {
    return std::numeric_limits<T>::max();
  }
  struct Elements {
    size_t width;
    size_t height;
    ElementType lower_bound;
    ElementType upper_bound;

    std::vector<std::vector<ElementType>> source_rows;
    std::vector<std::vector<uint8_t>> expected_rows;

    Elements(size_t _width, size_t _height, ElementType _lower_bound,
             ElementType _upper_bound,
             std::vector<std::vector<ElementType>>&& _source_rows,
             std::vector<std::vector<uint8_t>>&& _expected_rows)
        : width(_width),
          height(_height),
          lower_bound(_lower_bound),
          upper_bound(_upper_bound),
          source_rows(std::move(_source_rows)),
          expected_rows(std::move(_expected_rows)) {}
  };

  static float floatval(uint32_t v) {
    float result;  // Avoid cppcoreguidelines-init-variables. NOLINT
    static_assert(sizeof(result) == sizeof(v));
    memcpy(&result, &v, sizeof(result));
    return result;
  }

  const float quietNaN = std::numeric_limits<float>::quiet_NaN();
  const float signalingNaN = std::numeric_limits<float>::signaling_NaN();
  const float posInfinity = std::numeric_limits<float>::infinity();
  const float negInfinity = -std::numeric_limits<float>::infinity();

  const float minusNaN = floatval(0xFF800001);
  const float plusNaN = floatval(0x7F800001);
  const float plusZero = 0.0F;
  const float minusZero = -0.0F;

  const float oneNaN = floatval(0x7FC00001);
  const float zeroDivZero = -std::numeric_limits<float>::quiet_NaN();
  const float floatMin = std::numeric_limits<float>::lowest();
  const float floatMax = std::numeric_limits<float>::max();

  const float posSubnormalMin = std::numeric_limits<float>::denorm_min();
  const float posSubnormalMax = floatval(0x007FFFFF);
  const float negSubnormalMin = -std::numeric_limits<float>::denorm_min();
  const float negSubnormalMax = floatval(0x807FFFFF);

  void calculate_expected(const test::Array2D<ElementType>& source,
                          test::Array2D<uint8_t>& expected,
                          ElementType lower_bound = 50,
                          ElementType upper_bound = 100) {
    for (size_t hindex = 0; hindex < source.height(); ++hindex) {
      for (size_t vindex = 0; vindex < source.width(); ++vindex) {
        uint8_t calculated = 0;
        // NOLINTBEGIN(clang-analyzer-core.uninitialized.Assign)
        ElementType current_element = *source.at(hindex, vindex);
        // NOLINTEND(clang-analyzer-core.uninitialized.Assign)
        if ((current_element >= lower_bound) &&
            (current_element <= upper_bound)) {
          calculated = 255;
        } else {
          calculated = 0;
        }
        *expected.at(hindex, vindex) = calculated;
      }
    }
  }

  template <typename T>
  size_t get_linear_height(size_t width, size_t minimum_size) {
    // Calling this with a float type would generate a value of infinity.
    static_assert(std::is_integral_v<T>);

    size_t image_size =
        std::max(minimum_size, static_cast<size_t>(max<T>() - lowest<T>()));
    size_t height = image_size / width + 1;

    return height;
  }

  template <typename T>
  std::tuple<test::Array2D<T>, test::Array2D<uint8_t>, test::Array2D<uint8_t>>
  get_linear_arrays(size_t width, size_t height) {
    test::Array2D<T> source(width, height, 1, 1);
    test::Array2D<uint8_t> expected(width, height, 1, 1);
    test::Array2D<uint8_t> actual(width, height, 1, 1);

    test::GenerateLinearSeries<T> generator(0);
    source.fill(generator);

    calculate_expected(source, expected);

    return {source, expected, actual};
  }

  std::array<size_t, 1> inputs_padding_{};
  std::array<size_t, 1> outputs_padding_{};

 public:
  // Tests special float values.
  template <typename T, std::enable_if_t<std::is_same_v<float, T>, bool> = true>
  const Elements& get_custom_test_elements() {
    static const Elements kTestElements = {
        // clang-format off
        4, 6,
        -11.4, 1111.10,
        {{
          {       quietNaN,    signalingNaN,     posInfinity,     negInfinity},
          {       minusNaN,         plusNaN,        plusZero,       minusZero},
          {         oneNaN,     zeroDivZero,        floatMin,        floatMax},
          {posSubnormalMin, posSubnormalMax, negSubnormalMin, negSubnormalMax},
          {        1111.11,        -1112.22,          113.33,          114.44},
          {           11.5,            12.5,           -11.5,           -12.5},
        }},
        {{
          { 0,    0,   0,   0},
          { 0,    0, 255, 255},
          {  0,   0,   0,   0},
          {255, 255, 255, 255},
          {  0,   0, 255, 255},
          {255, 255,   0,   0},
        }}
        // clang-format on
    };
    return kTestElements;
  }

  // float and uint8_t behave differently around their max values.
  template <typename T, std::enable_if_t<std::is_same_v<float, T>, bool> = true>
  const Elements& get_max_test_elements() {
    static const Elements kTestElements = {
        // clang-format off
        2, 3,
        max<T>() - 100, max<T>() - 1,
        {{
          {            10, max<T>() - 105},
          {max<T>() - 100,   max<T>() - 2},
          {  max<T>() - 1,       max<T>()},
        }},
        {{
          {  0, 255},
          {255, 255},
          {255, 255},
        }}
        // clang-format on
    };
    return kTestElements;
  }

  // More extensive test for uint8_t values.
  template <typename T,
            std::enable_if_t<std::is_same_v<uint8_t, T>, bool> = true>
  const Elements& get_custom_test_elements() {
    static const Elements kTestElements = {
        // clang-format off
        16, 16,
        1, 10,
        {{
          {177,  48,   1,  58, 223, 153,  36, 183, 113, 234, 218, 216, 129, 243, 230, 185},
          { 45, 187,  58,  65, 247,  68, 122, 158, 239,  84, 171,  15, 180, 156, 227,  21},
          { 53,  95, 117, 138, 197, 251,  23, 120, 201, 223, 219,  50,  37,   4, 112, 129},
          {240, 245, 130,   3,  65, 182,  19, 254, 241, 216, 198,  22, 101, 189, 151,  60},
          {234, 147, 201,  67, 121,  42, 173, 151, 108, 220, 190, 230,  23, 135, 139,  66},
          { 62, 180, 191,  65,  74, 190, 237, 102,   6, 196,  14,  76, 192,  12, 223, 232},
          { 66, 167,  77,  86,  85, 228, 104, 127, 251, 126,  56, 107,  93, 153, 222,   2},
          {160,  74, 225, 116, 243, 193, 166, 157, 193, 233, 109, 141,  86,  83, 156, 203},
          {  5, 225,  87,  53,   5, 104, 122, 157,  68,  45,  39, 229, 133, 138,  47, 231},
          {225,   4, 246,   0,  89, 227, 150, 114,  45, 144,  98, 192, 245, 232, 214,  39},
          {124, 119, 166,  74,  34, 237, 107, 200, 243, 139, 241, 147,  99,  34, 101,  73},
          {234, 157, 175, 116,  51,  73,  93, 125, 125, 110,  96, 252, 209,  60, 169, 138},
          {248, 224, 173,  23,  55, 103,  90,  71, 125, 224,  46, 212,  11,  25, 102, 151},
          {183, 199, 239,  93,  17, 125,   4, 109,  55,  89, 196, 136, 164,  76, 159, 223},
          {172,  12,  50,  36, 123, 102, 127, 119, 251,  46,  85,  25,  64, 140, 111, 163},
          {120,  19,  56, 222,  99, 198, 180, 115, 150, 168, 222, 198,  91, 154,  35, 133},
        }},
        {{
          {  0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
          {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
          {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255,   0,   0},
          {  0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
          {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
          {  0,   0,   0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0},
          {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255},
          {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
          {255,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
          {  0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
          {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
          {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
          {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
          {  0,   0,   0,   0,   0,   0, 255,   0,   0,   0,   0,   0,   0,   0,   0,   0},
          {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
          {  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0},
        }}
        // clang-format on
    };
    return kTestElements;
  }

  // float and uint8_t behave differently around their max values.
  template <typename T,
            std::enable_if_t<std::is_same_v<uint8_t, T>, bool> = true>
  const Elements& get_max_test_elements() {
    static const Elements kTestElements = {
        // clang-format off
        2, 3,
        max<T>() - 100, max<T>() - 1,
        {{
          {max<T>() - 105,           10},
          {max<T>() - 100, max<T>() - 2},
          {  max<T>() - 1,     max<T>()},
        }},
        {{
          {  0,   0},
          {255, 255},
          {255,   0},
        }}
        // clang-format on
    };
    return kTestElements;
  }

  // lower_bound > upperbound -> no values within bounds
  const Elements& get_empty_interval_test_elements() {
    static const Elements kTestElements = {
        // clang-format off
        3, 1,
        13, 0,
        {{
          {12, 13, 14}
        }},
        {{
          { 0, 0, 0}
        }}
        // clang-format on
    };
    return kTestElements;
  }

  const Elements& get_same_bounds_test_elements() {
    static const Elements kTestElements = {
        // clang-format off
        3, 1,
        13, 13,
        {{
          {12, 13, 14}
        }},
        {{
          { 0, 255, 0}
        }}
        // clang-format on
    };
    return kTestElements;
  }

  static uint8_t increment(uint8_t x) { return x + 1; }

  static float increment(float x) { return std::nextafter(x, FLT_MAX); }

  const Elements& get_min_test_elements() {
    static const Elements kTestElements = {
        // clang-format off
        3, 1,
        lowest<ElementType>(), increment(lowest<ElementType>()),
        {{
          {lowest<ElementType>(),
           increment(lowest<ElementType>()),
           increment(increment(lowest<ElementType>()))}
        }},
        {{
          { 255, 255, 0}
        }}
        // clang-format on
    };
    return kTestElements;
  }

  const Elements& get_zero_one_test_elements() {
    static const Elements kTestElements = {
        // clang-format off
        3, 1,
        0, 1,
        {{
          {0, 1, 2}
        }},
        {{
          { 255, 255, 0}
        }}
        // clang-format on
    };
    return kTestElements;
  }

  // minimum_size set by caller to trigger the 'big' conversion path.
  template <typename T>
  void test_linear(size_t width, size_t minimum_size = 1, T lower_bound = 50,
                   T upper_bound = 100) {
    size_t height = 0;
    height = get_linear_height<uint8_t>(width, minimum_size);

    auto arrays = get_linear_arrays<T>(width, height);

    test::Array2D<T>& source = std::get<0>(arrays);
    test::Array2D<uint8_t>& expected = std::get<1>(arrays);
    test::Array2D<uint8_t>& actual = std::get<2>(arrays);

    ASSERT_EQ(KLEIDICV_OK, (in_range<T>()(source.data(), source.stride(),
                                          actual.data(), actual.stride(), width,
                                          height, lower_bound, upper_bound)));

    EXPECT_EQ_ARRAY2D(expected, actual);
  }

  void test(const Elements& elements_list) {
    const size_t width = elements_list.width;
    const size_t height = elements_list.height;
    const ElementType lower_bound = elements_list.lower_bound;
    const ElementType upper_bound = elements_list.upper_bound;

    test::Array2D<ElementType> source(width, height);
    test::Array2D<uint8_t> expected(width, height);
    test::Array2D<uint8_t> actual(width, height);

    for (size_t i = 0; i < height; i++) {
      source.set(i, 0, elements_list.source_rows[i]);
      expected.set(i, 0, elements_list.expected_rows[i]);
    }

    ASSERT_EQ(KLEIDICV_OK,
              (in_range<ElementType>()(source.data(), source.stride(),
                                       actual.data(), actual.stride(), width,
                                       height, lower_bound, upper_bound)));

    EXPECT_EQ_ARRAY2D(expected, actual);
  }

  InRangeTest<ElementType>& with_paddings(
      std::initializer_list<size_t> inputs_padding,
      std::initializer_list<size_t> outputs_padding) {
    size_t i = 0;
    for (size_t p : inputs_padding) {
      inputs_padding_[i++] = p;
    }
    size_t j = 0;
    for (size_t q : outputs_padding) {
      outputs_padding_[j++] = q;
    }
    return *this;
  }

  InRangeTest() {
    inputs_padding_.fill(0);
    outputs_padding_.fill(0);
  }

  virtual ~InRangeTest() = default;
};

template <typename ElementType>
class InRange : public testing::Test {};

using ElementTypes = ::testing::Types<uint8_t, float>;

TYPED_TEST_SUITE(InRange, ElementTypes);

// Tests various padding combinations.
TYPED_TEST(InRange, Padding) {
  auto elements_list = InRangeTest<TypeParam>{}.get_same_bounds_test_elements();
  InRangeTest<TypeParam>{}.with_paddings({0}, {0}).test(elements_list);
  InRangeTest<TypeParam>{}.with_paddings({0}, {1}).test(elements_list);
  InRangeTest<TypeParam>{}.with_paddings({1}, {0}).test(elements_list);
  InRangeTest<TypeParam>{}.with_paddings({1}, {1}).test(elements_list);
}

TYPED_TEST(InRange, Scalar) {
  InRangeTest<TypeParam>{}.template test_linear<TypeParam>(
      test::Options::vector_length() - 1);
}

TYPED_TEST(InRange, Vector) {
  InRangeTest<TypeParam>{}.template test_linear<TypeParam>(
      test::Options::vector_length() * 2);
}

TYPED_TEST(InRange, TestEmptyInterval) {
  auto elements_list =
      InRangeTest<TypeParam>{}.get_empty_interval_test_elements();
  InRangeTest<TypeParam>{}.test(elements_list);
}

TYPED_TEST(InRange, TestSameBounds) {
  auto elements_list = InRangeTest<TypeParam>{}.get_same_bounds_test_elements();
  InRangeTest<TypeParam>{}.test(elements_list);
}

TYPED_TEST(InRange, TestCustom) {
  auto elements_list =
      InRangeTest<TypeParam>{}.template get_custom_test_elements<TypeParam>();
  InRangeTest<TypeParam>{}.test(elements_list);
}

TYPED_TEST(InRange, TestMax) {
  auto elements_list =
      InRangeTest<TypeParam>{}.template get_max_test_elements<TypeParam>();
  InRangeTest<TypeParam>{}.test(elements_list);
}

TYPED_TEST(InRange, TestMin) {
  auto elements_list = InRangeTest<TypeParam>{}.get_min_test_elements();
  InRangeTest<TypeParam>{}.test(elements_list);
}

TYPED_TEST(InRange, TestZeroOne) {
  auto elements_list = InRangeTest<TypeParam>{}.get_zero_one_test_elements();
  InRangeTest<TypeParam>{}.test(elements_list);
}

TYPED_TEST(InRange, NullPointer) {
  const TypeParam src[1] = {};
  uint8_t dst[1];
  test::test_null_args(in_range<TypeParam>(), src, sizeof(uint8_t), dst,
                       sizeof(TypeParam), 1, 1, 1, 1);
}

TYPED_TEST(InRange, Misalignment) {
  if (sizeof(TypeParam) == 1) {
    // misalignment impossible
    return;
  }
  TypeParam src[2] = {};
  uint8_t dst[2];
  EXPECT_EQ(KLEIDICV_ERROR_ALIGNMENT,
            in_range<TypeParam>()(src, sizeof(TypeParam) + 1, dst,
                                  sizeof(uint8_t), 2, 2, 1, 1));
}

TYPED_TEST(InRange, ZeroImageSize) {
  TypeParam src[1] = {};
  uint8_t dst[1];
  EXPECT_EQ(KLEIDICV_OK, in_range<TypeParam>()(src, sizeof(TypeParam), dst,
                                               sizeof(uint8_t), 0, 1, 1, 1));
  EXPECT_EQ(KLEIDICV_OK, in_range<TypeParam>()(src, sizeof(TypeParam), dst,
                                               sizeof(uint8_t), 1, 0, 1, 1));
}

TYPED_TEST(InRange, OversizeImage) {
  TypeParam src[1] = {};
  uint8_t dst[1];
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            in_range<TypeParam>()(src, sizeof(TypeParam), dst, sizeof(uint8_t),
                                  KLEIDICV_MAX_IMAGE_PIXELS + 1, 1, 1, 1));
  EXPECT_EQ(KLEIDICV_ERROR_RANGE,
            in_range<TypeParam>()(src, sizeof(TypeParam), dst, sizeof(uint8_t),
                                  KLEIDICV_MAX_IMAGE_PIXELS,
                                  KLEIDICV_MAX_IMAGE_PIXELS, 1, 1));
}
