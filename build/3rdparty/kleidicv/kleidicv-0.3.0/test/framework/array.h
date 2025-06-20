// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_TEST_FRAMEWORK_ARRAY_H_
#define KLEIDICV_TEST_FRAMEWORK_ARRAY_H_

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "framework/abstract.h"
#include "framework/types.h"
#include "framework/utils.h"

namespace test {

// A simple two-dimensional array representation.
template <typename ElementType>
class Array2D : public TwoDimensional<ElementType> {
 public:
  Array2D() = default;

  explicit Array2D(size_t width, size_t height) : Array2D(width, height, 0) {}

  explicit Array2D(size_t width, size_t height, size_t padding)
      : Array2D(width, height, padding, 1) {}

  explicit Array2D(ArrayLayout layout)
      : Array2D(layout.width, layout.height, layout.padding, layout.channels) {}

  explicit Array2D(size_t width, size_t height, size_t padding, size_t channels)
      : width_{width},
        height_{height},
        channels_{channels},
        stride_{(width + padding) * sizeof(ElementType)} {
    try_allocate();
    fill_padding();
  }

  // Destructor checks that padding bytes are not overwritten.
  ~Array2D() override { check_padding(); }

  // Copy constructor.
  Array2D(const Array2D &other) { this->operator=(other); }

  // Move constructor.
  Array2D(Array2D &&other) noexcept { this->operator=(other); }

  // Copy assignment operator.
  Array2D &operator=(const Array2D &other) {
    if (this == &other) {
      return *this;
    }
    width_ = other.width_;
    height_ = other.height_;
    channels_ = other.channels_;
    stride_ = other.stride_;

    try_allocate();

    EXPECT_TRUE(valid());
    if (valid()) {
      std::memcpy(data(), other.data(), height_ * stride_);
    }

    return *this;
  }

  // Move assignment operator.
  Array2D &operator=(Array2D &&other) noexcept {
    if (this == &other) {
      return *this;
    }
    buffer_ = std::move(other.buffer_);
    data_ = other.data_;
    width_ = other.width_;
    height_ = other.height_;
    channels_ = other.channels_;
    stride_ = other.stride_;
    other.reset();
    return *this;
  }

  // Fills the underlying memory range with a given value skipping padding
  // bytes.
  void fill(ElementType value) {
    ASSERT_EQ(valid(), true);

    ElementType *ptr = data();
    for (size_t row = 0; row < height(); ++row) {
      for (size_t column = 0; column < width(); ++column) {
        ptr[column] = value;
      }

      ptr = add_stride(ptr, 1);
    }
  }

  // Fills the underlying memory range with a given generator skipping padding
  // bytes.
  void fill(Generator<ElementType> &generator) {
    ASSERT_EQ(valid(), true);

    ElementType *ptr = data();
    for (size_t row = 0; row < height(); ++row) {
      for (size_t column = 0; column < width(); ++column) {
        std::optional<ElementType> optional_value = generator.next();
        ASSERT_NE(optional_value, std::nullopt);
        if (optional_value.has_value()) {
          ptr[column] = optional_value.value();
        }
      }

      ptr = add_stride(ptr, 1);
    }
  }

  // Fills the underlying memory range with the output of a caller provided
  // callable object skipping padding bytes.
  void fill(const std::function<std::optional<ElementType>(size_t, size_t)>
                &value_at) {
    ASSERT_EQ(valid(), true);

    ElementType *ptr = data();
    for (size_t row = 0; row < height(); ++row) {
      for (size_t column = 0; column < width(); ++column) {
        std::optional<ElementType> optional_value = value_at(row, column);
        ASSERT_NE(optional_value, std::nullopt);
        if (optional_value.has_value()) {
          ptr[column] = optional_value.value();
        }
      }

      ptr = add_stride(ptr, 1);
    }
  }

  // Sets values in a row starting at a given column.
  void set(size_t row, size_t column,
           std::initializer_list<ElementType> values) {
    ASSERT_EQ(valid(), true) << "Array is invalid.";
    ASSERT_GE(width() - column, values.size());

    ElementType *ptr = at(row, column);
    if (!ptr) {
      return;
    }

    size_t index = 0;
    for (ElementType value : values) {
      ptr[index++] = value;
    }
  }

  // Sets values in a row starting at a given column from a const vector.
  void set(size_t row, size_t column, const std::vector<ElementType> &values) {
    ASSERT_EQ(valid(), true) << "Array is invalid.";
    ASSERT_GE(width() - column, values.size());

    ElementType *ptr = at(row, column);
    if (!ptr) {
      return;
    }

    size_t index = 0;
    for (ElementType value : values) {
      ptr[index++] = value;
    }
  }

  // Sets values starting in a given row starting at a given column.
  //
  // The layout of the input TwoDimensional object is not altered, meaning that
  // it must fit as-is at the given row and column position into the array.
  void set(size_t row, size_t column,
           const TwoDimensional<ElementType> *elements) {
    ASSERT_NE(elements, nullptr);
    ASSERT_GE(width(), column);
    ASSERT_GE(height(), row);
    ASSERT_GE(width() - column, elements->width());
    ASSERT_GE(height() - row, elements->height());

    for (size_t row_offset = 0; row_offset < elements->height(); ++row_offset) {
      const ElementType *src = elements->at(row_offset, 0);
      ElementType *dst = at(row + row_offset, column);
      std::memcpy(dst, src, elements->width() * sizeof(ElementType));
    }
  }

  // Compares two instances for equality considering only element bytes.
  // Returns the location of the first mismatch, if any.
  std::optional<std::tuple<size_t, size_t>> compare_to(
      const Array2D<ElementType> &other) const {
    for (size_t row = 0; row < height(); ++row) {
      for (size_t column = 0; column < width(); ++column) {
        const ElementType *lhs = at(row, column);
        const ElementType *rhs = other.at(row, column);
        if (!lhs || !rhs || (lhs[0] != rhs[0])) {
          return std::make_tuple(row, column);
        }
      }
    }

    return std::nullopt;
  }

  // Returns a pointer to the first element.
  ElementType *data() { return data_; }

  // Returns a const pointer to the first element.
  const ElementType *data() const { return data_; }

  // Returns the width of this array.
  size_t width() const override { return width_; }

  // Returns the height of this array.
  size_t height() const override { return height_; }

  // Returns the number of channels.
  size_t channels() const override { return channels_; };

  // Returns the stride of this array.
  size_t stride() const { return stride_; }

  // Returns true if this object holds actual memory, otherwise false.
  bool valid() const { return data() != nullptr; }

  // Returns a pointer to a data element at a given row and column position, or
  // nullptr if the requested position is invalid.
  ElementType *at(size_t row, size_t column) override {
    return const_cast<ElementType *>(
        const_cast<const Array2D<ElementType> *>(this)->at(row, column));
  }

  // Returns a const pointer to a data element at a given row and column
  // position, or nullptr if the requested position is invalid.
  const ElementType *at(size_t row, size_t column) const override {
    if (!check_access(row, column)) {
      TEST_FAIL_WITH(nullptr,
                     "Access is either out-of-bounds or the array is invalid.");
    }

    const ElementType *ptr = add_stride(data(), row);
    return &ptr[column];
  }

 private:
  // Returns the number of elements between the end of one row and the start of
  // the next row.
  size_t padding() { return stride() / sizeof(ElementType) - width(); }

  // Returns the offset to the first padding byte within a row.
  size_t padding_offset() const { return width() * sizeof(ElementType); }

  // Returns true if a row has padding, otherwise false.
  size_t has_padding() const { return padding_offset() != stride(); }

  // Checks that an access is valid or not.
  bool check_access(size_t row, size_t column) const {
    return valid() && (row < height()) && (column < width());
  }

  // Fills padding bytes, if present.
  void fill_padding() {
    if (!valid() || !has_padding()) {
      return;
    }

    uint8_t *ptr = reinterpret_cast<uint8_t *>(data());
    for (size_t row = 0; row < height(); ++row) {
      for (size_t column = padding_offset(); column < stride(); ++column) {
        ptr[column] = kPaddingValue;
      }

      ptr += stride();
    }
  }

  // Checks for clobbered padding bytes, if present.
  void check_padding() const {
    if (!valid() || !has_padding()) {
      return;
    }

    const uint8_t *ptr = reinterpret_cast<const uint8_t *>(data());
    for (size_t row = 0; row < height(); ++row) {
      for (size_t offset = padding_offset(); offset < stride(); ++offset) {
        if (ptr[offset] != kPaddingValue) {
          GTEST_FAIL() << "Padding byte was overwritten at (row=" << row
                       << ", offset=" << offset << ")" << std::endl;
        }
      }

      ptr += stride();
    }
  }

  // Adds stride to a pointer.
  ElementType *add_stride(ElementType *ptr, size_t count) {
    char *address = reinterpret_cast<char *>(ptr);
    address += count * stride();
    return reinterpret_cast<ElementType *>(address);
  }

  // Adds stride to a pointer.
  const ElementType *add_stride(const ElementType *ptr, size_t count) const {
    const char *address = reinterpret_cast<const char *>(ptr);
    address += count * stride();
    return reinterpret_cast<const ElementType *>(address);
  }

  // Resets the instance to the default instance.
  void reset() {
    width_ = height_ = channels_ = stride_ = 0;
    data_ = nullptr;
    buffer_.reset();
  }

  // Tries to allocate backing memory.
  void try_allocate() {
    size_t element_count = height_ * (width_ + padding());
    // Allocate extra to allow weakening alignment.
    size_t allocation_count = element_count + 1;

    try {
      buffer_ =
          std::unique_ptr<ElementType[]>(new ElementType[allocation_count]);
      // Weaken alignment to flush out potential alignment issues.
      // buffer_.get() will contain a pointer that is at least 16-byte aligned.
      // By adding a small offset to that we get a pointer that is only aligned
      // to sizeof(ElementType).
      data_ = buffer_.get() + 1;
    } catch (...) {
      reset();
      GTEST_FAIL() << "Failed to allocate memory of " << allocation_count
                   << " bytes";
    }
  }

  // Constant value of row padding bytes.
  static constexpr uint8_t kPaddingValue = std::numeric_limits<uint8_t>::max();

  // Smart pointer to the managed memory.
  std::unique_ptr<ElementType[]> buffer_;
  // Pointer to the start of the data. This is offset from the start of buffer_
  // to flush out potential alignment issues.
  ElementType *data_{nullptr};
  // Width a row in the array.
  size_t width_{0};
  // Number of rows in the array.
  size_t height_{0};
  // Number of channels.
  size_t channels_{0};
  // Stride in bytes between the first elements of two consecutive rows.
  size_t stride_{0};
};  // end of class Array2D<ElementType>

// Compares two Array2D objects for equality.
// Unary + is used to ensure values are printed as integers, not chars
#define EXPECT_EQ_ARRAY2D(lhs, rhs)                                 \
  do {                                                              \
    ASSERT_EQ((lhs).width(), (rhs).width())                         \
        << "Mismatch in width." << std::endl;                       \
    ASSERT_EQ((lhs).height(), (rhs).height())                       \
        << "Mismatch in height." << std::endl;                      \
    ASSERT_EQ((lhs).channels(), (rhs).channels())                   \
        << "Mismatch in channels." << std::endl;                    \
    auto mismatch = (lhs).compare_to((rhs));                        \
    if (mismatch) {                                                 \
      auto [row, col] = *mismatch;                                  \
      GTEST_FAIL() << "Mismatch at (row=" << row << ", col=" << col \
                   << "): " << std::hex << std::showbase            \
                   << +(lhs).at(row, col)[0] << " vs "              \
                   << +(rhs).at(row, col)[0] << "." << std::endl;   \
    }                                                               \
  } while (0 != 0)

// Compares two Array2D objects for inequality.
#define EXPECT_NE_ARRAY2D(lhs, rhs)                                \
  do {                                                             \
    ASSERT_EQ((lhs).width(), (rhs).width())                        \
        << "Mismatch in width." << std::endl;                      \
    ASSERT_EQ((lhs).height(), (rhs).height())                      \
        << "Mismatch in height." << std::endl;                     \
    ASSERT_EQ((lhs).channels(), (rhs).channels())                  \
        << "Mismatch in channels." << std::endl;                   \
    auto mismatch = (lhs).compare_to((rhs));                       \
    if (!mismatch) {                                               \
      GTEST_FAIL() << "Objects are equal, but expected to differ." \
                   << std::endl;                                   \
    }                                                              \
  } while (0 != 0)

}  // namespace test

#endif  // KLEIDICV_TEST_FRAMEWORK_ARRAY_H_
