// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_TYPES_H
#define KLEIDICV_TYPES_H

#include <cstring>
#include <memory>
#include <utility>

#include "kleidicv/config.h"
#include "kleidicv/ctypes.h"
#include "kleidicv/utils.h"

namespace KLEIDICV_TARGET_NAMESPACE {

// Represents a point on a 2D plane.
class Point final {
 public:
  explicit Point(size_t x, size_t y) KLEIDICV_STREAMING_COMPATIBLE : x_{x},
                                                                     y_{y} {}

  size_t x() const KLEIDICV_STREAMING_COMPATIBLE { return x_; }
  size_t y() const KLEIDICV_STREAMING_COMPATIBLE { return y_; }

 private:
  size_t x_;
  size_t y_;
};  // end of class Point

// Represents an area given by its width and height.
class Rectangle final {
 public:
  explicit Rectangle(size_t width, size_t height) KLEIDICV_STREAMING_COMPATIBLE
      : width_(width),
        height_(height) {}

  explicit Rectangle(int width, int height) KLEIDICV_STREAMING_COMPATIBLE
      : Rectangle(static_cast<size_t>(width), static_cast<size_t>(height)) {}

  explicit Rectangle(kleidicv_rectangle_t rect) KLEIDICV_STREAMING_COMPATIBLE
      : Rectangle(rect.width, rect.height) {}

  size_t width() const KLEIDICV_STREAMING_COMPATIBLE { return width_; }
  size_t height() const KLEIDICV_STREAMING_COMPATIBLE { return height_; }
  size_t area() const KLEIDICV_STREAMING_COMPATIBLE {
    return width() * height();
  }

  void flatten() KLEIDICV_STREAMING_COMPATIBLE {
    width_ = area();
    height_ = 1;
  }

  bool operator==(const Rectangle &rhs) const KLEIDICV_STREAMING_COMPATIBLE {
    return width() == rhs.width() && height() == rhs.height();
  }

  bool operator!=(const Rectangle &rhs) const KLEIDICV_STREAMING_COMPATIBLE {
    return !operator==(rhs);
  }

 private:
  size_t width_;
  size_t height_;
};  // end of class Rectangle

// Represents margins around a two dimensional area.
class Margin final {
 public:
  explicit constexpr Margin(size_t left, size_t top, size_t right,
                            size_t bottom) KLEIDICV_STREAMING_COMPATIBLE
      : left_(left),
        top_(top),
        right_(right),
        bottom_(bottom) {}

  explicit constexpr Margin(size_t margin) KLEIDICV_STREAMING_COMPATIBLE
      : left_(margin),
        top_(margin),
        right_(margin),
        bottom_(margin) {}

  explicit Margin(kleidicv_rectangle_t kernel,
                  kleidicv_point_t anchor) KLEIDICV_STREAMING_COMPATIBLE
      : Margin(anchor.x, anchor.y, kernel.width - anchor.x - 1,
               kernel.height - anchor.y - 1) {}

  explicit Margin(Rectangle kernel, Point anchor) KLEIDICV_STREAMING_COMPATIBLE
      : Margin(anchor.x(), anchor.y(), kernel.width() - anchor.x() - 1,
               kernel.height() - anchor.y() - 1) {}

  size_t left() const KLEIDICV_STREAMING_COMPATIBLE { return left_; }
  size_t top() const KLEIDICV_STREAMING_COMPATIBLE { return top_; }
  size_t right() const KLEIDICV_STREAMING_COMPATIBLE { return right_; }
  size_t bottom() const KLEIDICV_STREAMING_COMPATIBLE { return bottom_; }

 private:
  size_t left_;
  size_t top_;
  size_t right_;
  size_t bottom_;
};  // end of class Margin

// Represents constant border values around a two dimensional area.
template <typename T>
class Border final {
 public:
  Border() KLEIDICV_STREAMING_COMPATIBLE = default;

  Border(double left, double top, double right,
         double bottom) KLEIDICV_STREAMING_COMPATIBLE
      : left_{saturating_cast<double, T>(left)},
        top_{saturating_cast<double, T>(top)},
        right_{saturating_cast<double, T>(right)},
        bottom_{saturating_cast<double, T>(bottom)} {}

  explicit Border(kleidicv_border_values_t border_values)
      KLEIDICV_STREAMING_COMPATIBLE
      : Border(border_values.left, border_values.top, border_values.right,
               border_values.bottom) {}

  T left() const KLEIDICV_STREAMING_COMPATIBLE { return left_; }
  T top() const KLEIDICV_STREAMING_COMPATIBLE { return top_; }
  T right() const KLEIDICV_STREAMING_COMPATIBLE { return right_; }
  T bottom() const KLEIDICV_STREAMING_COMPATIBLE { return bottom_; }

 private:
  T left_;
  T top_;
  T right_;
  T bottom_;
};  // end of class Border<T>

// Describes the layout of one row given by a base pointer and channel count.
template <typename T>
class Columns final {
 public:
  explicit Columns(T *ptr, size_t channels) KLEIDICV_STREAMING_COMPATIBLE
      : ptr_{ptr},
        channels_{channels} {}

  // Subscript operator to return an arbitrary column at an index. To account
  // for channel count use at() method.
  T &operator[](ptrdiff_t index) KLEIDICV_STREAMING_COMPATIBLE {
    return ptr_[index];
  }

  // Addition assignment operator to step across the columns.
  Columns &operator+=(ptrdiff_t diff) KLEIDICV_STREAMING_COMPATIBLE {
    ptr_ += static_cast<ptrdiff_t>(channels()) * diff;
    return *this;
  }

  // Subtraction assignment operator to step across the columns.
  Columns &operator-=(ptrdiff_t diff) KLEIDICV_STREAMING_COMPATIBLE {
    ptr_ -= static_cast<ptrdiff_t>(channels()) * diff;
    return *this;
  }

  // Prefix increment operator to advance to the next column.
  Columns &operator++() KLEIDICV_STREAMING_COMPATIBLE { return operator+=(1); }

  // NOLINTBEGIN(hicpp-explicit-conversions)
  // Implicit conversion operator from Columns<T> to Columns<const T>.
  [[nodiscard]] operator Columns<const T>() const
      KLEIDICV_STREAMING_COMPATIBLE {
    return Columns<const T>{ptr_, channels()};
  }
  // NOLINTEND(hicpp-explicit-conversions)

  // Returns a new instance at a given column.
  [[nodiscard]] Columns<T> at(ptrdiff_t column) KLEIDICV_STREAMING_COMPATIBLE {
    return Columns<T>{&ptr_[column * static_cast<ptrdiff_t>(channels())],
                      channels()};
  }

  // Returns the number of channels in a row.
  size_t channels() const KLEIDICV_STREAMING_COMPATIBLE { return channels_; }

 private:
  // Pointer to the current position.
  T *ptr_;
  // Number of channels within a row.
  size_t channels_;
};  // end of class Columns<T>

// Describes the layout of one row given by a base pointer and channel count.
template <typename T>
class ParallelColumns final {
 public:
  explicit ParallelColumns(Columns<T> columns_0,
                           Columns<T> columns_1) KLEIDICV_STREAMING_COMPATIBLE
      : columns_{columns_0, columns_1} {}

  // Addition assignment operator to step across the columns.
  ParallelColumns &operator+=(ptrdiff_t diff) KLEIDICV_STREAMING_COMPATIBLE {
    columns_[0] += diff;
    columns_[1] += diff;
    return *this;
  }

  // Subtraction assignment operator to navigate among rows.
  ParallelColumns &operator-=(ptrdiff_t diff) KLEIDICV_STREAMING_COMPATIBLE {
    return operator+=(-1 * diff);
  }

  // Prefix increment operator to advance to the next column.
  ParallelColumns &operator++() KLEIDICV_STREAMING_COMPATIBLE {
    return operator+=(1);
  }

  // Returns the columns belonging to the first row.
  [[nodiscard]] Columns<T> first() const KLEIDICV_STREAMING_COMPATIBLE {
    return columns_[0];
  }

  // Returns the columns belonging to the second row.
  [[nodiscard]] Columns<T> second() const KLEIDICV_STREAMING_COMPATIBLE {
    return columns_[1];
  }

 private:
  // The columns this instance handles.
  Columns<T> columns_[2];
};  // end of class ParallelColumns<T>

// Base class of different row implementations.
template <typename T>
class RowBase {
 public:
  // Returns the distance in bytes between two consecutive rows.
  size_t stride() const KLEIDICV_STREAMING_COMPATIBLE { return stride_; }

  // Returns the number of channels in a row.
  size_t channels() const KLEIDICV_STREAMING_COMPATIBLE { return channels_; }

  // Returns true if rows are continuous for a given length, otherwise false.
  bool is_continuous(size_t length) const KLEIDICV_STREAMING_COMPATIBLE {
    return stride() == (length * channels() * sizeof(T));
  }

  // When handling multiple rows this switches to a single row in an
  // implementation defined way.
  void make_single_row() const KLEIDICV_STREAMING_COMPATIBLE {}

  // Returns false if is_continuous() always returns false, otherwise true.
  static constexpr bool maybe_continuous() KLEIDICV_STREAMING_COMPATIBLE {
    return true;
  }

 protected:
  // TODO: default initialise members.
  // NOLINTBEGIN(hicpp-member-init)
  //  The default constructor creates an uninitialized instance.
  RowBase() KLEIDICV_STREAMING_COMPATIBLE = default;
  // NOLINTEND(hicpp-member-init)

  RowBase(size_t stride, size_t channels) KLEIDICV_STREAMING_COMPATIBLE
      : stride_(stride),
        channels_(channels) {}

  // Adds a stride to a pointer, and returns the new pointer.
  template <typename P>
  [[nodiscard]] static P *add_stride(P *ptr, ptrdiff_t stride)
      KLEIDICV_STREAMING_COMPATIBLE {
    uintptr_t intptr = reinterpret_cast<uintptr_t>(ptr);
    intptr += stride;
    // NOLINTBEGIN(performance-no-int-to-ptr)
    return reinterpret_cast<P *>(intptr);
    // NOLINTEND(performance-no-int-to-ptr)
  }

  // Subtracts a stride to a pointer, and returns the new pointer.
  template <typename P>
  [[nodiscard]] static P *subtract_stride(P *ptr, ptrdiff_t stride)
      KLEIDICV_STREAMING_COMPATIBLE {
    uintptr_t intptr = reinterpret_cast<uintptr_t>(ptr);
    intptr -= stride;
    // NOLINTBEGIN(performance-no-int-to-ptr)
    return reinterpret_cast<P *>(intptr);
    // NOLINTEND(performance-no-int-to-ptr)
  }

 private:
  // Distance in bytes between two consecutive rows.
  size_t stride_;
  // Number of channels within a row.
  size_t channels_;
};  // end of class RowBase<T>

// Describes the layout of rows given by a base pointer, channel count and a
// stride in bytes.
template <typename T>
class Rows final : public RowBase<T> {
 public:
  // Shorten code: no need for 'this->'.
  using RowBase<T>::channels;
  using RowBase<T>::stride;

  // The default constructor creates an uninitialized instance.
  Rows() KLEIDICV_STREAMING_COMPATIBLE : RowBase<T>() {}

  explicit Rows(T *ptr, size_t stride,
                size_t channels) KLEIDICV_STREAMING_COMPATIBLE
      : RowBase<T>(stride, channels),
        ptr_{ptr} {}

  explicit Rows(T *ptr, size_t stride) KLEIDICV_STREAMING_COMPATIBLE
      : Rows(ptr, stride, 1) {}

  explicit Rows(T *ptr) KLEIDICV_STREAMING_COMPATIBLE : Rows(ptr, 0, 0) {}

  // Subscript operator to return an arbitrary position within the current row.
  // To account for stride and channel count use at() method.
  T &operator[](ptrdiff_t index) KLEIDICV_STREAMING_COMPATIBLE {
    return ptr_[index];
  }

  // Addition assignment operator to navigate among rows.
  Rows<T> &operator+=(ptrdiff_t diff) KLEIDICV_STREAMING_COMPATIBLE {
    ptr_ = get_pointer_at(diff);
    return *this;
  }

  // Prefix increment operator to advance to the next row.
  Rows<T> &operator++() KLEIDICV_STREAMING_COMPATIBLE { return operator+=(1); }

  // NOLINTBEGIN(hicpp-explicit-conversions)
  // Returns a const variant of this instance.
  [[nodiscard]] operator Rows<const T>() KLEIDICV_STREAMING_COMPATIBLE {
    return Rows<const T>{ptr_, stride(), channels()};
  }
  // NOLINTEND(hicpp-explicit-conversions)

  // Returns a new instance at a given row and column.
  [[nodiscard]] Rows<T> at(ptrdiff_t row,
                           ptrdiff_t column = 0) KLEIDICV_STREAMING_COMPATIBLE {
    return Rows<T>{get_pointer_at(row, column), stride(), channels()};
  }

  // Returns a view on columns within the current row.
  [[nodiscard]] Columns<T> as_columns() const KLEIDICV_STREAMING_COMPATIBLE {
    return Columns{ptr_, channels()};
  }

  // Translates a logical one-dimensional element index into physical byte
  // offset for that element with a given row width.
  [[nodiscard]] size_t offset_for_index(size_t index, size_t width) const
      KLEIDICV_STREAMING_COMPATIBLE {
    size_t row = index / width;
    size_t column = index % width;
    return row * stride() + column * sizeof(T);
  }

 private:
  // Returns a column in a row at a given index taking stride and channels into
  // account.
  [[nodiscard]] T *get_pointer_at(ptrdiff_t row, ptrdiff_t column = 0)
      KLEIDICV_STREAMING_COMPATIBLE {
    T *ptr =
        RowBase<T>::add_stride(ptr_, row * static_cast<ptrdiff_t>(stride()));
    return &ptr[column * static_cast<ptrdiff_t>(channels())];
  }

  // Pointer to the first row.
  T *ptr_;
};  // end of class Rows<T>

// Similar to Rows<T>, but in this case rows are indirectly addressed.
template <typename T>
class IndirectRows : public RowBase<T> {
 public:
  // Shorten code: no need for 'this->'.
  using RowBase<T>::channels;
  using RowBase<T>::stride;

  // The default constructor creates an uninitialized instance.
  IndirectRows() KLEIDICV_STREAMING_COMPATIBLE : RowBase<T>() {}

  explicit IndirectRows(T **ptr_storage, size_t stride,
                        size_t channels) KLEIDICV_STREAMING_COMPATIBLE
      : RowBase<T>(stride, channels),
        ptr_storage_(ptr_storage) {}

  explicit IndirectRows(T **ptr_storage, size_t depth,
                        Rows<T> rows) KLEIDICV_STREAMING_COMPATIBLE
      : RowBase<T>(rows.stride(), rows.channels()),
        ptr_storage_(ptr_storage) {
    for (size_t index = 0; index < depth; ++index) {
      ptr_storage_[index] = &rows.at(index, 0)[0];
    }
  }

  // Subscript operator to return a position within the current row. To account
  // for stride and channel count use at() method.
  T &operator[](ptrdiff_t index) KLEIDICV_STREAMING_COMPATIBLE {
    return ptr_storage_[0][index];
  }

  // Addition assignment operator to navigate among rows.
  IndirectRows<T> &operator+=(ptrdiff_t diff) KLEIDICV_STREAMING_COMPATIBLE {
    ptr_storage_ += diff;
    return *this;
  }

  // Prefix increment operator to advance to the next row.
  IndirectRows<T> &operator++() KLEIDICV_STREAMING_COMPATIBLE {
    return this->operator+=(1);
  }

  // Returns a new instance at a given row and column.
  [[nodiscard]] Rows<T> at(ptrdiff_t row,
                           ptrdiff_t column = 0) KLEIDICV_STREAMING_COMPATIBLE {
    auto rows = Rows<T>{ptr_storage_[row], stride(), channels()};
    return rows.at(0, column);
  }

  // Returns a view on columns within the current row.
  [[nodiscard]] Columns<T> as_columns() const KLEIDICV_STREAMING_COMPATIBLE {
    return Columns{ptr_storage_[0], channels()};
  }

 protected:
  // Pointer to the pointer storage.
  T **ptr_storage_;
};  // end of class IndirectRows<T>

// Same as IndirectRows<T> but with double buffering. Requires 3 times the depth
// of pointers.
template <typename T>
class DoubleBufferedIndirectRows final : public IndirectRows<T> {
 public:
  // Shorten code: no need for 'this->'.
  using IndirectRows<T>::channels;
  using IndirectRows<T>::stride;

  explicit DoubleBufferedIndirectRows(
      T **ptr_storage, size_t depth, Rows<T> rows) KLEIDICV_STREAMING_COMPATIBLE
      : IndirectRows<T>(ptr_storage, 2 * depth, rows) {
    // Fill the second half of the pointer storage.
    for (size_t index = 0; index < 2 * depth; ++index) {
      this->ptr_storage_[2 * depth + index] = this->ptr_storage_[index];
    }

    db_ptr_storage_[0] = &this->ptr_storage_[0];
    db_ptr_storage_[1] = &this->ptr_storage_[depth];
  }

  // Swaps the double buffered indirect rows.
  void swap() KLEIDICV_STREAMING_COMPATIBLE {
    std::swap(db_ptr_storage_[0], db_ptr_storage_[1]);
  }

  // Returns indirect rows where write is allowed.
  [[nodiscard]] IndirectRows<T> write_at() KLEIDICV_STREAMING_COMPATIBLE {
    return IndirectRows<T>{db_ptr_storage_[0], stride(), channels()};
  }

  // Returns indirect rows where read is allowed.
  [[nodiscard]] IndirectRows<T> read_at() KLEIDICV_STREAMING_COMPATIBLE {
    return IndirectRows<T>{db_ptr_storage_[1], stride(), channels()};
  }

 private:
  // The double buffer.
  T **db_ptr_storage_[2];
};  // end of class DoubleBufferedIndirectRows<T>

// Describes the layout of two parallel rows.
template <typename T>
class ParallelRows final : public RowBase<T> {
 public:
  // Shorten code: no need for 'this->'.
  using RowBase<T>::channels;
  using RowBase<T>::stride;

  explicit ParallelRows(T *ptr, size_t stride,
                        size_t channels) KLEIDICV_STREAMING_COMPATIBLE
      : RowBase<T>(2 * stride, channels),
        ptrs_{ptr, RowBase<T>::add_stride(ptr, stride)} {}

  explicit ParallelRows(T *ptr, size_t stride) KLEIDICV_STREAMING_COMPATIBLE
      : ParallelRows(ptr, stride, 1) {}

  // Addition assignment operator to navigate among rows.
  ParallelRows<T> &operator+=(ptrdiff_t diff) KLEIDICV_STREAMING_COMPATIBLE {
    ptrs_[0] = RowBase<T>::add_stride(ptrs_[0], diff * stride());
    ptrs_[1] = RowBase<T>::add_stride(ptrs_[1], diff * stride());
    return *this;
  }

  // Prefix increment operator to advance to the next row.
  ParallelRows<T> &operator++() KLEIDICV_STREAMING_COMPATIBLE {
    return operator+=(1);
  }

  // Returns views on columns within the current rows.
  [[nodiscard]] ParallelColumns<T> as_columns() const
      KLEIDICV_STREAMING_COMPATIBLE {
    Columns columns_0{ptrs_[0], channels()};
    Columns columns_1{ptrs_[1], channels()};
    return ParallelColumns{columns_0, columns_1};
  }

  // Instructs the logic to drop the second row.
  void make_single_row() KLEIDICV_STREAMING_COMPATIBLE { ptrs_[1] = ptrs_[0]; }

 private:
  // Pointers to the two parallel rows.
  T *ptrs_[2];
};  // end of class ParallelRows<T>

template <typename OperationType, typename... RowTypes>
void zip_rows(OperationType &operation, Rectangle rect,
              RowTypes... rows) KLEIDICV_STREAMING_COMPATIBLE {
  // Unary left fold. Evaluates the expression for every part of the unexpanded
  // parameter pack 'rows'.
  if ((... && (rows.is_continuous(rect.width())))) {
    rect.flatten();
  }

  for (size_t row_index = 0; row_index < rect.height(); ++row_index) {
    operation.process_row(rect.width(), rows.as_columns()...);
    // Call pre-increment operator on all elements in the parameter pack.
    ((++rows), ...);
  }
}

template <typename OperationType, typename... RowTypes>
void zip_parallel_rows(OperationType &operation, Rectangle rect,
                       RowTypes... rows) KLEIDICV_STREAMING_COMPATIBLE {
  for (size_t row_index = 0; row_index < rect.height(); row_index += 2) {
    // Handle the last odd row in a special way.
    if (KLEIDICV_UNLIKELY(row_index == (rect.height() - 1))) {
      ((rows.make_single_row(), ...));
    }

    operation.process_row(rect.width(), rows.as_columns()...);
    // Call pre-increment operator on all elements in the parameter pack.
    ((++rows), ...);
  }
}

// Copy rows with support for overlapping memory.
template <typename T>
class CopyRows final {
 public:
  void process_row(size_t length, Columns<const T> src,
                   Columns<T> dst) KLEIDICV_STREAMING_COMPATIBLE {
    memmove(static_cast<void *>(&dst[0]), static_cast<const void *>(&src[0]),
            length * sizeof(T) * dst.channels());
  }

  template <typename S, typename D>
  static void copy_rows(Rectangle rect, S src,
                        D dst) KLEIDICV_STREAMING_COMPATIBLE {
    CopyRows<T> operation;
    zip_rows(operation, rect, src, dst);
  }
};  // end of class CopyRows<T>

// Copy non-verlapping rows.
template <typename T>
class CopyNonOverlappingRows final {
 public:
  void process_row(size_t length, Columns<const T> src,
                   Columns<T> dst) KLEIDICV_STREAMING_COMPATIBLE {
    memcpy(static_cast<void *>(&dst[0]), static_cast<const void *>(&src[0]),
           length * sizeof(T) * dst.channels());
  }

  static void copy_rows(Rectangle rect, Rows<const T> src,
                        Rows<T> dst) KLEIDICV_STREAMING_COMPATIBLE {
    CopyNonOverlappingRows<T> operation;
    zip_rows(operation, rect, src, dst);
  }
};  // end of class CopyNonOverlappingRows<T>

// Sets the margins to zero. It takes both channel count and element size into
// account. For example, margin.left() = 1 means that one pixel worth of space,
// that is sizeof(T) * channels, will be set to zero. The first argument, rect,
// describes the total available memory, including all margins.
template <typename T>
void make_zero_border_border(Rectangle rect, Rows<T> rows, Margin margin) {
  if (margin.left()) {
    size_t margin_width_in_bytes = margin.left() * sizeof(T) * rows.channels();
    for (size_t index = 0; index < rect.height(); ++index) {
      memset(&rows.at(index)[0], 0, margin_width_in_bytes);
    }
  }

  if (margin.top()) {
    size_t top_width = rect.width() - margin.left() - margin.right();
    size_t top_width_in_bytes = top_width * sizeof(T) * rows.channels();
    for (size_t index = 0; index < margin.top(); ++index) {
      memset(&rows.at(index, margin.left())[0], 0, top_width_in_bytes);
    }
  }

  if (margin.right()) {
    size_t margin_width_in_bytes = margin.right() * sizeof(T) * rows.channels();
    for (size_t index = 0; index < rect.height(); ++index) {
      memset(&rows.at(index, rect.width() - margin.right())[0], 0,
             margin_width_in_bytes);
    }
  }

  if (margin.bottom()) {
    size_t bottom_width = rect.width() - margin.left() - margin.right();
    size_t bottom_width_in_bytes = bottom_width * sizeof(T) * rows.channels();
    for (size_t index = rect.height() - margin.bottom(); index < rect.height();
         ++index) {
      memset(&rows.at(index, margin.left())[0], 0, bottom_width_in_bytes);
    }
  }
}

// Struct for providing Rows object over memory managed by std::unique_ptr.
template <typename T>
class RowsOverUniquePtr {
 public:
  // Returns a rectangle which describes the layout of the allocated memory.
  Rectangle rect() const { return rect_; }

  // Returns a raw pointer to the allocated memory.
  T *data() const { return data_.get(); }

  // Returns a Rows instance over the allocated memory.
  Rows<T> rows() const { return rows_; }

  // Like rows() but without margins.
  Rows<T> rows_without_margin() const { return rows_without_margin_; }

 protected:
  RowsOverUniquePtr(Rectangle rect, Margin margin)
      : rect_{get_rectangle(rect, margin)},
        data_{std::unique_ptr<T[]>(new(std::nothrow) T[rect_.area()])} {
    if (!data_) {
      // Code that uses this class is required to check that data() is valid.
      return;
    }

    rows_ = Rows<T>{&data_[0], rect_.width() * sizeof(T)};
    rows_without_margin_ = rows_.at(margin.top(), margin.left());
    make_zero_border_border<T>(rect_, rows_, margin);
  }

 private:
  static Rectangle get_rectangle(Rectangle rect, Margin margin) {
    return Rectangle{margin.left() + rect.width() + margin.right(),
                     margin.top() + rect.height() + margin.bottom()};
  }

  Rectangle rect_;
  Rows<T> rows_;
  Rows<T> rows_without_margin_;
  std::unique_ptr<T[]> data_;
};

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_TYPES_H
