// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_MORPHOLOGY_WORKSPACE_H
#define KLEIDICV_MORPHOLOGY_WORKSPACE_H

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <optional>

#include "kleidicv/kleidicv.h"
#include "kleidicv/types.h"

namespace KLEIDICV_TARGET_NAMESPACE {

// Forward declarations.
class MorphologyWorkspace;

// Deleter for MorphologyWorkspace instances.
class MorphologyWorkspaceDeleter {
 public:
  void operator()(MorphologyWorkspace *ptr) const
      KLEIDICV_STREAMING_COMPATIBLE {
    std::free(ptr);
  };
};

// Workspace for morphological operations.
class MorphologyWorkspace final {
 public:
  // Shorthand for std::unique_ptr<> holding a workspace.
  using Pointer =
      std::unique_ptr<MorphologyWorkspace, MorphologyWorkspaceDeleter>;

  enum class BorderType {
    CONSTANT,
    REPLICATE,
  };

  static std::optional<BorderType> get_border_type(
      kleidicv_border_type_t border_type) KLEIDICV_STREAMING_COMPATIBLE {
    switch (border_type) {
      case KLEIDICV_BORDER_TYPE_REPLICATE:
        return BorderType::REPLICATE;
      case KLEIDICV_BORDER_TYPE_CONSTANT:
        return BorderType::CONSTANT;
      default:
        return std::optional<BorderType>();
    }
  }

  template <typename T>
  class CopyDataMemcpy {
   public:
    constexpr void operator()(Rows<const T> src_rows, Rows<T> dst_rows,
                              size_t length) const
        KLEIDICV_STREAMING_COMPATIBLE {
      std::memcpy(static_cast<void *>(&dst_rows[0]),
                  static_cast<const void *>(&src_rows[0]),
                  length * sizeof(T) * dst_rows.channels());
    }
  };

  // MorphologyWorkspace is only constructible with create().
  MorphologyWorkspace() = delete;

  // Creates a workspace on the heap.
  static kleidicv_error_t create(
      Pointer &workspace, kleidicv_rectangle_t kernel, kleidicv_point_t anchor,
      BorderType border_type, kleidicv_border_values_t border_values,
      size_t channels, size_t iterations, size_t type_size,
      kleidicv_rectangle_t image) KLEIDICV_STREAMING_COMPATIBLE {
    // These values are arbitrarily choosen.
    const size_t rows_per_iteration = std::max(2 * kernel.height, 32UL);
    // To avoid load/store penalties.
    const size_t kAlignment = 16;

    if (anchor.x >= kernel.width || anchor.y >= kernel.height) {
      return KLEIDICV_ERROR_RANGE;
    }

    Rectangle image_size{image};
    Margin margin{kernel, anchor};

    // A single wide row which can hold one row worth of data in addition
    // to left and right margins.
    size_t wide_rows_width =
        margin.left() + image_size.width() + margin.right();
    size_t wide_rows_stride = wide_rows_width * channels;
    wide_rows_stride = align_up(wide_rows_stride, kAlignment);
    size_t wide_rows_height = 1UL;  // There is only one wide row.
    size_t wide_rows_size = wide_rows_stride * wide_rows_height;
    wide_rows_size += kAlignment - 1;

    // Multiple buffer rows to hold rows without any borders.
    size_t buffer_rows_width = type_size * image_size.width();
    size_t buffer_rows_stride = buffer_rows_width * channels;
    buffer_rows_stride = align_up(buffer_rows_stride, kAlignment);
    size_t buffer_rows_height = 2 * rows_per_iteration;
    size_t buffer_rows_size = 0UL;
    if (__builtin_mul_overflow(buffer_rows_stride, buffer_rows_height,
                               &buffer_rows_size)) {
      return KLEIDICV_ERROR_RANGE;
    }
    buffer_rows_size += kAlignment - 1;

    // Storage for indirect row access.
    size_t indirect_row_storage_size = 3 * rows_per_iteration * sizeof(void *);

    // Try to allocate workspace at once.
    size_t allocation_size = sizeof(MorphologyWorkspace) +
                             indirect_row_storage_size + buffer_rows_size +
                             wide_rows_size;
    void *allocation = std::malloc(allocation_size);
    workspace = MorphologyWorkspace::Pointer{
        reinterpret_cast<MorphologyWorkspace *>(allocation)};
    if (!workspace) {
      return KLEIDICV_ERROR_ALLOCATION;
    }

    workspace->rows_per_iteration_ = rows_per_iteration;
    workspace->wide_rows_src_width_ = image_size.width();
    workspace->channels_ = channels;

    auto *buffer_rows_address = &workspace->data_[indirect_row_storage_size];
    buffer_rows_address = align_up(buffer_rows_address, kAlignment);
    workspace->buffer_rows_offset_ = buffer_rows_address - &workspace->data_[0];
    workspace->buffer_rows_stride_ = buffer_rows_stride;

    auto *wide_rows_address =
        &workspace->data_[indirect_row_storage_size + buffer_rows_size];
    wide_rows_address += margin.left() * channels;
    wide_rows_address = align_up(wide_rows_address, kAlignment);
    wide_rows_address -= margin.left() * channels;
    workspace->wide_rows_offset_ = wide_rows_address - &workspace->data_[0];
    workspace->wide_rows_stride_ = wide_rows_stride;

    workspace->kernel_ = kernel;
    workspace->anchor_ = anchor;
    workspace->border_type_ = border_type;
    workspace->border_values_ = border_values;
    workspace->channels_ = channels;
    workspace->iterations_ = iterations;
    workspace->type_size_ = type_size;
    workspace->image_size_ = image_size;

    return KLEIDICV_OK;
  }

  kleidicv_rectangle_t kernel() const { return kernel_; }
  kleidicv_point_t anchor() const { return anchor_; }
  BorderType border_type() const { return border_type_; }
  kleidicv_border_values_t border_values() const { return border_values_; }
  size_t channels() const { return channels_; }
  size_t iterations() const { return iterations_; }
  size_t type_size() const { return type_size_; }
  Rectangle image_size() const { return image_size_; }

  // This function is too complex, but disable the warning for now.
  // NOLINTBEGIN(readability-function-cognitive-complexity)
  template <typename O>
  void process(Rectangle rect, Rows<const typename O::SourceType> src_rows,
               Rows<typename O::DestinationType> dst_rows, Margin margin,
               Border<typename O::SourceType> border, BorderType border_type,
               O operation) KLEIDICV_STREAMING_COMPATIBLE {
    using S = typename O::SourceType;
    using B = typename O::BufferType;
    typename O::CopyData copy_data{};

    if (KLEIDICV_UNLIKELY(rect.width() == 0 || rect.height() == 0)) {
      return;
    }

    // Wide rows which can hold data with left and right margins.
    auto wide_rows = Rows{reinterpret_cast<S *>(&data_[wide_rows_offset_]),
                          wide_rows_stride_, channels_};

    // Double buffered indirect rows to access the buffer rows.
    auto db_indirect_rows = DoubleBufferedIndirectRows{
        reinterpret_cast<B **>(&data_[0]), rows_per_iteration_,
        Rows{reinterpret_cast<B *>(&data_[buffer_rows_offset_]),
             buffer_rows_stride_, channels_}};

    // [Step 1] Initialize workspace.
    horizontal_height_ = margin.top() + rect.height() + margin.bottom();
    vertical_height_ = rect.height();
    row_index_ = 0;

    // Constant border values.
    auto left_border_value = saturating_cast<double, S>(border.left());
    auto top_border_value = saturating_cast<double, S>(border.top());
    auto right_border_value = saturating_cast<double, S>(border.right());
    auto bottom_border_value = saturating_cast<double, S>(border.bottom());

    // Used by replicate border type.
    auto first_src_rows = src_rows;
    auto last_src_rows = src_rows.at(rect.height() - 1);

    size_t horizontal_height = get_next_horizontal_height();
    for (size_t index = 0; index < horizontal_height; ++index) {
      switch (border_type) {
        case BorderType::CONSTANT: {
          make_constant_border(wide_rows, 0, margin.left(), left_border_value);

          if (row_index_ < margin.top()) {
            make_constant_border(wide_rows, margin.left(), wide_rows_src_width_,
                                 top_border_value);
          } else if (row_index_ < (margin.top() + rect.height())) {
            copy_data(src_rows, wide_rows.at(0, margin.left()),
                      wide_rows_src_width_);
            // Advance source rows.
            ++src_rows;
          } else {
            make_constant_border(wide_rows, margin.left(), wide_rows_src_width_,
                                 bottom_border_value);
          }

          make_constant_border(wide_rows, margin.left() + wide_rows_src_width_,
                               margin.right(), right_border_value);

          // Advance counters.
          ++row_index_;
        } break;

        case BorderType::REPLICATE: {
          Rows<const S> current_src_row;

          if (row_index_ < margin.top()) {
            current_src_row = first_src_rows;
          } else if (row_index_ < (margin.top() + rect.height())) {
            current_src_row = src_rows;
            // Advance source rows.
            ++src_rows;
          } else {
            current_src_row = last_src_rows;
          }

          replicate_border(current_src_row, wide_rows, 0, 0, margin.left());
          copy_data(current_src_row, wide_rows.at(0, margin.left()),
                    wide_rows_src_width_);
          replicate_border(current_src_row, wide_rows, wide_rows_src_width_ - 1,
                           margin.left() + wide_rows_src_width_,
                           margin.right());

          // Advance counters.
          ++row_index_;
        } break;
      }  // switch (border_type)

      // [Step 2] Process the preloaded data.
      operation.process_horizontal(Rectangle{rect.width(), 1UL}, wide_rows,
                                   db_indirect_rows.write_at().at(index));
    }  // for (...; index < horizontal_height; ...)

    db_indirect_rows.swap();

    // [Step 3] Process any remaining data.
    while (vertical_height_) {
      size_t horizontal_height = get_next_horizontal_height();
      for (size_t index = 0; index < horizontal_height; ++index) {
        switch (border_type) {
          case BorderType::CONSTANT: {
            if (row_index_ < (margin.top() + rect.height())) {
              // Constant left and right borders with source data.
              copy_data(src_rows, wide_rows.at(0, margin.left()),
                        wide_rows_src_width_);
              // Advance source rows.
              ++src_rows;
            } else {
              make_constant_border(wide_rows, margin.left(),
                                   wide_rows_src_width_, bottom_border_value);
            }

            // Advance row counter.
            ++row_index_;
          } break;

          case BorderType::REPLICATE: {
            Rows<const S> current_src_row;

            if (row_index_ < (margin.top() + rect.height())) {
              current_src_row = src_rows;
              // Advance source rows.
              ++src_rows;
            } else {
              current_src_row = last_src_rows;
            }

            replicate_border(current_src_row, wide_rows, 0, 0, margin.left());
            copy_data(current_src_row, wide_rows.at(0, margin.left()),
                      wide_rows_src_width_);
            replicate_border(
                current_src_row, wide_rows, wide_rows_src_width_ - 1,
                margin.left() + wide_rows_src_width_, margin.right());

            // Advance counters.
            ++row_index_;
          } break;
        }  // switch (border_type)

        operation.process_horizontal(Rectangle{rect.width(), 1UL}, wide_rows,
                                     db_indirect_rows.write_at().at(index));
      }  // for (...; index < horizontal_height; ...)

      size_t next_vertical_height = get_next_vertical_height();
      operation.process_vertical(Rectangle{rect.width(), next_vertical_height},
                                 db_indirect_rows.read_at(), dst_rows);
      dst_rows += next_vertical_height;

      db_indirect_rows.swap();
    }
  }
  // NOLINTEND(readability-function-cognitive-complexity)

 private:
  // The number of wide rows to process in the next iteration.
  [[nodiscard]] size_t get_next_horizontal_height()
      KLEIDICV_STREAMING_COMPATIBLE {
    size_t height = std::min(horizontal_height_, rows_per_iteration_);
    horizontal_height_ -= height;
    return height;
  }

  // The number of indirect rows to process in the next iteration.
  [[nodiscard]] size_t get_next_vertical_height()
      KLEIDICV_STREAMING_COMPATIBLE {
    size_t height = std::min(vertical_height_, rows_per_iteration_);
    vertical_height_ -= height;
    return height;
  }

  template <typename T, typename BorderType>
  void make_constant_border(Rows<T> dst_rows, size_t dst_index, size_t count,
                            BorderType value) KLEIDICV_STREAMING_COMPATIBLE {
    auto dst = &dst_rows.at(0, dst_index)[0];
    for (size_t index = 0; index < count * dst_rows.channels(); ++index) {
      dst[index] = value;
    }
  }

  template <typename T>
  void replicate_border(Rows<const T> src_rows, Rows<T> dst_rows,
                        size_t src_index, size_t dst_index,
                        size_t count) KLEIDICV_STREAMING_COMPATIBLE {
    if (!count) {
      return;
    }

    for (size_t channel = 0; channel < src_rows.channels(); ++channel) {
      for (size_t index = dst_index; index < dst_index + count; ++index) {
        dst_rows.at(0, index)[channel] = src_rows.at(0, src_index)[channel];
      }
    }
  }

  static_assert(sizeof(Pointer) == sizeof(void *), "Unexpected type size");

  kleidicv_rectangle_t kernel_;
  kleidicv_point_t anchor_;
  BorderType border_type_;
  kleidicv_border_values_t border_values_;
  size_t iterations_;
  size_t type_size_;
  Rectangle image_size_;

  // Number of wide rows in this workspace.
  size_t rows_per_iteration_;
  // Size of the data in bytes within a row.
  size_t wide_rows_src_width_;
  // The number of channels.
  size_t channels_;
  // Remaining height to process in horizontal direction.
  size_t horizontal_height_;
  // Remaining height to process in vertical direction.
  size_t vertical_height_;
  // Index of the processed row.
  size_t row_index_;
  // Offset in bytes to the buffer rows from &data_[0].
  size_t buffer_rows_offset_;
  // Stride of the buffer rows.
  size_t buffer_rows_stride_;
  // Offset in bytes to the wide rows from &data_[0].
  size_t wide_rows_offset_;
  // Stride of the wide rows.
  size_t wide_rows_stride_;
  // Workspace area begins here.
  uint8_t data_[0] KLEIDICV_ATTR_ALIGNED(sizeof(void *));
};  // end of class MorphologyWorkspace

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_MORPHOLOGY_WORKSPACE_H
