// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_WORKSPACE_SEPARABLE_H
#define KLEIDICV_WORKSPACE_SEPARABLE_H

#include <cstdlib>
#include <memory>

#include "border_types.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/types.h"

namespace KLEIDICV_TARGET_NAMESPACE {

// Forward declarations.
class SeparableFilterWorkspace;

// Deleter for SeparableFilterWorkspace instances.
class SeparableFilterWorkspaceDeleter {
 public:
  void operator()(SeparableFilterWorkspace *ptr) const
      KLEIDICV_STREAMING_COMPATIBLE {
    std::free(ptr);
  };
};

// Workspace for separable fixed-size filters.
//
// Theory of operation
//
// Given an NxM input matrix and a separable filter AxB = V x H, this workspace
// first processes N rows vertically into a separate horizontal buffer. Right
// after the vertical operation, the horizontal operation is applied and the
// result is written to the destination.
//
// Limitations
//
//  1. In-place operations are not supported.
//  2. The input's width and height have to be at least `filter's width - 1` and
//  `filter's height - 1`, respectively.
//
// Example
//
//  N = 2, M = 3, A = B = 3, border type replicate and 'x' is multiplication.
//
//  Input:              Separated filters:
//    [ M00, M01, M02 ]     V = [ V0 ]  H = [H0, H1, H2 ]
//    [ M10, M11, M12 ]         [ V1 ]
//    [ M20, M21, M22 ]         [ V2 ]
//
//  Buffer contents in iteration 0 after applying the vertical operation
//  taking "replicate" border type into account:
//
//    [ B0, B1, B2 ] =
//      [ M{0, 0, 1}0 x V, M{0, 0, 1}1 x V, M{0, 0, 1}2 x V ]
//
//  The horizontal operation is then semantically performed on the following
//  input taking "replicate" border type into account:
//
//    [ B0, B0, B1, B2, B2 ]
//
//  The destination contents after the 0th iteration is then:
//
//    [ D00, D01, D02 ] =
//      [ B{0, 0, 1} x H, B{0, 1, 2} x H, B{1, 2, 2} x H]
//
// Handling of borders is calculated based on offsets rather than setting up
// suitably-sized buffers which could hold both borders and data.
class SeparableFilterWorkspace {
 public:
  // To avoid load/store penalties.
  static constexpr size_t kAlignment = 16UL;

  // Shorthand for std::unique_ptr<> holding a workspace.
  using Pointer = std::unique_ptr<SeparableFilterWorkspace,
                                  SeparableFilterWorkspaceDeleter>;

  // Workspace is only constructible with create().
  SeparableFilterWorkspace() = delete;

  // Creates a workspace on the heap.
  static Pointer create(Rectangle rect, size_t channels,
                        size_t intermediate_size)
      KLEIDICV_STREAMING_COMPATIBLE {
    size_t buffer_rows_number_of_elements = rect.width() * channels;
    // Adding more elements because of SVE, where interleaving stores are
    // governed by one predicate. For example, if a predicate requires 7 uint8_t
    // elements and an algorithm performs widening to 16 bits, the resulting
    // interleaving store will still be governed by the same predicate, thus
    // storing 8 elements. Choosing '3' to account for svst4().
    buffer_rows_number_of_elements += 3;

    size_t buffer_rows_stride =
        buffer_rows_number_of_elements * intermediate_size;
    size_t buffer_rows_size = buffer_rows_stride;
    buffer_rows_size += kAlignment - 1;

    // Try to allocate workspace at once.
    size_t allocation_size =
        sizeof(SeparableFilterWorkspace) + buffer_rows_size;
    void *allocation = std::malloc(allocation_size);
    auto workspace = SeparableFilterWorkspace::Pointer{
        reinterpret_cast<SeparableFilterWorkspace *>(allocation)};

    if (!workspace) {
      return workspace;
    }

    auto *buffer_rows_address = &workspace->data_[0];
    buffer_rows_address = align_up(buffer_rows_address, kAlignment);
    workspace->buffer_rows_offset_ = buffer_rows_address - &workspace->data_[0];
    workspace->buffer_rows_stride_ = buffer_rows_stride;
    workspace->image_size_ = rect;
    workspace->channels_ = channels;
    workspace->intermediate_size_ = intermediate_size;

    return workspace;
  }

  size_t channels() const { return channels_; }
  Rectangle image_size() const { return image_size_; }
  size_t intermediate_size() const { return intermediate_size_; }

  // Processes rows vertically first along the full width
  template <typename FilterType>
  void process(Rectangle rect, size_t y_begin, size_t y_end,
               Rows<const typename FilterType::SourceType> src_rows,
               Rows<typename FilterType::DestinationType> dst_rows,
               size_t channels, typename FilterType::BorderType border_type,
               FilterType filter) KLEIDICV_STREAMING_COMPATIBLE {
    // Border helper which calculates border offsets.
    typename FilterType::BorderInfoType vertical_border{rect.height(),
                                                        border_type};
    typename FilterType::BorderInfoType horizontal_border{rect.width(),
                                                          border_type};

    // Buffer rows which hold intermediate widened data.
    auto buffer_rows = Rows{reinterpret_cast<typename FilterType::BufferType *>(
                                &data_[buffer_rows_offset_]),
                            buffer_rows_stride_, channels};

    // Vertical processing loop.
    for (size_t vertical_index = y_begin; vertical_index < y_end;
         ++vertical_index) {
      // Recalculate vertical border offsets.
      auto offsets = vertical_border.offsets_with_border(vertical_index);
      // Process in the vertical direction first.
      filter.process_vertical(rect.width(), src_rows.at(vertical_index),
                              buffer_rows, offsets);
      // Process in the horizontal direction last.
      process_horizontal(rect.width(), buffer_rows, dst_rows.at(vertical_index),
                         filter, horizontal_border);
    }
  }

 protected:
  template <typename FilterType>
  void process_horizontal(size_t width,
                          Rows<typename FilterType::BufferType> buffer_rows,
                          Rows<typename FilterType::DestinationType> dst_rows,
                          FilterType filter,
                          typename FilterType::BorderInfoType horizontal_border)
      KLEIDICV_STREAMING_COMPATIBLE {
    // Margin associated with the filter.
    constexpr size_t margin = filter.margin;

    // Process data affected by left border.
    KLEIDICV_FORCE_LOOP_UNROLL
    for (size_t horizontal_index = 0; horizontal_index < margin;
         ++horizontal_index) {
      auto offsets =
          horizontal_border.offsets_with_left_border(horizontal_index);
      filter.process_horizontal_borders(buffer_rows.at(0, horizontal_index),
                                        dst_rows.at(0, horizontal_index),
                                        offsets);
    }

    // Process data which is not affected by any borders in bulk.
    {
      size_t width_without_borders = width - (2 * margin);
      auto offsets = horizontal_border.offsets_without_border();
      filter.process_horizontal(width_without_borders,
                                buffer_rows.at(0, margin),
                                dst_rows.at(0, margin), offsets);
    }

    // Process data affected by right border.
    KLEIDICV_FORCE_LOOP_UNROLL
    for (size_t horizontal_index = 0; horizontal_index < margin;
         ++horizontal_index) {
      size_t index = width - margin + horizontal_index;
      auto offsets = horizontal_border.offsets_with_right_border(index);
      filter.process_horizontal_borders(buffer_rows.at(0, index),
                                        dst_rows.at(0, index), offsets);
    }
  }

  // Offset in bytes to the buffer rows from &data_[0].
  size_t buffer_rows_offset_;
  // Stride of the buffer rows.
  size_t buffer_rows_stride_;

  Rectangle image_size_;
  size_t channels_;
  size_t intermediate_size_;

  // Workspace area begins here.
  uint8_t data_[0] KLEIDICV_ATTR_ALIGNED(kAlignment);
};  // end of class SeparableFilterWorkspace

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_WORKSPACE_SEPARABLE_H
