// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_WORKSPACE_BLUR_AND_DOWNSAMPLE_WS_H
#define KLEIDICV_WORKSPACE_BLUR_AND_DOWNSAMPLE_WS_H

#include "separable.h"

namespace KLEIDICV_TARGET_NAMESPACE {

// Alter SeparableFilterWorkspace's behavior to only process elements in even
// rows and columns
class BlurAndDownsampleFilterWorkspace final : public SeparableFilterWorkspace {
 public:
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
         vertical_index += 2) {
      // Recalculate vertical border offsets.
      auto offsets = vertical_border.offsets_with_border(vertical_index);
      // Process in the vertical direction first.
      filter.process_vertical(rect.width(), src_rows.at(vertical_index),
                              buffer_rows, offsets);
      // Process in the horizontal direction last.
      process_horizontal(rect.width(), buffer_rows,
                         dst_rows.at(vertical_index / 2), filter,
                         horizontal_border);
    }
  }

 private:
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
         horizontal_index += 2) {
      auto offsets =
          horizontal_border.offsets_with_left_border(horizontal_index);
      filter.process_horizontal_borders(buffer_rows.at(0, horizontal_index),
                                        dst_rows.at(0, horizontal_index / 2),
                                        offsets);
    }

    // Process data which is not affected by any borders in bulk.
    {
      size_t width_without_borders = width - (2 * margin);
      auto offsets = horizontal_border.offsets_without_border();
      size_t start = align_up(margin, 2);
      filter.process_horizontal(width_without_borders, buffer_rows.at(0, start),
                                dst_rows.at(0, start / 2), offsets);
    }

    // Process data affected by right border.
    for (size_t index = align_up(width - margin, 2); index < width;
         index += 2) {
      auto offsets = horizontal_border.offsets_with_right_border(index);
      filter.process_horizontal_borders(buffer_rows.at(0, index),
                                        dst_rows.at(0, index / 2), offsets);
    }
  }
};  // end of class BlurAndDownsampleFilterWorkspace

// BlurAndDownsampleFilterWorkspace and SeparableFilterWorkspace must have the
// same size because through the API of this library only
// SeparableFilterWorkspace can be created. So, child classes of
// SeparableFilterWorkspace can only add functionality but cannot add member
// variables.
static_assert(sizeof(BlurAndDownsampleFilterWorkspace) ==
              sizeof(SeparableFilterWorkspace));

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_WORKSPACE_BLUR_AND_DOWNSAMPLE_WS_H
