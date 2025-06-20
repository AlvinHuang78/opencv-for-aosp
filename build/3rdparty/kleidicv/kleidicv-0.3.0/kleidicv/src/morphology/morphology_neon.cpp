// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <limits>

#include "kleidicv/kleidicv.h"
#include "kleidicv/morphology/workspace.h"
#include "kleidicv/neon.h"
#include "kleidicv/types.h"

namespace kleidicv::neon {

template <typename ScalarType, typename O>
class VerticalOp final {
 public:
  using VecTraits = neon::VecTraits<ScalarType>;

  VerticalOp(Rectangle rect, Rectangle kernel) : rect_(rect), kernel_(kernel) {}

  void process_rows(IndirectRows<ScalarType> src_rows,
                    Rows<ScalarType> dst_rows) {
    if (KLEIDICV_UNLIKELY(kernel_.height()) == 1) {
      CopyRows<ScalarType>::copy_rows(rect_, src_rows, dst_rows);
      return;
    }

    // Iterate across the rows from top to bottom. This implementation can
    // handle two rows at once.
    for (size_t height = 0; height < rect_.height(); height += 2) {
      // Iterate across the columns from left to right.
      LoopUnroll2<TryToAvoidTailLoop> loop{rect_.width() * src_rows.channels(),
                                           VecTraits::num_lanes()};
      // clang-format off
      loop
          .unroll_four_times([&](size_t index) {
            vector_path_4x(src_rows, dst_rows, index, height);
          })
          .unroll_twice([&](size_t index) {
            vector_path_2x(src_rows, dst_rows, index, height);
          })
          .unroll_once([&](size_t index) {
            vector_path(src_rows, dst_rows, index, height);
          })
          .tail([&](size_t index) {
            scalar_path(src_rows, dst_rows, index, height);
          });
      // clang-format on
      src_rows += 2;
      dst_rows += 2;
    }
  }

 private:
  void vector_path_4x(IndirectRows<ScalarType> src_rows,
                      Rows<ScalarType> dst_rows, const size_t index,
                      const size_t height) {
    const ScalarType *src_row = &src_rows[index];
    auto first_row0 = vld1q(&src_row[0 * VecTraits::num_lanes()]);
    auto first_row1 = vld1q(&src_row[1 * VecTraits::num_lanes()]);
    auto first_row2 = vld1q(&src_row[2 * VecTraits::num_lanes()]);
    auto first_row3 = vld1q(&src_row[3 * VecTraits::num_lanes()]);
    ++src_rows;

    src_row = &src_rows[index];
    auto acc0 = vld1q(&src_row[0 * VecTraits::num_lanes()]);
    auto acc1 = vld1q(&src_row[1 * VecTraits::num_lanes()]);
    auto acc2 = vld1q(&src_row[2 * VecTraits::num_lanes()]);
    auto acc3 = vld1q(&src_row[3 * VecTraits::num_lanes()]);
    ++src_rows;

    LoopUnroll loop{kernel_.height() - 2, 2};

    loop.unroll_once([&](size_t step) {
      const ScalarType *src_row0 = &src_rows.at(0)[index];
      const ScalarType *src_row1 = &src_rows.at(1)[index];
      auto row00 = vld1q(&src_row0[0 * VecTraits::num_lanes()]);
      auto row01 = vld1q(&src_row0[1 * VecTraits::num_lanes()]);
      auto row02 = vld1q(&src_row0[2 * VecTraits::num_lanes()]);
      auto row03 = vld1q(&src_row0[3 * VecTraits::num_lanes()]);
      auto row10 = vld1q(&src_row1[0 * VecTraits::num_lanes()]);
      auto row11 = vld1q(&src_row1[1 * VecTraits::num_lanes()]);
      auto row12 = vld1q(&src_row1[2 * VecTraits::num_lanes()]);
      auto row13 = vld1q(&src_row1[3 * VecTraits::num_lanes()]);
      acc0 = O::operation(acc0, O::operation(row00, row10));
      acc1 = O::operation(acc1, O::operation(row01, row11));
      acc2 = O::operation(acc2, O::operation(row02, row12));
      acc3 = O::operation(acc3, O::operation(row03, row13));
      src_rows += step;
    });

    loop.tail([&](size_t /* index */) {
      const ScalarType *src_row = &src_rows[index];
      auto row0 = vld1q(&src_row[0 * VecTraits::num_lanes()]);
      auto row1 = vld1q(&src_row[1 * VecTraits::num_lanes()]);
      auto row2 = vld1q(&src_row[2 * VecTraits::num_lanes()]);
      auto row3 = vld1q(&src_row[3 * VecTraits::num_lanes()]);
      acc0 = O::operation(acc0, row0);
      acc1 = O::operation(acc1, row1);
      acc2 = O::operation(acc2, row2);
      acc3 = O::operation(acc3, row3);
      ++src_rows;
    });

    // Save partial results which do not contain the first row.
    auto partial_acc0 = acc0;
    auto partial_acc1 = acc1;
    auto partial_acc2 = acc2;
    auto partial_acc3 = acc3;

    // Take the first row into account.
    acc0 = O::operation(acc0, first_row0);
    acc1 = O::operation(acc1, first_row1);
    acc2 = O::operation(acc2, first_row2);
    acc3 = O::operation(acc3, first_row3);

    // Store the results.
    ScalarType *dst_row = &dst_rows[index];
    vst1q(&dst_row[0 * VecTraits::num_lanes()], acc0);
    vst1q(&dst_row[1 * VecTraits::num_lanes()], acc1);
    vst1q(&dst_row[2 * VecTraits::num_lanes()], acc2);
    vst1q(&dst_row[3 * VecTraits::num_lanes()], acc3);

    // Try to process one more row, because it is relatively cheap to do so.
    if (KLEIDICV_UNLIKELY((height + 1) >= rect_.height())) {
      return;
    }

    ++dst_rows;

    src_row = &src_rows[index];
    auto next_row0 = vld1q(&src_row[0 * VecTraits::num_lanes()]);
    auto next_row1 = vld1q(&src_row[1 * VecTraits::num_lanes()]);
    auto next_row2 = vld1q(&src_row[2 * VecTraits::num_lanes()]);
    auto next_row3 = vld1q(&src_row[3 * VecTraits::num_lanes()]);

    acc0 = O::operation(partial_acc0, next_row0);
    acc1 = O::operation(partial_acc1, next_row1);
    acc2 = O::operation(partial_acc2, next_row2);
    acc3 = O::operation(partial_acc3, next_row3);

    dst_row = &dst_rows[index];
    vst1q(&dst_row[0 * VecTraits::num_lanes()], acc0);
    vst1q(&dst_row[1 * VecTraits::num_lanes()], acc1);
    vst1q(&dst_row[2 * VecTraits::num_lanes()], acc2);
    vst1q(&dst_row[3 * VecTraits::num_lanes()], acc3);
  }

  void vector_path_2x(IndirectRows<ScalarType> src_rows,
                      Rows<ScalarType> dst_rows, const size_t index,
                      const size_t height) {
    const ScalarType *src_row = &src_rows[index];
    auto first_row0 = vld1q(&src_row[0]);
    auto first_row1 = vld1q(&src_row[VecTraits::num_lanes()]);
    ++src_rows;

    src_row = &src_rows[index];
    auto acc0 = vld1q(&src_row[0]);
    auto acc1 = vld1q(&src_row[VecTraits::num_lanes()]);
    ++src_rows;

    LoopUnroll loop{kernel_.height() - 2, 2};

    loop.unroll_once([&](size_t step) {
      const ScalarType *src_row0 = &src_rows.at(0)[index];
      const ScalarType *src_row1 = &src_rows.at(1)[index];
      auto row00 = vld1q(&src_row0[0]);
      auto row01 = vld1q(&src_row0[VecTraits::num_lanes()]);
      auto row10 = vld1q(&src_row1[0]);
      auto row11 = vld1q(&src_row1[VecTraits::num_lanes()]);
      acc0 = O::operation(acc0, O::operation(row00, row10));
      acc1 = O::operation(acc1, O::operation(row01, row11));
      src_rows += step;
    });

    loop.tail([&](size_t /* index */) {
      const ScalarType *src_row = &src_rows[index];
      auto row0 = vld1q(&src_row[0]);
      auto row1 = vld1q(&src_row[VecTraits::num_lanes()]);
      acc0 = O::operation(acc0, row0);
      acc1 = O::operation(acc1, row1);
      ++src_rows;
    });

    // Save partial results which do not contain the first row.
    auto partial_acc0 = acc0;
    auto partial_acc1 = acc1;

    // Take the first row into account.
    acc0 = O::operation(acc0, first_row0);
    acc1 = O::operation(acc1, first_row1);

    // Store the results.
    ScalarType *dst_row = &dst_rows[index];
    vst1q(&dst_row[0], acc0);
    vst1q(&dst_row[VecTraits::num_lanes()], acc1);

    // Try to process one more row, because it is relatively cheap to do so.
    if (KLEIDICV_UNLIKELY((height + 1) >= rect_.height())) {
      return;
    }

    ++dst_rows;

    src_row = &src_rows[index];
    auto next_row0 = vld1q(&src_row[0]);
    auto next_row1 = vld1q(&src_row[VecTraits::num_lanes()]);

    acc0 = O::operation(partial_acc0, next_row0);
    acc1 = O::operation(partial_acc1, next_row1);

    dst_row = &dst_rows[index];
    vst1q(&dst_row[0], acc0);
    vst1q(&dst_row[VecTraits::num_lanes()], acc1);
  }

  void vector_path(IndirectRows<ScalarType> src_rows, Rows<ScalarType> dst_rows,
                   const size_t index, const size_t height) {
    auto first_row = vld1q(&src_rows[index]);
    ++src_rows;

    auto acc = vld1q(&src_rows[index]);
    ++src_rows;

    LoopUnroll loop{kernel_.height() - 2, 2};

    loop.unroll_once([&](size_t step) {
      auto row0 = vld1q(&src_rows.at(0)[index]);
      auto row1 = vld1q(&src_rows.at(1)[index]);
      acc = O::operation(acc, O::operation(row0, row1));
      src_rows += step;
    });

    loop.tail([&](size_t /* index */) {
      auto row = vld1q(&src_rows[index]);
      acc = O::operation(acc, row);
      ++src_rows;
    });

    // Save partial result which does not contain the first row.
    auto partial_acc = acc;

    // Take the first row into account.
    acc = O::operation(acc, first_row);

    // Store the results.
    vst1q(&dst_rows[index], acc);

    // Try to process one more row, because it is relatively cheap to do so.
    if (KLEIDICV_UNLIKELY((height + 1) >= rect_.height())) {
      return;
    }

    ++dst_rows;

    auto next_row = vld1q(&src_rows[index]);
    acc = O::operation(partial_acc, next_row);
    vst1q(&dst_rows[index], acc);
  }

  void scalar_path(IndirectRows<ScalarType> src_rows, Rows<ScalarType> dst_rows,
                   const size_t index, const size_t height) {
    disable_loop_vectorization();

    ScalarType first_row = src_rows[index];
    ++src_rows;

    ScalarType acc = src_rows[index];
    ++src_rows;

    LoopUnroll loop{kernel_.height() - 2, 2};

    loop.unroll_once([&](size_t step) {
      auto row0 = src_rows.at(0)[index];
      auto row1 = src_rows.at(1)[index];
      acc = O::operation(acc, O::operation(row0, row1));
      src_rows += step;
    });

    loop.tail([&](size_t /* index */) {
      auto row = src_rows[index];
      acc = O::operation(acc, row);
      ++src_rows;
    });

    // Save partial result which does not contain the first row.
    auto partial_acc = acc;

    // Take the first row into account.
    acc = O::operation(acc, first_row);

    // Store the results.
    dst_rows[index] = acc;

    // Try to process one more row, because it is relatively cheap to do so.
    if (KLEIDICV_UNLIKELY((height + 1) >= rect_.height())) {
      return;
    }

    ++dst_rows;

    auto next_row = src_rows[index];
    acc = O::operation(partial_acc, next_row);
    dst_rows[index] = acc;
  }

  Rectangle rect_;
  Rectangle kernel_;
};  // end of class VerticalOp<ScalarType, O>

template <typename ScalarType, typename O>
class HorizontalOp final {
 public:
  using VecTraits = neon::VecTraits<ScalarType>;

  HorizontalOp(Rectangle rect, Rectangle kernel)
      : rect_(rect), kernel_(kernel) {}

  void process_rows(Rows<const ScalarType> src_rows,
                    Rows<ScalarType> dst_rows) {
    // Iterate across the rows from top to bottom.
    for (size_t height = 0; height < rect_.height(); ++height) {
      // Iterate across the columns from left to right.
      LoopUnroll2<TryToAvoidTailLoop> loop{rect_.width() * src_rows.channels(),
                                           VecTraits::num_lanes()};
      // clang-format off
      loop
          .unroll_four_times([&](size_t index) {
            vector_path_4x(src_rows, dst_rows, index);
          })
          .unroll_twice([&](size_t index) {
            vector_path_2x(src_rows, dst_rows, index);
          })
          .unroll_once([&](size_t index) {
            vector_path(src_rows, dst_rows, index);
          })
          .tail([&](size_t index) {
            scalar_path(src_rows, dst_rows, index);
          });
      // clang-format on
      ++src_rows;
      ++dst_rows;
    }
  }

 private:
  void vector_path_4x(Rows<const ScalarType> src_rows,
                      Rows<ScalarType> dst_rows, const size_t index) {
    const auto *src_row = &src_rows[index];
    auto acc0 = vld1q(&src_row[0 * VecTraits::num_lanes()]);
    auto acc1 = vld1q(&src_row[1 * VecTraits::num_lanes()]);
    auto acc2 = vld1q(&src_row[2 * VecTraits::num_lanes()]);
    auto acc3 = vld1q(&src_row[3 * VecTraits::num_lanes()]);

    for (size_t width = 1; width < kernel_.width(); ++width) {
      src_row = &src_rows[index + width * src_rows.channels()];
      auto row0 = vld1q(&src_row[0 * VecTraits::num_lanes()]);
      auto row1 = vld1q(&src_row[1 * VecTraits::num_lanes()]);
      auto row2 = vld1q(&src_row[2 * VecTraits::num_lanes()]);
      auto row3 = vld1q(&src_row[3 * VecTraits::num_lanes()]);
      acc0 = O::operation(acc0, row0);
      acc1 = O::operation(acc1, row1);
      acc2 = O::operation(acc2, row2);
      acc3 = O::operation(acc3, row3);
    }

    auto dst_row = &dst_rows[index];
    vst1q(&dst_row[0 * VecTraits::num_lanes()], acc0);
    vst1q(&dst_row[1 * VecTraits::num_lanes()], acc1);
    vst1q(&dst_row[2 * VecTraits::num_lanes()], acc2);
    vst1q(&dst_row[3 * VecTraits::num_lanes()], acc3);
  }

  void vector_path_2x(Rows<const ScalarType> src_rows,
                      Rows<ScalarType> dst_rows, const size_t index) {
    const auto *src_row = &src_rows[index];
    auto acc0 = vld1q(&src_row[0]);
    auto acc1 = vld1q(&src_row[VecTraits::num_lanes()]);

    for (size_t width = 1; width < kernel_.width(); ++width) {
      src_row = &src_rows[index + width * src_rows.channels()];
      auto row0 = vld1q(&src_row[0 * VecTraits::num_lanes()]);
      auto row1 = vld1q(&src_row[1 * VecTraits::num_lanes()]);
      acc0 = O::operation(acc0, row0);
      acc1 = O::operation(acc1, row1);
    }

    auto dst_row = &dst_rows[index];
    vst1q(&dst_row[0], acc0);
    vst1q(&dst_row[VecTraits::num_lanes()], acc1);
  }

  void vector_path(Rows<const ScalarType> src_rows, Rows<ScalarType> dst_rows,
                   const size_t index) {
    auto acc = vld1q(&src_rows[index]);

    for (size_t width = 1; width < kernel_.width(); ++width) {
      // TODO: Check if EXT was any faster.
      const auto *src_row = &src_rows[index + width * src_rows.channels()];
      acc = O::operation(acc, vld1q(&src_row[0]));
    }

    vst1q(&dst_rows[index], acc);
  }

  void scalar_path(Rows<const ScalarType> src_rows, Rows<ScalarType> dst_rows,
                   const size_t index) {
    auto acc = src_rows[index];

    for (size_t width = 1; width < kernel_.width(); ++width) {
      disable_loop_vectorization();
      acc = O::operation(acc, src_rows[index + width * src_rows.channels()]);
    }

    dst_rows[index] = acc;
  }

  Rectangle rect_;
  Rectangle kernel_;
};  // end of class HorizontalOp<ScalarType, O>

template <typename ScalarType>
class Min final {
 public:
  using VecTraits = neon::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;

  static VectorType operation(VectorType lhs, VectorType rhs) {
    return vminq_u8(lhs, rhs);
  }

  static ScalarType operation(ScalarType lhs, ScalarType rhs) {
    return std::min(lhs, rhs);
  }
};  // end of class Min<ScalarType>

template <typename ScalarType>
class Max final {
 public:
  using VecTraits = neon::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;

  static VectorType operation(VectorType lhs, VectorType rhs) {
    return vmaxq_u8(lhs, rhs);
  }

  static ScalarType operation(ScalarType lhs, ScalarType rhs) {
    return std::max(lhs, rhs);
  }
};  // end of class Max<ScalarType>

template <typename T>
using VerticalMin = VerticalOp<T, Min<T>>;
template <typename T>
using VerticalMax = VerticalOp<T, Max<T>>;

template <typename T>
using HorizontalMin = HorizontalOp<T, Min<T>>;
template <typename T>
using HorizontalMax = HorizontalOp<T, Max<T>>;

// Helper structure for dilate.
template <typename ScalarType>
class DilateOperation final {
 public:
  using SourceType = ScalarType;
  using BufferType = ScalarType;
  using DestinationType = ScalarType;
  using CopyData = MorphologyWorkspace::CopyDataMemcpy<ScalarType>;

  explicit DilateOperation(Rectangle kernel) : kernel_{kernel} {}

  void process_horizontal(Rectangle rect, Rows<const SourceType> src_rows,
                          Rows<BufferType> dst_rows) {
    neon::HorizontalMax<ScalarType>{rect, kernel_}.process_rows(src_rows,
                                                                dst_rows);
  }

  void process_vertical(Rectangle rect, IndirectRows<BufferType> src_rows,
                        Rows<DestinationType> dst_rows) {
    neon::VerticalMax<ScalarType>{rect, kernel_}.process_rows(src_rows,
                                                              dst_rows);
  }

 private:
  Rectangle kernel_;
};  // end of class DilateOperation<ScalarType>

template <typename T>
kleidicv_error_t dilate(const T *src, size_t src_stride, T *dst,
                        size_t dst_stride, size_t width, size_t height,
                        kleidicv_morphology_context_t *context) {
  CHECK_POINTERS(context);
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  auto *workspace = reinterpret_cast<MorphologyWorkspace *>(context);

  if (workspace->type_size() != sizeof(T)) {
    return KLEIDICV_ERROR_CONTEXT_MISMATCH;
  }

  Rectangle rect{width, height};
  if (workspace->image_size() != rect) {
    return KLEIDICV_ERROR_CONTEXT_MISMATCH;
  }

  // Currently valid, will need to be changed if morphology supports more border
  // types, like KLEIDICV_BORDER_TYPE_REVERSE.
  Rectangle kernel{workspace->kernel()};
  if (width < kernel.width() - 1 || height < kernel.height() - 1) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  Rows<const T> src_rows{src, src_stride, workspace->channels()};
  Rows<T> dst_rows{dst, dst_stride, workspace->channels()};
  Margin margin{workspace->kernel(), workspace->anchor()};
  Border<T> border{workspace->border_values()};

  Rows<const T> current_src_rows = src_rows;
  Rows<T> current_dst_rows = dst_rows;
  for (size_t iteration = 0; iteration < workspace->iterations(); ++iteration) {
    DilateOperation<T> operation{kernel};
    workspace->process(rect, current_src_rows, current_dst_rows, margin, border,
                       workspace->border_type(), operation);
    // Update source for the next iteration.
    current_src_rows = dst_rows;
  }
  return KLEIDICV_OK;
}

// Helper structure for erode.
template <typename ScalarType>
class ErodeOperation final {
 public:
  using SourceType = ScalarType;
  using BufferType = ScalarType;
  using DestinationType = ScalarType;
  using CopyData = MorphologyWorkspace::CopyDataMemcpy<ScalarType>;

  explicit ErodeOperation(Rectangle kernel) : kernel_{kernel} {}

  void process_horizontal(Rectangle rect, Rows<const SourceType> src_rows,
                          Rows<BufferType> dst_rows) {
    neon::HorizontalMin<ScalarType>{rect, kernel_}.process_rows(src_rows,
                                                                dst_rows);
  }

  void process_vertical(Rectangle rect, IndirectRows<BufferType> src_rows,
                        Rows<DestinationType> dst_rows) {
    neon::VerticalMin<ScalarType>{rect, kernel_}.process_rows(src_rows,
                                                              dst_rows);
  }

 private:
  Rectangle kernel_;
};  // end of class ErodeOperation<ScalarType>

template <typename T>
kleidicv_error_t erode(const T *src, size_t src_stride, T *dst,
                       size_t dst_stride, size_t width, size_t height,
                       kleidicv_morphology_context_t *context) {
  CHECK_POINTERS(context);
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  auto *workspace = reinterpret_cast<MorphologyWorkspace *>(context);

  if (workspace->type_size() != sizeof(T)) {
    return KLEIDICV_ERROR_CONTEXT_MISMATCH;
  }

  Rectangle rect{width, height};
  if (workspace->image_size() != rect) {
    return KLEIDICV_ERROR_CONTEXT_MISMATCH;
  }

  // Currently valid, will need to be changed if morphology supports more border
  // types, like KLEIDICV_BORDER_TYPE_REVERSE.
  Rectangle kernel{workspace->kernel()};
  if (width < kernel.width() - 1 || height < kernel.height() - 1) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  Rows<const T> src_rows{src, src_stride, workspace->channels()};
  Rows<T> dst_rows{dst, dst_stride, workspace->channels()};
  Margin margin{workspace->kernel(), workspace->anchor()};
  Border<T> border{workspace->border_values()};

  Rows<const T> current_src_rows = src_rows;
  Rows<T> current_dst_rows = dst_rows;
  for (size_t iteration = 0; iteration < workspace->iterations(); ++iteration) {
    ErodeOperation<T> operation{kernel};
    workspace->process(rect, current_src_rows, current_dst_rows, margin, border,
                       workspace->border_type(), operation);
    // Update source for the next iteration.
    current_src_rows = dst_rows;
  }
  return KLEIDICV_OK;
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(name, type)                       \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t name<type>(        \
      const type *src, size_t src_stride, type *dst, size_t dst_stride, \
      size_t width, size_t height, kleidicv_morphology_context_t *context)

KLEIDICV_INSTANTIATE_TEMPLATE(dilate, uint8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(erode, uint8_t);

}  // namespace kleidicv::neon
