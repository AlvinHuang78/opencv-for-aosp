// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/arithmetics/transpose.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"

namespace kleidicv::neon {

template <const size_t BufferSize, const size_t Order, typename DstVectorType,
          typename SrcType>
static void transpose_vectors_recursively(DstVectorType *dst_vectors,
                                          Rows<const SrcType> src_rows) {
  // order is halved at every recursive call, once it is 2 the recursion should
  // stop and the input data needs to be read.
  if constexpr (Order == 2) {
    KLEIDICV_FORCE_LOOP_UNROLL
    for (size_t index = 0; index < BufferSize; index += Order) {
      using SrcVectorType = typename VecTraits<SrcType>::VectorType;
      SrcVectorType src_vector[2];

      src_vector[0] = vld1q(&src_rows.at(index + 0)[0]);
      src_vector[1] = vld1q(&src_rows.at(index + 1)[0]);

      // If order is 2 than SrcVectorType is the same as DstVectorType
      dst_vectors[index + 0] = vtrn1q(src_vector[0], src_vector[1]);
      dst_vectors[index + 1] = vtrn2q(src_vector[0], src_vector[1]);
    }
  } else {
    // First we need to create the input for the current transpose stage, which
    // is the output of the previous stage. The previous stage transposes
    // elements half the size of the current stage and its order is also half of
    // the current one.
    half_element_width_t<DstVectorType> input[BufferSize];
    constexpr size_t previous_order = Order / 2;

    transpose_vectors_recursively<BufferSize, previous_order>(input, src_rows);

    constexpr size_t half_order = Order / 2;

    KLEIDICV_FORCE_LOOP_UNROLL
    for (size_t outer_i = 0; outer_i < BufferSize; outer_i += Order) {
      KLEIDICV_FORCE_LOOP_UNROLL
      for (size_t inner_i = 0; inner_i < half_order; ++inner_i) {
        dst_vectors[outer_i + inner_i] =
            vtrn1q(reinterpret_cast<DstVectorType>(input[outer_i + inner_i]),
                   reinterpret_cast<DstVectorType>(
                       input[outer_i + inner_i + half_order]));

        dst_vectors[outer_i + half_order + inner_i] =
            vtrn2q(reinterpret_cast<DstVectorType>(input[outer_i + inner_i]),
                   reinterpret_cast<DstVectorType>(
                       input[outer_i + inner_i + half_order]));
      }
    }
  }
}

// Transposes one tile of data with vector instructions. The tile's width and
// height are the number of NEON lanes for the given type.
template <typename ScalarType>
static void vector_path(Rows<const ScalarType> src_rows,
                        Rows<ScalarType> dst_rows) {
  constexpr size_t num_of_lanes = VecTraits<ScalarType>::num_lanes();

  // The number of vectors read and write is the same as the lane count of the
  // given element size
  constexpr size_t buffer_size = num_of_lanes;

  // Last transpose step is always done on 64 bit elements
  uint64x2_t trn_result_b64[buffer_size];  // NOLINT(runtime/arrays)

  // The 64 bit transpose spans through all the vectors, so its "order" is the
  // same as the number of vectors
  constexpr size_t transpose_order_b64 = num_of_lanes;

  transpose_vectors_recursively<buffer_size, transpose_order_b64>(
      trn_result_b64, src_rows);

  KLEIDICV_FORCE_LOOP_UNROLL
  for (size_t index = 0; index < buffer_size; ++index) {
    vst1q(&dst_rows.at(index)[0], trn_result_b64[index]);
  }
}

template <typename ScalarType>
static void scalar_path(Rows<const ScalarType> src_rows,
                        Rows<ScalarType> dst_rows, size_t height,
                        size_t width) {
  for (size_t vindex = 0; vindex < height; ++vindex) {
    disable_loop_vectorization();
    for (size_t hindex = 0; hindex < width; ++hindex) {
      disable_loop_vectorization();
      dst_rows.at(hindex)[vindex] = src_rows.at(vindex)[hindex];
    }
  }
}

template <typename ScalarType>
static kleidicv_error_t transpose(Rectangle rect,
                                  Rows<const ScalarType> src_rows,
                                  Rows<ScalarType> dst_rows) {
  constexpr size_t num_of_lanes = VecTraits<ScalarType>::num_lanes();

  auto handle_lane_number_of_rows = [&](size_t vindex) {
    LoopUnroll2<TryToAvoidTailLoop> horizontal_loop(rect.width(), num_of_lanes);

    horizontal_loop.unroll_once([&](size_t hindex) {
      // if the input is big enough handle it tile by tile
      vector_path<ScalarType>(src_rows.at(vindex, hindex),
                              dst_rows.at(hindex, vindex));
    });

    horizontal_loop.remaining([&](size_t hindex, size_t final_hindex) {
      scalar_path(src_rows.at(vindex, hindex), dst_rows.at(hindex, vindex),
                  num_of_lanes, final_hindex - hindex);
    });
  };

  LoopUnroll2<TryToAvoidTailLoop> vertical_loop(rect.height(), num_of_lanes);

  vertical_loop.unroll_once(handle_lane_number_of_rows);

  vertical_loop.remaining([&](size_t hindex, size_t final_hindex) {
    scalar_path(src_rows.at(hindex), dst_rows.at(0, hindex),
                final_hindex - hindex, rect.width());
  });
  return KLEIDICV_OK;
}

template <typename ScalarType>
static kleidicv_error_t transpose(Rectangle rect, Rows<ScalarType> data_rows) {
  constexpr size_t num_of_lanes = VecTraits<ScalarType>::num_lanes();

  // rect.width() needs to be equal to rect.height()
  LoopUnroll2 outer_loop(rect.width(), num_of_lanes);

  outer_loop.unroll_once([&](size_t vindex) {
    // Handle tiles on the diagonal line
    vector_path<ScalarType>(data_rows.at(vindex, vindex),
                            data_rows.at(vindex, vindex));

    // Handle the top right half
    if (rect.width() > (vindex + num_of_lanes)) {
      // Indexes are running through only the top right half
      LoopUnroll2 inner_loop(vindex + num_of_lanes, rect.width(), num_of_lanes);

      inner_loop.unroll_once([&](size_t hindex) {
        // Allocate temporary memory for one tile
        ScalarType tmp[num_of_lanes * num_of_lanes];  // NOLINT(runtime/arrays)
        Rows<ScalarType> tmp_rows{tmp, num_of_lanes * sizeof(ScalarType)};

        // Transpose a tile from the top right area, save the result
        // into temporary memory
        vector_path<ScalarType>(data_rows.at(vindex, hindex), tmp_rows);
        // Transpose its mirror tile from the left bottom area, save the
        // result to its final space
        vector_path<ScalarType>(data_rows.at(hindex, vindex),
                                data_rows.at(vindex, hindex));
        // Copy the temprory result to its final destination
        Rows<const ScalarType> const_tmp_rows{
            tmp, num_of_lanes * sizeof(ScalarType)};
        CopyNonOverlappingRows<ScalarType>::copy_rows(
            Rectangle{num_of_lanes, num_of_lanes}, const_tmp_rows,
            data_rows.at(hindex, vindex));
      });

      inner_loop.remaining([&](size_t hindex, size_t final_hindex) {
        // As this is the unroll_once path of the outer_loop there is
        // num_of_lanes worth of data in the vertical direction
        for (size_t i = vindex; i < vindex + num_of_lanes; ++i) {
          disable_loop_vectorization();
          for (size_t j = hindex; j < final_hindex; ++j) {
            disable_loop_vectorization();
            std::swap(data_rows.at(i)[j], data_rows.at(j)[i]);
          }
        }
      });
    }
  });

  outer_loop.remaining([&](size_t vindex, size_t final_vindex) {
    for (size_t i = vindex; i < final_vindex; ++i) {
      disable_loop_vectorization();
      // Only the top right half pixels need to be indexed
      for (size_t j = i + 1; j < final_vindex; ++j) {
        disable_loop_vectorization();
        std::swap(data_rows.at(i)[j], data_rows.at(j)[i]);
      }
    }
  });
  return KLEIDICV_OK;
}

template <typename T>
static kleidicv_error_t transpose(const void *src_void, size_t src_stride,
                                  void *dst_void, size_t dst_stride,
                                  size_t src_width, size_t src_height) {
  MAKE_POINTER_CHECK_ALIGNMENT(const T, src, src_void);
  MAKE_POINTER_CHECK_ALIGNMENT(T, dst, dst_void);
  CHECK_POINTER_AND_STRIDE(src, src_stride, src_height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, src_width);
  CHECK_IMAGE_SIZE(src_width, src_height);

  Rectangle rect{src_width, src_height};
  Rows<T> dst_rows{dst, dst_stride};

  if (src == dst) {
    if (src_width != src_height) {
      // Inplace transpose only implemented if width and height are the same
      return KLEIDICV_ERROR_NOT_IMPLEMENTED;
    }
    return transpose(rect, dst_rows);
  }
  Rows<const T> src_rows{src, src_stride};
  return transpose(rect, src_rows, dst_rows);
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t transpose(const void *src, size_t src_stride, void *dst,
                           size_t dst_stride, size_t src_width,
                           size_t src_height, size_t element_size) {
  switch (element_size) {
    case sizeof(uint8_t):
      return transpose<uint8_t>(src, src_stride, dst, dst_stride, src_width,
                                src_height);
    case sizeof(uint16_t):
      return transpose<uint16_t>(src, src_stride, dst, dst_stride, src_width,
                                 src_height);
    case sizeof(uint32_t):
      return transpose<uint32_t>(src, src_stride, dst, dst_stride, src_width,
                                 src_height);
    case sizeof(uint64_t):
      return transpose<uint64_t>(src, src_stride, dst, dst_stride, src_width,
                                 src_height);
    default:
      return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }
}

}  // namespace kleidicv::neon
