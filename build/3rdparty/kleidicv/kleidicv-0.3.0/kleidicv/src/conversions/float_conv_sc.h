// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_FLOAT_CONV_SC_H
#define KLEIDICV_FLOAT_CONV_SC_H

#include <limits>
#include <type_traits>

#include "kleidicv/kleidicv.h"
#include "kleidicv/sve2.h"

namespace KLEIDICV_TARGET_NAMESPACE {

template <typename InputType, typename OutputType>
class float_conversion_operation;

template <typename OutputType>
class float_conversion_operation<float, OutputType> {
 public:
  using SrcVecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<float>;
  using SrcVectorType = typename SrcVecTraits::VectorType;
  using IntermediateVecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<
      std::conditional_t<std::is_signed_v<OutputType>, int32_t, uint32_t>>;
  using IntermediateVectorType = typename IntermediateVecTraits::VectorType;
  using DstVecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<OutputType>;
  using DstVectorType = typename DstVecTraits::VectorType;

  explicit float_conversion_operation(svuint8_t& index)
      KLEIDICV_STREAMING_COMPATIBLE : index_(index) {
    // Index generation to reorder converted values by tbl instruction
    auto index0 = svindex_u8(0, 4);
    auto index1 = svindex_u8(1, 4);
    auto index2 = svindex_u8(2, 4);
    auto index3 = svindex_u8(3, 4);

    svbool_t pg = svwhilelt_b8(uint64_t(0), svcntb() / 4);

    index_ = svsplice(pg, index3, svdup_u8(0));
    index_ = svsplice(pg, index2, index_);
    index_ = svsplice(pg, index1, index_);
    index_ = svsplice(pg, index0, index_);
  }

  void process_row(size_t width, Columns<const float> src,
                   Columns<OutputType> dst) KLEIDICV_STREAMING_COMPATIBLE {
    LoopUnroll{width, SrcVecTraits::num_lanes()}
        .unroll_n_times<4>([&](size_t step) KLEIDICV_STREAMING_COMPATIBLE {
          svbool_t pg = DstVecTraits::svptrue();
          SrcVectorType src_v0 = svld1(pg, &src[0]);
          SrcVectorType src_v1 = svld1_vnum(pg, &src[0], 1);
          SrcVectorType src_v2 = svld1_vnum(pg, &src[0], 2);
          SrcVectorType src_v3 = svld1_vnum(pg, &src[0], 3);
          DstVectorType res0 = vector_path(pg, src_v0, src_v1, src_v2, src_v3);
          svst1(pg, &dst[0], res0);
          src += ptrdiff_t(step);
          dst += ptrdiff_t(step);
        })
        .remaining([&](size_t length, size_t) KLEIDICV_STREAMING_COMPATIBLE {
          size_t index = 0;
          svbool_t pg = SrcVecTraits::svwhilelt(index, length);
          while (svptest_first(SrcVecTraits::svptrue(), pg)) {
            SrcVectorType src_vector = svld1(pg, &src[ptrdiff_t(index)]);
            IntermediateVectorType result_vector =
                remaining_path<OutputType>(pg, src_vector);
            svst1b(pg, &dst[ptrdiff_t(index)], result_vector);
            // Update loop counter and calculate the next governing predicate.
            index += SrcVecTraits::num_lanes();
            pg = SrcVecTraits::svwhilelt(index, length);
          }
        });
  }

 private:
  template <
      typename O,
      std::enable_if_t<std::is_integral_v<O> && std::is_signed_v<O>, int> = 0>
  decltype(auto) convert(svbool_t full_pg,
                         SrcVectorType in) KLEIDICV_STREAMING_COMPATIBLE {
    return svcvt_s32_f32_x(full_pg, in);
  }

  template <
      typename O,
      std::enable_if_t<std::is_integral_v<O> && !std::is_signed_v<O>, int> = 0>
  decltype(auto) convert(svbool_t full_pg,
                         SrcVectorType in) KLEIDICV_STREAMING_COMPATIBLE {
    return svcvt_u32_f32_x(full_pg, in);
  }

  DstVectorType vector_path(svbool_t full_pg, SrcVectorType fsrc0,
                            SrcVectorType fsrc1, SrcVectorType fsrc2,
                            SrcVectorType fsrc3) KLEIDICV_STREAMING_COMPATIBLE {
    fsrc0 = svrinti_f32_x(full_pg, fsrc0);
    fsrc1 = svrinti_f32_x(full_pg, fsrc1);
    fsrc2 = svrinti_f32_x(full_pg, fsrc2);
    fsrc3 = svrinti_f32_x(full_pg, fsrc3);

    auto _32bit_res0 = convert<OutputType>(full_pg, fsrc0);
    auto _32bit_res1 = convert<OutputType>(full_pg, fsrc1);
    auto _32bit_res2 = convert<OutputType>(full_pg, fsrc2);
    auto _32bit_res3 = convert<OutputType>(full_pg, fsrc3);

    auto _16bit_res0 = svqxtnb(_32bit_res0);
    _16bit_res0 = svqxtnt(_16bit_res0, _32bit_res2);

    auto _16bit_res1 = svqxtnb(_32bit_res1);
    _16bit_res1 = svqxtnt(_16bit_res1, _32bit_res3);

    auto _8bit_res = svqxtnb(_16bit_res0);
    _8bit_res = svqxtnt(_8bit_res, _16bit_res1);

    return svtbl(_8bit_res, index_);
  }

  template <
      typename O,
      std::enable_if_t<std::is_integral_v<O> && std::is_signed_v<O>, int> = 0>
  IntermediateVectorType remaining_path(svbool_t& pg, SrcVectorType src)
      KLEIDICV_STREAMING_COMPATIBLE {
    constexpr float min_val = std::numeric_limits<O>::lowest();
    constexpr float max_val = std::numeric_limits<O>::max();

    src = svrinti_f32_x(pg, src);

    svbool_t less = svcmplt_n_f32(pg, src, min_val);
    src = svdup_n_f32_m(src, less, min_val);

    svbool_t greater = svcmpgt_n_f32(pg, src, max_val);
    src = svdup_n_f32_m(src, greater, max_val);

    return svcvt_s32_f32_x(pg, src);
  }

  template <
      typename O,
      std::enable_if_t<std::is_integral_v<O> && !std::is_signed_v<O>, int> = 0>
  IntermediateVectorType remaining_path(svbool_t& pg, SrcVectorType src)
      KLEIDICV_STREAMING_COMPATIBLE {
    constexpr float max_val = std::numeric_limits<O>::max();

    src = svrinti_f32_x(pg, src);

    svbool_t greater = svcmpgt_n_f32(pg, src, max_val);
    src = svdup_n_f32_m(src, greater, max_val);

    return svcvt_u32_f32_x(pg, src);
  }

  svuint8_t& index_;
};  // end of class float_conversion_operation<float, OutputType>

template <typename InputType>
class float_conversion_operation<InputType, float> {
 public:
  using VecTraits = KLEIDICV_TARGET_NAMESPACE::VecTraits<float>;
  using VectorType = typename VecTraits::VectorType;

  explicit float_conversion_operation(svuint8_t&) {}

  void process_row(size_t width, Columns<const InputType> src,
                   Columns<float> dst) KLEIDICV_STREAMING_COMPATIBLE {
    LoopUnroll{width, VecTraits::num_lanes()}
        .unroll_twice([&](size_t step) KLEIDICV_STREAMING_COMPATIBLE {
          svbool_t pg = VecTraits::svptrue();
          auto src_vect1 = load_src(pg, &src[0], 0);
          auto src_vect2 = load_src(pg, &src[0], 1);

          VectorType dst_vector1 = vector_path(pg, src_vect1);
          VectorType dst_vector2 = vector_path(pg, src_vect2);
          svst1(pg, &dst[0], dst_vector1);
          svst1_vnum(pg, &dst[0], 1, dst_vector2);
          src += ptrdiff_t(step);
          dst += ptrdiff_t(step);
        })
        .remaining([&](size_t length, size_t) KLEIDICV_STREAMING_COMPATIBLE {
          size_t index = 0;
          svbool_t pg = VecTraits::svwhilelt(index, length);
          while (svptest_first(VecTraits::svptrue(), pg)) {
            auto src_vect = load_src(pg, &src[ptrdiff_t(index)], 0);
            VectorType dst_vector = vector_path(pg, src_vect);
            svst1(pg, &dst[ptrdiff_t(index)], dst_vector);
            // Update loop counter and calculate the next governing predicate.
            index += VecTraits::num_lanes();
            pg = VecTraits::svwhilelt(index, length);
          }
        });
  }

 private:
  template <typename I, std::enable_if_t<std::is_same_v<I, svint32_t>, int> = 0>
  VectorType vector_path(svbool_t& pg,
                         I src_vector) KLEIDICV_STREAMING_COMPATIBLE {
    return svcvt_f32_s32_x(pg, src_vector);
  }
  template <typename I,
            std::enable_if_t<std::is_same_v<I, svuint32_t>, int> = 0>
  VectorType vector_path(svbool_t& pg,
                         I src_vector) KLEIDICV_STREAMING_COMPATIBLE {
    return svcvt_f32_u32_x(pg, src_vector);
  }

  template <
      typename I,
      std::enable_if_t<std::is_integral_v<I> && std::is_signed_v<I>, int> = 0>
  svint32_t load_src(svbool_t& pg, const I* src,
                     size_t vnum) KLEIDICV_STREAMING_COMPATIBLE {
    svint32_t src_vect = svld1sb_vnum_s32(pg, src, vnum);
    return src_vect;
  }

  template <
      typename I,
      std::enable_if_t<std::is_integral_v<I> && !std::is_signed_v<I>, int> = 0>
  svuint32_t load_src(svbool_t& pg, const I* src,
                      size_t vnum) KLEIDICV_STREAMING_COMPATIBLE {
    svuint32_t src_vect = svld1ub_vnum_u32(pg, src, vnum);
    return src_vect;
  }
};  // end of class float_conversion_operation<InputType, float>

template <typename InputType, typename OutputType>
static kleidicv_error_t float_conversion_sc(
    const InputType* src, size_t src_stride, OutputType* dst, size_t dst_stride,
    size_t width, size_t height) KLEIDICV_STREAMING_COMPATIBLE {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  svuint8_t index;
  float_conversion_operation<InputType, OutputType> operation{index};
  Rectangle rect{width, height};
  Rows<const InputType> src_rows{src, src_stride};
  Rows<OutputType> dst_rows{dst, dst_stride};
  zip_rows(operation, rect, src_rows, dst_rows);

  return KLEIDICV_OK;
}

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_FLOAT_CONV_SC_H
