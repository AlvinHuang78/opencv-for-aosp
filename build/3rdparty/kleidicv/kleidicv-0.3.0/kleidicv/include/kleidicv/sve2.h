// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_SVE2_H
#define KLEIDICV_SVE2_H

#include <arm_sve.h>

#include <utility>

#include "kleidicv/operations.h"
#include "kleidicv/utils.h"

// It is used by SVE2 and SME2, the actual namespace will reflect it.
namespace KLEIDICV_TARGET_NAMESPACE {

// Context associated with SVE operations.
class Context {
 public:
  explicit Context(svbool_t &pg) KLEIDICV_STREAMING_COMPATIBLE : pg_{pg} {}

  // Sets the predicate associated with the context to a given predicate.
  void set_predicate(svbool_t pg) KLEIDICV_STREAMING_COMPATIBLE { pg_ = pg; }

  // Returns predicate associated with the context.
  svbool_t predicate() const KLEIDICV_STREAMING_COMPATIBLE { return pg_; }

 protected:
  // Hold a reference to an svbool_t because a sizeless type cannot be a member.
  svbool_t &pg_;
};  // end of class Context

// Primary template to describe logically grouped properties of vectors.
template <typename ScalarType>
class VectorTypes;

template <>
class VectorTypes<int8_t> {
 public:
  using ScalarType = int8_t;
  using VectorType = svint8_t;
  using Vector2Type = svint8x2_t;
  using Vector3Type = svint8x3_t;
  using Vector4Type = svint8x4_t;
};  // end of class VectorTypes<int8_t>

template <>
class VectorTypes<uint8_t> {
 public:
  using ScalarType = uint8_t;
  using VectorType = svuint8_t;
  using Vector2Type = svuint8x2_t;
  using Vector3Type = svuint8x3_t;
  using Vector4Type = svuint8x4_t;
};  // end of class VectorTypes<uint8_t>

template <>
class VectorTypes<int16_t> {
 public:
  using ScalarType = int16_t;
  using VectorType = svint16_t;
  using Vector2Type = svint16x2_t;
  using Vector3Type = svint16x3_t;
  using Vector4Type = svint16x4_t;
};  // end of class VectorTypes<int16_t>

template <>
class VectorTypes<uint16_t> {
 public:
  using ScalarType = uint16_t;
  using VectorType = svuint16_t;
  using Vector2Type = svuint16x2_t;
  using Vector3Type = svuint16x3_t;
  using Vector4Type = svuint16x4_t;
};  // end of class VectorTypes<uint16_t>

template <>
class VectorTypes<int32_t> {
 public:
  using ScalarType = int32_t;
  using VectorType = svint32_t;
  using Vector2Type = svint32x2_t;
  using Vector3Type = svint32x3_t;
  using Vector4Type = svint32x4_t;
};  // end of class VectorTypes<int32_t>

template <>
class VectorTypes<uint32_t> {
 public:
  using ScalarType = uint32_t;
  using VectorType = svuint32_t;
  using Vector2Type = svuint32x2_t;
  using Vector3Type = svuint32x3_t;
  using Vector4Type = svuint32x4_t;
};  // end of class VectorTypes<uint32_t>

template <>
class VectorTypes<int64_t> {
 public:
  using ScalarType = int64_t;
  using VectorType = svint64_t;
  using Vector2Type = svint64x2_t;
  using Vector3Type = svint64x3_t;
  using Vector4Type = svint64x4_t;
};  // end of class VectorTypes<int64_t>

template <>
class VectorTypes<uint64_t> {
 public:
  using ScalarType = uint64_t;
  using VectorType = svuint64_t;
  using Vector2Type = svuint64x2_t;
  using Vector3Type = svuint64x3_t;
  using Vector4Type = svuint64x4_t;
};  // end of class VectorTypes<uint64_t>

template <>
class VectorTypes<float> {
 public:
  using ScalarType = float;
  using VectorType = svfloat32_t;
  using Vector2Type = svfloat32x2_t;
  using Vector3Type = svfloat32x3_t;
  using Vector4Type = svfloat32x4_t;
};  // end of class VectorTypes<float>

template <>
class VectorTypes<double> {
 public:
  using ScalarType = double;
  using VectorType = svfloat64_t;
  using Vector2Type = svfloat64x2_t;
  using Vector3Type = svfloat64x3_t;
  using Vector4Type = svfloat64x4_t;
};  // end of class VectorTypes<double>

// Base class for all SVE vector traits.
template <typename ScalarType>
class VecTraitsBase : public VectorTypes<ScalarType> {
 public:
  using typename VectorTypes<ScalarType>::VectorType;

  // Number of lanes in a vector.
  static inline size_t num_lanes() KLEIDICV_STREAMING_COMPATIBLE {
    return static_cast<size_t>(svcnt());
  }

  // Maximum number of lanes in a vector.
  static constexpr size_t max_num_lanes() KLEIDICV_STREAMING_COMPATIBLE {
    return 256 / sizeof(ScalarType);
  }

  // Loads a single vector from 'src'.
  static inline void load(Context ctx, const ScalarType *src,
                          VectorType &vec) KLEIDICV_STREAMING_COMPATIBLE {
    vec = svld1(ctx.predicate(), &src[0]);
  }

  // Loads two consecutive vectors from 'src'.
  static inline void load_consecutive(Context ctx, const ScalarType *src,
                                      VectorType &vec_0, VectorType &vec_1)
      KLEIDICV_STREAMING_COMPATIBLE {
    vec_0 = svld1(ctx.predicate(), &src[0]);
    vec_1 = svld1_vnum(ctx.predicate(), &src[0], 1);
  }

  // Stores a single vector to 'dst'.
  static inline void store(Context ctx, VectorType vec,
                           ScalarType *dst) KLEIDICV_STREAMING_COMPATIBLE {
    svst1(ctx.predicate(), &dst[0], vec);
  }

  // Stores two consecutive vectors to 'dst'.
  static inline void store_consecutive(Context ctx, VectorType vec_0,
                                       VectorType vec_1, ScalarType *dst)
      KLEIDICV_STREAMING_COMPATIBLE {
    svst1(ctx.predicate(), &dst[0], vec_0);
    svst1_vnum(ctx.predicate(), &dst[0], 1, vec_1);
  }

  template <typename T = ScalarType>
  static std::enable_if_t<sizeof(T) == sizeof(int8_t), uint64_t> svcnt()
      KLEIDICV_STREAMING_COMPATIBLE {
    return svcntb();
  }

  template <typename T = ScalarType>
  static std::enable_if_t<sizeof(T) == sizeof(int16_t), uint64_t> svcnt()
      KLEIDICV_STREAMING_COMPATIBLE {
    return svcnth();
  }

  template <typename T = ScalarType>
  static std::enable_if_t<sizeof(T) == sizeof(int32_t), uint64_t> svcnt()
      KLEIDICV_STREAMING_COMPATIBLE {
    return svcntw();
  }

  template <typename T = ScalarType>
  static std::enable_if_t<sizeof(T) == sizeof(int64_t), uint64_t> svcnt()
      KLEIDICV_STREAMING_COMPATIBLE {
    return svcntd();
  }

  template <typename T = ScalarType>
  static std::enable_if_t<sizeof(T) == sizeof(int8_t), uint64_t> svcntp(
      svbool_t pg) KLEIDICV_STREAMING_COMPATIBLE {
    return svcntp_b8(pg, pg);
  }

  template <typename T = ScalarType>
  static std::enable_if_t<sizeof(T) == sizeof(int16_t), uint64_t> svcntp(
      svbool_t pg) KLEIDICV_STREAMING_COMPATIBLE {
    return svcntp_b16(pg, pg);
  }

  template <typename T = ScalarType>
  static std::enable_if_t<sizeof(T) == sizeof(int32_t), uint64_t> svcntp(
      svbool_t pg) KLEIDICV_STREAMING_COMPATIBLE {
    return svcntp_b32(pg, pg);
  }

  template <typename T = ScalarType>
  static std::enable_if_t<sizeof(T) == sizeof(int64_t), uint64_t> svcntp(
      svbool_t pg) KLEIDICV_STREAMING_COMPATIBLE {
    return svcntp_b64(pg, pg);
  }

  template <typename T = ScalarType>
  static std::enable_if_t<sizeof(T) == sizeof(int8_t), svbool_t> svptrue()
      KLEIDICV_STREAMING_COMPATIBLE {
    return svptrue_b8();
  }

  template <typename T = ScalarType>
  static std::enable_if_t<sizeof(T) == sizeof(int16_t), svbool_t> svptrue()
      KLEIDICV_STREAMING_COMPATIBLE {
    return svptrue_b16();
  }

  template <typename T = ScalarType>
  static std::enable_if_t<sizeof(T) == sizeof(int32_t), svbool_t> svptrue()
      KLEIDICV_STREAMING_COMPATIBLE {
    return svptrue_b32();
  }

  template <typename T = ScalarType>
  static std::enable_if_t<sizeof(T) == sizeof(int64_t), svbool_t> svptrue()
      KLEIDICV_STREAMING_COMPATIBLE {
    return svptrue_b64();
  }

  template <typename IndexType, typename T = ScalarType>
  static std::enable_if_t<sizeof(T) == sizeof(int8_t), svbool_t> svwhilelt(
      IndexType index, IndexType max_index) KLEIDICV_STREAMING_COMPATIBLE {
    return svwhilelt_b8(index, max_index);
  }

  template <typename IndexType, typename T = ScalarType>
  static std::enable_if_t<sizeof(T) == sizeof(int16_t), svbool_t> svwhilelt(
      IndexType index, IndexType max_index) KLEIDICV_STREAMING_COMPATIBLE {
    return svwhilelt_b16(index, max_index);
  }

  template <typename IndexType, typename T = ScalarType>
  static std::enable_if_t<sizeof(T) == sizeof(int32_t), svbool_t> svwhilelt(
      IndexType index, IndexType max_index) KLEIDICV_STREAMING_COMPATIBLE {
    return svwhilelt_b32(index, max_index);
  }

  template <typename IndexType, typename T = ScalarType>
  static std::enable_if_t<sizeof(T) == sizeof(int64_t), svbool_t> svwhilelt(
      IndexType index, IndexType max_index) KLEIDICV_STREAMING_COMPATIBLE {
    return svwhilelt_b64(index, max_index);
  }

  // Transforms a single predicate into three other predicates that then can be
  // used for consecutive operations. The input predicate can only have
  // consecutive ones starting at the lowest element.
  static void make_consecutive_predicates(svbool_t pg, svbool_t &pg_0,
                                          svbool_t &pg_1, svbool_t &pg_2)
      KLEIDICV_STREAMING_COMPATIBLE {
    // Length of data. Must be signed because of the unconditional subtraction
    // of fixed values.
    int64_t length = 3 * svcntp(pg);
    // Handle up to VL length worth of data with the first predicated operation.
    pg_0 = svwhilelt(int64_t{0}, length);
    // Handle up to VL length worth of data with the second predicated operation
    // taking into account data stored in the first predicated operation.
    length -= num_lanes();
    pg_1 = svwhilelt(int64_t{0}, length);
    // Handle up to VL length worth of data with the second predicated operation
    // taking into account data stored in the first and second predicated
    // operations.
    length -= num_lanes();
    pg_2 = svwhilelt(int64_t{0}, length);
  }

  // Transforms a single predicate into four other predicates that then can be
  // used for consecutive operations. The input predicate can only have
  // consecutive ones starting at the lowest element.
  static void make_consecutive_predicates(
      svbool_t pg, svbool_t &pg_0, svbool_t &pg_1, svbool_t &pg_2,
      svbool_t &pg_3) KLEIDICV_STREAMING_COMPATIBLE {
    // Length of data. Must be signed because of the unconditional subtraction
    // of fixed values.
    int64_t length = 4 * svcntp(pg);
    // Handle up to VL length worth of data with the first predicated operation.
    pg_0 = svwhilelt(int64_t{0}, length);
    // Handle up to VL length worth of data with the second predicated operation
    // taking into account data stored in the first predicated operation.
    length -= num_lanes();
    pg_1 = svwhilelt(int64_t{0}, length);
    // Handle up to VL length worth of data with the second predicated operation
    // taking into account data stored in the first and second predicated
    // operations.
    length -= num_lanes();
    pg_2 = svwhilelt(int64_t{0}, length);
    // Handle up to VL length worth of data with the third predicated operation
    // taking into account data stored in the first, second and third predicated
    // operations.
    length -= num_lanes();
    pg_3 = svwhilelt(int64_t{0}, length);
  }
};  // end of class VecTraitsBase<ScalarType>

// Primary template for SVE vector traits.
template <typename ScalarType>
class VecTraits : public VecTraitsBase<ScalarType> {};

template <>
class VecTraits<int8_t> : public VecTraitsBase<int8_t> {
 public:
  static inline svint8_t svdup(int8_t v) KLEIDICV_STREAMING_COMPATIBLE {
    return svdup_s8(v);
  }
  static inline svint8_t svreinterpret(svuint8_t v)
      KLEIDICV_STREAMING_COMPATIBLE {
    return svreinterpret_s8(v);
  }
  static inline svint8_t svasr_n(svbool_t pg, svint8_t v,
                                 uint8_t s) KLEIDICV_STREAMING_COMPATIBLE {
    return svasr_n_s8_x(pg, v, s);
  }
};  // end of class VecTraits<int8_t>

template <>
class VecTraits<uint8_t> : public VecTraitsBase<uint8_t> {
 public:
  static inline svuint8_t svdup(uint8_t v) KLEIDICV_STREAMING_COMPATIBLE {
    return svdup_u8(v);
  }
  static inline svuint8_t svreinterpret(svint8_t v)
      KLEIDICV_STREAMING_COMPATIBLE {
    return svreinterpret_u8(v);
  }
  static inline svuint8_t svsub(svbool_t pg, svuint8_t v,
                                svuint8_t u) KLEIDICV_STREAMING_COMPATIBLE {
    return svsub_u8_x(pg, v, u);
  }
  static inline svuint8_t svhsub(svbool_t pg, svuint8_t v,
                                 svuint8_t u) KLEIDICV_STREAMING_COMPATIBLE {
    return svhsub_u8_x(pg, v, u);
  }
};  // end of class VecTraits<uint8_t>

template <>
class VecTraits<int16_t> : public VecTraitsBase<int16_t> {
 public:
  static inline svint16_t svdup(int16_t v) KLEIDICV_STREAMING_COMPATIBLE {
    return svdup_s16(v);
  }
  static inline svint16_t svreinterpret(svuint16_t v)
      KLEIDICV_STREAMING_COMPATIBLE {
    return svreinterpret_s16(v);
  }
};  // end of class VecTraits<int16_t>

template <>
class VecTraits<uint16_t> : public VecTraitsBase<uint16_t> {
 public:
  static inline svuint16_t svdup(uint16_t v) KLEIDICV_STREAMING_COMPATIBLE {
    return svdup_u16(v);
  }
  static inline svuint16_t svreinterpret(svint16_t v)
      KLEIDICV_STREAMING_COMPATIBLE {
    return svreinterpret_u16(v);
  }
};  // end of class VecTraits<uint16_t>

template <>
class VecTraits<int32_t> : public VecTraitsBase<int32_t> {
 public:
  static inline svint32_t svdup(int32_t v) KLEIDICV_STREAMING_COMPATIBLE {
    return svdup_s32(v);
  }
  static inline svint32_t svreinterpret(svuint32_t v)
      KLEIDICV_STREAMING_COMPATIBLE {
    return svreinterpret_s32(v);
  }
};  // end of class VecTraits<int32_t>

template <>
class VecTraits<uint32_t> : public VecTraitsBase<uint32_t> {
 public:
  static inline svuint32_t svdup(uint32_t v) KLEIDICV_STREAMING_COMPATIBLE {
    return svdup_u32(v);
  }
  static inline svuint32_t svreinterpret(svint32_t v)
      KLEIDICV_STREAMING_COMPATIBLE {
    return svreinterpret_u32(v);
  }
};  // end of class VecTraits<uint32_t>

template <>
class VecTraits<int64_t> : public VecTraitsBase<int64_t> {
 public:
  static inline svint64_t svdup(int64_t v) KLEIDICV_STREAMING_COMPATIBLE {
    return svdup_s64(v);
  }
  static inline svint64_t svreinterpret(svuint64_t v)
      KLEIDICV_STREAMING_COMPATIBLE {
    return svreinterpret_s64(v);
  }
};  // end of class VecTraits<int64_t>

template <>
class VecTraits<uint64_t> : public VecTraitsBase<uint64_t> {
 public:
  static inline svuint64_t svdup(uint64_t v) KLEIDICV_STREAMING_COMPATIBLE {
    return svdup_u64(v);
  }
  static inline svuint64_t svreinterpret(svint64_t v)
      KLEIDICV_STREAMING_COMPATIBLE {
    return svreinterpret_u64(v);
  }
};  // end of class VecTraits<uint64_t>

template <>
class VecTraits<float> : public VecTraitsBase<float> {
 public:
  static inline svfloat32_t svdup(float v) KLEIDICV_STREAMING_COMPATIBLE {
    return svdup_f32(v);
  }
  static inline svfloat32_t svsub(svbool_t pg, svfloat32_t v,
                                  svfloat32_t u) KLEIDICV_STREAMING_COMPATIBLE {
    return svsub_f32_x(pg, v, u);
  }
};  // end of class VecTraits<float>

template <>
class VecTraits<double> : public VecTraitsBase<double> {
 public:
  static inline svfloat64_t svdup(double v) KLEIDICV_STREAMING_COMPATIBLE {
    return svdup_f64(v);
  }
};  // end of class VecTraits<double>

// Adapter which adds context and forwards arguments.
template <typename OperationType>
class OperationContextAdapter : public OperationBase<OperationType> {
  // Shorten rows: no need to write 'this->'.
  using OperationBase<OperationType>::operation;
  using OperationBase<OperationType>::num_lanes;

 public:
  using ContextType = Context;
  using VecTraits = typename OperationBase<OperationType>::VecTraits;

  explicit OperationContextAdapter(OperationType &operation)
      KLEIDICV_STREAMING_COMPATIBLE : OperationBase<OperationType>(operation) {}

  // Forwards vector_path_2x() calls to the inner operation.
  template <typename... ArgTypes>
  void vector_path_2x(ArgTypes &&...args) KLEIDICV_STREAMING_COMPATIBLE {
    svbool_t ctx_pg;
    ContextType ctx{ctx_pg};
    ctx.set_predicate(VecTraits::svptrue());
    operation().vector_path_2x(ctx, std::forward<ArgTypes>(args)...);
  }

  // Forwards vector_path() calls to the inner operation.
  template <typename... ArgTypes>
  void vector_path(ArgTypes &&...args) KLEIDICV_STREAMING_COMPATIBLE {
    svbool_t ctx_pg;
    ContextType ctx{ctx_pg};
    ctx.set_predicate(VecTraits::svptrue());
    operation().vector_path(ctx, std::forward<ArgTypes>(args)...);
  }

  // Forwards remaining_path() calls to the inner operation if the concrete
  // operation is unrolled once.
  template <typename... ColumnTypes, typename T = OperationType>
  std::enable_if_t<T::is_unrolled_once()> remaining_path(
      size_t length, ColumnTypes &&...columns) KLEIDICV_STREAMING_COMPATIBLE {
    svbool_t ctx_pg;
    ContextType ctx{ctx_pg};
    ctx.set_predicate(VecTraits::svwhilelt(size_t{0}, length));
    operation().remaining_path(ctx, std::forward<ColumnTypes>(columns)...);
  }

  // Forwards remaining_path() calls to the inner operation if the concrete
  // operation is not unrolled once.
  template <typename... ColumnTypes, typename T = OperationType>
  std::enable_if_t<!T::is_unrolled_once()> remaining_path(
      size_t length, ColumnTypes... columns) KLEIDICV_STREAMING_COMPATIBLE {
    svbool_t ctx_pg;
    ContextType ctx{ctx_pg};

    size_t index = 0;
    ctx.set_predicate(VecTraits::svwhilelt(index, length));
    while (svptest_first(VecTraits::svptrue(), ctx.predicate())) {
      operation().remaining_path(ctx, columns.at(index)...);
      // Update loop counter and calculate the next governing predicate.
      index += num_lanes();
      ctx.set_predicate(VecTraits::svwhilelt(index, length));
    }
  }
};  // end of class OperationContextAdapter<OperationType>

// Adapter which implements remaining_path() for general SVE2 operations.
template <typename OperationType>
class RemainingPathAdapter : public OperationBase<OperationType> {
 public:
  using ContextType = Context;

  explicit RemainingPathAdapter(OperationType &operation)
      KLEIDICV_STREAMING_COMPATIBLE : OperationBase<OperationType>(operation) {}

  // Forwards remaining_path() to either vector_path() or tail_path() of the
  // inner operation depending on what is requested by the innermost operation.
  template <typename... ArgTypes>
  void remaining_path(ArgTypes... args) KLEIDICV_STREAMING_COMPATIBLE {
    if constexpr (OperationType::uses_tail_path()) {
      this->operation().tail_path(std::forward<ArgTypes>(args)...);
    } else {
      this->operation().vector_path(std::forward<ArgTypes>(args)...);
    }
  }
};  // end of class RemainingPathAdapter<OperationType>

// Shorthand for applying a generic unrolled SVE2 operation.
template <typename OperationType, typename... ArgTypes>
void apply_operation_by_rows(OperationType &operation,
                             ArgTypes &&...args) KLEIDICV_STREAMING_COMPATIBLE {
  ForwardingOperation forwarding_operation{operation};
  OperationAdapter operation_adapter{forwarding_operation};
  RemainingPathAdapter remaining_path_adapter{operation_adapter};
  OperationContextAdapter context_adapter{remaining_path_adapter};
  RowBasedOperation row_based_operation{context_adapter};
  zip_rows(row_based_operation, std::forward<ArgTypes>(args)...);
}

// Swap two variables, since some C++ Standard Library implementations do not
// allow using std::swap for SVE vectors.
template <typename T>
static inline void swap_scalable(T &a, T &b) KLEIDICV_STREAMING_COMPATIBLE {
  T tmp = a;
  a = b;
  b = tmp;
}

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_SVE2_H
