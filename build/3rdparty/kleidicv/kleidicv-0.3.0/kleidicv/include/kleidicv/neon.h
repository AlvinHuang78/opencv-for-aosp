// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_NEON_H
#define KLEIDICV_NEON_H

#include <utility>

#include "kleidicv/neon_intrinsics.h"
#include "kleidicv/operations.h"
#include "kleidicv/utils.h"

namespace kleidicv::neon {

template <>
class half_element_width<uint16x8_t> {
 public:
  using type = uint8x16_t;
};

template <>
class half_element_width<uint32x4_t> {
 public:
  using type = uint16x8_t;
};

template <>
class half_element_width<uint64x2_t> {
 public:
  using type = uint32x4_t;
};

template <>
class double_element_width<uint8x16_t> {
 public:
  using type = uint16x8_t;
};

template <>
class double_element_width<uint16x8_t> {
 public:
  using type = uint32x4_t;
};

template <>
class double_element_width<uint32x4_t> {
 public:
  using type = uint64x2_t;
};

// Primary template to describe logically grouped peroperties of vectors.
template <typename ScalarType>
class VectorTypes;

template <>
class VectorTypes<int8_t> {
 public:
  using ScalarType = int8_t;
  using VectorType = int8x16_t;
  using Vector2Type = int8x16x2_t;
  using Vector3Type = int8x16x3_t;
  using Vector4Type = int8x16x4_t;
};  // end of class VectorTypes<int8_t>

template <>
class VectorTypes<uint8_t> {
 public:
  using ScalarType = uint8_t;
  using VectorType = uint8x16_t;
  using Vector2Type = uint8x16x2_t;
  using Vector3Type = uint8x16x3_t;
  using Vector4Type = uint8x16x4_t;
};  // end of class VectorTypes<uint8_t>

template <>
class VectorTypes<int16_t> {
 public:
  using ScalarType = int16_t;
  using VectorType = int16x8_t;
  using Vector2Type = int16x8x2_t;
  using Vector3Type = int16x8x3_t;
  using Vector4Type = int16x8x4_t;
};  // end of class VectorTypes<int16_t>

template <>
class VectorTypes<uint16_t> {
 public:
  using ScalarType = uint16_t;
  using VectorType = uint16x8_t;
  using Vector2Type = uint16x8x2_t;
  using Vector3Type = uint16x8x3_t;
  using Vector4Type = uint16x8x4_t;
};  // end of class VectorTypes<uint16_t>

template <>
class VectorTypes<int32_t> {
 public:
  using ScalarType = int32_t;
  using VectorType = int32x4_t;
  using Vector2Type = int32x4x2_t;
  using Vector3Type = int32x4x3_t;
  using Vector4Type = int32x4x4_t;
};  // end of class VectorTypes<int32_t>

template <>
class VectorTypes<uint32_t> {
 public:
  using ScalarType = uint32_t;
  using VectorType = uint32x4_t;
  using Vector2Type = uint32x4x2_t;
  using Vector3Type = uint32x4x3_t;
  using Vector4Type = uint32x4x4_t;
};  // end of class VectorTypes<uint32_t>

template <>
class VectorTypes<int64_t> {
 public:
  using ScalarType = int64_t;
  using VectorType = int64x2_t;
  using Vector2Type = int64x2x2_t;
  using Vector3Type = int64x2x3_t;
  using Vector4Type = int64x2x4_t;
};  // end of class VectorTypes<int64_t>

template <>
class VectorTypes<uint64_t> {
 public:
  using ScalarType = uint64_t;
  using VectorType = uint64x2_t;
  using Vector2Type = uint64x2x2_t;
  using Vector3Type = uint64x2x3_t;
  using Vector4Type = uint64x2x4_t;
};  // end of class VectorTypes<uint64_t>

template <>
class VectorTypes<float> {
 public:
  using ScalarType = float;
  using VectorType = float32x4_t;
  using Vector2Type = float32x4x2_t;
  using Vector3Type = float32x4x3_t;
  using Vector4Type = float32x4x4_t;
};  // end of class VectorTypes<float>

template <>
class VectorTypes<double> {
 public:
  using ScalarType = double;
  using VectorType = float64x2_t;
  using Vector2Type = float64x2x2_t;
  using Vector3Type = float64x2x3_t;
  using Vector4Type = float64x2x4_t;
};  // end of class VectorTypes<double>

// Base class for all NEON vector traits.
template <typename ScalarType>
class VecTraitsBase : public VectorTypes<ScalarType> {
 public:
  using typename VectorTypes<ScalarType>::VectorType;
  using typename VectorTypes<ScalarType>::Vector2Type;
  using typename VectorTypes<ScalarType>::Vector3Type;
  using typename VectorTypes<ScalarType>::Vector4Type;

  // Vector length in bytes.
  static constexpr size_t kVectorLength = 16;

  // Number of lanes in a vector.
  static constexpr size_t num_lanes() {
    return kVectorLength / sizeof(ScalarType);
  }

  // Maximum number of lanes in a vector.
  static constexpr size_t max_num_lanes() { return num_lanes(); }

  // Loads a single vector from 'src'.
  static inline void load(const ScalarType *src, VectorType &vec) {
    vec = vld1q(&src[0]);
  }

  // Loads two consecutive vectors from 'src'.
  static inline void load(const ScalarType *src, Vector2Type &vec) {
    vec = vld1q_x2(&src[0]);
  }

  // Loads three consecutive vectors from 'src'.
  static inline void load(const ScalarType *src, Vector3Type &vec) {
    vec = vld1q_x3(&src[0]);
  }

  // Loads four consecutive vectors from 'src'.
  static inline void load(const ScalarType *src, Vector4Type &vec) {
    vec = vld1q_x4(&src[0]);
  }

  // Loads two consecutive vectors from 'src'.
  static inline void load_consecutive(const ScalarType *src, VectorType &vec_0,
                                      VectorType &vec_1) {
    vec_0 = vld1q(&src[0]);
    vec_1 = vld1q(&src[num_lanes()]);
  }

  // Loads 2x2 consecutive vectors from 'src'.
  static inline void load_consecutive(const ScalarType *src, Vector2Type &vec_0,
                                      Vector2Type &vec_1) {
    vec_0 = vld1q_x2(&src[0]);
    vec_1 = vld1q_x2(&src[num_lanes() * 2]);
  }

  // Loads 2x3 consecutive vectors from 'src'.
  static inline void load_consecutive(const ScalarType *src, Vector3Type &vec_0,
                                      Vector3Type &vec_1) {
    vec_0 = vld1q_x3(&src[0]);
    vec_1 = vld1q_x3(&src[num_lanes() * 3]);
  }

  // Loads 2x4 consecutive vectors from 'src'.
  static inline void load_consecutive(const ScalarType *src, Vector4Type &vec_0,
                                      Vector4Type &vec_1) {
    vec_0 = vld1q_x4(&src[0]);
    vec_1 = vld1q_x4(&src[num_lanes() * 4]);
  }

  // Stores a single vector to 'dst'.
  static inline void store(VectorType vec, ScalarType *dst) {
    vst1q(&dst[0], vec);
  }

  // Stores two consecutive vectors to 'dst'.
  static inline void store(Vector2Type vec, ScalarType *dst) {
    vst1q_x2(&dst[0], vec);
  }

  // Stores two consecutive vectors to 'dst'.
  static inline void store_consecutive(VectorType vec_0, VectorType vec_1,
                                       ScalarType *dst) {
    vst1q(&dst[0], vec_0);
    vst1q(&dst[num_lanes()], vec_1);
  }
};  // end of class VecTraitsBase<ScalarType>

// Available NEON vector traits.
template <typename ScalarType>
class VecTraits : public VecTraitsBase<ScalarType> {};

// NEON has no associated context yet.
using NeonContextType = Monostate;

// Adapter which simply adds context and forwards all arguments.
template <typename OperationType>
class OperationContextAdapter : public OperationBase<OperationType> {
  // Shorten rows: no need to write 'this->'.
  using OperationBase<OperationType>::operation;

 public:
  using ContextType = NeonContextType;

  explicit OperationContextAdapter(OperationType &operation)
      : OperationBase<OperationType>(operation) {}

  // Forwards vector_path_2x() calls to the inner operation.
  template <typename... ArgTypes>
  void vector_path_2x(ArgTypes &&...args) {
    operation().vector_path_2x(ContextType{}, std::forward<ArgTypes>(args)...);
  }

  // Forwards vector_path() calls to the inner operation.
  template <typename... ArgTypes>
  void vector_path(ArgTypes &&...args) {
    operation().vector_path(ContextType{}, std::forward<ArgTypes>(args)...);
  }

  // Forwards remaining_path() calls to the inner operation.
  template <typename... ArgTypes>
  void remaining_path(ArgTypes &&...args) {
    operation().remaining_path(ContextType{}, std::forward<ArgTypes>(args)...);
  }
};  // end of class OperationContextAdapter<OperationType>

// Adapter which implements remaining_path() for general NEON operations.
template <typename OperationType>
class RemainingPathAdapter : public OperationBase<OperationType> {
 public:
  using ContextType = NeonContextType;

  explicit RemainingPathAdapter(OperationType &operation)
      : OperationBase<OperationType>(operation) {}

  // Forwards remaining_path() calls to scalar_path() of the inner operation
  // element by element.
  template <typename... ColumnTypes>
  void remaining_path(ContextType ctx, size_t length, ColumnTypes... columns) {
    for (size_t index = 0; index < length; ++index) {
      disable_loop_vectorization();
      this->operation().scalar_path(ctx, columns.at(index)...);
    }
  }
};  // end of class RemainingPathAdapter<OperationType>

// Adapter which implements remaining_path() for NEON operations which
// implementation custom processing of remaining elements.
template <typename OperationType>
class RemainingPathToScalarPathAdapter : public OperationBase<OperationType> {
 public:
  using ContextType = NeonContextType;

  explicit RemainingPathToScalarPathAdapter(OperationType &operation)
      : OperationBase<OperationType>(operation) {}

  // Forwards remaining_path() calls to scalar_path() of the inner operation.
  template <typename... ArgTypes>
  void remaining_path(ArgTypes &&...args) {
    this->operation().scalar_path(std::forward<ArgTypes>(args)...);
  }
};  // end of class RemainingPathToScalarPathAdapter<OperationType>

// Shorthand for applying a generic unrolled NEON operation.
template <typename OperationType, typename... ArgTypes>
void apply_operation_by_rows(OperationType &operation, ArgTypes &&...args) {
  RemoveContextAdapter remove_context_adapter{operation};
  OperationAdapter operation_adapter{remove_context_adapter};
  RemainingPathAdapter remaining_path_adapter{operation_adapter};
  OperationContextAdapter context_adapter{remaining_path_adapter};
  RowBasedOperation row_based_operation{context_adapter};
  zip_rows(row_based_operation, std::forward<ArgTypes>(args)...);
}

// Shorthand for applying a generic unrolled and block-based NEON operation.
template <typename OperationType, typename... ArgTypes>
void apply_block_operation_by_rows(OperationType &operation,
                                   ArgTypes &&...args) {
  RemoveContextAdapter remove_context_adapter{operation};
  OperationAdapter operation_adapter{remove_context_adapter};
  RemainingPathAdapter remaining_path_adapter{operation_adapter};
  OperationContextAdapter context_adapter{remaining_path_adapter};
  RowBasedBlockOperation block_operation{context_adapter};
  zip_rows(block_operation, std::forward<ArgTypes>(args)...);
}

}  // namespace kleidicv::neon

#endif  // KLEIDICV_NEON_H
