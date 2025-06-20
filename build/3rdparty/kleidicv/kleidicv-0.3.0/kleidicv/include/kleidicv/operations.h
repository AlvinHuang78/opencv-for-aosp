// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_OPERATIONS_H
#define KLEIDICV_OPERATIONS_H

#include <algorithm>
#include <utility>

#include "kleidicv/traits.h"
#include "kleidicv/types.h"

namespace KLEIDICV_TARGET_NAMESPACE {

// Base for classes that change the behaviour or extend the interface of an
// operation.
//
// This class conveniently forwards common types and methods without having to
// explicitly define them in all derived classes.
template <typename T>
class OperationBase {
 public:
  // Type of the inner operation.
  using OperationType = T;

  // The vector traits used by the inner operation.
  using VecTraits = typename OperationType::VecTraits;

  // The context type used by the inner operation, if any.
  using ContextType = context_type_t<OperationType>;

  // Returns a reference to the inner operation.
  OperationType &operation() KLEIDICV_STREAMING_COMPATIBLE {
    return operation_;
  }

  // Forwards num_lanes() calls to the inner operation.
  static size_t num_lanes() KLEIDICV_STREAMING_COMPATIBLE {
    return VecTraits::num_lanes();
  }

  // Forwards vector_path_2x() calls to the inner operation.
  template <typename... ArgTypes>
  decltype(auto) vector_path_2x(ArgTypes &&...args)
      KLEIDICV_STREAMING_COMPATIBLE {
    return operation().vector_path_2x(std::forward<ArgTypes>(args)...);
  }

  // Forwards vector_path() calls to the inner operation.
  template <typename... ArgTypes>
  decltype(auto) vector_path(ArgTypes &&...args) KLEIDICV_STREAMING_COMPATIBLE {
    return operation().vector_path(std::forward<ArgTypes>(args)...);
  }

  // Forwards remaining_path() calls to the inner operation.
  template <typename... ArgTypes>
  decltype(auto) remaining_path(ArgTypes &&...args)
      KLEIDICV_STREAMING_COMPATIBLE {
    return operation().remaining_path(std::forward<ArgTypes>(args)...);
  }

  // Forwards tail_path() calls to the inner operation.
  template <typename... ArgTypes>
  decltype(auto) tail_path(ArgTypes &&...args) KLEIDICV_STREAMING_COMPATIBLE {
    return operation().tail_path(std::forward<ArgTypes>(args)...);
  }

  // Forwards scalar_path() calls to the inner operation.
  template <typename... ArgTypes>
  decltype(auto) scalar_path(ArgTypes &&...args) KLEIDICV_STREAMING_COMPATIBLE {
    return operation().scalar_path(std::forward<ArgTypes>(args)...);
  }

  template <typename... ArgTypes>
  decltype(auto) load(ArgTypes &&...args) KLEIDICV_STREAMING_COMPATIBLE {
    return VecTraits::load(std::forward<ArgTypes>(args)...);
  }

  template <typename... ArgTypes>
  decltype(auto) load_consecutive(ArgTypes &&...args)
      KLEIDICV_STREAMING_COMPATIBLE {
    return VecTraits::load_consecutive(std::forward<ArgTypes>(args)...);
  }

  template <typename... ArgTypes>
  decltype(auto) store(ArgTypes &&...args) KLEIDICV_STREAMING_COMPATIBLE {
    return VecTraits::store(std::forward<ArgTypes>(args)...);
  }

  template <typename... ArgTypes>
  decltype(auto) store_consecutive(ArgTypes &&...args)
      KLEIDICV_STREAMING_COMPATIBLE {
    return VecTraits::store_consecutive(std::forward<ArgTypes>(args)...);
  }

  // Forwards max_vectors_per_block() calls to the inner operation.
  size_t max_vectors_per_block() KLEIDICV_STREAMING_COMPATIBLE {
    return operation_.max_vectors_per_block();
  }

  // Forwards on_block_finished() calls to the inner operation.
  void on_block_finished(size_t vectors_in_block)
      KLEIDICV_STREAMING_COMPATIBLE {
    return operation_.on_block_finished(vectors_in_block);
  }

  // Returns true if the innermost operation is unrolled twice, otherwise false.
  static constexpr bool is_unrolled_twice() KLEIDICV_STREAMING_COMPATIBLE {
    return ::KLEIDICV_TARGET_NAMESPACE::is_unrolled_twice<
        concrete_operation_type_t<OperationType>>;
  }

  // Returns true if the innermost operation is unrolled once, otherwise false.
  static constexpr bool is_unrolled_once() KLEIDICV_STREAMING_COMPATIBLE {
    return ::KLEIDICV_TARGET_NAMESPACE::is_unrolled_once<
        concrete_operation_type_t<OperationType>>;
  }

  // Returns true if the innermost operation uses tail path, otherwise false.
  static constexpr bool uses_tail_path() KLEIDICV_STREAMING_COMPATIBLE {
    return ::KLEIDICV_TARGET_NAMESPACE::uses_tail_path<
        concrete_operation_type_t<OperationType>>;
  }

  // Returns true if the innermost operation tries to avoid tail loop, otherwise
  // false.
  static constexpr bool try_to_avoid_tail_loop() KLEIDICV_STREAMING_COMPATIBLE {
    return ::KLEIDICV_TARGET_NAMESPACE::try_to_avoid_tail_loop<
        concrete_operation_type_t<OperationType>>;
  }

 protected:
  // Constructor is protected so that only derived classes can instantiate.
  explicit OperationBase(OperationType &operation) KLEIDICV_STREAMING_COMPATIBLE
      : operation_{operation} {}

 private:
  // Reference to the inner operation.
  OperationType &operation_;
};  // end of class OperationBase<OperationType>

// Mixin which simply forwards all operations as-is.
template <typename OperationType>
class ForwardingOperation : public OperationBase<OperationType> {
 public:
  explicit ForwardingOperation(OperationType &operation)
      KLEIDICV_STREAMING_COMPATIBLE : OperationBase<OperationType>(operation) {}
};  // end of class ForwardingOperation<OperationType>

// Facade to offer a simplified row-based operation interface.
//
// A row-based operation is executed row-by-row and is unrolled within the row.
//
// Facade layout:
//                 |- operation.vector_path_<N>x()
//  process_row() -|- operation.vector_path()
//                 |- operation.remaining_path()
//                 |- operation.num_lanes()
template <typename OperationType>
class RowBasedOperation : public OperationBase<OperationType> {
 public:
  explicit RowBasedOperation(OperationType &operation)
      KLEIDICV_STREAMING_COMPATIBLE : OperationBase<OperationType>(operation) {}

  // NOLINTBEGIN(cppcoreguidelines-avoid-goto, hicpp-avoid-goto)
  template <typename... ColumnTypes>
  void process_row(size_t length,
                   ColumnTypes... columns) KLEIDICV_STREAMING_COMPATIBLE {
    LoopUnroll loop{length, this->num_lanes()};

    // clang-format off

    loop.unroll_twice_if<OperationType::is_unrolled_twice()>(
      [&](size_t step) KLEIDICV_STREAMING_COMPATIBLE {
        this->operation().vector_path_2x(columns...);
        ((columns += step), ...);
      });

  avoid_tail_loop:

    loop.unroll_once_if<OperationType::is_unrolled_once()>(
      [&](size_t step) KLEIDICV_STREAMING_COMPATIBLE {
        this->operation().vector_path(columns...);
        ((columns += step), ...);
      });

    // Process leftover bytes on the unrolled-once vector path, if requested and
    // possible.
    if constexpr (OperationType::is_unrolled_once() && OperationType::try_to_avoid_tail_loop()) {
        if (loop.try_avoid_tail_loop(
            [&](size_t backward_step) KLEIDICV_STREAMING_COMPATIBLE {
              // Adjust pointers backwards to include
              // the leftover bytes.
              ((columns -= backward_step), ...);
            })) {
        goto avoid_tail_loop;
      }
    }

    loop.remaining(
      [&](size_t length, size_t /* step */) KLEIDICV_STREAMING_COMPATIBLE {
        this->operation().remaining_path(length, columns...);
      });

    // clang-format on
  }
  // NOLINTEND(cppcoreguidelines-avoid-goto, hicpp-avoid-goto)
};  // end of class RowBasedOperation<OperationType>

// Facade to offer a simplified row-based operation interface.
//
// A row-based operation is executed row-by-row, is unrolled within the row and
// split into blocks.
//
// Facade layout:
//                 |- operation.vector_path_<N>x()
//  process_row() -|- operation.vector_path()
//                 |- operation.remaining_path()
//                 |- operation.num_lanes()
//                 |- operation.max_vectors_per_block()
//                 |- operation.on_block_finished()
template <typename OperationType>
class RowBasedBlockOperation : public OperationBase<OperationType> {
 public:
  explicit RowBasedBlockOperation(OperationType &operation)
      KLEIDICV_STREAMING_COMPATIBLE : OperationBase<OperationType>(operation) {}

  template <typename... ColumnTypes>
  void process_row(size_t length,
                   ColumnTypes... columns) KLEIDICV_STREAMING_COMPATIBLE {
    if constexpr (OperationType::is_unrolled_twice()) {
      process_blocks<2>(length, [&](size_t step) KLEIDICV_STREAMING_COMPATIBLE {
        this->operation().vector_path_2x(columns...);
        ((columns += step), ...);
      });
    }

    if constexpr (OperationType::is_unrolled_once()) {
      process_blocks<1>(length, [&](size_t step) KLEIDICV_STREAMING_COMPATIBLE {
        this->operation().vector_path(columns...);
        ((columns += step), ...);
      });
    }

    // clang-format off
    LoopUnroll loop{length, this->num_lanes()};
    loop.remaining(
      [&](size_t length, size_t /* step */) KLEIDICV_STREAMING_COMPATIBLE {
        this->operation().remaining_path(length, columns...);
      });
    // clang-format on
  }

 private:
  template <size_t UnrollFactor, typename CallbackType>
  void process_blocks(size_t &length,
                      CallbackType callback) KLEIDICV_STREAMING_COMPATIBLE {
    // The number of elements a single iteration would process.
    const size_t elements_per_iteration = UnrollFactor * this->num_lanes();
    // The number of elements which will be processed when this method returns.
    const size_t processed_length =
        (length / elements_per_iteration) * elements_per_iteration;
    // The number of vectors which can be processed safely with the current
    // unroll factor.
    const size_t max_vectors_per_block =
        (this->max_vectors_per_block() / UnrollFactor) * UnrollFactor;

    // Process data up to allowed number of blocks.
    for (size_t index = 0, block_length = 0; index < processed_length;
         index += block_length) {
      // Process max_vectors_per_block vectors, or less in the last iteration
      // (remaining size).
      size_t remaining_vectors = (processed_length - index) / this->num_lanes();
      size_t vectors_in_block =
          std::min(max_vectors_per_block, remaining_vectors);
      block_length = vectors_in_block * this->num_lanes();

      // clang-format off
      // Process data with the appropriate unroll factor.
      LoopUnroll loop{block_length, this->num_lanes()};
      loop.unroll_n_times<UnrollFactor>(
        [&](size_t step) KLEIDICV_STREAMING_COMPATIBLE {
          callback(step);
          // Adjust remaining length here.
          // This improves generated code.
          length -= step;
        });
      // clang-format on

      // Notify the operation that a block was processed.
      this->on_block_finished(vectors_in_block);
    }
  }
};  // end of class RowBasedBlockOperation<OperationType>

// Adapter to offer an interface which allows users to use ParallelRows and to
// unpack them.
template <typename OperationType>
class ParallelRowsAdapter : public OperationBase<OperationType> {
 public:
  using VecTraits = typename OperationBase<OperationType>::VecTraits;
  using ScalarType = typename VecTraits::ScalarType;
  using ConstParallelColumnType = ParallelColumns<const ScalarType>;
  using ParallelColumnType = ParallelColumns<ScalarType>;
  using ConstColumnType = Columns<const ScalarType>;

  explicit ParallelRowsAdapter(OperationType &operation)
      KLEIDICV_STREAMING_COMPATIBLE : OperationBase<OperationType>(operation) {}

  // Forwards vector_path_2x() calls to the inner operation with one source and
  // destination parallel columns.
  void vector_path_2x(ConstParallelColumnType src_a, ConstColumnType src_b,
                      ParallelColumnType dst) KLEIDICV_STREAMING_COMPATIBLE {
    this->operation().vector_path_2x(src_a.first(), src_a.second(), src_b,
                                     dst.first(), dst.second());
  }

  // Forwards vector_path() calls to the inner operation with one source and
  // destination parallel columns.
  void vector_path(ConstParallelColumnType src_a, ConstColumnType src_b,
                   ParallelColumnType dst) KLEIDICV_STREAMING_COMPATIBLE {
    this->operation().vector_path(src_a.first(), src_a.second(), src_b,
                                  dst.first(), dst.second());
  }

  // Forwards remaining_path() calls to the inner operation with one source and
  // destination parallel columns.
  void remaining_path(size_t length, ConstParallelColumnType src_a,
                      ConstColumnType src_b,
                      ParallelColumnType dst) KLEIDICV_STREAMING_COMPATIBLE {
    this->operation().remaining_path(length, src_a.first(), src_a.second(),
                                     src_b, dst.first(), dst.second());
  }
};  // end of class ParallelRowsAdapter<OperationType>

// Adapter to offer an interface which allows users to apply unrolling to the
// adaptee operation. This adapter binds to the interface of the innermost
// operation in the chain.
//
// Provides specialized implementations using SFINAE principle for
//  - vector_path_2x(),
//  - vector_path(), and
//  - tail_path(), and
//  - scalar_path()
// methods.
template <typename OperationType>
class OperationAdapter : public OperationBase<OperationType> {
  // Shorten rows: no need to write 'this->'.
  using OperationBase<OperationType>::operation;
  using OperationBase<OperationType>::num_lanes;
  using OperationBase<OperationType>::load;
  using OperationBase<OperationType>::load_consecutive;
  using OperationBase<OperationType>::store;
  using OperationBase<OperationType>::store_consecutive;
  // The innermost operation in the chain.
  using ConcreteOperationType = concrete_operation_type_t<OperationType>;

 public:
  using VecTraits = typename OperationBase<OperationType>::VecTraits;
  using ContextType = typename OperationBase<OperationType>::ContextType;
  using ScalarType = typename VecTraits::ScalarType;
  using VectorType = typename VecTraits::VectorType;
  using Vector2Type = typename VecTraits::Vector2Type;
  using Vector3Type = typename VecTraits::Vector3Type;
  using Vector4Type = typename VecTraits::Vector4Type;
  using ConstColumnType = Columns<const ScalarType>;
  using ColumnType = Columns<ScalarType>;

  explicit OperationAdapter(OperationType &operation)
      KLEIDICV_STREAMING_COMPATIBLE : OperationBase<OperationType>(operation) {}

  // ---------------------------------------------------------------------------
  // Forwarding implementations for vector_path_2x().
  // ---------------------------------------------------------------------------

  // Required method:
  // void T::vector_path([ContextType,] VectorType);
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<void, T, ContextType, VectorType> vector_path_2x(
      ContextType ctx, ConstColumnType src) KLEIDICV_STREAMING_COMPATIBLE {
    VectorType src_0, src_1;
    operation().load_consecutive(ctx, &src[0], src_0, src_1);
    operation().vector_path(ctx, src_0);
    operation().vector_path(ctx, src_1);
  }

  // Required method:
  // VectorType T::vector_path([ContextType,] VectorType);
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<VectorType, T, ContextType, VectorType>
  vector_path_2x(ContextType ctx, ConstColumnType src,
                 ColumnType dst) KLEIDICV_STREAMING_COMPATIBLE {
    VectorType src_0, src_1;
    operation().load_consecutive(ctx, &src[0], src_0, src_1);
    VectorType res_0 = operation().vector_path(ctx, src_0);
    VectorType res_1 = operation().vector_path(ctx, src_1);
    operation().store_consecutive(ctx, res_0, res_1, &dst[0]);
  }

  // Required method:
  // VectorType T::vector_path([ContextType,] VectorType, VectorType);
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<VectorType, T, ContextType, VectorType,
                              VectorType>
  vector_path_2x(ContextType ctx, ConstColumnType src_a, ConstColumnType src_b,
                 ColumnType dst) KLEIDICV_STREAMING_COMPATIBLE {
    VectorType src_a_0, src_a_1, src_b_0, src_b_1;
    operation().load_consecutive(ctx, &src_a[0], src_a_0, src_a_1);
    operation().load_consecutive(ctx, &src_b[0], src_b_0, src_b_1);
    VectorType res_0 = operation().vector_path(ctx, src_a_0, src_b_0);
    VectorType res_1 = operation().vector_path(ctx, src_a_1, src_b_1);
    operation().store_consecutive(ctx, res_0, res_1, &dst[0]);
  }

  // Required method:
  // void T::vector_path([ContextType,] VectorType, ScalarType *);
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<void, T, ContextType, VectorType, ScalarType *>
  vector_path_2x(ContextType ctx, ConstColumnType src,
                 ColumnType dst) KLEIDICV_STREAMING_COMPATIBLE {
    VectorType src_0, src_1;
    operation().load_consecutive(ctx, &src[0], src_0, src_1);
    operation().vector_path(ctx, src_0, &dst[0]);
    operation().vector_path(ctx, src_1, &dst.at(num_lanes())[0]);
  }

  // Required method:
  // void T::vector_path([ContextType,] VectorType, VectorType, ScalarType *);
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<void, T, ContextType, VectorType, VectorType,
                              ScalarType *>
  vector_path_2x(ContextType ctx, ConstColumnType src_a, ConstColumnType src_b,
                 ColumnType dst) KLEIDICV_STREAMING_COMPATIBLE {
    VectorType src_a_0, src_a_1, src_b_0, src_b_1;
    operation().load_consecutive(ctx, &src_a[0], src_a_0, src_a_1);
    operation().load_consecutive(ctx, &src_b[0], src_b_0, src_b_1);
    operation().vector_path(ctx, src_a_0, src_b_0, &dst[0]);
    operation().vector_path(ctx, src_a_1, src_b_1, &dst.at(num_lanes())[0]);
  }

  // Required method:
  // void T::vector_path([ContextType,] VectorType, VectorType, VectorType,
  //                     ScalarType *);
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<void, T, ContextType, VectorType, VectorType,
                              VectorType, ScalarType *>
  vector_path_2x(ContextType ctx, ConstColumnType src_a, ConstColumnType src_b,
                 ConstColumnType src_c,
                 ColumnType dst) KLEIDICV_STREAMING_COMPATIBLE {
    VectorType src_a_0, src_a_1, src_b_0, src_b_1, src_c_0, src_c_1;
    operation().load_consecutive(ctx, &src_a[0], src_a_0, src_a_1);
    operation().load_consecutive(ctx, &src_b[0], src_b_0, src_b_1);
    operation().load_consecutive(ctx, &src_c[0], src_c_0, src_c_1);
    operation().vector_path(ctx, src_a_0, src_b_0, src_c_0, &dst[0]);
    operation().vector_path(ctx, src_a_1, src_b_1, src_c_1,
                            &dst.at(num_lanes())[0]);
  }

  // Required method:
  // void T::vector_path([ContextType,] VectorType, VectorType, VectorType,
  // ScalarType *,
  //                     ScalarType *);
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<void, T, ContextType, VectorType, VectorType,
                              VectorType, ScalarType *, ScalarType *>
  vector_path_2x(ContextType ctx, ConstColumnType src_a, ConstColumnType src_b,
                 ConstColumnType src_c, ColumnType dst_a,
                 ColumnType dst_b) KLEIDICV_STREAMING_COMPATIBLE {
    VectorType src_a_0, src_a_1, src_b_0, src_b_1, src_c_0, src_c_1;
    operation().load_consecutive(ctx, &src_a[0], src_a_0, src_a_1);
    operation().load_consecutive(ctx, &src_b[0], src_b_0, src_b_1);
    operation().load_consecutive(ctx, &src_c[0], src_c_0, src_c_1);
    operation().vector_path(ctx, src_a_0, src_b_0, src_c_0, &dst_a[0],
                            &dst_b[0]);
    operation().vector_path(ctx, src_a_1, src_b_1, src_c_1,
                            &dst_a.at(num_lanes())[0],
                            &dst_b.at(num_lanes())[0]);
  }

  // Required method:
  // void T::vector_path([ContextType,] VectorType, VectorType, VectorType,
  // VectorType,
  //                     ScalarType *);
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<void, T, ContextType, VectorType, VectorType,
                              VectorType, VectorType, ScalarType *>
  vector_path_2x(ContextType ctx, ConstColumnType src_a, ConstColumnType src_b,
                 ConstColumnType src_c, ConstColumnType src_d,
                 ColumnType dst) KLEIDICV_STREAMING_COMPATIBLE {
    VectorType src_a_0, src_a_1, src_b_0, src_b_1;
    VectorType src_c_0, src_c_1, src_d_0, src_d_1;
    operation().load_consecutive(ctx, &src_a[0], src_a_0, src_a_1);
    operation().load_consecutive(ctx, &src_b[0], src_b_0, src_b_1);
    operation().load_consecutive(ctx, &src_c[0], src_c_0, src_c_1);
    operation().load_consecutive(ctx, &src_d[0], src_d_0, src_d_1);
    operation().vector_path(ctx, src_a_0, src_b_0, src_c_0, src_d_0, &dst[0]);
    operation().vector_path(ctx, src_a_1, src_b_1, src_c_1, src_d_1,
                            &dst.at(num_lanes())[0]);
  }

  // Required method:
  // void T::vector_path([ContextType,] const ScalarType *, ScalarType *);
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<void, T, ContextType, const ScalarType *,
                              ScalarType *>
  vector_path_2x(ContextType ctx, ConstColumnType src,
                 ColumnType dst) KLEIDICV_STREAMING_COMPATIBLE {
    operation().vector_path(ctx, &src[0], &dst[0]);
    operation().vector_path(ctx, &src.at(num_lanes())[0],
                            &dst.at(num_lanes())[0]);
  }

  // Required method:
  // void T::vector_path([ContextType,] const ScalarType *, ScalarType *);
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<void, T, ContextType, const ScalarType *,
                              const ScalarType *, ScalarType *>
  vector_path_2x(ContextType ctx, ConstColumnType src_a, ConstColumnType src_b,
                 ColumnType dst) KLEIDICV_STREAMING_COMPATIBLE {
    operation().vector_path(ctx, &src_a[0], &src_b[0], &dst[0]);
    operation().vector_path(ctx, &src_a.at(num_lanes())[0],
                            &src_b.at(num_lanes())[0], &dst.at(num_lanes())[0]);
  }

  // Required method:
  // void T::vector_path([ContextType,] const ScalarType *, const ScalarType *,
  //                     const ScalarType *, ScalarType *, ScalarType *);
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<void, T, ContextType, const ScalarType *,
                              const ScalarType *, const ScalarType *,
                              ScalarType *, ScalarType *>
  vector_path_2x(ContextType ctx, ConstColumnType src_a, ConstColumnType src_b,
                 ConstColumnType src_c, ColumnType dst_a,
                 ColumnType dst_b) KLEIDICV_STREAMING_COMPATIBLE {
    operation().vector_path(ctx, &src_a[0], &src_b[0], &src_c[0], &dst_a[0],
                            &dst_b[0]);
    operation().vector_path(
        ctx, &src_a.at(num_lanes())[0], &src_b.at(num_lanes())[0],
        &src_c.at(num_lanes())[0], &dst_a.at(num_lanes())[0],
        &dst_b.at(num_lanes())[0]);
  }

  // Required method:
  // void T::vector_path([ContextType,] const ScalarType *, ScalarType *,
  //                     ScalarType *);
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<void, T, ContextType, const ScalarType *,
                              ScalarType *, ScalarType *>
  vector_path_2x(ContextType ctx, ConstColumnType src, ColumnType dst_a,
                 ColumnType dst_b) KLEIDICV_STREAMING_COMPATIBLE {
    operation().vector_path(ctx, &src[0], &dst_a[0], &dst_b[0]);
    operation().vector_path(ctx, &src.at(num_lanes())[0],
                            &dst_a.at(num_lanes())[0],
                            &dst_b.at(num_lanes())[0]);
  }

  // Required method:
  // void T::vector_path([ContextType,] const ScalarType *, VectorType &,
  // VectorType &);
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<void, T, ContextType, const ScalarType *,
                              VectorType &, VectorType &>
  vector_path_2x(ContextType ctx, ConstColumnType src, ColumnType dst_a,
                 ColumnType dst_b) KLEIDICV_STREAMING_COMPATIBLE {
    Vector2Type vdst_a, vdst_b;

    operation().vector_path(ctx, &src[0], vdst_a.val[0], vdst_b.val[0]);
    operation().vector_path(ctx, &src.at(num_lanes() * 2)[0], vdst_a.val[1],
                            vdst_b.val[1]);

    store(vdst_a, &dst_a[0]);
    store(vdst_b, &dst_b[0]);
  }

  // Required method:
  // void T::vector_path([ContextType,] Vector2Type,
  //                            VectorType &, VectorType &);
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<void, T, ContextType, Vector2Type, VectorType &,
                              VectorType &>
  vector_path_2x(ContextType ctx, ConstColumnType src, ColumnType dst_a,
                 ColumnType dst_b) KLEIDICV_STREAMING_COMPATIBLE {
    Vector2Type vsrc_0, vsrc_1;
    Vector2Type vdst_a, vdst_b;

    load_consecutive(&src[0], vsrc_0, vsrc_1);

    operation().vector_path(ctx, vsrc_0, vdst_a.val[0], vdst_b.val[0]);
    operation().vector_path(ctx, vsrc_1, vdst_a.val[1], vdst_b.val[1]);

    store(vdst_a, &dst_a[0]);
    store(vdst_b, &dst_b[0]);
  }

  // Required method:
  // void T::vector_path([ContextType,] const ScalarType *, ScalarType *,
  //                     ScalarType *, ScalarType *);
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<void, T, ContextType, const ScalarType *,
                              ScalarType *, ScalarType *, ScalarType *>
  vector_path_2x(ContextType ctx, ConstColumnType src, ColumnType dst_a,
                 ColumnType dst_b,
                 ColumnType dst_c) KLEIDICV_STREAMING_COMPATIBLE {
    operation().vector_path(ctx, &src[0], &dst_a[0], &dst_b[0], &dst_c[0]);
    operation().vector_path(
        ctx, &src.at(num_lanes() * 3)[0], &dst_a.at(num_lanes())[0],
        &dst_b.at(num_lanes())[0], &dst_c.at(num_lanes())[0]);
  }

  // Required method:
  // void T::vector_path([ContextType,] const ScalarType *, VectorType &,
  //                      VectorType &, VectorType &);
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<void, T, ContextType, const ScalarType *,
                              VectorType &, VectorType &, VectorType &>
  vector_path_2x(ContextType ctx, ConstColumnType src, ColumnType dst_a,
                 ColumnType dst_b,
                 ColumnType dst_c) KLEIDICV_STREAMING_COMPATIBLE {
    Vector2Type vdst_a, vdst_b, vdst_c;

    operation().vector_path(ctx, &src[0], vdst_a.val[0], vdst_b.val[0],
                            vdst_c.val[0]);
    operation().vector_path(ctx, &src.at(num_lanes() * 3)[0], vdst_a.val[1],
                            vdst_b.val[1], vdst_c.val[1]);

    store(vdst_a, &dst_a[0]);
    store(vdst_b, &dst_b[0]);
    store(vdst_c, &dst_c[0]);
  }

  // Required method:
  // void T::vector_path([ContextType,] , Vector3Type , VectorType &
  //                     VectorType &, VectorType &);
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<void, T, ContextType, Vector3Type, VectorType &,
                              VectorType &, VectorType &>
  vector_path_2x(ContextType ctx, ConstColumnType src, ColumnType dst_a,
                 ColumnType dst_b,
                 ColumnType dst_c) KLEIDICV_STREAMING_COMPATIBLE {
    Vector3Type vsrc_0, vsrc_1;
    Vector2Type vdst_a, vdst_b, vdst_c;

    load_consecutive(&src[0], vsrc_0, vsrc_1);

    operation().vector_path(ctx, vsrc_0, vdst_a.val[0], vdst_b.val[0],
                            vdst_c.val[0]);
    operation().vector_path(ctx, vsrc_1, vdst_a.val[1], vdst_b.val[1],
                            vdst_c.val[1]);

    store(vdst_a, &dst_a[0]);
    store(vdst_b, &dst_b[0]);
    store(vdst_c, &dst_c[0]);
  }

  // Required method:
  // void T::vector_path([ContextType,] const ScalarType *, ScalarType *,
  //                     ScalarType *, ScalarType *, ScalarType *);
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<void, T, ContextType, const ScalarType *,
                              ScalarType *, ScalarType *, ScalarType *,
                              ScalarType *>
  vector_path_2x(ContextType ctx, ConstColumnType src, ColumnType dst_a,
                 ColumnType dst_b, ColumnType dst_c,
                 ColumnType dst_d) KLEIDICV_STREAMING_COMPATIBLE {
    operation().vector_path(ctx, &src[0], &dst_a[0], &dst_b[0], &dst_c[0],
                            &dst_d[0]);
    operation().vector_path(
        ctx, &src.at(num_lanes() * 4)[0], &dst_a.at(num_lanes())[0],
        &dst_b.at(num_lanes())[0], &dst_c.at(num_lanes())[0],
        &dst_d.at(num_lanes())[0]);
  }

  // Required method:
  // void T::vector_path([ContextType,] const ScalarType *,
  //                          VectorType &, VectorType &, VectorType &,
  //                          VectorType &);
  // Operation with multiple destination address, and with store. This
  // gives the possibility to store the 2 result vectors in 1 step
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<void, T, ContextType, const ScalarType *,
                              VectorType &, VectorType &, VectorType &,
                              VectorType &>
  vector_path_2x(ContextType ctx, ConstColumnType src, ColumnType dst_a,
                 ColumnType dst_b, ColumnType dst_c,
                 ColumnType dst_d) KLEIDICV_STREAMING_COMPATIBLE {
    Vector2Type vdst_a, vdst_b, vdst_c, vdst_d;

    operation().vector_path(ctx, &src[0], vdst_a.val[0], vdst_b.val[0],
                            vdst_c.val[0], vdst_d.val[0]);

    operation().vector_path(ctx, &src.at(num_lanes() * 4)[0], vdst_a.val[1],
                            vdst_b.val[1], vdst_c.val[1], vdst_d.val[1]);

    store(vdst_a, &dst_a[0]);
    store(vdst_b, &dst_b[0]);
    store(vdst_c, &dst_c[0]);
    store(vdst_d, &dst_d[0]);
  }

  // Required method:
  // void T::vector_path([ContextType,] const Vector4Type
  //                          VectorType &, VectorType &, VectorType &,
  //                          VectorType &);
  // Operation with multiple destination address, and with load and store.
  // This gives the possibility to load the source x4 vectors and store
  // the 2x4 result vectors in 1 step
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<void, T, ContextType, const Vector4Type,
                              VectorType &, VectorType &, VectorType &,
                              VectorType &>
  vector_path_2x(ContextType ctx, ConstColumnType src, ColumnType dst_a,
                 ColumnType dst_b, ColumnType dst_c,
                 ColumnType dst_d) KLEIDICV_STREAMING_COMPATIBLE {
    Vector4Type vsrc_0, vsrc_1;
    Vector2Type vdst_a, vdst_b, vdst_c, vdst_d;

    load_consecutive(&src[0], vsrc_0, vsrc_1);

    operation().vector_path(ctx, vsrc_0, vdst_a.val[0], vdst_b.val[0],
                            vdst_c.val[0], vdst_d.val[0]);

    operation().vector_path(ctx, vsrc_1, vdst_a.val[1], vdst_b.val[1],
                            vdst_c.val[1], vdst_d.val[1]);

    store(vdst_a, &dst_a[0]);
    store(vdst_b, &dst_b[0]);
    store(vdst_c, &dst_c[0]);
    store(vdst_d, &dst_d[0]);
  }

  // ---------------------------------------------------------------------------
  // Forwarding implementations for vector_path().
  // ---------------------------------------------------------------------------

  // Required method:
  // void T::vector_path([ContextType,] VectorType);
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<void, T, ContextType, VectorType> vector_path(
      ContextType ctx, ConstColumnType src) KLEIDICV_STREAMING_COMPATIBLE {
    VectorType src_0;
    operation().load(ctx, &src[0], src_0);
    operation().vector_path(ctx, src_0);
  }

  // Required method:
  // VectorType T::vector_path([ContextType,] VectorType);
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<VectorType, T, ContextType, VectorType>
  vector_path(ContextType ctx, ConstColumnType src,
              ColumnType dst) KLEIDICV_STREAMING_COMPATIBLE {
    VectorType src_0;
    operation().load(ctx, &src[0], src_0);
    VectorType res_0 = operation().vector_path(ctx, src_0);
    operation().store(ctx, res_0, &dst[0]);
  }

  // Required method:
  // VectorType T::vector_path([ContextType,] VectorType, VectorType);
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<VectorType, T, ContextType, VectorType,
                              VectorType>
  vector_path(ContextType ctx, ConstColumnType src_a, ConstColumnType src_b,
              ColumnType dst) KLEIDICV_STREAMING_COMPATIBLE {
    VectorType src_a_0, src_b_0;
    operation().load(ctx, &src_a[0], src_a_0);
    operation().load(ctx, &src_b[0], src_b_0);
    VectorType res_0 = operation().vector_path(ctx, src_a_0, src_b_0);
    operation().store(ctx, res_0, &dst[0]);
  }

  // Required method:
  // void T::vector_path([ContextType,] VectorType, ScalarType *);
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<void, T, ContextType, VectorType, ScalarType *>
  vector_path(ContextType ctx, ConstColumnType src,
              ColumnType dst) KLEIDICV_STREAMING_COMPATIBLE {
    VectorType src_0;
    operation().load(ctx, &src[0], src_0);
    operation().vector_path(ctx, src_0, &dst[0]);
  }

  // Required method:
  // void T::vector_path([ContextType,] VectorType, VectorType, ScalarType *);
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<void, T, ContextType, VectorType, VectorType,
                              ScalarType *>
  vector_path(ContextType ctx, ConstColumnType src_a, ConstColumnType src_b,
              ColumnType dst) KLEIDICV_STREAMING_COMPATIBLE {
    VectorType src_a_0, src_b_0;
    operation().load(ctx, &src_a[0], src_a_0);
    operation().load(ctx, &src_b[0], src_b_0);
    operation().vector_path(ctx, src_a_0, src_b_0, &dst[0]);
  }

  // Required method:
  // void T::vector_path([ContextType,] VectorType, VectorType, VectorType,
  //                     ScalarType *);
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<void, T, ContextType, VectorType, VectorType,
                              VectorType, ScalarType *>
  vector_path(ContextType ctx, ConstColumnType src_a, ConstColumnType src_b,
              ConstColumnType src_c,
              ColumnType dst) KLEIDICV_STREAMING_COMPATIBLE {
    VectorType src_a_0, src_b_0, src_c_0;
    operation().load(ctx, &src_a[0], src_a_0);
    operation().load(ctx, &src_b[0], src_b_0);
    operation().load(ctx, &src_c[0], src_c_0);
    operation().vector_path(ctx, src_a_0, src_b_0, src_c_0, &dst[0]);
  }

  // Required method:
  // void T::vector_path([ContextType,] VectorType, VectorType, VectorType,
  //                     ScalarType *, ScalarType *);
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<void, T, ContextType, VectorType, VectorType,
                              VectorType, ScalarType *, ScalarType *>
  vector_path(ContextType ctx, ConstColumnType src_a, ConstColumnType src_b,
              ConstColumnType src_c, ColumnType dst_a,
              ColumnType dst_b) KLEIDICV_STREAMING_COMPATIBLE {
    VectorType src_a_0, src_b_0, src_c_0;
    operation().load(ctx, &src_a[0], src_a_0);
    operation().load(ctx, &src_b[0], src_b_0);
    operation().load(ctx, &src_c[0], src_c_0);
    operation().vector_path(ctx, src_a_0, src_b_0, src_c_0, &dst_a[0],
                            &dst_b[0]);
  }

  // Required method:
  // void T::vector_path([ContextType,] VectorType, VectorType, VectorType,
  //                     VectorType, ScalarType *);
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<void, T, ContextType, VectorType, VectorType,
                              VectorType, VectorType, ScalarType *>
  vector_path(ContextType ctx, ConstColumnType src_a, ConstColumnType src_b,
              ConstColumnType src_c, ConstColumnType src_d,
              ColumnType dst) KLEIDICV_STREAMING_COMPATIBLE {
    VectorType src_a_0, src_b_0, src_c_0, src_d_0;
    operation().load(ctx, &src_a[0], src_a_0);
    operation().load(ctx, &src_b[0], src_b_0);
    operation().load(ctx, &src_c[0], src_c_0);
    operation().load(ctx, &src_d[0], src_d_0);
    operation().vector_path(ctx, src_a_0, src_b_0, src_c_0, src_d_0, &dst[0]);
  }

  // Required method:
  // void T::vector_path([ContextType,] const ScalarType *, ScalarType *);
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<void, T, ContextType, const ScalarType *,
                              ScalarType *>
  vector_path(ContextType ctx, ConstColumnType src,
              ColumnType dst) KLEIDICV_STREAMING_COMPATIBLE {
    operation().vector_path(ctx, &src[0], &dst[0]);
  }

  // Required method:
  // void T::vector_path([ContextType,] const ScalarType *, const ScalarType *,
  //                     ScalarType *);
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<void, T, ContextType, const ScalarType *,
                              const ScalarType *, ScalarType *>
  vector_path(ContextType ctx, ConstColumnType src_a, ConstColumnType src_b,
              ColumnType dst) KLEIDICV_STREAMING_COMPATIBLE {
    operation().vector_path(ctx, &src_a[0], &src_b[0], &dst[0]);
  }

  // Required method:
  // void T::vector_path([ContextType,] const ScalarType *, const ScalarType *,
  //                     const ScalarType *, ScalarType *, ScalarType *);
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<void, T, ContextType, const ScalarType *,
                              const ScalarType *, const ScalarType *,
                              ScalarType *, ScalarType *>
  vector_path(ContextType ctx, ConstColumnType src_a, ConstColumnType src_b,
              ConstColumnType src_c, ColumnType dst_a,
              ColumnType dst_b) KLEIDICV_STREAMING_COMPATIBLE {
    operation().vector_path(ctx, &src_a[0], &src_b[0], &src_c[0], &dst_a[0],
                            &dst_b[0]);
  }

  // Required method:
  // void T::vector_path([ContextType,] const ScalarType *, ScalarType *,
  //                     ScalarType *);
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<void, T, ContextType, const ScalarType *,
                              ScalarType *, ScalarType *>
  vector_path(ContextType ctx, ConstColumnType src, ColumnType dst_a,
              ColumnType dst_b) KLEIDICV_STREAMING_COMPATIBLE {
    operation().vector_path(ctx, &src[0], &dst_a[0], &dst_b[0]);
  }

  // Required method:
  // void T::vector_path([ContextType,] const ScalarType *, VectorType &,
  //                     VectorType &);
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<void, T, ContextType, const ScalarType *,
                              VectorType &, VectorType &>
  vector_path(ContextType ctx, ConstColumnType src, ColumnType dst_a,
              ColumnType dst_b) KLEIDICV_STREAMING_COMPATIBLE {
    VectorType vdst_a, vdst_b;

    operation().vector_path(ctx, &src[0], vdst_a, vdst_b);

    store(vdst_a, &dst_a[0]);
    store(vdst_b, &dst_b[0]);
  }

  // Required method:
  // void T::vector_path([ContextType,] Vector2Type, VectorType &,
  //                     VectorType &);
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<void, T, ContextType, Vector2Type, VectorType &,
                              VectorType &>
  vector_path(ContextType ctx, ConstColumnType src, ColumnType dst_a,
              ColumnType dst_b) KLEIDICV_STREAMING_COMPATIBLE {
    VectorType vdst_a, vdst_b;
    Vector2Type vsrc;

    load(&src[0], vsrc);

    operation().vector_path(ctx, vsrc, vdst_a, vdst_b);

    store(vdst_a, &dst_a[0]);
    store(vdst_b, &dst_b[0]);
  }

  // Required method:
  // void T::vector_path([ContextType,] const ScalarType *, ScalarType *,
  //                     ScalarType *, ScalarType *);
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<void, T, ContextType, const ScalarType *,
                              ScalarType *, ScalarType *, ScalarType *>
  vector_path(ContextType ctx, ConstColumnType src, ColumnType dst_a,
              ColumnType dst_b,
              ColumnType dst_c) KLEIDICV_STREAMING_COMPATIBLE {
    operation().vector_path(ctx, &src[0], &dst_a[0], &dst_b[0], &dst_c[0]);
  }

  // Required method:
  // void T::vector_path([ContextType,] const ScalarType *, VectorType &,
  //                     VectorType &, VectorType &);
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<void, T, ContextType, const ScalarType *,
                              VectorType &, VectorType &, VectorType &>
  vector_path(ContextType ctx, ConstColumnType src, ColumnType dst_a,
              ColumnType dst_b,
              ColumnType dst_c) KLEIDICV_STREAMING_COMPATIBLE {
    VectorType vdst_a, vdst_b, vdst_c;

    operation().vector_path(ctx, &src[0], vdst_a, vdst_b, vdst_c);

    store(vdst_a, &dst_a[0]);
    store(vdst_b, &dst_b[0]);
    store(vdst_c, &dst_c[0]);
  }

  // Required method:
  // void T::vector_path([ContextType,] Vector3Type, VectorType &,
  //                     VectorType &, VectorType &);
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<void, T, ContextType, Vector3Type, VectorType &,
                              VectorType &, VectorType &>
  vector_path(ContextType ctx, ConstColumnType src, ColumnType dst_a,
              ColumnType dst_b,
              ColumnType dst_c) KLEIDICV_STREAMING_COMPATIBLE {
    VectorType vdst_a, vdst_b, vdst_c;
    Vector3Type vsrc;

    load(&src[0], vsrc);

    operation().vector_path(ctx, vsrc, vdst_a, vdst_b, vdst_c);

    store(vdst_a, &dst_a[0]);
    store(vdst_b, &dst_b[0]);
    store(vdst_c, &dst_c[0]);
  }

  // Required method:
  // void T::vector_path([ContextType,] const ScalarType *, ScalarType *,
  //                     ScalarType *, ScalarType *, ScalarType *);
  // Operation type with multiple destination, store in the vectorpath
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<void, T, ContextType, const ScalarType *,
                              ScalarType *, ScalarType *, ScalarType *,
                              ScalarType *>
  vector_path(ContextType ctx, ConstColumnType src, ColumnType dst_a,
              ColumnType dst_b, ColumnType dst_c,
              ColumnType dst_d) KLEIDICV_STREAMING_COMPATIBLE {
    operation().vector_path(ctx, &src[0], &dst_a[0], &dst_b[0], &dst_c[0],
                            &dst_d[0]);
  }

  // Required method:
  // void T::vector_path([ContextType,] const ScalarType *,
  //                          VectorType &, VectorType &, VectorType &,
  //                          VectorType &);
  // Operation type with multiple destination, and with store
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<void, T, ContextType, const ScalarType *,
                              VectorType &, VectorType &, VectorType &,
                              VectorType &>
  vector_path(ContextType ctx, ConstColumnType src, ColumnType dst_a,
              ColumnType dst_b, ColumnType dst_c,
              ColumnType dst_d) KLEIDICV_STREAMING_COMPATIBLE {
    VectorType vdst_a, vdst_b, vdst_c, vdst_d;

    operation().vector_path(ctx, &src[0], vdst_a, vdst_b, vdst_c, vdst_d);

    store(vdst_a, &dst_a[0]);
    store(vdst_b, &dst_b[0]);
    store(vdst_c, &dst_c[0]);
    store(vdst_d, &dst_d[0]);
  }

  // Required method:
  // void T::vector_path([ContextType,] const Vector4Type,
  //                          VectorType &, VectorType &, VectorType &,
  //                          VectorType &);
  // Operation type with multiple destination, and with load and store
  template <typename T = ConcreteOperationType>
  enable_if_has_vector_path_t<void, T, ContextType, const Vector4Type,
                              VectorType &, VectorType &, VectorType &,
                              VectorType &>
  vector_path(ContextType ctx, ConstColumnType src, ColumnType dst_a,
              ColumnType dst_b, ColumnType dst_c,
              ColumnType dst_d) KLEIDICV_STREAMING_COMPATIBLE {
    VectorType vdst_a, vdst_b, vdst_c, vdst_d;
    Vector4Type vsrc;

    load(&src[0], vsrc);

    operation().vector_path(ctx, vsrc, vdst_a, vdst_b, vdst_c, vdst_d);

    store(vdst_a, &dst_a[0]);
    store(vdst_b, &dst_b[0]);
    store(vdst_c, &dst_c[0]);
    store(vdst_d, &dst_d[0]);
  }

  // ---------------------------------------------------------------------------
  // Forwarding implementations for tail_path().
  // ---------------------------------------------------------------------------

  // Required method:
  // void T::tail_path([ContextType,] VectorType, ScalarType *);
  template <typename T = ConcreteOperationType>
  enable_if_has_tail_path_t<void, T, ContextType, VectorType, ScalarType *>
  tail_path(ContextType ctx, ConstColumnType src,
            ColumnType dst) KLEIDICV_STREAMING_COMPATIBLE {
    VectorType src_0;
    operation().load(ctx, &src[0], src_0);
    operation().tail_path(ctx, src_0, &dst[0]);
  }

  // Required method:
  // void T::tail_path([ContextType,] const ScalarType *, ScalarType *);
  template <typename T = ConcreteOperationType>
  enable_if_has_tail_path_t<void, T, ContextType, const ScalarType *,
                            ScalarType *>
  tail_path(ContextType ctx, ConstColumnType src,
            ColumnType dst) KLEIDICV_STREAMING_COMPATIBLE {
    operation().tail_path(ctx, &src[0], &dst[0]);
  }

  // ---------------------------------------------------------------------------
  // Forwarding implementations for scalar_path().
  // ---------------------------------------------------------------------------

  // Required method:
  // void T::scalar_path([ContextType,] ScalarType);
  template <typename T = ConcreteOperationType>
  enable_if_has_scalar_path_t<void, T, ContextType, ScalarType> scalar_path(
      ContextType ctx, ConstColumnType src) KLEIDICV_STREAMING_COMPATIBLE {
    operation().scalar_path(ctx, src[0]);
  }

  // Required method:
  // ScalarType T::scalar_path([ContextType,] ScalarType);
  template <typename T = ConcreteOperationType>
  enable_if_has_scalar_path_t<ScalarType, T, ContextType, ScalarType>
  scalar_path(ContextType ctx, ConstColumnType src,
              ColumnType dst) KLEIDICV_STREAMING_COMPATIBLE {
    dst[0] = operation().scalar_path(ctx, src[0]);
  }

  // Required method:
  // ScalarType T::scalar_path([ContextType,] ScalarType, ScalarType);
  template <typename T = ConcreteOperationType>
  enable_if_has_scalar_path_t<ScalarType, T, ContextType, ScalarType,
                              ScalarType>
  scalar_path(ContextType ctx, ConstColumnType src_a, ConstColumnType src_b,
              ColumnType dst) KLEIDICV_STREAMING_COMPATIBLE {
    dst[0] = operation().scalar_path(ctx, src_a[0], src_b[0]);
  }

  // Required method:
  // void T::scalar_path([ContextType,] const ScalarType *, ScalarType *);
  template <typename T = ConcreteOperationType>
  enable_if_has_scalar_path_t<void, T, ContextType, const ScalarType *,
                              ScalarType *>
  scalar_path(ContextType ctx, ConstColumnType src,
              ColumnType dst) KLEIDICV_STREAMING_COMPATIBLE {
    operation().scalar_path(ctx, &src[0], &dst[0]);
  }

  // Required method:
  // void T::scalar_path([ContextType,] const ScalarType *, const ScalarType *,
  //                     ScalarType *);
  template <typename T = ConcreteOperationType>
  enable_if_has_scalar_path_t<void, T, ContextType, const ScalarType *,
                              const ScalarType *, ScalarType *>
  scalar_path(ContextType ctx, ConstColumnType src_a, ConstColumnType src_b,
              ColumnType dst) KLEIDICV_STREAMING_COMPATIBLE {
    operation().scalar_path(ctx, &src_a[0], &src_b[0], &dst[0]);
  }

  // Required method:
  // void T::scalar_path([ContextType,] const ScalarType *, const ScalarType *,
  //                     const ScalarType *, ScalarType *);
  template <typename T = ConcreteOperationType>
  enable_if_has_scalar_path_t<void, T, ContextType, const ScalarType *,
                              const ScalarType *, const ScalarType *,
                              ScalarType *>
  scalar_path(ContextType ctx, ConstColumnType src_a, ConstColumnType src_b,
              ConstColumnType src_c,
              ColumnType dst) KLEIDICV_STREAMING_COMPATIBLE {
    operation().scalar_path(ctx, &src_a[0], &src_b[0], &src_c[0], &dst[0]);
  }

  // Required method:
  // void T::scalar_path([ContextType,] const ScalarType *, const ScalarType *,
  //                     const ScalarType *, const ScalarType *, ScalarType *);
  template <typename T = ConcreteOperationType>
  enable_if_has_scalar_path_t<void, T, ContextType, const ScalarType *,
                              const ScalarType *, const ScalarType *,
                              const ScalarType *, ScalarType *>
  scalar_path(ContextType ctx, ConstColumnType src_a, ConstColumnType src_b,
              ConstColumnType src_c, ConstColumnType src_d,
              ColumnType dst) KLEIDICV_STREAMING_COMPATIBLE {
    operation().scalar_path(ctx, &src_a[0], &src_b[0], &src_c[0], &src_d[0],
                            &dst[0]);
  }

  // Required method:
  // void T::scalar_path([ContextType,] size_t, const ScalarType *,
  //                     const ScalarType *, const ScalarType *, ScalarType *,
  //                     ScalarType *);
  template <typename T = ConcreteOperationType>
  enable_if_has_scalar_path_t<void, T, ContextType, size_t, const ScalarType *,
                              const ScalarType *, const ScalarType *,
                              ScalarType *, ScalarType *>
  scalar_path(ContextType ctx, size_t length, ConstColumnType src_a,
              ConstColumnType src_b, ConstColumnType src_c, ColumnType dst_a,
              ColumnType dst_b) KLEIDICV_STREAMING_COMPATIBLE {
    operation().scalar_path(ctx, length, &src_a[0], &src_b[0], &src_c[0],
                            &dst_a[0], &dst_b[0]);
  }

  // Required method:
  // void T::scalar_path([ContextType,] const ScalarType *, ScalarType *,
  //                     ScalarType *);
  template <typename T = ConcreteOperationType>
  enable_if_has_scalar_path_t<void, T, ContextType, const ScalarType *,
                              ScalarType *, ScalarType *>
  scalar_path(ContextType ctx, ConstColumnType src, ColumnType dst_a,
              ColumnType dst_b) KLEIDICV_STREAMING_COMPATIBLE {
    operation().scalar_path(ctx, &src[0], &dst_a[0], &dst_b[0]);
  }

  // Required method:
  // void T::scalar_path([ContextType,] const ScalarType *, ScalarType *,
  //                     ScalarType *, ScalarType *);
  template <typename T = ConcreteOperationType>
  enable_if_has_scalar_path_t<void, T, ContextType, const ScalarType *,
                              ScalarType *, ScalarType *, ScalarType *>
  scalar_path(ContextType ctx, ConstColumnType src, ColumnType dst_a,
              ColumnType dst_b,
              ColumnType dst_c) KLEIDICV_STREAMING_COMPATIBLE {
    operation().scalar_path(ctx, &src[0], &dst_a[0], &dst_b[0], &dst_c[0]);
  }

  // Required method:
  // void T::scalar_path([ContextType,] const ScalarType *, ScalarType *,
  //                     ScalarType *, ScalarType *, ScalarType *);
  template <typename T = ConcreteOperationType>
  enable_if_has_scalar_path_t<void, T, ContextType, const ScalarType *,
                              ScalarType *, ScalarType *, ScalarType *,
                              ScalarType *>
  scalar_path(ContextType ctx, ConstColumnType src, ColumnType dst_a,
              ColumnType dst_b, ColumnType dst_c,
              ColumnType dst_d) KLEIDICV_STREAMING_COMPATIBLE {
    operation().scalar_path(ctx, &src[0], &dst_a[0], &dst_b[0], &dst_c[0],
                            &dst_d[0]);
  }
};  // end of class OperationAdapter<OperationType>

// Removes context before forwarding calls.
template <typename OperationType>
class RemoveContextAdapter : public OperationBase<OperationType> {
 public:
  using ContextType = typename OperationBase<OperationType>::ContextType;

  explicit RemoveContextAdapter(OperationType &operation)
      KLEIDICV_STREAMING_COMPATIBLE : OperationBase<OperationType>(operation) {}

  template <typename... ArgTypes>
  decltype(auto) load(ContextType,
                      ArgTypes &&...args) KLEIDICV_STREAMING_COMPATIBLE {
    return OperationBase<OperationType>::load(std::forward<ArgTypes>(args)...);
  }

  template <typename... ArgTypes>
  decltype(auto) load_consecutive(ContextType, ArgTypes &&...args)
      KLEIDICV_STREAMING_COMPATIBLE {
    return OperationBase<OperationType>::load_consecutive(
        std::forward<ArgTypes>(args)...);
  }

  template <typename... ArgTypes>
  decltype(auto) store(ContextType,
                       ArgTypes &&...args) KLEIDICV_STREAMING_COMPATIBLE {
    return OperationBase<OperationType>::store(std::forward<ArgTypes>(args)...);
  }

  template <typename... ArgTypes>
  decltype(auto) store_consecutive(ContextType, ArgTypes &&...args)
      KLEIDICV_STREAMING_COMPATIBLE {
    return OperationBase<OperationType>::store_consecutive(
        std::forward<ArgTypes>(args)...);
  }

  template <typename... ArgTypes>
  decltype(auto) vector_path(ContextType,
                             ArgTypes &&...args) KLEIDICV_STREAMING_COMPATIBLE {
    return this->operation().vector_path(std::forward<ArgTypes>(args)...);
  }

  // Forwards remaining_path() calls to the inner operation.
  template <typename... ArgTypes>
  decltype(auto) remaining_path(ContextType, ArgTypes &&...args)
      KLEIDICV_STREAMING_COMPATIBLE {
    return this->operation().remaining_path(std::forward<ArgTypes>(args)...);
  }

  template <typename... ArgTypes>
  decltype(auto) tail_path(ContextType,
                           ArgTypes &&...args) KLEIDICV_STREAMING_COMPATIBLE {
    return this->operation().tail_path(std::forward<ArgTypes>(args)...);
  }

  template <typename... ArgTypes>
  decltype(auto) scalar_path(ContextType,
                             ArgTypes &&...args) KLEIDICV_STREAMING_COMPATIBLE {
    return this->operation().scalar_path(std::forward<ArgTypes>(args)...);
  }
};  // end of class RemoveContextAdapter<OperationType>

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_OPERATIONS_H
