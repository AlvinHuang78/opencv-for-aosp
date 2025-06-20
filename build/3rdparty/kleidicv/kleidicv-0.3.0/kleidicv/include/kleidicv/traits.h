// SPDX-FileCopyrightText: 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_TYPE_TRAITS_H
#define KLEIDICV_TYPE_TRAITS_H

#include <type_traits>

#include "kleidicv/config.h"

namespace KLEIDICV_TARGET_NAMESPACE {

// An empty class.
class Monostate {};

template <typename FnType>
class remove_streaming_compatible;

#if KLEIDICV_TARGET_SME2
template <typename Ret, typename Impl, typename... Args>
class remove_streaming_compatible<Ret (Impl::*)(Args...)
                                      KLEIDICV_STREAMING_COMPATIBLE> {
 public:
  using type = Ret (Impl::*)(Args...);
};
#endif

template <typename Ret, typename Impl, typename... Args>
class remove_streaming_compatible<Ret (Impl::*)(Args...)> {
 public:
  using type = Ret (Impl::*)(Args...);
};

template <typename FnType>
using remove_streaming_compatible_t =
    typename remove_streaming_compatible<FnType>::type;

// Tags an operation to process data of one vector per iteration.
class UnrollOnce {};

// Returns true if an instance derives from UnrollOnce, otherwise false.
template <typename T>
constexpr bool is_unrolled_once = std::is_base_of_v<UnrollOnce, T>;

// Tags an operation to process data of two vectors per iteration.
class UnrollTwice {};

// Returns true if an instance derives from UnrollTwice, otherwise false.
template <typename T>
constexpr bool is_unrolled_twice = std::is_base_of_v<UnrollTwice, T>;

// Tags an operation to use a different vector path to process the tail.
class UsesTailPath {};

// Returns true if an instance derives from UsesTailPath, otherwise false.
template <typename T>
constexpr bool uses_tail_path = std::is_base_of_v<UsesTailPath, T>;

// Tags an operation to avoid the tail loop if possible, by rewinding and doing
// a vector path.
class TryToAvoidTailLoop {};

// Returns true if an instance derives from TryToAvoidTailLoop, otherwise false.
template <typename T>
constexpr bool try_to_avoid_tail_loop =
    std::is_base_of_v<TryToAvoidTailLoop, T>;

// Primary template to handle types without T::vector_path() method.
template <typename, typename, typename = void>
constexpr bool has_vector_path = false;

// Specialization to handle types with T::vector_path() method.
template <typename T, typename FnType>
constexpr bool
    has_vector_path<T, FnType, std::void_t<decltype(&T::vector_path)>> =
        std::is_same<remove_streaming_compatible_t<decltype(&T::vector_path)>,
                     FnType>::value;

// Helper to simplify usage of std::enable_if_t with has_vector_path<T>.
template <class ReturnType, class ImplType, class ContextType,
          class... ArgTypes>
using enable_if_has_vector_path_t = std::enable_if_t<
    has_vector_path<ImplType, ReturnType (ImplType::*)(ArgTypes...)> ||
    has_vector_path<ImplType,
                    ReturnType (ImplType::*)(ContextType, ArgTypes...)>>;

// Primary template to handle types without T::tail_path() method.
template <typename, typename, typename = void>
constexpr bool has_tail_path = false;

// Specialization to handle types with T::tail_path() method.
template <typename T, typename FnType>
constexpr bool has_tail_path<T, FnType, std::void_t<decltype(&T::tail_path)>> =
    std::is_same<remove_streaming_compatible_t<decltype(&T::tail_path)>,
                 FnType>::value;

// Helper to simplify usage of std::enable_if_t with has_tail_path<T>.
template <class ReturnType, class ImplType, class ContextType,
          class... ArgTypes>
using enable_if_has_tail_path_t = std::enable_if_t<
    has_tail_path<ImplType, ReturnType (ImplType::*)(ArgTypes...)> ||
    has_tail_path<ImplType,
                  ReturnType (ImplType::*)(ContextType, ArgTypes...)>>;

// Primary template to handle types without T::scalar_path() method.
template <typename, typename, typename = void>
constexpr bool has_scalar_path = false;

// Specialization to handle types with T::scalar_path() method.
template <typename T, typename FnType>
constexpr bool
    has_scalar_path<T, FnType, std::void_t<decltype(&T::scalar_path)>> =
        std::is_same<remove_streaming_compatible_t<decltype(&T::scalar_path)>,
                     FnType>::value;

// Helper to simplify usage of std::enable_if_t with has_scalar_path<T>.
template <class ReturnType, class ImplType, class ContextType,
          class... ArgTypes>
using enable_if_has_scalar_path_t = std::enable_if_t<
    has_scalar_path<ImplType, ReturnType (ImplType::*)(ArgTypes...)> ||
    has_scalar_path<ImplType,
                    ReturnType (ImplType::*)(ContextType, ArgTypes...)>>;

// Primary template to handle types without T::OperationType member.
template <class T, class = void>
class concrete_operation_type {
 public:
  using type = T;
};

// Specialization to handle types with T::OperationType member.
template <class T>
class concrete_operation_type<T, std::void_t<typename T::OperationType>> {
 public:
  using type =
      typename concrete_operation_type<typename T::OperationType>::type;
};

// Helper to simplify usage of has_operation_type_member<T>::type.
template <class T>
using concrete_operation_type_t = typename concrete_operation_type<T>::type;

// Primary template to handle types without T::ContextType member.
template <class, class = void>
class has_context_type_member : public std::false_type {
 public:
  using type = Monostate;
};

// Specialization to handle types with T::ContextType member.
template <class T>
class has_context_type_member<T, std::void_t<typename T::ContextType>>
    : public std::true_type {
 public:
  using type = typename T::ContextType;
};

// Helper to simplify usage of has_context_type_member<T>::type.
template <class T>
using context_type_t = typename has_context_type_member<T>::type;

// Returns a type which has half the element size of that of type T.
template <class T>
class half_element_width;

template <class T>
using half_element_width_t = typename half_element_width<T>::type;

// Returns a type which has double the element size of that of type T.
template <class T>
class double_element_width;

template <class T>
using double_element_width_t = typename double_element_width<T>::type;

template <>
class double_element_width<int8_t> {
 public:
  using type = int16_t;
};

template <>
class double_element_width<uint8_t> {
 public:
  using type = uint16_t;
};

template <>
class double_element_width<int16_t> {
 public:
  using type = int32_t;
};

template <>
class double_element_width<uint16_t> {
 public:
  using type = uint32_t;
};

template <>
class double_element_width<int32_t> {
 public:
  using type = int64_t;
};

template <>
class double_element_width<uint32_t> {
 public:
  using type = uint64_t;
};

// Trait which disables object copy.
class NonCopyable {
 public:
  NonCopyable() = default;
  NonCopyable(const NonCopyable &) = delete;
  NonCopyable &operator=(const NonCopyable &) = delete;
};  // end of class NonCopyable

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_TYPE_TRAITS_H
