// SPDX-FileCopyrightText: 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_CONTAINERS_STACK_H
#define KLEIDICV_CONTAINERS_STACK_H

#include <cinttypes>
#include <cstddef>
#include <memory>

#include "kleidicv/config.h"
#include "kleidicv/traits.h"
#include "kleidicv/utils.h"

namespace KLEIDICV_TARGET_NAMESPACE {

// A simple last-in first-out container.
template <typename T>
class Stack final : public NonCopyable {
 public:
  // As per the usual container interface requirements.
  using value_type = T;
  using reference = value_type &;
  using const_reference = const value_type &;

  // NOLINTBEGIN(cppcoreguidelines-prefer-member-initializer)
  explicit Stack() noexcept {
    current_block_ = Block::make_block(nullptr);
    // Note: push_back() first increments the back_ pointer.
    back_min_ = current_block_->data() - 1;
    back_max_ = &back_min_[kBlockCapacity];
    back_ = back_min_;
    empty_back_ = back_min_;
    first_block_ = current_block_;
  }
  // NOLINTEND(cppcoreguidelines-prefer-member-initializer)

  ~Stack() noexcept {
    Block *block = first_block_;
    while (block) {
      Block *next_block = block->next();
      delete[] block;
      block = next_block;
    }
  }

  // Appends an element to the end of the container.
  void push_back(const_reference value) noexcept {
    if (back_ == back_max_) {
      next_block();
    }

    *++back_ = value;
  }

  // Returns a reference to the last element in the container.  It is undefined
  // behavior to call back() if the container is empty.
  reference back() noexcept { return *back_; }

  // Returns a reference to the last element in the container.  It is undefined
  // behavior to call back() if the container is empty.
  const_reference back() const noexcept { return *back_; }

  // Removes the last element from the container. It is undefined behavior to
  // call pop_back() if the container is empty.
  void pop_back() noexcept {
    --back_;
    if ((back_ == back_min_) && (back_ != empty_back_)) {
      prev_block();
    }
  }

  // Retruns true if the container is empty, otherwise false.
  [[nodiscard]] bool empty() const noexcept { return back_ == empty_back_; }

 private:
  // Granularity of blocks in bytes. Blocks are a multiple of this value in
  // size. It matches the smallest page size, therefore it should work well with
  // systems with larger page size too.
  static constexpr size_t kBlockGranule = 2 * 4096UL;

  static_assert(sizeof(T) <= sizeof(void *),
                "Current implementation is limited. Please improve it.");

  // Alignment of elements.
  static constexpr size_t kElementAlignment = alignof(void *);

  static_assert(alignof(T) <= alignof(void *),
                "Current implementation is limited. Please improve it.");

  // Doubly-linked list to manage allocated data blocks.
  class Block final {
   public:
    // Returns true if the current block is followed by another block, otherwise
    // false.
    bool has_next() const noexcept { return next_ != nullptr; }

    // Returns the previous block (if any) in the chain.
    Block *prev() const noexcept { return prev_; }

    // Returns the next block (if any) in the chain.
    Block *next() const noexcept { return next_; }

    // Returns a pointer to the first data element in the block.
    T *data() noexcept { return reinterpret_cast<T *>(&data_[0]); }

    // Allocates and initializes a new block.
    [[nodiscard]] static Block *make_block(Block *prev) noexcept {
      uint8_t *block_data = new (std::nothrow) uint8_t[kBlockGranule];
      if (KLEIDICV_UNLIKELY(!block_data)) {
        __builtin_trap();
      }

      Block *block = new (block_data) Block(prev);
      if (prev) {
        prev->next_ = block;
      }

      return block;
    }

   private:
    explicit Block(Block *prev) : prev_{prev}, next_{nullptr} {}

    // Pointer to the previous block in the doubly-linked list.
    Block *prev_;
    // Pointer to the next block in the doubly-linked list.
    Block *next_;
    // Address of data. This does not actually occupy any storage.
    alignas(kElementAlignment) uint8_t data_[0];
  };  // end of class Block

  // Advances to the next block, optionally allocating a new one.
  void next_block() noexcept {
    if (current_block_->has_next()) {
      current_block_ = current_block_->next();
    } else {
      current_block_ = Block::make_block(current_block_);
    }

    // Note: push_back() first increments the back_ pointer.
    back_min_ = current_block_->data() - 1;
    back_max_ = &back_min_[kBlockCapacity];
    back_ = back_min_;
  }

  // Advances to the previous block.
  void prev_block() noexcept {
    current_block_ = current_block_->prev();
    back_min_ = current_block_->data() - 1;
    back_max_ = &back_min_[kBlockCapacity];
    back_ = back_max_;
  }

  // Number of elements a block can hold.
  static constexpr size_t kBlockCapacity =
      (kBlockGranule - sizeof(Block)) / sizeof(T);

  // Pointer to the last entry in the container.
  T *back_;
  // Pointer which signals that the stack is empty.
  T *empty_back_;
  // Pointer to mark where back_ is invalid upon a pop_back() call.
  T *back_min_;
  // Pointer to the last entry in the current block which is still valid.
  T *back_max_;
  // Pointer to the current block.
  Block *current_block_;
  // Pointer to the first block.
  Block *first_block_;
};  // end of class Stack<T>

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_CONTAINERS_STACK_H
