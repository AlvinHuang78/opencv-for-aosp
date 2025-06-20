// SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include <climits>
#include <limits>

#include "kleidicv/kleidicv.h"
#include "kleidicv/neon.h"

// This algorithm calculates the index of element in src array of the global
// minimum and maximum values.
//
// Theory of operation
//
//  At most max_vectors_per_block() number of consecutive vector registers are
//  processed with the vector path before performing some additional
//  calculations. In every vector path iteration the algorithm remembers
//  the following data per lane for the currently processed block:
//    - [1] offsets of the vector where the last min/max value was seen, and
//    - [2] overall min/max values.
//
//  Block index
//
//  Since our offsets only contain an element-size value, we need to
//  extend this offset with a base value, to form a full index.
//  This base value is the block's starting index:
//
//    final index = block_index + offset * num_lanes_in_vector + lane_number
//
//  Once a block is completed, min/max_block_indices are updated, and the
//  per-lane offsets and min/max values are also copied to the global vectors,
//  whichever lane has been updated.
//
//  Which lane has been updated?
//
//  Offsets are relative to block index, and are numbered from 1 to
//  max_vectors_per_block(). At start of a block, offsets are zeroed,
//  so after the block a non-zero check is enough to flag if an offset
//  (and the min/max value) is updated.
//
//  How is the update done?
//
//  From the offset lanes, a single flag is made: are there any updates.
//  If yes, then the min/max_block_index is copied from running_index
//  (this is done effectively by a CSEL instruction).
//  Min/max values and relative lane offsets are copied by counting a
//  no_update vector lane-by-lane (comparing each lane to zero), so only the
//  changed lanes are copied to the global min/max and offset vector. This is
//  important to copy only the values that belong to the actual block index
//  and are smaller/bigger than last global min/max.
//  Otherwise we could get false results, see this example:
//  vec: [ 4, 4, 4, 4]  -->  base_index: 0       offsets: 1 1 1 1
//   vmax: 4, 4, 4, 4
//  vec: [ 1,10, 2, 3]  -->  base_index: 1   new_offsets: 0 3 0 0
//   vmax: 4,10, 4, 4    max_base_index: 1   --> offsets: 1 3 1 1
//  vec: [10, 1, 1, 1]  -->
//   vmax:10,10, 4, 4   --- but we did not update the base index + offset!
//    we would return base_index 1 offset 1, which is completely wrong
//
//  Min/max values updated
//
//  For all the above to work, we have to make sure we don't overwrite the
//  global min/max values and offsets with an updated value of a non global
//  min/max. To ensure this, after each block global min/max are calculated
//  horizontally across from min/max per lane values, and these new global
//  min/max values copied to all lanes and are used to compare values with
//  in the next block.
//
// Example of 'min' for hypothetical 4 lanes and uint8_t type and
// max_block_size() = 2.
//
//  Initial state:
//    vrunning_offset_:  [ 0x01, 0x01, 0x01, 0x01 ]
//    vmin_offsets_:     [ 0x00, 0x00, 0x00, 0x00 ]
//    vmin_offsets_new_: [ 0x00, 0x00, 0x00, 0x00 ]
//    vmin_new_:         [ 0xFF, 0xFF, 0xFF, 0xFF ]
//    vmin_:             [ 0xFF, 0xFF, 0xFF, 0xFF ]
//
//  1st block:
//  -> vector_path()
//    vec:               [ 0x12, 0xFF, 0x20, 0x42 ]
//    is_smaller:        [ 0xFF, 0x00, 0xFF, 0xFF ]
//    vmin_offsets_new_: [ 0x01, 0x00, 0x01, 0x01 ]
//    vmin_new_:         [ 0x12, 0xFF, 0x20, 0x42 ]
//    vrunning_offset_:  [ 0x02, 0x02, 0x02, 0x02 ]
//  -> vector_path()
//    vec:               [ 0x22, 0xFF, 0x55, 0x23 ]
//    is_smaller:        [ 0x00, 0x00, 0x00, 0xFF ]
//    vmin_offsets_new_: [ 0x01, 0x00, 0x01, 0x02 ]
//    vmin_new_:         [ 0x12, 0xFF, 0x00, 0x23 ]
//    vrunning_offset_:  [ 0x03, 0x03, 0x03, 0x03 ]
//  -> on_block_finished(1)
//    vno_update:        [ 0x00, 0xFF, 0x00, 0x00 ]
//    vmin_offsets_:     [ 0x01, 0x00, 0x01, 0x02 ]
//    vmin_:             [ 0x12, 0xFF, 0x20, 0x23 ]
//    vmin_new_:         [ 0x12, 0x12, 0x12, 0x12 ]

namespace kleidicv::neon {

template <typename ScalarType>
class MinMaxLoc final : public UnrollTwice {
 public:
  using VecTraits = neon::VecTraits<ScalarType>;
  using VectorType = typename VecTraits::VectorType;
  using UnsignedVectorType =
      typename neon::VecTraits<std::make_unsigned_t<ScalarType>>::VectorType;

  // NOLINTBEGIN(cppcoreguidelines-prefer-member-initializer)
  MinMaxLoc() {
    vmin_offsets_ = vmax_offsets_ = vmin_offsets_new_ = vmax_offsets_new_ =
        vdupq_n(ScalarType{0});
    vrunning_offset_ = vdupq_n(ScalarType{1});
    vmin_ = vmin_new_ = vdupq_n(std::numeric_limits<ScalarType>::max());
    vmax_ = vmax_new_ = vdupq_n(std::numeric_limits<ScalarType>::lowest());
    min_block_index_ = max_block_index_ = 0;
    min_vector_ = min_scalar_ = std::numeric_limits<ScalarType>::max();
    max_vector_ = max_scalar_ = std::numeric_limits<ScalarType>::lowest();
    min_scalar_index_ = max_scalar_index_ = 0;
    running_index_ = 0;
  }
  // NOLINTEND(cppcoreguidelines-prefer-member-initializer)

  void vector_path(VectorType src) {
    VectorType v_is_smaller = vcltq_u8(src, vmin_new_);
    vmin_offsets_new_ =
        vbslq_u8(v_is_smaller, vrunning_offset_, vmin_offsets_new_);
    vmin_new_ = vminq_u8(vmin_new_, src);

    VectorType v_is_bigger = vcgtq_u8(src, vmax_new_);
    vmax_offsets_new_ =
        vbslq_u8(v_is_bigger, vrunning_offset_, vmax_offsets_new_);
    vmax_new_ = vmaxq_u8(vmax_new_, src);

    vrunning_offset_ = vaddq_u8(vrunning_offset_, vdupq_n(ScalarType{1}));
  }

  void scalar_path(ScalarType src) {
    if (src < min_scalar_) {
      min_scalar_ = src;
      min_scalar_index_ = running_index_;
    }

    if (src > max_scalar_) {
      max_scalar_ = src;
      max_scalar_index_ = running_index_;
    }

    ++running_index_;
  }

  void on_block_finished(size_t vectors_in_block) {
    min_vector_ = vminvq_u8(vmin_new_);
    // Does any lane have a new min? Then update the block number
    // 32-bit horizontal max is the fastest way to do a logical OR
    min_block_index_ = vmaxvq_u32(vreinterpretq_u32_u8(vmin_offsets_new_))
                           ? running_index_
                           : min_block_index_;
    // Which lane is zero, that was not updated in last block
    uint8x16_t vno_update = vceqzq_u8(vmin_offsets_new_);
    // Copy the updated (new min!) offsets into the ultimate offset vector
    vmin_offsets_ = vbslq_u8(vno_update, vmin_offsets_, vmin_offsets_new_);
    // Copy the updated new min values into the ultimate min vector
    vmin_ = vbslq_u8(vno_update, vmin_, vmin_new_);
    // Copy new min to all lanes, to prevent overwriting it with inferior values
    vmin_new_ = vdupq_n(min_vector_);

    // Just like min, but for max
    max_vector_ = vmaxvq_u8(vmax_new_);
    max_block_index_ = vmaxvq_u32(vreinterpretq_u32_u8(vmax_offsets_new_))
                           ? running_index_
                           : max_block_index_;
    vno_update = vceqzq_u8(vmax_offsets_new_);
    vmax_offsets_ = vbslq_u8(vno_update, vmax_offsets_, vmax_offsets_new_);
    vmax_ = vbslq_u8(vno_update, vmax_, vmax_new_);
    vmax_new_ = vdupq_n(max_vector_);

    running_index_ += vectors_in_block * VecTraits::num_lanes();
    // Starts from 1, so if an offset is updated, we can tell from it is nonzero
    vrunning_offset_ = vdupq_n(ScalarType{1});
    // Offsets are zeroed so we'll see when they updated
    vmax_offsets_new_ = vmin_offsets_new_ = vdupq_n(ScalarType{0});
  }

  size_t max_vectors_per_block() const {
    // -1: see description of vrunning_offset_
    return std::numeric_limits<ScalarType>::max() - 1;
  }

  size_t min_index() const {
    // Which minimum is the smaller one, scalar path or vector path?
    if (min_scalar_ < min_vector_) {
      return min_scalar_index_;
    }

    // To find the minimum location:
    // 1. Get the lanes that have the same min value as the global min.
    // 2. From the offsets of these lanes, find the smallest one.
    // 3. From the smallest ones (it can happen if they were in the same vector)
    //    take the first one (LSB, because little endian)
    // Example offsets:  [ 0x01, 0x00, 0x01, 0x02, 0x01, 0x03 ]
    //           vmin_:  [ 0x12, 0xFF, 0x20, 0x23, 0x12, 0x12 ]
    // Fill not selected offsets with maximum value, they won't be the minimum
    const uint8x16_t v_ignore = vdupq_n(static_cast<ScalarType>(
        std::numeric_limits<std::make_unsigned_t<ScalarType>>::max()));
    // [1] Test which lanes contain the global minimum value
    //         is_min:   [ 0xFF, 0x00, 0x00, 0x00, 0xFF, 0xFF ]
    VectorType is_min = vcleq(vmin_, vmin_new_);
    // Select those offsets that contain the minimum
    // selected_offsets: [ 0x01, 0xFF, 0xFF, 0xFF, 0x01, 0x03 ]
    UnsignedVectorType v_selected_offsets =
        vbslq_u8(is_min, vmin_offsets_, v_ignore);
    // [2] Take the smallest offset, that's the earliest, that is needed
    //  earliest_offset:   0x01
    ScalarType earliest_offset = vminvq(v_selected_offsets);
    // In the rare case the vector_path did not run, no selected lane
    // so the scalar_min is the global min (e.g. when src[0] is the min)
    if (earliest_offset == 0) {
      return min_scalar_index_;
    }

    // [3] Calculate the lane for the first one
    //             lane: 0  (smaller than 4)
    size_t lane = lane_of_first_smallest(v_selected_offsets, earliest_offset);
    size_t min_vector_index = min_block_index_ +
                              (earliest_offset - 1) * VecTraits::num_lanes() +
                              lane;
    // If the minimum value is not less, then they are equal, so the earlier one
    // is the winner
    return (min_vector_ < min_scalar_ || min_vector_index < min_scalar_index_
                ? min_vector_index
                : min_scalar_index_);
  }

  // Just like min, but for max
  size_t max_index() const {
    if (max_scalar_ > max_vector_) {
      return max_scalar_index_;
    }

    const uint8x16_t v_ignore = vdupq_n(static_cast<ScalarType>(
        std::numeric_limits<std::make_unsigned_t<ScalarType>>::max()));
    VectorType is_max = vcgeq(vmax_, vmax_new_);
    UnsignedVectorType v_selected_offsets =
        vbslq_u8(is_max, vmax_offsets_, v_ignore);
    ScalarType earliest_offset = vminvq(v_selected_offsets);
    if (earliest_offset == 0) {
      return max_scalar_index_;
    }

    size_t lane = lane_of_first_smallest(v_selected_offsets, earliest_offset);
    size_t max_vector_index = max_block_index_ +
                              (earliest_offset - 1) * VecTraits::num_lanes() +
                              lane;
    return (max_vector_ > max_scalar_ || max_vector_index < max_scalar_index_
                ? max_vector_index
                : max_scalar_index_);
  }

 private:
  inline static size_t lane_of_first_smallest(UnsignedVectorType v_values,
                                              ScalarType smallest) {
    // Flag all values that equal to the smallest one
    // example values: [ 0x01, 0xFF, 0xFF, 0xFF, 0x01, 0x03 ]
    //     v_smallest: [ 0x01, 0x01, 0x01, 0x01, 0x01, 0x01 ]
    UnsignedVectorType v_smallest = vdupq_n(smallest);
    //        v_flags: [ 0xFF, 0x00, 0x00, 0x00, 0xFF, 0x00 ]
    UnsignedVectorType v_flags = vcleq(v_values, v_smallest);
    // Find the leftmost of them all
    uint64_t halfvector = vgetq_lane_u64(vreinterpretq_u64(v_flags), 0);
    size_t lane = !halfvector * (sizeof(uint64_t) / sizeof(ScalarType));
    // If the left half is 0, it must be on the right side, take that 64bit
    if (lane) {
      halfvector = vgetq_lane_u64(vreinterpretq_u64(v_flags), 1);
    }

    // Counting the trailing zeroes will tell us where is the leftmost element
    // (Little Endian) ---> example returns 0, this is the leftmost byte
    lane += __builtin_ctzll(halfvector) / (CHAR_BIT * sizeof(ScalarType));
    return lane;
  }

  // Either the actual min/max in the current block or so far global min/max,
  // per lane.
  VectorType vmin_new_, vmax_new_;
  // Offsets of the vectors where the last min/max values were seen, per lane.
  // It contains values from 0 (no new) to max_vectors_per_block().
  UnsignedVectorType vmin_offsets_new_, vmax_offsets_new_;
  // Actual offset of the vector on the vector path, copied to each lane.
  // It contains values from 1 to max_vectors_per_block()
  UnsignedVectorType vrunning_offset_;
  // Global min/max per lane.
  VectorType vmin_, vmax_;
  // Offsets per lane where the first global min/max value was seen.
  UnsignedVectorType vmin_offsets_, vmax_offsets_;
  // Index of the current vector block or scalar element.
  uint64_t running_index_;
  // Index of the current vector block where the first global min/max value was
  // seen.
  uint64_t min_block_index_, max_block_index_;
  // Index of global scalar min and max elements.
  uint64_t min_scalar_index_, max_scalar_index_;
  // Global min/max on vector path.
  ScalarType min_vector_, max_vector_;
  // Global min/max on scalar path.
  ScalarType min_scalar_, max_scalar_;
};  // end of class MinMaxLoc<T>

template <typename ScalarType>
kleidicv_error_t min_max_loc(const ScalarType *src, size_t src_stride,
                             size_t width, size_t height, size_t *min_offset,
                             size_t *max_offset) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  if (KLEIDICV_UNLIKELY(width == 0 || height == 0)) {
    return KLEIDICV_ERROR_RANGE;
  }

  Rectangle rect{width, height};
  Rows<const ScalarType> src_rows{src, src_stride};
  MinMaxLoc<ScalarType> operation;
  apply_block_operation_by_rows(operation, rect, src_rows);
  if (min_offset) {
    *min_offset = src_rows.offset_for_index(operation.min_index(), width);
  }

  if (max_offset) {
    *max_offset = src_rows.offset_for_index(operation.max_index(), width);
  }
  return KLEIDICV_OK;
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(type)                             \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t min_max_loc<type>( \
      const type *src, size_t src_stride, size_t width, size_t height,  \
      size_t *min_offset, size_t *max_offset)

KLEIDICV_INSTANTIATE_TEMPLATE(uint8_t);

}  // namespace kleidicv::neon
