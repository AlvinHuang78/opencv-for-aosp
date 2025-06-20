// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_SIGMA_H
#define KLEIDICV_SIGMA_H

#include <array>
#include <cmath>

#include "kleidicv/config.h"

namespace KLEIDICV_TARGET_NAMESPACE {

static constexpr size_t get_half_kernel_size(size_t kernel_size)
    KLEIDICV_STREAMING_COMPATIBLE {
  // since kernel sizes are odd, "half" here means that
  // the extra element is included
  return (kernel_size >> 1) + 1;
}

// This function is not marked as streaming compatible, as std::round is also
// not streaming compatible.
template <size_t HalfKernelSize>
static std::array<uint16_t, HalfKernelSize> generate_gaussian_half_kernel(
    float sigma) {
  // Define the mid point of the full kernel range.
  constexpr size_t kMid = HalfKernelSize - 1;

  // Define the full kernel size.
  constexpr size_t KernelSize = kMid * 2 + 1;

  // Calculate the sigma manually in case it is not defined.
  if (sigma == 0.0) {
    sigma = static_cast<float>(KernelSize) * 0.15 + 0.35;
  }

  // Temporary float half-kernel.
  std::array<float, HalfKernelSize> half_kernel_float{};

  // Prepare the sigma value for later multiplication inside a loop.
  float coefficient = 1 / -(2 * sigma * sigma);

  float sum = 0.0;
  size_t j = kMid;
  for (size_t i = 0; i < kMid; i++, j--) {
    half_kernel_float[i] =
        std::exp(static_cast<float>(j) * static_cast<float>(j) * coefficient);
    sum += half_kernel_float[i];
  }

  // This multiplier is used for two things:
  // * For normalizing the kernel values, so the sum of the final values is 1.
  //   (The 'sum' variable only accounts for the half of the kernel values
  //   without the mid point. That is the reason for the division by
  //   '(sum * 2 + 1)'.)
  // * For converting the values to fixed-point (uint16_t), where 8 bits are
  //   used for the fractional part. That is the reason for the multiplication
  //   by 256.
  float multiplier = 256 / (sum * 2 + 1);

  // Result half-kernel
  std::array<uint16_t, HalfKernelSize> half_kernel{};

  // Normalize the kernel and convert it to the fixed-point format. Rounding
  // errors are diffused in the kernel.
  float error = 0.0;
  for (size_t i = 0; i < kMid; i++) {
    float value = half_kernel_float[i] * multiplier - error;
    float value_rounded = std::round(value);
    half_kernel[i] = static_cast<uint16_t>(value_rounded);
    error = value_rounded - value;
  }
  half_kernel[kMid] = static_cast<uint16_t>(std::round(multiplier - error));

  return half_kernel;
}

}  // namespace KLEIDICV_TARGET_NAMESPACE

#endif  // KLEIDICV_SIGMA_H
