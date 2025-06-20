// SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0

#include "multithreading_fake.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <vector>

#include "framework/utils.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv_thread/kleidicv_thread.h"

// Fake multithreading to allow running tests in environments that don't support
// multithreading.
// We rely on other testsuites to exercise the system with real multithreading.
static kleidicv_error_t parallel_fake(kleidicv_thread_callback callback,
                                      void *callback_data, void *parallel_data,
                                      unsigned task_count) {
  const unsigned max_thread_count =
      static_cast<unsigned>(reinterpret_cast<uintptr_t>(parallel_data));
  const unsigned thread_count = std::min(task_count, max_thread_count);

  std::vector<kleidicv_error_t> results(thread_count,
                                        KLEIDICV_ERROR_CONTEXT_MISMATCH);

  std::vector<unsigned> task_indices(thread_count);
  // initialize task_indices with 0, 1, 2...
  std::iota(task_indices.begin(), task_indices.end(), 0);
  // Put task_indices in random order to simulate the unpredictability of
  // multithreading.
  std::shuffle(task_indices.begin(), task_indices.end(),
               std::minstd_rand{test::Options::seed()});

  for (unsigned i : task_indices) {
    unsigned task_begin = i * task_count / thread_count;
    unsigned task_end = i + 1 == thread_count
                            ? task_count
                            : (i + 1) * task_count / thread_count;
    results[i] = callback(task_begin, task_end, callback_data);
  }

  for (auto r : results) {
    if (r != KLEIDICV_OK) {
      return r;
    }
  }
  return KLEIDICV_OK;
}

kleidicv_thread_multithreading get_multithreading_fake(uintptr_t thread_count) {
  // NOLINTBEGIN(performance-no-int-to-ptr)
  void *parallel_data = reinterpret_cast<void *>(thread_count);
  // NOLINTEND(performance-no-int-to-ptr)
  return kleidicv_thread_multithreading{parallel_fake, parallel_data};
}
