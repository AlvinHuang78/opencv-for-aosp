<!--
SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>

SPDX-License-Identifier: Apache-2.0
-->

# Benchmarking KleidiCV

KleidiCV has a small set of benchmarks to test the performance of some of its APIs.

Building of the benchmarks can be enabled with the `cmake` argument `-DKLEIDICV_BENCHMARK=ON`.

Benchmarks should be built in Release mode via the `CMAKE_BUILD_TYPE` parameter.
Building is otherwise as described in the [building documentation](building.md).

For example, on Linux:
```
cmake -S /path/to/kleidicv -B build-kleidicv-linux \
  -DCMAKE_BUILD_TYPE=Release \
  -DKLEIDICV_BENCHMARK=ON
cmake --build build-kleidicv-linux --parallel
```

The resulting benchmark binary will be available in the build folder at `benchmark/kleidicv-benchmark`.

The benchmarks are based on [Google Benchmark](https://github.com/google/benchmark).
All the standard Google Benchmark command line arguments can be used.

By default, benchmarks will be run on an image size of 1280*720.
The image size can be changed via the command line options
`--image_width=` and `--image_height=`. For example, to run the
benchmarks for a "4K" image size use the options
`--image_width=3840 --image_height=2160`.
