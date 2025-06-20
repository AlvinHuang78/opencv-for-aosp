#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

set -exu

# Ensure we're at the root of the repo.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Ensure we're doing a clean build
rm -rf build

if ! command -v qemu-aarch64; then
  apt-get update
  apt-get -y --no-install-recommends install qemu-user
fi

# Force ccache for all CMake builds
export CMAKE_CXX_COMPILER_LAUNCHER=ccache

# Check format of C++ files
CHECK_ONLY=ON VERBOSE=ON scripts/format.sh

scripts/cpplint.sh

# Check format of shell scripts
# shellcheck disable=SC2046
# Word splitting is essential here.
shellcheck $(find scripts -name '*.sh' | tr '\n' ' ')

# Check license headers
reuse lint

# Generate documentation
doxygen

# Build
cmake -S . -B build -G Ninja \
  -DCMAKE_COMPILE_WARNING_AS_ERROR=ON \
  -DCMAKE_CXX_CLANG_TIDY=clang-tidy-19 \
  -DCMAKE_CXX_FLAGS="--target=aarch64-linux-gnu --coverage" \
  -DCMAKE_EXE_LINKER_FLAGS="--rtlib=compiler-rt -static -fuse-ld=lld" \
  -DKLEIDICV_ENABLE_SME2=ON \
  -DKLEIDICV_LIMIT_SME2_TO_SELECTED_ALGORITHMS=OFF \
  -DKLEIDICV_LIMIT_SVE2_TO_SELECTED_ALGORITHMS=OFF \
  -DKLEIDICV_CHECK_BANNED_FUNCTIONS=ON

# Workaround to avoid applying clang-tidy to files in external projects.
echo '{"Checks": "-*,cppcoreguidelines-avoid-goto"}'>build/_deps/.clang-tidy

ninja -C build

# Build with GCC
CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ cmake -S . -B build/gcc -G Ninja \
  -DCMAKE_COMPILE_WARNING_AS_ERROR=ON \
  -DCMAKE_EXE_LINKER_FLAGS="-static"
ninja -C build/gcc

# Run tests
LONG_VECTOR_TESTS="GRAY2.*:RGB*:Yuv*:Rgb*:Resize*"
TESTRESULT=0
qemu-aarch64     build/test/framework/kleidicv-framework-test --gtest_output=xml:build/test-results/ || TESTRESULT=1
qemu-aarch64 -cpu cortex-a35 build/test/api/kleidicv-api-test --gtest_output=xml:build/test-results/clang-neon/ || TESTRESULT=1
qemu-aarch64 -cpu max,sve128=on,sme=off \
  build/test/api/kleidicv-api-test --gtest_output=xml:build/test-results/clang-sve128/ --vector-length=16 || TESTRESULT=1
qemu-aarch64 -cpu max,sve2048=on,sve-default-vector-length=256,sme=off \
  build/test/api/kleidicv-api-test --gtest_filter="${LONG_VECTOR_TESTS}" --gtest_output=xml:build/test-results/clang-sve2048/ --vector-length=256 || TESTRESULT=1
qemu-aarch64 -cpu max,sve128=on,sme512=on \
  build/test/api/kleidicv-api-test --gtest_output=xml:build/test-results/clang-sme/ --vector-length=64 || TESTRESULT=1
qemu-aarch64 -cpu cortex-a35 build/gcc/test/api/kleidicv-api-test --gtest_output=xml:build/test-results/gcc-neon/ || TESTRESULT=1

scripts/prefix_testsuite_names.py build/test-results/clang-neon/kleidicv-api-test.xml "clang-neon."
scripts/prefix_testsuite_names.py build/test-results/clang-sve128/kleidicv-api-test.xml "clang-sve128."
scripts/prefix_testsuite_names.py build/test-results/clang-sve2048/kleidicv-api-test.xml "clang-sve2048."
scripts/prefix_testsuite_names.py build/test-results/clang-sme/kleidicv-api-test.xml "clang-sme."
scripts/prefix_testsuite_names.py build/test-results/gcc-neon/kleidicv-api-test.xml "gcc-neon."

# Generate test coverage report
LLVM_COV=llvm-cov scripts/generate_coverage_report.py

# Sanitizers don't work when run through qemu so must be run natively.
if [[ $(dpkg --print-architecture) = arm64 ]]; then
  # Clang address & undefined behaviour sanitizers
  cmake -S . -B build/sanitize -G Ninja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_COMPILE_WARNING_AS_ERROR=ON \
    -DKLEIDICV_ENABLE_SME2=OFF \
    -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-sanitize-recover=all -Wno-pass-failed"
  ninja -C build/sanitize kleidicv-api-test
  build/sanitize/test/api/kleidicv-api-test
fi

# Build benchmarks, just to prevent bitrot.
cmake -S . -B build/build-benchmark -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_COMPILE_WARNING_AS_ERROR=ON \
  -DCMAKE_CROSSCOMPILING_EMULATOR=qemu-aarch64 \
  -DCMAKE_CXX_COMPILER_TARGET=aarch64-linux-gnu \
  -DCMAKE_EXE_LINKER_FLAGS="--rtlib=compiler-rt -static -fuse-ld=lld" \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DKLEIDICV_BENCHMARK=ON \
  -DKLEIDICV_ENABLE_SME2=ON \
  -DKLEIDICV_LIMIT_SME2_TO_SELECTED_ALGORITHMS=OFF \
  -DKLEIDICV_LIMIT_SVE2_TO_SELECTED_ALGORITHMS=OFF
ninja -C build/build-benchmark kleidicv-benchmark

# TODO: Cross-build OpenCV
if [[ $(dpkg --print-architecture) = arm64 ]]; then
  # Check OpenCV-KleidiCV integration
  scripts/ci-opencv.sh
fi

exit $TESTRESULT
