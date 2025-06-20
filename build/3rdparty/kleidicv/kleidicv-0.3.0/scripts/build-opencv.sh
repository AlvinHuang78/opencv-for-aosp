#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

# Builds a given target for OpenCV.
#
# The build artifacts are placed in the `build` directory.
#
# Arguments:
#   1: Target to build. Defaults to default target'.
#
# To target Android devices the following command can be used:
#   BUILD_ID=opencv-android \
#     OPENCV_PATH=/path/to/opencv \
#     CMAKE_TOOLCHAIN_FILE=/path/to/android-ndk/build/cmake/android.toolchain.cmake \
#     EXTRA_CMAKE_ARGS="-DANDROID_ABI=arm64-v8a" \
#     scripts/build-opencv.sh
#
# To target Android devices with KleidiCV the following command can be used:
#   BUILD_ID=opencv-android-kleidicv \
#     OPENCV_PATH=/path/to/opencv \
#     CMAKE_TOOLCHAIN_FILE=/path/to/android-ndk/build/cmake/android.toolchain.cmake \
#     EXTRA_CMAKE_ARGS="-DANDROID_ABI=arm64-v8a -DWITH_KLEIDICV=ON" \
#     scripts/benchmark/build-opencv.sh
#
# Options:
#   BUILD_ID:                      Identifier of the build, defaults to 'opencv'.
#   BUILD_PATH:                    Directory for all the files associated with a build.
#   CLEAN:                         Clean builds if set to 'ON'. Defaults to 'OFF'.
#   CMAKE:                         Full path of the CMake executable.
#   CMAKE_BUILD_TYPE:              Specifies the build type. Defaults to 'Release'.
#   CMAKE_CROSSCOMPILING_EMULATOR: If set, it is the full path to an emulator that can run the binary target.
#   CMAKE_CXX_FLAGS:               General C++ flags for all compiler commands.
#   CMAKE_EXE_LINKER_FLAGS:        General flags for all linker commands for executables.
#   CMAKE_GENERATOR:               Generator to use, see cmake documentation for details. Defaults to 'Ninja'.
#   CMAKE_SHARED_LINKER_FLAGS:     General flags for all linker commands for shared libraries.
#   CMAKE_TOOLCHAIN_FILE:          If set, it is the full path to the CMake toolchain file.
#   CMAKE_VERBOSE_MAKEFILE:        Enables verbose logs during builds. Defaults to 'OFF'.
#   EXTRA_CMAKE_ARGS:              Any additional args to pass to CMake.
#   SOURCE_PATH:                   Path of KleidiCV source directory. Defaults to parent directory of this script.
#   OPENCV_PATH:                   Path of OpenCV source directory. Mandatory!
# ------------------------------------------------------------------------------

set -exu

# ------------------------------------------------------------------------------
# Automatic configuration
# ------------------------------------------------------------------------------

SCRIPT_PATH="$(realpath "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"

: "${BUILD_ID:=opencv}"
: "${BUILD_PATH:=${SCRIPT_PATH}/../build/${BUILD_ID}}"
: "${CLEAN:=OFF}"
: "${CMAKE:=cmake}"
: "${CMAKE_BUILD_TYPE:=Release}"
: "${CMAKE_CROSSCOMPILING_EMULATOR:=}"
: "${CMAKE_CXX_FLAGS:=}"
: "${CMAKE_EXE_LINKER_FLAGS:=}"
: "${CMAKE_GENERATOR:=Ninja}"
: "${CMAKE_SHARED_LINKER_FLAGS:=}"
: "${CMAKE_TOOLCHAIN_FILE:=}"
: "${CMAKE_VERBOSE_MAKEFILE:=OFF}"
: "${EXTRA_CMAKE_ARGS:=}"
: "${SOURCE_PATH:="$(realpath "${SCRIPT_PATH}"/../)"}"

if [[ -z "${OPENCV_PATH:-}" ]]; then
  echo "Please specify the local path to a checked out (cloned) OpenCV repo in the OPENCV_PATH env variable"
  exit 1
fi

IFS=' ' read -r -a CMAKE_TARGETS <<< "${1:-}"
IFS=' ' read -r -a EXTRA_CMAKE_ARGS_ARRAY <<< "${EXTRA_CMAKE_ARGS}"

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

if [[ "${CLEAN}" == "ON" ]]; then
    rm -rf "${BUILD_PATH}"
fi

mkdir -p "${BUILD_PATH}"
BUILD_PATH="$(realpath "${BUILD_PATH}")"

cmake_config_args=(
    -Wno-dev
    -S "${OPENCV_PATH}"
    -B "${BUILD_PATH}"
    -G "${CMAKE_GENERATOR}"
    "-DCMAKE_VERBOSE_MAKEFILE=${CMAKE_VERBOSE_MAKEFILE}"
    "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
    "-DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
    "-DCMAKE_EXE_LINKER_FLAGS=${CMAKE_EXE_LINKER_FLAGS}"
    "-DCMAKE_SHARED_LINKER_FLAGS=${CMAKE_SHARED_LINKER_FLAGS}"
    "-DCMAKE_CXX_STANDARD=14"
)

if [[ -n "${CMAKE_CROSSCOMPILING_EMULATOR}" ]]; then
    cmake_config_args+=(
        "-DCMAKE_CROSSCOMPILING_EMULATOR=${CMAKE_CROSSCOMPILING_EMULATOR}"
    )
fi

if [[ -n "${CMAKE_TOOLCHAIN_FILE}" ]]; then
    cmake_config_args+=(
        "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
    )
fi

if [[ -n "${EXTRA_CMAKE_ARGS_ARRAY-}" ]]; then
    cmake_config_args+=("${EXTRA_CMAKE_ARGS_ARRAY[@]}")
fi

"${CMAKE}" "${cmake_config_args[@]}"

# ------------------------------------------------------------------------------
# Build
# ------------------------------------------------------------------------------

cmake_build_args=(
    --build "${BUILD_PATH}"
    --parallel
)

if [[ -n "${CMAKE_TARGETS[*]}" ]]; then
    cmake_build_args+=(--target "${CMAKE_TARGETS[@]}")
fi

"${CMAKE}" "${cmake_build_args[@]}"

# ------------------------------------------------------------------------------
# End of script
# ------------------------------------------------------------------------------
