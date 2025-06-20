#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

# Environment variables used:
#   ADB:                 adb executable. Must be set.
#   BUILD_ROOT_PATH:     Directory of the different builds to push.
#   CUSTOM_BUILD_SUFFIX: Try this build suffix for the extra custom build. Defaults to 'custom'.
# Note:
#   Use standard ADB env vars (like ANDROID_SERIAL, ANDROID_ADB_SERVER_ADDRESS and
#   ANDROID_ADB_SERVER_PORT) to customize ADB calls.

set -exu

if [[ -z "${ADB:-}" ]]; then
  echo "Required variable 'ADB' is not set. Please set it to point to adb command."
  exit 1
fi

BENCHMARK_SCRIPT_PATH="$(realpath "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"
BUILD_ROOT_PATH="${BUILD_ROOT_PATH:-"${BENCHMARK_SCRIPT_PATH}/../../build"}"
CUSTOM_BUILD_SUFFIX="${CUSTOM_BUILD_SUFFIX:-custom}"

: "${DEV_DIR:=/data/local/tmp}"

"${ADB}" push "${BUILD_ROOT_PATH}"/opencv-vanilla/bin/opencv_perf_core    "${DEV_DIR}"/opencv_perf_core_vanilla
"${ADB}" push "${BUILD_ROOT_PATH}"/opencv-vanilla/bin/opencv_perf_imgproc "${DEV_DIR}"/opencv_perf_imgproc_vanilla
"${ADB}" push "${BUILD_ROOT_PATH}"/opencv-vanilla/bin/opencv_perf_video "${DEV_DIR}"/opencv_perf_video_vanilla

"${ADB}" push "${BUILD_ROOT_PATH}"/opencv-kleidicv/bin/opencv_perf_core    "${DEV_DIR}"/opencv_perf_core_kleidicv
"${ADB}" push "${BUILD_ROOT_PATH}"/opencv-kleidicv/bin/opencv_perf_imgproc "${DEV_DIR}"/opencv_perf_imgproc_kleidicv
"${ADB}" push "${BUILD_ROOT_PATH}"/opencv-kleidicv/bin/opencv_perf_video "${DEV_DIR}"/opencv_perf_video_kleidicv

if [[ -f "${BUILD_ROOT_PATH}"/opencv-kleidicv-"${CUSTOM_BUILD_SUFFIX}"/bin/opencv_perf_core ]] && \
   [[ -f "${BUILD_ROOT_PATH}"/opencv-kleidicv-"${CUSTOM_BUILD_SUFFIX}"/bin/opencv_perf_imgproc ]]; then
  "${ADB}" push "${BUILD_ROOT_PATH}"/opencv-kleidicv-"${CUSTOM_BUILD_SUFFIX}"/bin/opencv_perf_core    "${DEV_DIR}"/opencv_perf_core_kleidicv_"${CUSTOM_BUILD_SUFFIX}"
  "${ADB}" push "${BUILD_ROOT_PATH}"/opencv-kleidicv-"${CUSTOM_BUILD_SUFFIX}"/bin/opencv_perf_imgproc "${DEV_DIR}"/opencv_perf_imgproc_kleidicv_"${CUSTOM_BUILD_SUFFIX}"
  "${ADB}" push "${BUILD_ROOT_PATH}"/opencv-kleidicv-"${CUSTOM_BUILD_SUFFIX}"/bin/opencv_perf_video "${DEV_DIR}"/opencv_perf_video_kleidicv_"${CUSTOM_BUILD_SUFFIX}"
fi

"${ADB}" push "${BENCHMARK_SCRIPT_PATH}"/perf_test_op.sh "${BENCHMARK_SCRIPT_PATH}"/run_benchmarks.sh "${BENCHMARK_SCRIPT_PATH}"/benchmarks*.txt "${DEV_DIR}"/
