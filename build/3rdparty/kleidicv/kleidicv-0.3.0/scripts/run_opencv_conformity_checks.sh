#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

set -exu

: "${CLEAN:=OFF}"
: "${OPENCV_VERSION:=}"
: "${OPENCV_URL:=}"

SCRIPT_PATH="$(realpath "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"

SOURCE_PATH="${SCRIPT_PATH}/../conformity/opencv"
KLEIDICV_SOURCE_PATH="${SCRIPT_PATH}/.."
BUILD_PATH="${SCRIPT_PATH}/../build/conformity"
OPENCV_DEFAULT_PATH="${BUILD_PATH}/opencv_default"
OPENCV_KLEIDICV_PATH="${BUILD_PATH}/opencv_kleidicv"

if [[ "${CLEAN}" == "ON" ]]; then
    rm -rf "${BUILD_PATH}"
fi

common_cmake_args=(
    -Wno-dev
    -S "${SOURCE_PATH}"
    -G Ninja
    "-DBUILD_SHARED_LIBS=OFF"
    "-DBUILD_TESTS=ON"
    "-DBUILD_PERF_TESTS=OFF"
    "-DBUILD_LIST=imgproc,core,video,ts"

    "-DCV_TRACE=OFF"
    "-DBUILD_EXAMPLES=OFF"
    "-DBUILD_opencv_apps=OFF"
    "-DBUILD_ANDROID_EXAMPLES=OFF"
    "-DBUILD_ANDROID_PROJECTS=OFF"
    "-DBUILD_JAVA=OFF"
    "-DWITH_QT=OFF"
    "-DBUILD_OPENCV_PYTHON=NO"
    "-DBUILD_OPENCV_PYTHON2=NO"
    "-DBUILD_OPENCV_PYTHON3=NO"
    "-DWITH_VTK=OFF"
    "-DWITH_JASPER=OFF"
    "-DWITH_OPENJPEG=OFF"
    "-DWITH_JPEG=OFF"
    "-DWITH_WEBP=OFF"
    "-DWITH_PNG=OFF"
    "-DWITH_TIFF=OFF"
    "-DWITH_V4L=OFF"
    "-DWITH_OPENCL=OFF"
    "-DWITH_FLATBUFFERS=OFF"
    "-DWITH_PROTOBUF=OFF"
    "-DWITH_IMGCODEC_HDR=OFF"
    "-DWITH_IMGCODEC_SUNRASTER=OFF"
    "-DWITH_IMGCODEC_PXM=OFF"
    "-DWITH_IMGCODEC_PFM=OFF"
    "-DWITH_ADE=OFF"
    "-DWITH_LAPACK=OFF"
    "-DOPENCV_PYTHON_SKIP_DETECTION=ON"
)

if [[ -n "${OPENCV_VERSION}" ]]; then
    common_cmake_args+=(
        "-DOPENCV_VERSION=${OPENCV_VERSION}"
    )
fi

if [[ -n "${OPENCV_URL}" ]]; then
    common_cmake_args+=(
        "-DOPENCV_URL=${OPENCV_URL}"
    )
fi

cmake "${common_cmake_args[@]}" \
      -B "${OPENCV_DEFAULT_PATH}" \
      -DWITH_KLEIDICV=OFF
ninja -C "${OPENCV_DEFAULT_PATH}" subordinate

cmake "${common_cmake_args[@]}" \
      -B "${OPENCV_KLEIDICV_PATH}" \
      -DWITH_KLEIDICV=ON \
      -DKLEIDICV_SOURCE_PATH="${KLEIDICV_SOURCE_PATH}" \
      -DKLEIDICV_ENABLE_SME2=ON
ninja -C "${OPENCV_KLEIDICV_PATH}" manager

TESTRESULT=0
qemu-aarch64 -cpu cortex-a35 "${OPENCV_KLEIDICV_PATH}/bin/manager" "${OPENCV_DEFAULT_PATH}/bin/subordinate" || TESTRESULT=1
qemu-aarch64 -cpu max,sve128=on,sme=off "${OPENCV_KLEIDICV_PATH}/bin/manager" "${OPENCV_DEFAULT_PATH}/bin/subordinate" || TESTRESULT=1
qemu-aarch64 -cpu max,sve128=on,sme512=on "${OPENCV_KLEIDICV_PATH}/bin/manager" "${OPENCV_DEFAULT_PATH}/bin/subordinate" || TESTRESULT=1

exit $TESTRESULT
