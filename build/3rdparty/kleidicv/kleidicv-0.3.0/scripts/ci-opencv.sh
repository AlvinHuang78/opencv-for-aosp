#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

set -exu

# Ensure we're at the root of the repo.
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export OPENCV_VERSION="4.10.0"

# ------------------------------------------------------------------------------
# Try to build unpatched OpenCV with KleidiCV
# ------------------------------------------------------------------------------
mkdir -p build/unpatched-opencv-src
tar -xzf /opt/opencv-${OPENCV_VERSION}.tar.gz -C build/unpatched-opencv-src
BUILD_ID=unpatched-opencv \
OPENCV_PATH="$(pwd)/build/unpatched-opencv-src/opencv-${OPENCV_VERSION}" \
CMAKE_EXE_LINKER_FLAGS="--rtlib=compiler-rt -fuse-ld=lld" \
EXTRA_CMAKE_ARGS="\
  -DBUILD_SHARED_LIBS=OFF \
  -DWITH_KLEIDICV=ON \
  -DKLEIDICV_SOURCE_PATH=$(pwd) \
  -DKLEIDICV_ENABLE_SME2=ON \
  -DBUILD_ANDROID_EXAMPLES=OFF \
  -DBUILD_ANDROID_PROJECTS=OFF \
  -DOPENCV_DISABLE_THREAD_SUPPORT=ON \
  -DCV_TRACE=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_opencv_apps=OFF \
  -DBUILD_JAVA=OFF \
  -DWITH_QT=OFF \
  -DBUILD_OPENCV_PYTHON=NO \
  -DBUILD_OPENCV_PYTHON2=NO \
  -DBUILD_OPENCV_PYTHON3=NO \
  -DWITH_VTK=OFF \
  -DWITH_JASPER=OFF \
  -DWITH_OPENJPEG=OFF \
  -DWITH_JPEG=OFF \
  -DWITH_WEBP=OFF \
  -DWITH_PNG=OFF \
  -DWITH_TIFF=OFF \
  -DWITH_V4L=OFF \
  -DWITH_OPENCL=OFF \
  -DWITH_FLATBUFFERS=OFF \
  -DWITH_PROTOBUF=OFF \
  -DWITH_IMGCODEC_HDR=OFF \
  -DWITH_IMGCODEC_SUNRASTER=OFF \
  -DWITH_IMGCODEC_PXM=OFF \
  -DWITH_IMGCODEC_PFM=OFF \
  -DWITH_ADE=OFF \
  -DWITH_LAPACK=OFF \
  -DOPENCV_PYTHON_SKIP_DETECTION=ON" \
./scripts/build-opencv.sh

# ------------------------------------------------------------------------------
# Run OpenCV conformity checks
# ------------------------------------------------------------------------------
TESTRESULT=0
CLEAN="ON" \
  OPENCV_URL="/opt/opencv-${OPENCV_VERSION}.tar.gz" \
  LDFLAGS="--rtlib=compiler-rt -fuse-ld=lld" \
  ./scripts/run_opencv_conformity_checks.sh || TESTRESULT=1

# ------------------------------------------------------------------------------
# Run a subset of OpenCV's test suite
# ------------------------------------------------------------------------------
# Build OpenCV test executables from already configured conformity check project
# The OpenCV source is patched in this case
ninja -C build/conformity/opencv_kleidicv \
  opencv_test_imgproc \
  opencv_test_core \
  opencv_test_video

# Some tests require opencv_extra for the test images
tar xf /opt/opencv-extra-${OPENCV_VERSION}.tar.gz -C build
mv build/opencv_extra-${OPENCV_VERSION} build/opencv_extra

pushd build/opencv_extra/testdata/cv

join_strings_with_colon() {
  local array="${1}"
  # shellcheck disable=SC2068
  # Here array should be re-splitted.
  array="$(printf ":%s" ${array[@]})"
  echo "${array:1}"
}

IMGPROC_TEST_PATTERNS=(
  '*Imgproc_ColorGray*'
  '*Imgproc_ColorRGB*'
  '*Imgproc_ColorYUV*'
  '*Imgproc_cvtColor_BE.COLOR_YUV*'
  '*Imgproc_cvtColor_BE.COLOR_RGB2YUV'
  '*Imgproc_Threshold*'
  '*Imgproc_Morphology*'
  '*Imgproc_GaussianBlur*'
  '*Imgproc_Sobel*'
  '*Imgproc_Resize*'
  '*Imgproc_Dilate*'
  '*Imgproc_Erode*'
  '*Imgproc_PyramidDown*'
  '*Imgproc_Remap*'
  '*Imgproc_Warp*'
)
IMGPROC_TEST_PATTERNS_STR="$(join_strings_with_colon "${IMGPROC_TEST_PATTERNS[*]}")"
../../../conformity/opencv_kleidicv/bin/opencv_test_imgproc \
  --gtest_filter="${IMGPROC_TEST_PATTERNS_STR}" || TESTRESULT=1

CORE_TEST_PATTERNS=(
  '*Core_AbsDiff*'
  '*Core_Add*'
  '*Core_And*'
  '*Core_Mul*'
  '*Core_Sub*'
  '*Core_Transpose*'
  '*Core_MinMaxLoc*'
  '*MinMaxLoc*'
  '*Core_ConvertScale*'
  '*Core_Exp*'
  '*Core_MinMaxIdx*'
  '*Core_minMaxIdx*'
  '*Core_Array*'
  'Compare*'
  '*Core_InRange/*'
)
CORE_TEST_PATTERNS_STR="$(join_strings_with_colon "${CORE_TEST_PATTERNS[*]}")"
../../../conformity/opencv_kleidicv/bin/opencv_test_core \
  --gtest_filter="${CORE_TEST_PATTERNS_STR}" || TESTRESULT=1

VIDEO_TEST_PATTERNS=(
  'Video_OpticalFlowPyrLK.accuracy'
)
VIDEO_TEST_PATTERNS_STR="$(join_strings_with_colon "${VIDEO_TEST_PATTERNS[*]}")"
../../../conformity/opencv_kleidicv/bin/opencv_test_video \
  --gtest_filter="${VIDEO_TEST_PATTERNS_STR}" || TESTRESULT=1

popd

exit $TESTRESULT
