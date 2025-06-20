<!--
SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>

SPDX-License-Identifier: Apache-2.0
-->

# Building KleidiCV for Android

## Prerequisites
While the core functionality of the library does not rely on any third-party
libraries, there are build prerequisites that are essential for compiling the
source code and generating the executable. Please ensure that these tools are
installed on your system before proceeding with the build process.

To successfully build and compile this project for Android, you'll need the following tools:
- [Android NDK](https://developer.android.com/ndk/).
  See [the platform support page](platform-support.md) for supported versions.
- [CMake](https://cmake.org) 3.16 or higher.
- `make`
- `patch`

Running tests on Android devices requires [ADB](https://developer.android.com/tools/adb).
ADB is included in [Android SDK Platform-Tools](https://developer.android.com/tools/releases/platform-tools).

## Building OpenCV & KleidiCV for Android

For details of which OpenCV function are accelerated by KleidiCV see
[KleidiCV's OpenCV documentation](opencv.md).

### Get and patch OpenCV source

This version of KleidiCV is compatible with [OpenCV](https://opencv.org) version 4.10 and later.
Earlier versions of KleidiCV are compatible with earlier versions of OpenCV.
OpenCV 5.x support is experimental.

Integration consists of the following steps:
1. Download OpenCV sources:
```
wget https://github.com/opencv/opencv/archive/refs/tags/4.10.0.tar.gz
tar xf 4.10.0.tar.gz
cd opencv-4.10.0
```
2. Patch OpenCV:
```
patch -p1</path/to/kleidicv/adapters/opencv/opencv-4.10.patch
```

### Build

OpenCV can be built with KleidiCV enabled via the following CMake variables:
- `WITH_KLEIDICV` - set this to `ON` to enable KleidiCV.
- `KLEIDICV_SOURCE_PATH` - the top-level `kleidicv` directory.

```
cmake \
  -S /path/to/opencv \
  -B build-opencv-android \
  -DANDROID_ABI=arm64-v8a \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/android-ndk/build/cmake/android.toolchain.cmake \
  -DBUILD_ANDROID_PROJECTS=OFF \
  -DWITH_KLEIDICV=ON \
  -DKLEIDICV_SOURCE_PATH=/path/to/kleidicv
cmake --build build-opencv-android --parallel
```

(`BUILD_ANDROID_PROJECTS=OFF` is specified just to simplify these build instructions.
See the OpenCV project's documentation to learn the requirements for enabling the option.)

## Build KleidiCV standalone for Android

From the top-level `kleidicv` directory run:
```
cmake -S . -B build-kleidicv-android \
-DANDROID_ABI=arm64-v8a \
-DCMAKE_TOOLCHAIN_FILE=/path/to/android-ndk/build/cmake/android.toolchain.cmake
cmake --build build-kleidicv-android --parallel
```

# Building KleidiCV for AArch64 Linux

If your build machine is AArch64 Linux then you can build KleidiCV with the system toolchain.

Cross-building from another architecture to AArch64 in the standard way
defined by your toolchain is also possible. See your toolchain
documentation for cross-building instructions.

## Prerequisites
To successfully build and compile this project for AArch64 Linux, you'll need the following tools:
- Either GCC 9.4 or higher, or Clang 10 or higher.
- Binutils
- [CMake](https://cmake.org) 3.16 or higher.
- `make`
- `patch`

## Building OpenCV & KleidiCV for AArch64 Linux

This is similar to building for Android, just with fewer settings required.
First [get and patch the OpenCV source](#get-and-patch-opencv-source).
Then build:
```
cmake \
  -S /path/to/opencv \
  -B build-opencv-linux \
  -DWITH_KLEIDICV=ON \
  -DKLEIDICV_SOURCE_PATH=/path/to/kleidicv
cmake --build build-opencv-linux --parallel
```

## Build KleidiCV standalone for AArch64 Linux

This is similar to building for Android, just with fewer settings required.
```
cmake -S /path/to/kleidicv -B build-kleidicv-linux
cmake --build build-kleidicv-linux --parallel
```

# KleidiCV Build Options

In addition to the standard CMake settings, KleidiCV behaviour can be
modified at build time via the following CMake options:
- `KLEIDICV_BENCHMARK` - Enable building KleidiCV benchmarks. The benchmarks use Google Benchmark which will be downloaded automatically. Off by default.
- `KLEIDICV_ENABLE_SME2` - Enable Scalable Matrix Extension 2 and Streaming Scalable Vector Extension code paths. Off by default while the [ACLE SME specification is in beta](https://github.com/ARM-software/acle/blob/main/main/acle.md#sme-language-extensions-and-intrinsics).
  - `KLEIDICV_LIMIT_SME2_TO_SELECTED_ALGORITHMS` - Limit Scalable Matrix Extension 2 code paths to cases where it is expected to provide a benefit over other code paths. On by default. Has no effect if `KLEIDICV_ENABLE_SME2` is off.
- `KLEIDICV_ENABLE_SVE2` - Enable Scalable Vector Extension 2 code paths. This is on by default for some popular compilers known to support SVE2 but otherwise off by default.
  - `KLEIDICV_LIMIT_SVE2_TO_SELECTED_ALGORITHMS` - Limit Scalable Vector Extension 2 code paths to cases where it is expected to provide a benefit over other code paths. On by default. Has no effect if `KLEIDICV_ENABLE_SVE2` is off.
