<!--
SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>

SPDX-License-Identifier: Apache-2.0
-->

# Experimental benchmarking scripts to compare KleidiCV to vanilla OpenCV performance on Android

Use at your own risk, the stability of this solution is the best on release tags.

First, you need a Linux x86 machine to build this.
Next, to build for Android, you'll need [Android NDK](https://developer.android.com/ndk/).

Also, the OpenCV 4.10.0 source needs to be downloaded and patched. Assuming CWD is the root of OpenCV's
source directory please run:
```
patch -p1 < path/to/kleidicv/adapters/opencv/opencv-4.10.patch
patch -p1 < path/to/kleidicv/adapters/opencv/extra_benchmarks/opencv-4.10.patch
```

Let's assume you are building on a machine that has the phone attached to via USB.
Let's assume your CWD is this directory.

```
OPENCV_PATH=<path to OpenCV> \
CMAKE_TOOLCHAIN_FILE=<path to the NDK>/build/cmake/android.toolchain.cmake \
./build.sh
```

By default, the script will produce two flavours of builds. One without KleidiCV (opencv-vanilla) and
one with it (opencv-kleidicv). Both builds can be fine tuned by the `VANILLA_EXTRA_CMAKE_OPTIONS` and the
`KLEIDICV_EXTRA_CMAKE_OPTIONS` environment variables respectively.
If the `CUSTOM_CMAKE_OPTIONS` environment variable is set a third flavour is also build, next to vanilla
OpenCV and OpenCV+KleidiCV. The variable specifies the extra CMake variables for this
custom build and the `CUSTOM_BUILD_SUFFIX` environment variable can alter the `custom` build name suffix.
(If the `CUSTOM_BUILD_SUFFIX` was defined for the build, it should be provided to the further scripts as well,
like `push` and `run_*`.)

Then push the test binaries to the phone (replace 9A9A9A9A with your actual device ID, or skip it if you have only one phone attached):
```
ADB=adb ANDROID_SERIAL=9A9A9A9A ./push.sh
```
Now you can run the benchmark set for a given resolution:
```
adb -s 9A9A9A9A shell 'RESOLUTION=FHD /data/local/tmp/run_benchmarks.sh' > your_phone_benchmarks_FHD.tsv
```
To run on another core, set CPU and THERMAL_ZONE_ID to the desired values. THERMAL_ZONE_ID is important
to set correctly because benchmarks scripts check for changes of the temperature associated with selected CPU.
To find the correct thermal zone id, it's best to cat /sys/devices/virtual/thermal/thermal_zone${id}/type to find
the appropriate thermal zone identifier. E.g. it can be BIG for 0, MID for 1 and LITTLE for 2,
but it can be different for another device.
