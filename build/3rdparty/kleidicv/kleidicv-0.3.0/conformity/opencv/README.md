<!--
SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>

SPDX-License-Identifier: Apache-2.0
-->

# Conformity checks for OpenCV

This CMake project makes it possible to automatically compare KleidiCV
results with vanilla OpenCV for a given operation.

To achieve this the project needs to be built twice (vanilla version and
KleidiCV one) as the availabilty of KleidiCV for a given operation is a
compile time decision. Then, the built executables (`manager` and `subordinate`,
provided by different builds) perform the same operations, and the results are
compared. The communication between the executables is implemented with POSIX
IPC.

The tests can be run from the project's root like:
```
scripts/run_opencv_conformity_checks.sh
```

The script expects an environment where KleidiCV can be built natively with
`cmake` and `ninja`, and `qemu-aarch64` is available.
