<!--
SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>

SPDX-License-Identifier: Apache-2.0
-->

# Platform Support

Support for different platforms is organised in 3 tiers[^1].

## Tier 1

A released version of KleidiCV shall be tested for correctness and performance on a Tier 1 platform.

At present the only Tier 1 platform is Android on Samsung Galaxy S22
with an up-to-date operating system, where applications are built with
Android NDK r27c.

## Tier 2

KleidiCV is tested for correctness on a Tier 2 platform, however performance is not checked.

At present the only Tier 2 platform is AArch64 Ubuntu, as defined by `docker/Dockerfile` in the KleidiCV source tree.

## Tier 3

KleidiCV aims to support Tier 3 platforms but since no routine
testing is done on these platforms it may or may not work in practise.

At present the only Tier 3 platform is Mac computers with Apple silicon.


[^1] Inspired by [Rust's platform support](https://doc.rust-lang.org/nightly/rustc/platform-support.html).
