# SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)

project("KleidiCV" CXX)

add_subdirectory(kleidicv)
add_subdirectory(kleidicv_thread)
add_subdirectory(test)
add_subdirectory(benchmark)
