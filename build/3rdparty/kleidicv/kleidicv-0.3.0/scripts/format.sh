#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

# Runs clang-format or checks whether the source is well-formatted.
#
# Options:
#   CHECK_ONLY:             If set to 'ON', the script exists with non-zero value if source is not formatted. Defaults to 'OFF'.
#   CLANG_FORMAT_BIN_PATH:  Clang-format binary, defaults to 'clang-format=19'.
#   VERBOSE:                If set to 'ON', verbose output is printed. Defaults to 'OFF'.
# ------------------------------------------------------------------------------

set -eu

# ------------------------------------------------------------------------------

SCRIPT_PATH="$(realpath "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")"

# ------------------------------------------------------------------------------

KLEIDICV_ROOT_PATH="$(realpath "${SCRIPT_PATH}"/..)"

: "${CHECK_ONLY:=OFF}"
: "${CLANG_FORMAT_BIN_PATH:=clang-format-19}"
: "${VERBOSE:=OFF}"

# ------------------------------------------------------------------------------

SOURCES="$(find \
    "${KLEIDICV_ROOT_PATH}/adapters" \
    "${KLEIDICV_ROOT_PATH}/benchmark" \
    "${KLEIDICV_ROOT_PATH}/kleidicv" \
    "${KLEIDICV_ROOT_PATH}/kleidicv_thread" \
    "${KLEIDICV_ROOT_PATH}/test" \
    "${KLEIDICV_ROOT_PATH}/conformity/opencv" \
    \( -name \*.cpp -o -name \*.h -o -name \*.h.in \) \
    -print)"

if [[ "${CHECK_ONLY}" == "ON" ]]; then
  FORMAT_FLAGS="--dry-run -Werror"
else
  FORMAT_FLAGS="-i"
fi

if [[ "${VERBOSE}" == "ON" ]]; then
  FORMAT_FLAGS="${FORMAT_FLAGS} --verbose"
fi

# shellcheck disable=2086
# Split ${SOURCES} and ${FORMAT_FLAGS}.
"${CLANG_FORMAT_BIN_PATH}" ${FORMAT_FLAGS} ${SOURCES}

# ------------------------------------------------------------------------------
# End of script
# ------------------------------------------------------------------------------
