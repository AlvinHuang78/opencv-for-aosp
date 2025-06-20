#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

set -eu

# ------------------------------------------------------------------------------

cd "$(dirname "${BASH_SOURCE[0]}")/.."

set -x

# Run cpplint.
#
# Disabled categories rationale:
# * build/c++11 - cpplint's list of unapproved headers (like <thread>) doesn't
#   make much sense outside the projects it was originally designed for.
# * build/header_guard - our header guards are not coupled to the directory
#   structure because they're designed to be used from other projects.
# * build/include_subdir - it's OK to include a header in the same directory
#   and not prefix it with the directory name.
# * readability/todo - it's OK to add a comment stating that there's work to
#   be done without knowing who will do it.
# * runtime/references - it's OK to pass non-const references if an argument
#   is intended to be modified.
# * whitespace/indent - handled by clang-format.
# * whitespace/line_length - handled by clang-format.
#
# False positives should not be disabled but instead suppressed by adding a
# NOLINT(category) comment to the affected line.
cpplint \
    --recursive \
    --exclude=build \
    --counting=detailed \
    --filter=-build/c++11,-build/header_guard,-build/include_subdir,-readability/todo,-runtime/references,-whitespace/indent,-whitespace/line_length \
    .
