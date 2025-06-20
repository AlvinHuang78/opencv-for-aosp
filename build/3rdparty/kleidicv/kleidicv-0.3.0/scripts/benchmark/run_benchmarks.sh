#!/bin/sh

# SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

set -eu

# ------------------------------------------------------------------------------
# Runs benchmarks described in benchmark specification file(s).
#
# Benchmark specification should follow this pattern:
#   <NAME>: <BINARY NAME> '<GTEST_FILTER>' '<GTEST_PARAM_FILTER>'
#
# The number of spaces is meaningless in the pattern above. Empty lines and
# lines starting with '#' are ignored.
#
# The special word $PIXEL_FORMAT is replaced automatically with an appropriate MxN value
# depending on what RESOLUTION is set to. RESOLUTION defaults to 4K UHD.
#
# By default both the common 'benchmarks.txt' and resolution-specific 'benchmarks<_RES>.txt'
# are run, if exist.
#
# Environment:
#   DEV_DIR: Directory on the devicewhere all relevant files are stored.
#       Defaults to '/data/local/tmp'.
#   CPU: Identifier of the core to run benchmarks on.
#       Defaults to '4', which is most often a mid core.
#   CUSTOM_BUILD_SUFFIX: Build name suffix for the extra build.
#       Defaults to 'custom'.
#   RESOLUTION: Sets the resolution to bechmark. See select_resolution() below for
#       supported formats. Defaults to 4K.
#   THERMAL_ZONE_ID: Identifier of the thermal zone to watch for changes.
#       Defaults to '1', which is most often the zone associated with mid cores.
# ------------------------------------------------------------------------------

: "${DEV_DIR:=/data/local/tmp}"
: "${CPU:=4}"
: "${CUSTOM_BUILD_SUFFIX:=custom}"
: "${THERMAL_ZONE_ID:=1}"

# ------------------------------------------------------------------------------

# Pixel format in MxN.
PIXEL_FORMAT=
# The number of benchmarks.
NUM_BENCHMARKS=0
# The current benchmark which is being run.
CURRENT_BENCHMARK=0

# ------------------------------------------------------------------------------

# Prints a warning message.
# Arguments:
#   1: Message to print.
warn() {
    >&2 echo "WARNING: $1"
}

# Selects which resolution to use.
# Arguments: none
select_resolution() {
    if [ -z "${RESOLUTION:-}" ]; then
        warn "Resolution is not specified, falling back to 4K UHD."
        RESOLUTION="4K"
    fi

    case "${RESOLUTION:-}" in
        FHD | 1080p)
            PIXEL_FORMAT="1920x1080"
            ;;
        UHD | 4K | 2160p)
            PIXEL_FORMAT="3840x2160"
            ;;
        *)
            warn "Resolution is not recognized, falling back to 4K UHD."
            RESOLUTION="4K"
            PIXEL_FORMAT="3840x2160"
            ;;
    esac
}

# Calls a user-supplied function for every benchmark.
# Arguments:
#   1: Path to a benchmark specification.
#   2: Function to call for a benchmark.
foreach_benchmark() {
    [ ! -f "${1}" ] && return

    while IFS= read -r line; do
        # Skip empty lines
        [ -z "${line}" ] && continue
        # Skip lines starting with '#'
        [ -z "${line%\#*}" ] && continue

        "${2}" "${line}"
    done < "${1}"
}

# Counts the number of benchmarks.
# All arguments are ignored.
count_benchmark() {
    NUM_BENCHMARKS=$((NUM_BENCHMARKS+1))
}

# Runs a given benchmark.
# Arguments:
#   1: Benchmark specification to run.
run_benchmark() {
    # shellcheck disable=SC3043
    # Many shells support 'local', therefore this warning is ignored.
    local spec="$1"

    # Replace multiple spaces with one.
    spec="$(echo "${spec}" | tr -s ' ')"

    # Resolve $PIXEL_FORMAT
    # shellcheck disable=SC3060
    # Targeted shell support string replacement.
    spec="${spec//\$PIXEL_FORMAT/${PIXEL_FORMAT}}"

    # Be informative to user.
    CURRENT_BENCHMARK=$((CURRENT_BENCHMARK+1))
    >&2 echo RUNNING ["${CURRENT_BENCHMARK}"/"${NUM_BENCHMARKS}"]: "${spec}"

    # shellcheck disable=SC3043
    # Many shells support 'local', therefore this warning is ignored.
    local -r disp_name="${spec%%:*}"

    # shellcheck disable=SC2086
    # Word splitting at the end is required here to produce PERF_TEST_BINARY_BASENAME, GTEST_FILTER and GTEST_PARAM_FILTER.
    eval "${DEV_DIR}"/perf_test_op.sh "${CUSTOM_BUILD_SUFFIX}" "${CPU}" "${THERMAL_ZONE_ID}" "${disp_name}" ${spec#*:}
    >&2 echo
}

# ------------------------------------------------------------------------------

select_resolution

foreach_benchmark "${DEV_DIR}"/benchmarks.txt count_benchmark
foreach_benchmark "${DEV_DIR}"/benchmarks_"${RESOLUTION}".txt count_benchmark

# Print tsv header overwriting any existing files.
printf "Operation\tOpenCV\tstd\tKleidiCV\tstd\tKleidiCV_%s\tstd\n" "${CUSTOM_BUILD_SUFFIX}"

foreach_benchmark "${DEV_DIR}"/benchmarks.txt run_benchmark
foreach_benchmark "${DEV_DIR}"/benchmarks_"${RESOLUTION}".txt run_benchmark

# ------------------------------------------------------------------------------
