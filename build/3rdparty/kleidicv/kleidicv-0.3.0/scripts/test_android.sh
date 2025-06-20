#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0
#
# Runs a given executable on an Android device.
#
# Arguments:
#   1: Absolute path to build folder.
#   2: Absolute path to the executable to run.
#   3-*: Optional arguments to the executable.
#
# Options:
#   ANDROID_SERIAL:     Passed to adb it selects a device by its serial number. This is optional.
#   COVERAGE:           Enables collection of coverage metrics if set to 'ON'. Defaults to 'OFF'.
#   DEVICE_HOST:        Passed to adb it sets the host where the device is found. This is optional.
#   DEVICE_PORT:        Passed to adb it sets the port where the device is found. This is optional.
#   DEVICE_TMP_PATH:    Temporary path for generated files on the device. Defaults to '/data/local/tmp'.
#   GCOV_PREFIX:        Prefix to paths for gcov. Defaults to '${DEVICE_TMP_PATH}/gcov'.

set -exu

# ------------------------------------------------------------------------------
# Mandatory arguments check
# ------------------------------------------------------------------------------

if [[ -z "${ADB}" ]]; then
    echo "Required variable 'ADB' is not set. Please set it to point to adb command."
    exit 1
fi

# ------------------------------------------------------------------------------
# Automatic configuration
# ------------------------------------------------------------------------------

: "${COVERAGE:=OFF}"
: "${DEVICE_TMP_PATH:=/data/local/tmp}"
# Generated coverage metrics are placed into '${DEVICE_TMP_PATH}/gcov'.
: "${GCOV_PREFIX:=${DEVICE_TMP_PATH}/gcov}"

BUILD_PATH="$(realpath "${1}")"
BINARY_HOST_PATH="$(realpath "${2}")"
BINARY_ARGS="${*:3}"
BINARY_NAME="$(basename "${BINARY_HOST_PATH}")"
BINARY_DEVICE_PATH="${DEVICE_TMP_PATH}"/"${BINARY_NAME}"

# ------------------------------------------------------------------------------

# Returns a command which is suitable for collecting coverage info.
# Arguments:
#  1: Build directory on host.
#  2+: The command to wrap.
with_coverage() {
    local directory_levels="${1//[^\/]}"
    # To match host and device paths set GCOV_PREFIX_STRIP so that it removes
    # suffient number of directory levels from path names.
    echo "GCOV_PREFIX=${GCOV_PREFIX} GCOV_PREFIX_STRIP=${#directory_levels} ${*:2}"
}

# ------------------------------------------------------------------------------

run_adb_command() {
    local args=()
    if [[ -n "${DEVICE_HOST+x}" ]]; then
        args+=( "-H" "${DEVICE_HOST}" )
    fi
    if [[ -n "${DEVICE_PORT+x}" ]]; then
        args+=( "-P" "${DEVICE_PORT}" )
    fi
    if [[ -n "${ANDROID_SERIAL+x}" ]]; then
        args+=( "-s" "${ANDROID_SERIAL}" )
    fi

    "${ADB}" "${args[@]}" "$@"
}

# ------------------------------------------------------------------------------
# Cleanup from host.
# ------------------------------------------------------------------------------

# Remove gcov folder from device.
if [[ "${COVERAGE}" == "ON" ]]; then
    run_adb_command shell rm -rf "${GCOV_PREFIX}"
fi

# ------------------------------------------------------------------------------
# Compose and run the binary on the device.
# ------------------------------------------------------------------------------

# Push the binary to execute to the device.
run_adb_command push "${BINARY_HOST_PATH}" "${BINARY_DEVICE_PATH}"

# Compose command to run.
command="${BINARY_DEVICE_PATH} ${BINARY_ARGS}"

# Execute the binary with appropriate GCOV settings.
if [[ "${COVERAGE}" == "ON" ]]; then
    command="$(with_coverage "${BUILD_PATH}" "${command}")"
fi

run_adb_command shell "${command}"

if [[ "${COVERAGE}" == "ON" ]]; then
    # Pull generated .gcda files.
    run_adb_command pull "${GCOV_PREFIX}" "${BUILD_PATH}"

    # Move .gcda files next to .gcdo files.
    rsync -a "${BUILD_PATH}"/gcov/ "${BUILD_PATH}"
fi

# ------------------------------------------------------------------------------
# Cleanup from device.
# ------------------------------------------------------------------------------

run_adb_command shell rm -rf "${BINARY_DEVICE_PATH}"

if [[ "${COVERAGE}" == "ON" ]]; then
    run_adb_command shell rm -rf "${GCOV_PREFIX}"
fi

# ------------------------------------------------------------------------------
# Cleanup from host.
# ------------------------------------------------------------------------------

if [[ "${COVERAGE}" == "ON" ]]; then
    rm -rf "${BUILD_PATH}"/gcov
fi

# ------------------------------------------------------------------------------
# End of script
# ------------------------------------------------------------------------------
