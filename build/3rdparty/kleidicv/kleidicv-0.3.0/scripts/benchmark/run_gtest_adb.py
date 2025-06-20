#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
Runs OpenCV's gtest-based benchmarks via Android ADB

The script checks the CPU temperature before running each benchmark and
if it's found to be too high then waits to allow it to cool down, to
avoid the CPU being throttled.

This script provides similar functionality to perf_test_op.sh.
It should be run from the adb host whereas perf_test_op.sh is designed
to run directly on the device.
The benefit of this script is that it can be run with --gtest_filter
matching a large number of tests whereas perf_test_op.sh must be called
for each individual test.

Example invocation that runs all 640*480 multiply tests and subtract
tests, on executables with and without KleidiCV enabled, on mid and big
cores, and writes the mean & stddev to a tsv file that can be opened in a
text editor or spreadsheet editor for further analysis:

./run_gtest_adb.py \
    --taskset_masks 0x10 0x80 \
    --thermal_zones 1 0 \
    --serial 0123456789ABCDEF \
    --tsv multiply_subtract.tsv \
    --tsv_columns mean stddev \
    --executables opencv_perf_core_vanilla opencv_perf_core_kleidicv \
    --gtest_filter="*multiply*:*subtract*" \
    --gtest_param_filter="*640x480*"
"""

import argparse
import collections
import csv
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time


def int_hex(x):
    return int(x, 16)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--executables",
        nargs="+",
        help="Path to the gtest executables on the host. "
        "These will be copied to the device.",
    )
    parser.add_argument("--adb", default="adb", help="Path to adb")
    parser.add_argument(
        "--taskset_masks",
        type=int_hex,
        nargs="+",
        help="taskset masks to run with. "
        "Typically 0x80 for big, 0x10 for mid, 0x1 for little",
    )
    parser.add_argument(
        "--thermal_zones",
        type=int,
        nargs="+",
        help="Thermal zones of CPUs to run on, corresponding to --taskset_masks argument. "
        "Typically 0 for big CPU, 1 for mid CPU, 2 for little CPU",
    )
    parser.add_argument(
        "--tsv",
        default="gtest_adb.tsv",
        help="tab-separated-value output file",
    )
    parser.add_argument(
        "--json",
        default="gtest_adb.json",
        help="JSON output file",
    )
    parser.add_argument(
        "--serial",
        "-s",
        help="Serial of the device to connect to. "
        "Optional if only one device is connected",
    )
    parser.add_argument(
        "--tmpdir",
        default="/data/local/tmp",
        help="Temporary directory on the device. "
        "Executables and output files will be stored here, and it will be used"
        " as the current working directory when running the executables.",
    )
    parser.add_argument(
        "--tsv_columns",
        default=["median"],
        nargs="*",
        help="Which measurements to write to the TSV file "
        "e.g. median mean min gmean gstddev. "
        "All measurements are always available in JSON file",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Repeat the entire set of tests this many times. "
        "Useful for getting a sense of how much noise there is in the results",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--gtest_filter")
    parser.add_argument("--gtest_param_filter")
    parser.add_argument("--perf_min_samples", type=int, default=100)
    args = parser.parse_args()

    assert len(args.taskset_masks) == len(args.thermal_zones)

    return args


class ADBRunner:
    def __init__(self, *, adb_command, serial_number, verbose):
        self.adb_command = adb_command
        self.serial_number = serial_number
        self.verbose = verbose

    def _make_adb_command(self):
        result = [self.adb_command]
        if self.serial_number:
            result.extend(["-s", self.serial_number])
        return result

    def _print_command(self, command):
        if self.verbose:
            print("+ " + shlex.join(command))

    def check_output(self, script):
        command = self._make_adb_command() + ["shell", "su"]
        self._print_command(command)
        if self.verbose:
            print("+ " + script)
        try:
            return subprocess.check_output(
                command,
                stderr=subprocess.STDOUT,
                input=script.encode(),
            ).decode()
        except subprocess.CalledProcessError as e:
            out = e.stdout.decode()
            print(out, file=sys.stderr)
            raise

    def push(self, filenames, dst):
        command = self._make_adb_command() + ["push", *filenames, dst]
        self._print_command(command)
        subprocess.check_output(command).decode()

    def pull(self, filenames, dst="."):
        command = self._make_adb_command() + ["pull", *filenames, dst]
        self._print_command(command)
        subprocess.check_output(command).decode()

    def read_json(self, json_filename):
        with tempfile.TemporaryDirectory() as tmpdirname:
            dst = os.path.join(tmpdirname, "t.json")
            self.pull([json_filename], dst=dst)
            with open(dst) as f:
                return json.load(f)


def wait_for_cooldown(runner, thermal_zone):
    temp_filename = (
        f"/sys/devices/virtual/thermal/thermal_zone{thermal_zone}/temp"
    )
    while True:
        temp = int(runner.check_output(f"cat {shlex.quote(temp_filename)}"))
        if temp < 40000:
            return
        print(f"Temperature {temp} - waiting to cool down")
        time.sleep(3)


def get_run_name(rep, executable, taskset_mask):
    return f"{executable}-{taskset_mask:x}-#{rep}"


def host_filename_to_device_filename(args, host_filename):
    return os.path.join(args.tmpdir, host_filename.lstrip(os.sep).replace(os.sep, "-"))


def run_executable_tests(
    runner, args, host_executable, taskset_mask, thermal_zone
):
    executable = host_filename_to_device_filename(args, host_executable)

    output_file = f"{os.path.splitext(executable)[0]}-{taskset_mask:x}.json"

    command_args = [
        executable,
        "--gtest_list_tests",
        f"--gtest_output=json:{output_file}",
    ]
    if args.gtest_filter:
        command_args.append(f"--gtest_filter={args.gtest_filter}")
    if args.gtest_param_filter:
        command_args.append(f"--gtest_param_filter={args.gtest_param_filter}")

    runner.check_output(shlex.join(command_args))
    testsuite_dict = runner.read_json(output_file)

    test_list = []

    for testsuite in testsuite_dict["testsuites"]:
        testsuite_name = testsuite["name"]
        for test in testsuite["testsuite"]:
            test_name = test["name"]
            test_list.append(f"{testsuite_name}.{test_name}")

    if not test_list:
        print("Error: No tests found")
        sys.exit(1)

    results = {}
    testsuites = collections.OrderedDict()

    for test_name in test_list:
        wait_for_cooldown(runner, thermal_zone)
        try:
            runner.check_output(
                f"cd {shlex.quote(args.tmpdir)} && "
                + shlex.join(
                    [
                        "taskset",
                        f"{taskset_mask:x}",
                        executable,
                        f"--gtest_output=json:{output_file}",
                        f"--gtest_filter={test_name}",
                        f"--perf_min_samples={args.perf_min_samples}",
                    ]
                )
            )
        except subprocess.CalledProcessError:
            continue
        test_result = runner.read_json(output_file)

        if not results:
            results = test_result

        testsuite = test_result["testsuites"][0]
        testsuite_name = testsuite["name"]
        if testsuite_name not in testsuites:
            testsuites[testsuite_name] = testsuite
        else:
            testsuites[testsuite_name]["testsuite"].extend(
                testsuite["testsuite"]
            )

        test_result = testsuite["testsuite"][0]

        try:
            output = (
                f"{executable}-{taskset_mask:x}\t{test_name}"
                f"\t{test_result.get('value_param', '')}"
            )
            for key in args.tsv_columns:
                output += f"\t{test_result[key]}"
            print(output)
        except KeyError:
            pass

    results["testsuites"] = list(testsuites.values())

    return results


def run_tests_on_cpus(runner, args, rep, taskset_mask, thermal_zone):
    # Iterate through the CPUs enabled in the taskset mask
    for cpu in range(taskset_mask.bit_length()):
        if (1 << cpu) & taskset_mask == 0:
            continue

        scaling_governor_filename_quoted = shlex.quote(
            f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
        )
        prev_scaling_governor = runner.check_output(
            f"cat {scaling_governor_filename_quoted}"
        ).strip()
        try:
            runner.check_output(
                f"echo performance ~> {scaling_governor_filename_quoted}"
            )
            return {
                get_run_name(
                    rep, executable, taskset_mask
                ): run_executable_tests(
                    runner, args, executable, taskset_mask, thermal_zone
                )
                for executable in args.executables
            }
        finally:
            runner.check_output(
                f"echo {shlex.quote(prev_scaling_governor)} ~> {scaling_governor_filename_quoted}"
            )


def get_results_table(args, results):
    # If executables have a common prefix then strip it, or omit it
    # altogether if only one executable.
    exe_common_prefix_len = len(os.path.commonprefix(args.executables))

    field_names = ["name", "value_param"]
    for taskset_mask in args.taskset_masks:
        for key in args.tsv_columns:
            for executable in args.executables:
                for rep in range(args.repetitions):
                    field_names.append(
                        f"{taskset_mask:x} {executable[exe_common_prefix_len:]} {key} #{rep}"
                    )

    rows = [field_names]

    first_result = next(iter(results.values()))
    for testsuite_index, testsuite in enumerate(first_result["testsuites"]):
        testsuite_name = testsuite["name"]
        for test_index, test in enumerate(testsuite["testsuite"]):
            test_name = test["name"]
            value_param = test.get("value_param", "")
            row = [f"{testsuite_name}.{test_name}", value_param]

            try:
                for taskset_mask in args.taskset_masks:
                    for key in args.tsv_columns:
                        for executable in args.executables:
                            for rep in range(args.repetitions):
                                result = results[
                                    get_run_name(rep, executable, taskset_mask)
                                ]
                                exe_test = result["testsuites"][
                                    testsuite_index
                                ]["testsuite"][test_index]
                                row.append(exe_test[key])
                rows.append(row)
            except KeyError:
                pass

    return rows


def main():
    args = parse_args()

    runner = ADBRunner(
        adb_command=args.adb,
        serial_number=args.serial,
        verbose=args.verbose,
    )

    # Copy executables to device
    for host_filename in args.executables:
        runner.push(
            [host_filename],
            host_filename_to_device_filename(args, host_filename),
        )

    results = {}

    for rep in range(args.repetitions):
        for taskset_mask, thermal_zone in zip(
            args.taskset_masks, args.thermal_zones
        ):
            results.update(
                run_tests_on_cpus(
                    runner, args, rep, taskset_mask, thermal_zone
                )
            )

    with open(args.json, "w") as f:
        json.dump(results, f, indent="  ")

    with open(args.tsv, "w", newline="") as tsvfile:
        csv.writer(tsvfile, dialect="excel-tab").writerows(
            get_results_table(args, results)
        )


if __name__ == "__main__":
    main()
