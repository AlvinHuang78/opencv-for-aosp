#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

# Collects coverage data and generates Cobertura & HTML coverage reports.
#
# Arguments
#   1: Path to build directory. Defaults to default build directory.

import argparse
import os
import shlex
import subprocess
import sys


def percentage_from_line(line):
    # Given a line like this:
    # lines: 58.4% (2676 out of 4580)
    # extract the number 58.4
    return float(line.split()[1][:-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("build_directory", nargs="?")
    args = parser.parse_args()

    llvm_cov = os.environ.get("LLVM_COV")
    if not llvm_cov:
        print(
            """\
Required environment variable 'LLVM_COV' is not set.
Please set it to point to an llvm-cov instance which is compatible with
the toolchain with which the binaries were built."""
        )
        sys.exit(1)

    source_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    build_dir = os.path.abspath(
        args.build_directory or os.path.join(source_dir, "build")
    )

    cobertura = os.path.join(build_dir, "cobertura-coverage.xml")

    coverage_dir = os.path.join(build_dir, "html", "coverage")
    os.makedirs(coverage_dir, exist_ok=True)
    html = os.path.join(coverage_dir, "coverage_report.html")

    gcovr_command = [
        "gcovr",
        "-j",
        f"--gcov-executable={llvm_cov} gcov",
        f"--cobertura={cobertura}",
        f"--html-details={html}",
        "--html-title=KleidiCV Coverage Report",
        "--html-syntax-highlighting",
        "--html-tab-size=2",
        "--exclude-noncode-lines",
        "--exclude-unreachable-branches",
        f"--exclude={build_dir}",
        "--exclude=test",
        "--print-summary",
        build_dir,
    ]

    print("+", shlex.join(gcovr_command))

    gcovr_output = subprocess.check_output(
        gcovr_command, cwd=source_dir, text=True
    )

    print(gcovr_output, end="")

    lines = None
    branches = None
    for line in gcovr_output.splitlines():
        if line.startswith("lines:"):
            lines = percentage_from_line(line)
        if line.startswith("branches:"):
            branches = percentage_from_line(line)
    combined_coverage = lines * branches / 100
    print(f"line and branch coverage: {combined_coverage:.1f}%")


if __name__ == "__main__":
    main()
