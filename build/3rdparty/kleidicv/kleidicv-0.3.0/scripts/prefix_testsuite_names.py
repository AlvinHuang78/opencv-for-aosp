#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

"""
This is a script to help improve how tests are displayed in CI.
The same tests are run repeatedly with different features enabled.
To differentiate these test runs, this script prefixes the test names with the
given prefix.

Example:
  prefix_testsuite_names.py path/to/test.xml MyPrefix.
"""

import argparse
import xml.etree.ElementTree as ET

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("testsuite")
    parser.add_argument("prefix")
    args = parser.parse_args()
    tree = ET.parse(args.testsuite)
    for ts in tree.getroot().findall("testsuite"):
        for tc in ts.findall("testcase"):
            name = tc.get("classname")
            name = args.prefix + name
            tc.set("classname", name)
    tree.write(args.testsuite)


if __name__ == "__main__":
    main()
