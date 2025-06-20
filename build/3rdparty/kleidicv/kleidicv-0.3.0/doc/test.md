<!--
SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>

SPDX-License-Identifier: Apache-2.0
-->

# Testing KleidiCV

To verify whether all KleidiCV APIs behave as expected without running
OpenCV or any sample applications, you can run KleidiCV's API tests.

These tests are based on the [GoogleTest](https://github.com/google/googletest)
test framework.

Creating a standalone build as described in the [build documentation](build.md)
will create a command line executable in the build folder at
`test/api/kleidicv-api-test`.

Running the executable with no arguments will run all the tests. The
executable also accepts the standard GoogleTest command line options.
Run the executable with the argument `--help` to see the full list of
accepted options.
