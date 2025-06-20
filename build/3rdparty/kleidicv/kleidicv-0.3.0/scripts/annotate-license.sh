#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

# This is a helper script for developers to check and automate inclusion of
# correct license headers in their commits. It relies on `reuse` tool available
# at https://reuse.software
#
# The script is meant to be run before committing changes (after `git add`,
# before `git commit`) and can be turned into a pre-commit hook . It would check
# staging files for correct Arm copyright and amend copyright if needed. In case
# of outdated copyright year - extends the copyright.
# For non-recognisable file formats, such as .patch it would enforce c-style
# comment - it is recommended to check amended file to make sure this style is
# compatible with the file type.
# If any changes has been made as a result of running this script the user will
# be asked to add changes to commit.
# Script return values:
#   0 - default, nothing was changed
#   1 - reuse lint complains, but files haven't been modified
#   2 - files have been modified by the script and have to be staged for commit
#   3 - both 1 and 2

set -u

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

COPYRIGHT="Arm Limited and/or its affiliates <open-source-office@arm.com>"
LICENSE="Apache-2.0"

# Use `reuse` to check for copyrights.
# Suppresses `reuse` stderr not to spam console with `reuse -h` prompts for
# unrecognised formats
annotate() {
    local file=$1
    out=$(reuse annotate -c "${COPYRIGHT}" --license "${LICENSE}" --merge-copyrights "${file}" 2> >(cat))
    if [ $? -eq 2 ]; then
        echo -e "${RED}Failed to annotate ${file}, enforcing c-style comment${NC}"
        reuse annotate -c "${COPYRIGHT}" --license "${LICENSE}" --merge-copyrights --style c "${file}"
    else
        echo "${out}"
    fi
}

UNSTAGED=$(git diff --name-only)
STAGED=$(git diff --cached --name-only)
for file in $STAGED; do
    annotate "${file}"
done

# Run license check on the entire codebase
reuse lint
EXIT_CODE="${?}"

# Notify user if there were changes made to staging files
for file in $(git diff --name-only); do
    if ! echo "${UNSTAGED}" | grep -q "\b${file}\b"; then
        echo -e "${GREEN}Please stage ${file} for the commit${NC}"
        EXIT_CODE=$((EXIT_CODE | 2))
    fi
done

exit "${EXIT_CODE}"
