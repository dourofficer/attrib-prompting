#!/usr/bin/env bash
# Run the binary-search baseline. Usage:
#   MODEL=gpt-4o DATASET=ww SUBSET=hand-crafted bash scripts/binary_search.sh
# See scripts/README.md for all knobs.
METHOD="binary_search"
source "$(dirname "${BASH_SOURCE[0]}")/_method_common.sh"
