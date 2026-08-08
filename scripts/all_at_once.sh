#!/usr/bin/env bash
# Run the all-at-once baseline. Usage:
#   MODEL=gpt-4o DATASET=ww SUBSET=hand-crafted bash scripts/all_at_once.sh
# See scripts/README.md for all knobs.
METHOD="all_at_once"
source "$(dirname "${BASH_SOURCE[0]}")/_method_common.sh"
