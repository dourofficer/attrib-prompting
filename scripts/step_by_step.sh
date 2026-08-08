#!/usr/bin/env bash
# Run the step-by-step baseline. Usage:
#   MODEL=gpt-4o DATASET=ww SUBSET=hand-crafted bash scripts/step_by_step.sh
# See scripts/README.md for all knobs.
METHOD="step_by_step"
source "$(dirname "${BASH_SOURCE[0]}")/_method_common.sh"
