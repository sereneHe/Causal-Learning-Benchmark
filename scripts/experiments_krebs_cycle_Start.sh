#!/bin/sh

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR"

export PYTHONPATH=$PYTHONPATH:../

CMD="python3 main.py"


# Run Krebs_Cycle_1
# ${CMD} problem="krebs_cycle_1" paths.data_root="/Users/xiaoyuhe/Causal-Methods/krebcycle/data" solver="notebook_all" solver.skip_existing=false

# Run Krebs_Cycle_3
${CMD} problem="krebs_cycle_3" paths.data_root="/Users/xiaoyuhe/Causal-Methods/krebcycle/data" solver="notebook_all" solver.skip_existing=false

# Run Krebs_Cycle_1_Normalised
# ${CMD} problem="krebs_cycle_1_normalised" paths.data_root="/Users/xiaoyuhe/Causal-Methods/krebcycle/data" solver="notebook_all" solver.skip_existing=false

# Run Krebs_Cycle_3_Normalised
# ${CMD} problem="krebs_cycle_3_Normalised" paths.data_root="/Users/xiaoyuhe/Causal-Methods/krebcycle/data" solver="notebook_all" solver.skip_existing=false
