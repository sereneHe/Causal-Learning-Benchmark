#!/bin/sh

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SCRIPT_DIR/.."

export PYTHONPATH=$PYTHONPATH:../../

PROBLEM=${PROBLEM:-krebs_cycle_3}
DATA_ROOT=${DATA_ROOT:-/Users/xiaoyuhe/Causal-Methods/krebcycle/data}
GRAPH_FILE=${GRAPH_FILE:-/Users/xiaoyuhe/Causal-Methods/krebcycle/output/Results_Krebs_Cycle_3/adj_matrices/DyNotear_adj.csv}
OUTPUT_FILE=${OUTPUT_FILE:-}

CMD="python3 plot/krebcycle_heatmap.py"

# Example:
# PROBLEM=krebs_cycle_normalised_3 \
# GRAPH_FILE=/Users/xiaoyuhe/Causal-Methods/krebcycle/output/Results_Krebs_Cycle_Normalised_3/adj_matrices/DyNotear_adj.csv \
# sh ./experiments_krebcycle_heatmap.sh

if [ -n "$OUTPUT_FILE" ]; then
  ${CMD} --problem "$PROBLEM" --data-root "$DATA_ROOT" --graph-file "$GRAPH_FILE" --output-file "$OUTPUT_FILE"
else
  ${CMD} --problem "$PROBLEM" --data-root "$DATA_ROOT" --graph-file "$GRAPH_FILE"
fi
