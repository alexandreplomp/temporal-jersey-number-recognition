#!/usr/bin/env bash
# Every step of the hand-in, in order: about 3-4 hours on an M-series Mac (MPS).
# Run from the project root with the data in data/. Nothing is ever overwritten, and the script can be
# rerun after a stop: train.py skips finished runs (and refuses half-finished ones: delete those), and
# each run is scored on the test split only once.
set -euo pipefail

uv run pytest

# 1. Data checks and the player-level split (reproduces the committed results/splits.json).
uv run prepare.py

# 2. The comparison ladder with 49 held out, to measure reading a number never seen in training.
#    The first run that pastes digits trains the digit reader and finds the boxes (results/boxes/ex49/).
for approach in single_frame fused two_step_mean two_step_pool two_step_full parseq_gate; do
    uv run train.py --approach "$approach" --exclude 49
done
uv run evaluate.py single_frame_ex49 fused_ex49 two_step_mean_ex49 two_step_pool_ex49 two_step_full_ex49 parseq_gate_ex49

# 3. The final approach: the two_step_* or parseq_gate run with the best val macro accuracy (numbers with >= 5
#    sequences). Chosen on val only, never on the held-out numbers or the test split.
FINAL=$(uv run python - <<'PY'
import csv
with open("results/compare/table.csv") as file:
    rows = [r for r in csv.DictReader(file) if r["approach"].startswith(("two_step", "parseq")) and r["run"].endswith("_ex49")]
print(max(rows, key=lambda r: float(r["val_macro_5"]))["approach"])
PY
)
echo "final approach: $FINAL"

# 4. The final model on all ten numbers (makes results/boxes/all/ first).
uv run train.py --approach "$FINAL"
uv run evaluate.py "${FINAL}_all"

# 5. The test split, once: the final model, and the held-out ladder (49 test sequences are never trained on).
for run in "${FINAL}_all" single_frame_ex49 fused_ex49 two_step_mean_ex49 two_step_pool_ex49 two_step_full_ex49 parseq_gate_ex49; do
    [ -f "results/runs/$run/metrics_test.json" ] || uv run evaluate.py "$run" --split test
done

# 6. Ship the final weights (used by predict.py).
mkdir -p weights/final
cp "results/runs/${FINAL}_all/"{best.pt,config.json,splits.json,metrics_val.json,metrics_test.json} weights/final/
echo "done: results/compare/ holds the table and graphs; weights/final/ the shipped model"
