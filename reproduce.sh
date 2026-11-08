#!/usr/bin/env bash
# Every step of the hand-in, in order: about 5.5 hours on an Apple M4 Pro (MPS). The first training run
# downloads the ImageNet MobileNetV3 and PARSeq-Tiny weights, so it needs internet. MPS training is not
# bit-for-bit deterministic: a rerun gives slightly different numbers; the committed metrics are the record.
# Run from the project root with the data in data/. Nothing is ever overwritten, and the script can be
# rerun after a stop: train.py skips finished runs (and refuses half-finished ones: delete those), and
# each run is scored on the test split only once.
set -euo pipefail

# A fresh clone holds the reported runs' records in results/runs/ but not their weights: train anew elsewhere.
for done in results/runs/*/done.json; do
    if [ -f "$done" ] && [ ! -f "$(dirname "$done")/best.pt" ]; then
        echo "results/runs/ holds the committed record of the reported runs, without weights."
        echo "Move it aside to reproduce from scratch:  mv results/runs results/runs_reported"
        exit 1
    fi
done

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

# 6. Ship the final weights (used by predict.py) and check they give the recorded test answers.
mkdir -p weights/final
cp "results/runs/${FINAL}_all/"{best.pt,config.json,splits.json,metrics_val.json,metrics_test.json,predictions_test.csv} weights/final/
uv run evaluate.py --verify weights/final
#    Model B, trained only on the given data plus ImageNet: the two_step_* run picked by the same val rule.
TWO_STEP=$(uv run python - <<'PY'
import csv
with open("results/compare/table.csv") as file:
    rows = [r for r in csv.DictReader(file) if r["approach"].startswith("two_step") and r["run"].endswith("_ex49")]
print(max(rows, key=lambda r: float(r["val_macro_5"]))["run"])
PY
)
mkdir -p weights/two_step_ex49
cp "results/runs/$TWO_STEP/"{best.pt,config.json,splits.json,metrics_val.json,metrics_test.json,predictions_test.csv} weights/two_step_ex49/
uv run evaluate.py --verify weights/two_step_ex49

# 7. How much of the held-out result is the final approach, and how much its pretraining and its seed:
#    two more seeds and a random start, 49 held out as before (not candidates for the final approach).
for extra in "--seed 1" "--seed 2" "--no-pretrained"; do
    uv run train.py --approach "$FINAL" --exclude 49 $extra
done
for run in "${FINAL}_ex49_s1" "${FINAL}_ex49_s2" "${FINAL}_ex49_scratch"; do
    [ -f "results/runs/$run/metrics_val.json" ] || uv run evaluate.py "$run"
    [ -f "results/runs/$run/metrics_test.json" ] || uv run evaluate.py "$run" --split test
done

# 8. Extra evidence for the report: match overlap, paired tests, track-level intervals, gate weights,
#    int8 and fewer frames (results/compare/analysis_<run>.json; test predictions are only read).
uv run evaluate.py --analyse "${FINAL}_all"
echo "done: results/compare/ holds the table and graphs; weights/final/ the shipped model"
