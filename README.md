# Temporal jersey number recognition

A lightweight model that reads a jersey number from a sequence of player crops and gives one stable
answer per sequence. It reads every number as two positions, each a digit or blank, with one shared
digit classifier. So a number never seen in training, such as 94 after 4, 9 and 49, can still be read.
The design, the results and their limits are in [REPORT.md](REPORT.md).

## Setup

Requires [uv](https://docs.astral.sh/uv/); it installs Python 3.12 and the locked dependencies:

```bash
uv sync
```

Unzip the dataset into `data/`, so the layout is `data/<number>/<player>/<sequence>/<frames>.jpg`. Each
sequence may have one `*_anchor` image, used only in training. `data/` is never changed and is not in Git.

The first training run downloads pretrained weights, so it needs internet: MobileNetV3-Small (ImageNet,
torchvision) and PARSeq-Tiny (scene text, github.com/baudm/parseq). `predict.py` and `evaluate.py` on the
shipped weights need no download.

## Read a number

```bash
uv run predict.py data/4/<player>/<sequence>                       # the shipped model in weights/final/
uv run predict.py <folder> --weights results/runs/two_step_pool_ex49 --two-digit-share 0.5
                                                                    # two-step readers only: a league with more two-digit numbers
```

It prints one number and its confidence. The anchor is ignored, since a tracker has none. The shipped
PARSeq model has no count head, so `--two-digit-share` gives a warning and is ignored. The two-step
weights are not committed, so the second line works only after training that run.

To check the headline result from the shipped weights, run `uv run evaluate.py --verify weights/final`.
It scores the test split again into `weights/final/verify_test.json` and compares every answer with
`weights/final/predictions_test.csv`. It should report 0.9642 and the same answer on 866/866.

## Pipeline

Each script is one step; it reads the previous step's files and writes its own.

| Step | Command | Writes |
|---|---|---|
| Data | `uv run prepare.py` | `results/data_issues.json`, `results/splits.json` (by player, 70/15/15) |
| Train | `uv run train.py --approach two_step_mean --exclude 49` | `results/runs/two_step_mean_ex49/`, and `results/boxes/ex49/` if missing |
| Evaluate | `uv run evaluate.py two_step_mean_ex49` (add `--split test` once) | `metrics_val.json` in the run, and `results/compare/` |
| Analyse | `uv run evaluate.py --analyse parseq_gate_all` | `results/compare/analysis_parseq_gate_all.json` (match overlap, paired tests, track-level intervals, gate, int8) |
| Predict | `uv run predict.py <folder>` | one number |

`reproduce.sh` runs everything in order, in about 6 h on an Apple M4 Pro:

1. the tests
2. the comparison runs with 49 held out
3. the choice of the final approach on val
4. the final model on all numbers
5. the test split, once
6. copying the weights to `weights/final/` and verifying them
7. two more seeds and a random start of the final approach, 49 held out
8. the extra analysis

**Reruns differ slightly.** MPS training is not bit-for-bit deterministic, so a rerun gives slightly
different numbers. The record is the committed one:

- the shipped weights;
- each run's `config.json` (with its git commit), `history.json`, `metrics_*.json` and `predictions_*.csv` in `results/runs/` (weights not committed);
- `results/compare/`.

`uv run pytest` runs the tests on small fake data (about 2 minutes).

The approaches are the table at the top of `train.py`; each row changes one thing:

| Approach | What it shows |
|---|---|
| `single_frame` | One frame per sequence: what the temporal component adds |
| `fused` | Two digit heads on whole-crop features: the two-digit structure alone |
| `two_step_mean` | Where-then-what reader, frames averaged |
| `two_step_pool` | The same, with learned frame weights |
| `two_step_full` | The same, frames see each other (transformer) |
| `parseq_gate` | PARSeq-Tiny, a pretrained text reader, per frame; a learned gate per digit position over up to 10 frames; real data only |

`jersey/pairs.py` (digit boxes and spliced pairs) only serves the two-step readers.

## Results

From one run of `reproduce.sh` (`results/compare/table.csv`, `reproduce.log`). The final model is
`parseq_gate` trained on all ten numbers, shipped in `weights/final/`: **0.964 on the 866 test sequences**
(anchor alone 0.844, frame vote 0.947), 6.0M parameters, 22.9 MB, 36 ms per 10-frame sequence on one CPU
thread. The comparison below holds 49 out of training, so 49 is a number never seen in training:

- "Dev. hold-out" is 105 sequences of train- and val-split players, looked at during development;
- "Test hold-out" is 16 test-player sequences, the clean estimate.

| Run (49 held out) | Val accuracy | Two-digit | Dev. hold-out | Anchor alone | Frame vote | Test accuracy | Test hold-out | Size |
|---|---|---|---|---|---|---|---|---|
| `single_frame` | 0.815 | 0.205 | 11/105 | 0.836 | 0.815 | 0.815 | 2/16 | 1.5 MB |
| `fused` | 0.867 | 0.872 | 0/105 | 0.706 | 0.849 | 0.875 | 0/16 | 1.2 MB |
| `two_step_mean` | 0.932 | 0.590 | 23/105 | 0.827 | 0.924 | 0.928 | 5/16 | 1.5 MB |
| `two_step_pool` | 0.942 | 0.538 | 26/105 | 0.832 | 0.929 | 0.920 | 4/16 | 1.5 MB |
| `two_step_full` | 0.935 | 0.615 | 26/105 | 0.829 | 0.925 | 0.928 | 6/16 | 2.0 MB |
| **`parseq_gate`** | **0.960** | **0.949** | **59/105** | 0.893 | 0.949 | **0.959** | **11/16** | 22.9 MB |
| `parseq_gate`, seed 1 | 0.960 | 0.949 | 59/105 | 0.898 | 0.945 | 0.961 | 11/16 | 22.9 MB |
| `parseq_gate`, seed 2 | 0.957 | 0.949 | 59/105 | 0.889 | 0.946 | 0.961 | 10/16 | 22.9 MB |
| `parseq_gate`, random start | 0.462 | 0.513 | 0/105 | 0.435 | 0.454 | 0.409 | 0/16 | 22.9 MB |

Most of PARSeq's held-out result comes from its scene-text pretraining. REPORT.md, "Use of pretrained
weights", gives the details and the lightweight alternative trained only on this data plus ImageNet.

The graphs are in `results/compare/`; [REPORT.md](REPORT.md) discusses them.

## Layout

```text
prepare.py   train.py   evaluate.py   predict.py   reproduce.sh
jersey/
  data.py     labels, splits, the sequence dataset and its augmentation
  pairs.py    finding digit boxes without annotation, spliced two-digit sequences, the pair metric
  model.py    readers (fused, two-step, parseq), frame combiners (mean, pool, transformer), and the
              PARSeq-Tiny network (Apache-2.0, from baudm/parseq) cut to the ten digits
  engine.py   losses, one training epoch, evaluation, shared helpers
  metrics.py  scores with 95% intervals
  plots.py    every figure
tests/  test_parts.py  test_pipeline.py  helpers.py
weights/final/   the shipped model: parseq_gate on all numbers, with its config, split, metrics and test predictions
results/runs/    each run's config, history, metrics and predictions (weights not committed)
LICENSE-parseq  NOTICE   Apache-2.0 licence and attribution for the code taken from PARSeq
```
