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

## Read a number

```bash
uv run predict.py data/4/<player>/<sequence>                       # the shipped model in weights/final/
uv run predict.py <folder> --two-digit-share 0.5                    # a league with more two-digit numbers
```

It prints one number and its confidence. The anchor is ignored, since a tracker has none.

## Pipeline

Each script is one step; it reads the previous step's files and writes its own.

| Step | Command | Writes |
|---|---|---|
| Data | `uv run prepare.py` | `results/data_issues.json`, `results/splits.json` (by player, 70/15/15) |
| Train | `uv run train.py --approach two_step_mean --exclude 49` | `results/runs/two_step_mean_ex49/`, and `results/boxes/ex49/` if missing |
| Evaluate | `uv run evaluate.py two_step_mean_ex49` (add `--split test` once) | `metrics_val.json` in the run, and `results/compare/` |
| Predict | `uv run predict.py <folder>` | one number |

`reproduce.sh` runs everything in order (about 5-6 h on an M-series Mac):
1. the tests
2. the comparison runs with 49 held out
3. the choice of the final approach on val
4. the final model on all numbers
5. the test split, once
6. copying the weights to `weights/final/`

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

## Results

Filled in from `results/compare/table.csv` after `reproduce.sh`.

| Run | Val accuracy | Two-digit | Held-out 49 | Anchor alone | Frame vote | Test accuracy |
|---|---|---|---|---|---|---|
| TODO | | | | | | |

## Layout

```text
prepare.py   train.py   evaluate.py   predict.py   reproduce.sh
jersey/
  data.py     labels, splits, the sequence dataset and its augmentation
  pairs.py    finding digit boxes without annotation, spliced two-digit sequences, the pair metric
  model.py    readers (fused, two-step, parseq), frame combiners (mean, pool, transformer)
  parseq.py   the PARSeq-Tiny network (Apache-2.0, from baudm/parseq), cut to the ten digits
  engine.py   losses, one training epoch, evaluation, shared helpers
  metrics.py  scores with 95% intervals
  plots.py    every figure
tests/  test_parts.py  test_pipeline.py  helpers.py
weights/final/   the shipped model (after reproduce.sh)
```
