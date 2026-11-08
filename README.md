# Temporal jersey number recognition

A lightweight model that reads a jersey number from a sequence of player crops and gives one stable
answer per sequence. It reads every number as two positions, each a digit or blank, with one shared
digit classifier. So a number never seen in training, such as 94 after 4, 9 and 49, can still be read.
The design, results and limits are discussed in the separate written report; this README covers setup,
use and reproduction.

## Setup

1. **Get the code and install.** This needs [uv](https://docs.astral.sh/uv/), which installs Python 3.12
   and the locked dependencies:

   ```bash
   git clone https://github.com/alexandreplomp/trackbox_case.git
   cd trackbox_case
   uv sync
   ```

2. **Download the dataset.** It is not in this repository. Download the ZIP from the case brief's link
   (<https://drive.google.com/file/d/183CXUk99IyCRgfAPeRnpo5xBYCER1-an/view>) and unzip it. That gives a
   folder `temporal_jersey_nr_recognition_dataset_subset/`.

3. **Move it into the project and rename it `data`.** Every script looks for the dataset in `data/` at
   the project root, and the run configs store that path:

   ```bash
   mv ~/Downloads/temporal_jersey_nr_recognition_dataset_subset ./data
   ```

   The layout must then be `data/<number>/<player>/<sequence>/<frames>.jpg`, for example
   `data/4/1000_19764/0/1000_46888_0.jpg`. Each sequence may have one `*_anchor` image, used only in
   training. The scripts never change `data/`, and `.gitignore` keeps it out of Git.

Run every command from the project root (the folder with `README.md`).

**Internet:** the first *training* run downloads pretrained weights: MobileNetV3-Small (ImageNet, from
torchvision) and PARSeq-Tiny (scene text, from github.com/baudm/parseq). Reading numbers and checking
the shipped weights need no download.

## Read a number

```bash
uv run predict.py data/4/1000_19764/0                                  # model A, weights/final/
uv run predict.py data/4/1000_19764/0 --weights weights/two_step_ex49  # model B
```

It prints one number and its confidence, e.g. `4  (confidence 0.96)`. The anchor is ignored, since a
tracker has none.

**Your own sequences.** `predict.py` reads any folder of `.jpg`, `.jpeg` or `.png` crops of one player,
not only folders in `data/`:

```bash
uv run predict.py path/to/my_sequence
```

- **Frame order:** frames are read in time order, by the number after the first underscore
  (`<prefix>_<frame>_...`, as in the dataset). Other names are read in alphabetical order.
- **Many frames:** at most 10 frames are used, spread evenly over the sequence.
- **Anchors:** files ending in `_anchor` are skipped.
- **Errors:** a missing folder, or one with no frames, gives a one-line error.

Two models are shipped:
- **A, `weights/final/`:** `parseq_gate` trained on all ten numbers, the best accuracy; it starts from
  PARSeq's scene-text weights.
- **B, `weights/two_step_ex49/`:** `two_step_full` (0.51M parameters, 2.0 MB), trained with 49 held out;
  it uses only the given data plus ImageNet weights.

`--two-digit-share` sets B's count prior for a league with a different share of two-digit numbers. For
example, with about half the numbers two-digit:
`uv run predict.py <folder> --weights weights/two_step_ex49 --two-digit-share 0.5`. A has no count head,
so for A the flag gives a warning and is ignored.

To check the shipped weights, run `uv run evaluate.py --verify weights/final` (or
`weights/two_step_ex49`). It scores the test split again into `verify_test.json` in that folder and
compares every answer with the recorded `predictions_test.csv`. It should report 0.9642 on 866/866
identical answers for A, and 0.9282 on 850/850 for B.

## Pipeline

Each script is one step; it reads the previous step's files and writes its own.

| Step | Command | Writes |
|---|---|---|
| Data | `uv run prepare.py` | `results/data_issues.json`, `results/splits.json` (by track folder, 70/15/15) |
| Train | `uv run train.py --approach two_step_mean --exclude 49` | `results/runs/two_step_mean_ex49/`, and `results/boxes/ex49/` if missing |
| Evaluate | `uv run evaluate.py two_step_mean_ex49` (add `--split test` once) | `metrics_val.json` in the run, and `results/compare/` |
| Analyse | `uv run evaluate.py --analyse parseq_gate_all` | `results/compare/analysis_parseq_gate_all.json` (match overlap, paired tests, track-level intervals, gate, int8) |
| Predict | `uv run predict.py <folder>` | one number |

### Train and evaluate something new

```bash
uv run prepare.py                                           # once: checks the data, writes the split
uv run train.py --approach parseq_gate --exclude 49 --seed 3  # a new run: results/runs/parseq_gate_ex49_s3/
uv run evaluate.py parseq_gate_ex49_s3                      # val scores, and the table and graphs remade
uv run evaluate.py parseq_gate_ex49_s3 --split test         # test scores: allowed once per run
uv run predict.py data/49/100_5810/0 --weights results/runs/parseq_gate_ex49_s3
```

- **Approaches:** `--approach` is one of the six in the table below. `--exclude 4 49` holds out more
  numbers, and `--no-pretrained` starts from random weights. A smoke test: `--epochs 1`, which takes
  minutes. Delete its run folder afterwards, or the full run with the same settings is skipped.
- **Run names:** a run's folder name comes from its settings (`_ex49`, `_s<seed>`, `_scratch`; `_all`
  without `--exclude`).
- **Runs are never overwritten:** a finished run is skipped when asked again, and a half-finished one
  must be deleted first. To try a variant, change a setting, for example `--seed`. Retraining a reported
  run (e.g. `parseq_gate_ex49`) needs `results/runs/` moved aside, since its record is committed without
  weights.
- **Devices:** training uses CUDA, then Apple MPS, then CPU, whichever exists first (`--device` overrides
  it). `--workers 0` helps when data-loader processes misbehave.

`reproduce.sh` runs everything in order, in about 5.5 h on an Apple M4 Pro (3.7 h for steps 1-6, 1.6 h
for steps 7-8). `results/runs/` holds the committed record of the reported runs without their weights,
so move it aside first; the script refuses to start otherwise:

```bash
mv results/runs results/runs_reported
bash reproduce.sh
```

The steps:

1. the tests
2. the comparison runs with 49 held out
3. the choice of the final approach on val
4. the final model on all numbers
5. the test split, once
6. copying A's weights to `weights/final/` and B's (the best `two_step_*` run by the same rule) to
   `weights/two_step_ex49/`, and verifying both
7. two more seeds and a random start of the final approach, 49 held out
8. the extra analysis

**Reruns differ slightly.** MPS training is not bit-for-bit deterministic, so a rerun gives slightly
different numbers. The record is the committed one:

- the shipped weights;
- each run's `config.json` (with its git commit), `history.json`, `metrics_*.json` and
  `predictions_*.csv` in `results/runs/` (weights not committed);
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

From `reproduce.sh` (`results/compare/table.csv`, each run's metrics in `results/runs/`; the seed and
random-start rows come from step 7). The final model is
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

The random start reads no held-out 49 at all, so most of PARSeq's held-out result comes from its
scene-text pretraining. Model B, `two_step_full` (0.51M parameters, ImageNet only), is the alternative
trained on this data alone: 6 of 16 held-out 49s on test.

The graphs are in `results/compare/`.

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
weights/final/          model A: parseq_gate on all numbers, with its config, split, metrics and test predictions
weights/two_step_ex49/  model B: two_step_full with 49 held out, the same files
results/runs/    each run's config, history, metrics and predictions (weights not committed)
LICENSE-parseq  NOTICE   Apache-2.0 licence and attribution for the code taken from PARSeq
```
