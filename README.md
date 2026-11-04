# Temporal jersey number recognition

Recognise two-digit jersey numbers from sequences of player crops.

## Setup

Requires [uv](https://docs.astral.sh/uv/). It installs Python 3.12 and the locked dependencies:

```bash
uv sync
```

Unzip the dataset into `data/` so the layout is `data/<number>/<player>/<sequence>/<image>.jpg`.
Each sequence has one `*_anchor` image plus its frames. `data/` is never changed and is not in Git.

## Commands

```bash
uv run check_data.py    # writes results/data_issues.json and prints counts per number
uv run make_splits.py   # writes results/splits.json (player-level, leakage-free)
uv run pytest           # runs the tests on small fake data
uv run train.py --variant full   # trains one model, writes results/runs/<variant>_<date-time>/
```

```bash
uv run evaluate.py averaged_30 pooled_30 full_30   # scores runs on val, writes results/results_val.csv
uv run explain.py results/runs/full_30             # prediction.png for an existing run
```

`evaluate.py` scores three methods per run the same way: the sequence model, the anchor alone, and a
confidence-weighted vote of the frames read alone. Use `--split test` only once, on the final runs;
a second test evaluation of a run stops.

With `--exclude`, held-out numbers are scored apart: on val, their train and val sequences
(`predictions_holdout_val.csv`); on `--split test`, only their test sequences (`predictions_holdout_test.csv`),
since the others were already seen while developing.

`uv run train.py --help` lists the settings. The best epoch is always the one with the lowest val loss
(saved as `selected_by` in `config.json`), and val two-digit accuracy is reported beside it, since the loss
is dominated by one-digit numbers. Its per-number val scores, with 95% intervals, go to `per_number.json` in the run folder. Use `--schedule onecycle` for final runs and
`--max-frames 1 --variant averaged` for the single-frame baseline.

`results/splits.json` is committed, so you only need to rerun the two scripts if the data changes.

## Results

To be added.
