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
uv run evaluate.py averaged_final full_final   # scores runs on val, writes results/results_val.csv
uv run explain.py results/runs/full_final      # prediction.png for an existing run
uv run check_shortcuts.py averaged_final      # val accuracy without colour or without the number area
```

`evaluate.py` scores three methods per run the same way: the sequence model, the anchor alone, and a
confidence-weighted vote of the frames read alone. Use `--split test` only once, on the final runs;
a second test evaluation of a run stops.

With `--exclude`, held-out numbers are scored apart: on val, their train and val sequences
(`predictions_holdout_val.csv`); on `--split test`, only their test sequences (`predictions_holdout_test.csv`),
since the others were already seen while developing.

`uv run train.py --help` lists the settings. The best epoch is always the one with the lowest val loss
(saved as `selected_by` in `config.json`), and val two-digit accuracy is reported beside it, since the loss
is dominated by one-digit numbers. Its per-number val scores, with 95% intervals, go to `per_number.json` in the run folder. `--augment strong` (the default) scrambles colour, since colour predicts the number in this data (see `check_shortcuts.py`);
`--augment basic` is the earlier, colour-keeping policy, for comparison. Use `--schedule onecycle` for final runs and
`--max-frames 1 --variant averaged` for the single-frame baseline.

Two-digit numbers are rare in this data (a first 4 only in 48), so a model cannot learn to read an
unseen one such as 49 digit by digit. `--synthetic-pairs` splices extra two-digit training sequences from
single-digit ones, with no annotation (see `jersey/pairs.py`):

```bash
uv run make_pairs.py full_strong_ex49_keypos   # finds each single digit with that run, writes results/pairs/
uv run train.py --exclude 49 --synthetic-pairs 0.3 --schedule onecycle --epochs 40
```

`make_pairs.py` locates the digit by covering the torso and keeps the boxes the run's reading depends on.
Check `results/pairs/strips.png` (the boxes) and `spliced.png` (spliced frames) before training. Training stops
if the bank could splice a held-out number, so build it with a run that held out the same numbers.

A pasted patch is itself a cue: if only two-digit samples have one, the model learns "pasted means two
digits" (two_step_s2_ex49 reads pasted single digits right 45% of the time against 83% unpasted).
`--synthetic-singles` pastes a partner's digit alone, the same way and as wide, labelled as one digit, and
the blend width varies per sequence for both kinds. Use the same share for both, e.g.
`--synthetic-pairs 0.4 --synthetic-singles 0.4`. The pair metric's `pasted_singles_accuracy` checks it.

`results/splits.json` is committed, so you only need to rerun the two scripts if the data changes.

## Results

To be added.
