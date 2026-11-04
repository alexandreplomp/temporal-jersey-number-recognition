"""Score finished runs on val, or once on test.
Example: uv run evaluate.py averaged_30 pooled_30 full_30

Two steps, as in the SoccerNet jersey task and the Koshkina & Elder (2024) pipeline:
predict writes one row per sequence, and score turns rows into numbers. Every method
goes through the same scorer, and scoring needs no model. The methods:
    sequence  the model's answer
    anchor    the clearest frame read alone; the deployed model never gets it
    vote      each frame read alone, then a confidence-weighted vote, the tracklet
              rule of Koshkina & Elder: what the frames give without a learned combination

Writes per run, in its folder:
    predictions_<split>.csv  one row per sequence: every method's answer and confidence (for
                             the vote, its agreement), each frame's own reading and the answer
                             after 1, 2, ... frames
    predictions_holdout_<split>.csv
                             the same for held-out numbers, if the run has --exclude
    metrics_<split>.json     per method: scores, per number, shot bands, confusions;
                             stability; held-out numbers if the run has --exclude (train and
                             val sequences on val, only test sequences on --split test)
and one row per run in results/results_<split>.csv, the comparison table (as timm's
validate.py --results-file). Re-evaluating a run replaces its row.

The test split is scored once per run: a second --split test stops.
"""

from collections import defaultdict
from pathlib import Path
import argparse
import csv
import json
import math

import torch
from torch.utils.data import DataLoader

from jersey.data import JerseyDataset, encode_label, filter_numbers, run_splits
from jersey.engine import evaluate
from jersey.metrics import (
    UNSEEN, band_accuracy, confusions, count_flips, heldout_error_share, majority_baseline,
    per_number, scores, shot_bands, to_numbers,
)
from jersey.model import load_run, predict_digits
from train import pick_device

METHODS = ("sequence", "anchor", "vote")
# The column beside each answer. The vote's is the winner's share of the votes, not a
# probability, so it is never averaged or shown as a confidence.
SCORE_COLUMN = {"sequence": "sequence_confidence", "anchor": "anchor_confidence", "vote": "vote_agreement"}
SEPARATOR = "|"  # between the per-frame answers in one CSV cell
# accuracy_by_frames follows the sequences with at least this many frames, the same ones for every k.
PREFIX_FRAMES = 5
# Where the held-out numbers' sequences come from, per --split. Kept apart, never pooled.
HOLD_OUT_SPLITS = {"val": ("train", "val"), "test": ("test",)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="+", help="run folders under --runs, e.g. full_30")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--runs", type=Path, default=Path("results/runs"))
    parser.add_argument("--device", default="auto", help="auto picks cuda, then mps, then cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args(argv)


@torch.no_grad()
def prefix_answers(model, frames, mask):
    """The answer after the first 1, 2, ..., T frames of each sequence: digits (B, T, 2).

    Real frames come first (see data._pad), so the first k are mask & (t < k). Past a
    sequence's last real frame, the answer repeats its full answer.
    """
    steps = torch.arange(mask.shape[1], device=mask.device)
    return torch.stack([
        predict_digits(model(frames, mask & (steps < k))["logits"])[0]
        for k in range(1, mask.shape[1] + 1)
    ], dim=1)


def vote(readings, confidences):
    """Confidence-weighted vote: the number with the largest summed confidence, and its share of the total."""
    totals = defaultdict(float)
    for number, confidence in zip(readings, confidences):
        totals[number] += confidence
    winner = max(totals, key=totals.get)
    return winner, totals[winner] / sum(totals.values())


def predict(model, loader, device):
    """Return one row per sequence with every method's answer, and the sequence and anchor losses."""
    sequence = evaluate(model, loader, device)
    anchor = evaluate(model, loader, device, use_anchor=True)
    frame_rows = []
    with torch.no_grad():
        for batch in loader:
            frames, mask = batch["frames"].to(device), batch["mask"].to(device)
            frame_digits, frame_confidence = predict_digits(model(frames, mask)["frame_logits"])
            prefixes = prefix_answers(model, frames, mask).cpu()
            for i, count in enumerate(batch["mask"].sum(dim=1).tolist()):
                readings = to_numbers(frame_digits[i, :count])
                frame_rows.append({
                    "sequence_id": batch["sequence_id"][i],
                    "frames": count,
                    "readings": readings,
                    "confidences": frame_confidence[i, :count].tolist(),
                    "prefix": to_numbers(prefixes[i, :count]),
                })
    # The loader does not shuffle, so all three passes see the sequences in the same order.
    assert sequence["sequence_ids"] == anchor["sequence_ids"] == [row["sequence_id"] for row in frame_rows]

    rows = []
    columns = zip(
        frame_rows, to_numbers(sequence["labels"]),
        to_numbers(sequence["predictions"]), sequence["confidences"].tolist(),
        to_numbers(anchor["predictions"]), anchor["confidences"].tolist(),
    )
    for frame_row, true, answer, confidence, anchor_answer, anchor_confidence in columns:
        voted, vote_agreement = vote(frame_row["readings"], frame_row["confidences"])
        rows.append({
            "sequence_id": frame_row["sequence_id"],
            "true": true,
            "frames": frame_row["frames"],
            "sequence": answer,
            "sequence_confidence": round(confidence, 4),
            "anchor": anchor_answer,
            "anchor_confidence": round(anchor_confidence, 4),
            "vote": voted,
            "vote_agreement": round(vote_agreement, 4),
            "readings": SEPARATOR.join(frame_row["readings"]),
            "prefix": SEPARATOR.join(frame_row["prefix"]),
        })
    return rows, {"sequence": sequence["loss"], "anchor": anchor["loss"]}


def score(rows, train_numbers):
    """Turn prediction rows (from predict or the CSV) into metrics, the same way for every method."""
    labels = _digits(row["true"] for row in rows)
    bands = shot_bands(train_numbers)
    methods = {}
    for method in METHODS:
        predictions = _digits(row[method] for row in rows)
        key = "mean_agreement" if method == "vote" else "mean_confidence"
        methods[method] = {
            **scores({"loss": None, "predictions": predictions, "labels": labels}),
            key: _mean(float(row[SCORE_COLUMN[method]]) for row in rows),
            "per_number": per_number(predictions, labels),
            "groups": band_accuracy(predictions, labels, bands),
            "confusions": confusions(predictions, labels),
        }
    return {
        "sequences": len(rows),
        "baseline": majority_baseline(labels, train_numbers),
        "methods": methods,
        "stability": stability(rows),
    }


def stability(rows):
    """How often the answer changes as frames arrive, and accuracy after k frames.

    prefix is the model accumulating frames, the like-for-like comparison between runs.
    frame_readings is each frame alone, which changes by design: context, not a baseline.
    """
    prefixes = [row["prefix"].split(SEPARATOR) for row in rows]
    # One fixed group for every k, so the curve shows frames and not which sequences remain.
    # Capped at the longest sequence, so a --max-frames 1 run still gets k = 1.
    frames = min(PREFIX_FRAMES, max(map(len, prefixes)))
    group = [(answers, row["true"]) for answers, row in zip(prefixes, rows) if len(answers) >= frames]
    by_frames = {
        k: {"count": len(group), "accuracy": _mean(float(answers[k - 1] == true) for answers, true in group)}
        for k in range(1, frames + 1)
    }
    return {
        "prefix": _flips(prefixes),
        "frame_readings": _flips([row["readings"].split(SEPARATOR) for row in rows]),
        "accuracy_by_frames": by_frames,
    }


def hold_out(rows, train_numbers):
    """Scores on numbers the run never trained on, and how often a wrong answer is a training number."""
    labels = _digits(row["true"] for row in rows)
    predictions = _digits(row["sequence"] for row in rows)
    return {
        "sequences": len(rows),
        "accuracy": _mean(float(row["sequence"] == row["true"]) for row in rows),
        "per_number": per_number(predictions, labels),
        "wrong_answers_that_are_training_numbers": heldout_error_share(predictions, labels, train_numbers),
        "confusions": confusions(predictions, labels),
    }


def main(argv=None):
    args = parse_args(argv)
    device = pick_device(args.device)
    for name in args.names:
        evaluate_run(args, args.runs / name, device)


def evaluate_run(args, run_dir, device):
    metrics_path = run_dir / f"metrics_{args.split}.json"
    if args.split == "test" and metrics_path.exists():
        raise FileExistsError(f"{metrics_path} exists: the test split is scored only once per run")
    model, config, epoch = load_run(run_dir, device)
    splits = run_splits(run_dir, config["splits"])
    exclude = tuple(config["exclude"])
    # As trained: without held-out numbers and their linked partners. Older runs dropped no partners.
    dropped = set(config.get("linked_dropped", []))
    train_numbers = [sequence_id.split("/")[0] for sequence_id in filter_numbers(splits["train"], exclude)
                     if sequence_id not in dropped]

    def loader(ids):
        dataset = JerseyDataset(config["data"], ids, load_anchor=True, max_frames=config["max_frames"])
        return DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)

    rows, losses = predict(model, loader(filter_numbers(splits[args.split], exclude)), device)
    metrics = {
        "run": {
            "name": run_dir.name, "split": args.split, "variant": config["variant"],
            "max_frames": config["max_frames"], "epoch": epoch,
            "param_count": sum(p.numel() for p in model.parameters()),
        },
        **score(rows, train_numbers),
        "hold_out": None,
    }
    for method, loss in losses.items():
        metrics["methods"][method]["loss"] = loss
    _write_rows(run_dir / f"predictions_{args.split}.csv", rows)

    if exclude:
        # On val, the train and val sequences of held-out numbers: the model never trained on
        # them, and this score feeds decisions. On test, only the test ones, since the others
        # were already seen while developing, so the final score stays a clean test score.
        names = HOLD_OUT_SPLITS[args.split]
        held = sorted(s for name in names for s in splits[name] if s.split("/")[0] in exclude)
        held_rows, _ = predict(model, loader(held), device)
        metrics["hold_out"] = {"splits": list(names), **hold_out(held_rows, train_numbers)}
        _write_rows(run_dir / f"predictions_holdout_{args.split}.csv", held_rows)

    metrics_path.write_text(json.dumps(metrics, indent=2))
    _update_results(args.runs.parent / f"results_{args.split}.csv", _summary_row(metrics))
    _print_table(metrics, rows, shot_bands(train_numbers))


def _summary_row(metrics):
    """One flat row for the comparison table."""
    row = {key: metrics["run"][key] for key in ("name", "variant", "max_frames", "epoch", "param_count")}
    row["sequences"] = metrics["sequences"]
    row["sequence_loss"] = round(metrics["methods"]["sequence"]["loss"], 4)
    for method in METHODS:
        found = metrics["methods"][method]
        for key in ("accuracy", "two_digit_accuracy", "macro_accuracy", "macro_accuracy_5"):
            row[f"{method}_{key}"] = round(found[key], 4)
    row["stable_share"] = round(metrics["stability"]["prefix"]["stable_share"], 4)
    row["mean_flips"] = round(metrics["stability"]["prefix"]["flips"], 4)
    row["hold_out_accuracy"] = "" if metrics["hold_out"] is None else round(metrics["hold_out"]["accuracy"], 4)
    return row


def _update_results(path, row):
    """Add or replace this run's row in the shared results CSV."""
    rows = []
    if path.exists():
        with path.open(newline="") as file:
            rows = [old for old in csv.DictReader(file) if old["name"] != row["name"]]
    rows = sorted(rows + [row], key=lambda found: found["name"])
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row), restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _print_table(metrics, rows, bands):
    """Accuracy and mean confidence per group and method, with a combined row, as in PARSeq's test.py."""
    groups = [(band, lambda row, band=band: bands.get(row["true"], UNSEEN) == band)
              for band in ("many", "medium", "few", UNSEEN)]
    groups += [("one digit", lambda row: len(row["true"]) == 1), ("two digit", lambda row: len(row["true"]) == 2),
               ("combined", lambda row: True)]
    run = metrics["run"]
    print(f"\n{run['name']} ({run['variant']}, epoch {run['epoch']}) on {run['split']}")
    print(f"{'group':<10}{'count':>6}" + "".join(
        f"{method + ' acc':>14}{'agree' if method == 'vote' else 'conf':>6}" for method in METHODS))
    for label, keep in groups:
        kept = [row for row in rows if keep(row)]
        if not kept:
            continue
        cells = "".join(
            f"{_mean(float(row[m] == row['true']) for row in kept):>14.3f}"
            f"{_mean(float(row[SCORE_COLUMN[m]]) for row in kept):>6.2f}"
            for m in METHODS
        )
        print(f"{label:<10}{len(kept):>6}{cells}")
    found = metrics["methods"]["sequence"]
    flips = metrics["stability"]["prefix"]
    print(f"loss {found['loss']:.3f}, two-digit {found['two_digit_accuracy']:.3f},"
          f" macro {found['macro_accuracy']:.3f}, macro5 {found['macro_accuracy_5']:.3f},"
          f" stable share {flips['stable_share']:.3f}, mean flips {flips['flips']:.3f}")


def _write_rows(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _flips(answer_lists):
    """count_flips on per-sequence answer lists of different lengths."""
    length = max(map(len, answer_lists))
    padded = [answers + answers[-1:] * (length - len(answers)) for answers in answer_lists]
    return count_flips(torch.stack([_digits(answers) for answers in padded]),
                       torch.tensor([len(answers) for answers in answer_lists]))


def _digits(numbers):
    """Numbers as strings -> digit classes (S, 2)."""
    return torch.tensor([encode_label(number) for number in numbers])


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else math.nan


if __name__ == "__main__":
    main()
