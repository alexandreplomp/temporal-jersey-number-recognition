"""Score trained runs on val, or once on test, and compare every evaluated run.
Example: uv run evaluate.py two_step_mean_ex49 fused_ex49        (val)
         uv run evaluate.py two_step_mean_all --split test        (test: once per run)
         uv run evaluate.py --verify weights/final                (re-score shipped weights on test)
         uv run evaluate.py --analyse parseq_gate_all             (extra evidence, see analyse())

Three methods per run, scored the same way: the sequence model; the anchor read alone (the
deployed model never gets it); and each frame read alone, then a confidence-weighted vote. Plus
the answer after 1, 2, ... frames, to show how the answer settles. With held-out numbers (training
--exclude), those are scored apart: on val their train and val sequences, on test only test ones.
Two-step runs are also scored without their count offset (see jersey/model.count_prior).

Writes per run, in its folder:
    predictions_<split>.csv  one row per sequence: each method's answer, each frame's reading, the
                             answer after 1, 2, ... frames
    metrics_<split>.json     per method scores, per number with 95% intervals, confusions, stability,
                             hold-out; on val also the shortcut checks, size and CPU speed
    attention.png            (val) where each query looks on a few val frames
and across every run with metrics, in results/compare/:
    table.csv  curves.png  temporal.png  holdout.png  and curves_<run>.png for runs on all numbers
    (the graphs show one run per approach: seed 0 with pretrained weights; table.csv has every run)

--verify DIR re-scores the weights in DIR on test and writes DIR/verify_test.json, comparing each
answer with DIR/predictions_test.csv; it never touches metrics_test.json. --analyse RUN writes
results/compare/analysis_<RUN>.json: tests on the recorded test predictions (no new test pass) and
measurements on val (gate weights, int8, fewer frames, PARSeq head drift).
"""

from collections import Counter, defaultdict
from pathlib import Path
import argparse
import copy
import csv
import io
import json
import math
import re
import statistics
import time
import warnings

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import v2

from jersey import parseq, plots
from jersey.data import (
    IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, JerseyDataset, denormalize, encode_label, filter_numbers, number_of,
    read_splits, torso_mask,
)
from jersey.engine import evaluate, make_loader, pick_device
from jersey.metrics import (
    band_accuracy, confusions, count_flips, group_bootstrap, heldout_error_share, majority_baseline, match_of,
    mcnemar_exact, per_number, scores, shot_bands, to_numbers, track_of, wilson_interval,
)
from jersey.model import load_run, predict_digits
from train import APPROACHES

METHODS = ("sequence", "anchor", "vote")
SCORE_COLUMN = {"sequence": "sequence_confidence", "anchor": "anchor_confidence", "vote": "vote_agreement"}
SEPARATOR = "|"  # between per-frame answers in one CSV cell
PREFIX_FRAMES = 5  # accuracy after k frames follows the sequences with at least this many, the same for every k
HOLD_OUT_SPLITS = {"val": ("train", "val"), "test": ("test",)}  # where held-out sequences come from, never pooled
SPEED_PASSES = 50
ATTENTION_FRAMES = 6
FEW_FRAMES = 4  # --analyse: accuracy and speed with this many frames instead of max_frames


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="run folders under results/runs; none: only remake the comparison")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--verify", type=Path, help="folder with best.pt, splits.json and predictions_test.csv")
    parser.add_argument("--analyse", metavar="RUN", help="run folder under results/runs to analyse")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--device", default="auto", help="auto picks cuda, then mps, then cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args(argv)


# ----- predictions -----

@torch.no_grad()
def prefix_answers(model, frames, mask):
    """The answer after the first 1, 2, ..., T frames: digits (B, T, 2). Past the last real frame it repeats."""
    steps = torch.arange(mask.shape[1], device=mask.device)
    return torch.stack([predict_digits(model(frames, mask & (steps < k))["logits"])[0]
                        for k in range(1, mask.shape[1] + 1)], dim=1)


def vote(readings, confidences):
    """Confidence-weighted vote: the number with the largest summed confidence, and its share of the total."""
    totals = defaultdict(float)
    for number, confidence in zip(readings, confidences):
        totals[number] += confidence
    winner = max(totals, key=totals.get)
    return winner, totals[winner] / sum(totals.values())


def predict(model, loader, device):
    """One row per sequence with every method's answer, and the sequence and anchor losses."""
    sequence, anchor = evaluate(model, loader, device), evaluate(model, loader, device, use_anchor=True)
    frame_rows = []
    with torch.no_grad():
        for batch in loader:
            frames, mask = batch["frames"].to(device), batch["mask"].to(device)
            frame_digits, frame_confidence = predict_digits(model(frames, mask)["frame_logits"])
            prefixes = prefix_answers(model, frames, mask).cpu()
            for i, count in enumerate(batch["mask"].sum(dim=1).tolist()):
                frame_rows.append({"readings": to_numbers(frame_digits[i, :count]),
                                   "confidences": frame_confidence[i, :count].tolist(),
                                   "prefix": to_numbers(prefixes[i, :count]), "frames": count})
    rows = []
    for frame_row, sequence_id, true, answer, confidence, anchor_answer, anchor_confidence in zip(
            frame_rows, sequence["sequence_ids"], to_numbers(sequence["labels"]), to_numbers(sequence["predictions"]),
            sequence["confidences"].tolist(), to_numbers(anchor["predictions"]), anchor["confidences"].tolist()):
        voted, agreement = vote(frame_row["readings"], frame_row["confidences"])
        rows.append({"sequence_id": sequence_id, "true": true, "frames": frame_row["frames"], "sequence": answer,
                     "sequence_confidence": round(confidence, 4), "anchor": anchor_answer,
                     "anchor_confidence": round(anchor_confidence, 4), "vote": voted,
                     "vote_agreement": round(agreement, 4), "readings": SEPARATOR.join(frame_row["readings"]),
                     "prefix": SEPARATOR.join(frame_row["prefix"])})
    return rows, {"sequence": sequence["loss"], "anchor": anchor["loss"]}


# ----- scores -----

def score(rows, train_numbers):
    """Prediction rows -> metrics, the same way for every method."""
    labels, bands = _digits(row["true"] for row in rows), shot_bands(train_numbers)
    methods = {}
    for method in METHODS:
        predictions = _digits(row[method] for row in rows)
        methods[method] = {**scores({"loss": None, "predictions": predictions, "labels": labels}),
                           "mean_" + SCORE_COLUMN[method].split("_")[1]: _mean(float(r[SCORE_COLUMN[method]]) for r in rows),
                           "per_number": per_number(predictions, labels),
                           "groups": band_accuracy(predictions, labels, bands),
                           "confusions": confusions(predictions, labels)}
    return {"sequences": len(rows), "baseline": majority_baseline(labels, train_numbers), "methods": methods,
            "stability": stability(rows)}


def stability(rows):
    """How often the answer changes as frames arrive (prefix), and accuracy after k frames on one fixed group."""
    prefixes = [row["prefix"].split(SEPARATOR) for row in rows]
    frames = min(PREFIX_FRAMES, max(map(len, prefixes)))
    group = [(answers, row["true"]) for answers, row in zip(prefixes, rows) if len(answers) >= frames]
    return {"prefix": _flips(prefixes), "frame_readings": _flips([row["readings"].split(SEPARATOR) for row in rows]),
            "accuracy_by_frames": {k: {"count": len(group), "accuracy": _mean(float(a[k - 1] == t) for a, t in group)}
                                   for k in range(1, frames + 1)}}


def hold_out(rows, train_numbers):
    """Scores on numbers the run never trained on, and how its answers fail."""
    labels, predictions = _digits(row["true"] for row in rows), _digits(row["sequence"] for row in rows)
    return {"sequences": len(rows), "accuracy": _mean(float(row["sequence"] == row["true"]) for row in rows),
            "one_digit_share": _mean(float(len(row["sequence"]) == 1) for row in rows),
            "first_digit_hits": (predictions[:, 0] == labels[:, 0]).float().mean().item(),
            "second_digit_hits": (predictions[:, 1] == labels[:, 1]).float().mean().item(),
            "answer_counts": dict(Counter(row["sequence"] for row in rows).most_common()),
            "per_number": per_number(predictions, labels),
            "wrong_answers_that_are_training_numbers": heldout_error_share(predictions, labels, train_numbers)}


def speed(model, run_dir, max_frames):
    """Parameters, weights size in MB, and the median ms per sequence of max_frames frames on one CPU thread."""
    ms = cpu_ms(model, max_frames)
    return {"parameters": sum(p.numel() for p in model.parameters()),
            "weights_mb": round((run_dir / "best.pt").stat().st_size / 2 ** 20, 2),
            "ms_per_sequence_cpu_1_thread": ms, "ms_per_frame_cpu_1_thread": round(ms / max_frames, 2),
            "frames": max_frames}


@torch.no_grad()
def cpu_ms(model, frames_per_sequence):
    """Median ms per sequence of frames_per_sequence frames on one CPU thread (moves the model to the CPU)."""
    model = model.to("cpu").eval()
    frames = torch.randn(1, frames_per_sequence, 3, *IMAGE_SIZE)
    mask = torch.ones(1, frames_per_sequence, dtype=torch.bool)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for _ in range(5):  # warm-up
            model(frames, mask)
        times = []
        for _ in range(SPEED_PASSES):
            start = time.perf_counter()
            model(frames, mask)
            times.append(1000 * (time.perf_counter() - start))
    finally:
        torch.set_num_threads(threads)
    return round(statistics.median(times), 2)


# ----- shortcut checks: does the model read the digits, or lean on colour and context? -----
# Per position, with the input changed: grayscale (no colour), torso covered (the number area
# greyed out: what stays above "always <most common number>" is read from context), only torso
# (everything else greyed out: should stay near unchanged). Two baselines cannot read digits:
# always the most common training number, and a linear classifier on the colour histogram of the
# whole crop. A grey box is unusual input, so the covered rows are best compared between runs.

GREY = 0.5
HISTOGRAM_BINS = 8  # per channel, so 512 colours
NORMALIZE = v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def cover(images, torso):
    """Grey out the torso (torso=True) or everything else: images (N, 3, H, W) in [0, 1]."""
    mask = torso_mask(*images.shape[-2:])
    covered = images.clone()
    covered[..., mask if torso else ~mask] = GREY
    return covered


PERTURBATIONS = {  # images (N, 3, H, W) in [0, 1] -> same shape and range
    "unchanged": lambda images: images,
    "grayscale": lambda images: v2.functional.rgb_to_grayscale(images, num_output_channels=3),
    "torso covered": lambda images: cover(images, torso=True),
    "only torso": lambda images: cover(images, torso=False),
}


def position_accuracy(predictions, labels):
    """Share right per digit position and for the whole number: predictions and labels (S, 2)."""
    correct = predictions == labels
    return {"first_digit": correct[:, 0].float().mean().item(), "second_digit": correct[:, 1].float().mean().item(),
            "number": correct.all(dim=1).float().mean().item()}


@torch.no_grad()
def perturbed_predictions(model, loader, device):
    """{perturbation: predicted digits (S, 2)} and the labels (S, 2). Only real frames are changed."""
    model.eval()
    predictions, labels = {name: [] for name in PERTURBATIONS}, []
    for batch in loader:
        frames, mask = batch["frames"], batch["mask"]
        real = denormalize(frames[mask])
        for name, perturb in PERTURBATIONS.items():
            changed = frames.clone()
            changed[mask] = NORMALIZE(perturb(real))
            predictions[name].append(predict_digits(model(changed.to(device), mask.to(device))["logits"])[0].cpu())
        labels.append(batch["label"])
    return {name: torch.cat(found) for name, found in predictions.items()}, torch.cat(labels)


def colour_histograms(loader):
    """Colour histogram of each sequence's real frames, summing to 1: (S, 512), and labels (S, 2)."""
    histograms, labels = [], []
    for batch in loader:
        levels = (denormalize(batch["frames"]) * (HISTOGRAM_BINS - 1)).round().long()
        colours = (levels[:, :, 0] * HISTOGRAM_BINS + levels[:, :, 1]) * HISTOGRAM_BINS + levels[:, :, 2]
        for sequence_colours, mask in zip(colours, batch["mask"]):
            counts = torch.bincount(sequence_colours[mask].flatten(), minlength=HISTOGRAM_BINS ** 3).float()
            histograms.append(counts / counts.sum())
        labels.append(batch["label"])
    return torch.stack(histograms), torch.cat(labels)


def colour_baseline(train_loader, val_loader, steps=500, seed=0):
    """Fit a linear classifier on train colour histograms and return its val predictions (S, 2)."""
    generator = torch.Generator().manual_seed(seed)
    train_x, train_labels = colour_histograms(train_loader)
    val_x, _ = colour_histograms(val_loader)
    classes = sorted({tuple(label) for label in train_labels.tolist()})
    targets = torch.tensor([classes.index(tuple(label)) for label in train_labels.tolist()])
    mean, std = train_x.mean(dim=0), train_x.std(dim=0, correction=0) + 1e-6
    linear = torch.nn.Linear(train_x.shape[1], len(classes))
    with torch.no_grad():
        linear.weight.normal_(0, 0.01, generator=generator)
        linear.bias.zero_()
    optimizer = torch.optim.Adam(linear.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(steps):  # full batch: the histograms are small
        optimizer.zero_grad()
        F.cross_entropy(linear((train_x - mean) / std), targets).backward()
        optimizer.step()
    with torch.no_grad():
        return torch.tensor(classes)[linear((val_x - mean) / std).argmax(dim=1)]


def shortcut_report(model, train_loader, val_loader, train_ids, device):
    """The model under every perturbation and the two baselines, per position."""
    predictions, labels = perturbed_predictions(model, val_loader, device)
    most_common = Counter(number_of(sequence_id) for sequence_id in train_ids).most_common(1)[0][0]
    majority = torch.tensor([encode_label(most_common)] * len(labels))
    return {
        "sequences": len(labels),
        "baselines": {f"always {most_common}": position_accuracy(majority, labels),
                      "colour histogram": position_accuracy(colour_baseline(train_loader, val_loader), labels)},
        "model": {name: position_accuracy(found, labels) for name, found in predictions.items()},
    }


# ----- one run -----

def evaluate_run(args, run_dir, device):
    metrics_path = run_dir / f"metrics_{args.split}.json"
    if args.split == "test" and metrics_path.exists():
        raise FileExistsError(f"{metrics_path} exists: the test split is scored only once per run")
    model, config, epoch = load_run(run_dir, device)
    splits = read_splits(run_dir / "splits.json")
    exclude = tuple(config["exclude"])
    train_ids = _train_ids(config, splits)
    train_numbers = [number_of(s) for s in train_ids]

    def loader(ids):
        return _loader(config, ids, args, device)

    def scored(model):
        rows, losses = predict(model, loader(filter_numbers(splits[args.split], exclude)), device)
        found = score(rows, train_numbers)
        for method, loss in losses.items():
            found["methods"][method]["loss"] = loss
        held = sorted(s for name in HOLD_OUT_SPLITS[args.split] for s in splits[name] if number_of(s) in exclude)
        held_rows = predict(model, loader(held), device)[0] if held else []
        return rows, held_rows, found, ({"splits": list(HOLD_OUT_SPLITS[args.split]), **hold_out(held_rows, train_numbers)}
                                        if held_rows else None)

    rows, held_rows, metrics, held_metrics = scored(model)
    metrics = {"run": {"name": run_dir.name, "split": args.split, "epoch": epoch,
                       **{k: config[k] for k in ("approach", "reader", "temporal", "max_frames", "exclude", "count_offset")}},
               **metrics, "hold_out": held_metrics}
    if config["count_offset"]:
        offset, model.count_offset = model.count_offset, 0.0
        _, _, plain, plain_held = scored(model)
        model.count_offset = offset
        metrics["without_count_offset"] = {"methods": {"sequence": {k: v for k, v in plain["methods"]["sequence"].items()
                                                                    if k in ("accuracy", "one_digit_accuracy",
                                                                             "two_digit_accuracy", "macro_accuracy_5")}},
                                           "hold_out_accuracy": plain_held and plain_held["accuracy"]}
    _write_rows(run_dir / f"predictions_{args.split}.csv", rows)
    if held_rows:
        _write_rows(run_dir / f"predictions_holdout_{args.split}.csv", held_rows)
    if args.split == "val":
        val_loader = make_loader(JerseyDataset(config["data"], filter_numbers(splits["val"], exclude),
                                               max_frames=config["max_frames"]), args.batch_size, args.workers, device)
        train_loader = make_loader(JerseyDataset(config["data"], train_ids, max_frames=config["max_frames"]),
                                   args.batch_size, args.workers, device)
        metrics["shortcuts"] = shortcut_report(model, train_loader, val_loader, train_ids, device)
        save_attention(model, val_loader.dataset, run_dir / "attention.png", device)
        metrics["speed"] = speed(model, run_dir, config["max_frames"])
    metrics_path.write_text(json.dumps(metrics, indent=2))
    _print(metrics)


def _train_ids(config, splits):
    """The train sequences the run trained on: held-out numbers and their linked sequences left out."""
    dropped = set(config["linked_dropped"])
    return [s for s in filter_numbers(splits["train"], tuple(config["exclude"])) if s not in dropped]


def _loader(config, ids, args, device, max_frames=None, anchor=True):
    return make_loader(JerseyDataset(config["data"], ids, load_anchor=anchor, max_frames=max_frames or config["max_frames"]),
                       args.batch_size, args.workers, device)


# ----- the shipped weights, re-scored -----

def verify(args, folder, device):
    """Re-score the weights in folder on test; compare every answer with the recorded predictions_test.csv."""
    model, config, _ = load_run(folder, device)
    splits = read_splits(folder / "splits.json")
    rows, _ = predict(model, _loader(config, filter_numbers(splits["test"], tuple(config["exclude"])), args, device), device)
    found = score(rows, [number_of(s) for s in _train_ids(config, splits)])["methods"]["sequence"]
    recorded_path = folder / "predictions_test.csv"
    recorded = {row["sequence_id"]: row["sequence"] for row in _read_rows(recorded_path)} if recorded_path.exists() else {}
    same = sum(recorded.get(row["sequence_id"]) == row["sequence"] for row in rows)
    result = {"sequences": len(rows), "accuracy": found["accuracy"], "macro_accuracy_5": found["macro_accuracy_5"],
              "recorded": str(recorded_path) if recorded else None, "same_answers": same if recorded else None,
              "device": str(device), "torch": str(torch.__version__)}
    (folder / "verify_test.json").write_text(json.dumps(result, indent=2))
    print(f"{folder} on test: accuracy {found['accuracy']:.4f} of {len(rows)}"
          + (f"; same answer as recorded on {same}/{len(rows)}" if recorded else "; no recorded predictions to compare"))
    return result


# ----- extra evidence for the report (--analyse) -----

def analyse(args, run_dir, device):
    """Evidence beyond the per-run metrics, in results/compare/analysis_<run>.json.

    From the recorded predictions (the test split is not scored again):
      match_overlap   test accuracy for sequences whose match is, or is not, in the run's train split
      paired_tests    exact McNemar tests: sequence vs anchor, vs vote, first frame vs anchor
      track_bootstrap 95% intervals with whole tracks (player folders) resampled
      hold_out        every held-out run's development hold-out per origin split, and its test hold-out
    From new passes over val:
      gate            mean relative weight of frames read right vs wrong, per digit position
      few_frames      val accuracy and CPU speed with FEW_FRAMES frames
      int8            dynamic int8 quantisation of the linear layers: size, CPU speed, val accuracy
      head_drift      (parseq) cosine similarity per output row between fine-tuned and released weights
    """
    model, config, _ = load_run(run_dir, device)
    splits = read_splits(run_dir / "splits.json")
    train_ids = _train_ids(config, splits)
    result = {"run": run_dir.name}
    test_path = run_dir / "predictions_test.csv"
    if test_path.exists():
        rows = _read_rows(test_path)
        result["match_overlap"] = match_overlap(rows, train_ids)
        result["paired_tests"] = paired_tests(rows)
        result["track_bootstrap"] = {"test_accuracy": _bootstrap(rows)}
    result["hold_out"] = {d.name: hold_out_groups(d) for d in sorted((args.results / "runs").iterdir())
                          if (d / "predictions_holdout_val.csv").exists()}
    val_ids = filter_numbers(splits["val"], tuple(config["exclude"]))
    val_loader = _loader(config, val_ids, args, device, anchor=False)
    result["gate"] = gate_weights(model, val_loader, device)
    few_loader = _loader(config, val_ids, args, device, max_frames=FEW_FRAMES, anchor=False)
    result["few_frames"] = {
        str(frames): {"val_accuracy": scores(evaluate(model.to(device), found_loader, device))["accuracy"],
                      "ms_per_sequence_cpu_1_thread": cpu_ms(model, frames)}
        for frames, found_loader in ((FEW_FRAMES, few_loader), (config["max_frames"], val_loader))}
    result["int8"] = int8(model, config, val_ids, args)
    if config["reader"] == "parseq":
        result["head_drift"] = head_drift(model)
    out = args.results / "compare" / f"analysis_{run_dir.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"analysis in {out}")
    return result


def _group(flags):
    flags = list(flags)
    low, high = wilson_interval(sum(flags), len(flags)) if flags else (math.nan, math.nan)
    return {"count": len(flags), "correct": sum(flags), "accuracy": _mean(flags), "low": low, "high": high}


def _right(rows, method="sequence"):
    return [row[method] == row["true"] for row in rows]


def _bootstrap(rows, method="sequence"):
    low, high = group_bootstrap(_right(rows, method), [track_of(row["sequence_id"]) for row in rows])
    return {"accuracy": _mean(_right(rows, method)), "low": low, "high": high,
            "tracks": len({track_of(row["sequence_id"]) for row in rows}), "sequences": len(rows)}


def match_overlap(rows, train_ids):
    """Accuracy by whether the sequence's match (and its number in that match) appears in train."""
    matches = {match_of(s) for s in train_ids}
    numbers_in_match = {(match_of(s), number_of(s)) for s in train_ids}

    def group(row):
        match = match_of(row["sequence_id"])
        if match not in matches:
            return "match not in train"
        return "same match and number in train" if (match, row["true"]) in numbers_in_match else "match in train, other number"

    groups = defaultdict(list)
    for row in rows:
        groups[group(row)].append(row["sequence"] == row["true"])
    return {"all": _group(_right(rows)), **{name: _group(flags) for name, flags in sorted(groups.items())}}


def paired_tests(rows):
    """Exact McNemar: of the sequences where exactly one method is right, how many each gets."""
    first = [{**row, "first_frame": row["readings"].split(SEPARATOR)[0]} for row in rows]
    result = {"first_frame_accuracy": _mean(_right(first, "first_frame"))}
    for a, b in (("sequence", "anchor"), ("sequence", "vote"), ("first_frame", "anchor")):
        only_a = sum(x and not y for x, y in zip(_right(first, a), _right(first, b)))
        only_b = sum(y and not x for x, y in zip(_right(first, a), _right(first, b)))
        result[f"{a} vs {b}"] = {f"only {a} right": only_a, f"only {b} right": only_b,
                                 "p_exact": mcnemar_exact(only_a, only_b)}
    return result


def hold_out_groups(run_dir):
    """A held-out run's hold-out, apart per origin split (train / val players; test players), with intervals."""
    splits = read_splits(run_dir / "splits.json")
    origin = {s: name for name, ids in splits.items() for s in ids}
    result = {}
    for split in ("val", "test"):
        path = run_dir / f"predictions_holdout_{split}.csv"
        if not path.exists():
            continue
        rows = _read_rows(path)
        for name in HOLD_OUT_SPLITS[split]:
            result[f"{name} players"] = _group(_right([r for r in rows if origin[r["sequence_id"]] == name]))
        result[f"{split} pooled, track bootstrap"] = _bootstrap(rows)
    return result


@torch.no_grad()
def gate_weights(model, loader, device):
    """Per digit position: mean weight x frames (1 = uniform) of frames read right and wrong, in sequences
    where that position has both right and wrong frames; None for a reader that averages its frames."""
    model.to(device).eval()
    found = {position: {"right": [], "wrong": [], "sequences": 0} for position in (0, 1)}
    for batch in loader:
        mask = batch["mask"].to(device)
        out = model(batch["frames"].to(device), mask)
        if out["frame_weights"] is None:
            return None
        relative = (out["frame_weights"] * mask.sum(dim=1)[:, None, None]).cpu()  # (B, T, Q)
        digits = out["frame_logits"].argmax(dim=-1).cpu()  # (B, T, 2)
        for position in (0, 1):
            right = digits[..., position] == batch["label"][:, None, position]
            for i, count in enumerate(batch["mask"].sum(dim=1).tolist()):
                hits = right[i, :count]
                if hits.any() and not hits.all():
                    found[position]["sequences"] += 1
                    found[position]["right"] += relative[i, :count, position][hits].tolist()
                    found[position]["wrong"] += relative[i, :count, position][~hits].tolist()
    return {f"position {p + 1}": {"sequences_with_both": v["sequences"], "frames_right": len(v["right"]),
                                  "frames_wrong": len(v["wrong"]), "mean_weight_right": _mean(v["right"]),
                                  "mean_weight_wrong": _mean(v["wrong"])}
            for p, v in found.items()}


def int8(model, config, val_ids, args):
    """Dynamic int8 quantisation of every nn.Linear (the ViT is mostly linear layers), on the CPU."""
    if "qnnpack" in torch.backends.quantized.supported_engines:
        torch.backends.quantized.engine = "qnnpack"
    with warnings.catch_warnings():  # torch.ao.quantization is deprecated in favour of torchao, but still works
        warnings.simplefilter("ignore")
        quantized = torch.ao.quantization.quantize_dynamic(copy.deepcopy(model).to("cpu").eval(), {nn.Linear},
                                                           dtype=torch.qint8)
    buffer = io.BytesIO()
    torch.save(quantized.state_dict(), buffer)
    loader = _loader(config, val_ids, args, "cpu", anchor=False)
    return {"weights_mb": round(buffer.tell() / 2 ** 20, 2), "val_accuracy": scores(evaluate(quantized, loader, "cpu"))["accuracy"],
            "ms_per_sequence_cpu_1_thread": cpu_ms(quantized, config["max_frames"]),
            "fp32_ms_per_sequence_cpu_1_thread": cpu_ms(model, config["max_frames"])}


def head_drift(model):
    """Cosine similarity per output row (end, digits 0-9) between the fine-tuned and the released PARSeq head."""
    released = parseq.digit_weights()[1]["head.weight"]
    tuned = model.reader.text.head.weight.detach().cpu()
    similarity = F.cosine_similarity(tuned, released, dim=1).tolist()
    return {name: round(value, 4) for name, value in zip(["end", *map(str, range(10))], similarity)}


@torch.no_grad()
def save_attention(model, dataset, path, device):
    """Where each query looks on the first frame of a spread of val sequences."""
    model.eval()
    picks = torch.linspace(0, len(dataset) - 1, min(ATTENTION_FRAMES, len(dataset))).round().long().tolist()
    items = [dataset[i] for i in picks]
    images = torch.stack([item["frames"][0] for item in items])
    out = model.read(images.to(device))
    answers = to_numbers(predict_digits(out["logits"])[0].cpu())
    titles = [f"{number_of(item['sequence_id'])} read {answer}" for item, answer in zip(items, answers)]
    plots.attention(images, plots.cell_grid(out["attention"].cpu(), model.reader.attention_grid), titles, path)


def _print(metrics):
    run, found = metrics["run"], metrics["methods"]
    print(f"\n{run['name']} on {run['split']} ({metrics['sequences']} sequences, epoch {run['epoch']})")
    for method in METHODS:
        m = found[method]
        print(f"  {method:<8} accuracy {m['accuracy']:.3f}  one digit {m['one_digit_accuracy']:.3f}"
              f"  two digit {m['two_digit_accuracy']:.3f}  macro (>=5) {m['macro_accuracy_5']:.3f}")
    if metrics["hold_out"]:
        held = metrics["hold_out"]
        print(f"  hold-out {run['exclude']}: {held['accuracy']:.3f} of {held['sequences']}; answers {held['answer_counts']}")
    if "without_count_offset" in metrics:
        print(f"  without the count offset ({run['count_offset']:.2f}): accuracy "
              f"{metrics['without_count_offset']['methods']['sequence']['accuracy']:.3f}")


# ----- comparison across runs -----

def compare(results):
    """Table and graphs over every run in results/runs that has metrics_val.json."""
    runs = sorted(p.parent for p in (results / "runs").glob("*/metrics_val.json"))
    if not runs:
        return
    out = results / "compare"
    out.mkdir(parents=True, exist_ok=True)
    rows, val, segments, answers = [], {}, {}, {}
    for run_dir in runs:
        val[run_dir.name] = metrics = json.loads((run_dir / "metrics_val.json").read_text())
        segments[run_dir.name] = _stage_histories(run_dir)
        if metrics["hold_out"]:
            answers[run_dir.name] = {"counts": metrics["hold_out"]["answer_counts"], "sequences": metrics["hold_out"]["sequences"]}
        rows.append(_table_row(run_dir, metrics))
        if run_dir.name.endswith("_all"):  # the final model's train and val curves (stage switch marked if two stages)
            history = [entry for segment in segments[run_dir.name] for entry in segment]
            plots.curves(history, out / f"curves_{run_dir.name}.png",
                         stage_switch=segments[run_dir.name][-1][0]["step"] if len(segments[run_dir.name]) > 1 else None)
    with (out / "table.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ladder = ladder_runs([run_dir.name for run_dir in runs])  # the graphs: one run per approach, same val set
    plots.compare_curves({name: segments[name] for name in ladder}, out / "curves.png")
    plots.temporal({name: val[name] for name in ladder}, out / "temporal.png")
    if any(name in answers for name in ladder):
        plots.holdout({name: answers[name] for name in ladder if name in answers}, out / "holdout.png")
    print(f"comparison of {len(runs)} runs in {out}; graphs of {', '.join(ladder)}")


def ladder_runs(names):
    """One held-out run per approach, in APPROACHES order: seed 0 with pretrained weights, or the scratch one if
    there is no pretrained one. Other seeds, scratch twins and runs on all numbers are only in table.csv."""
    found = []
    for approach in APPROACHES:
        pattern = re.compile(rf"{approach}_ex[0-9_]+(_scratch)?")
        matches = sorted((name for name in names if pattern.fullmatch(name)), key=lambda name: name.endswith("_scratch"))
        found += matches[:1]
    return found


def _stage_histories(run_dir):
    """Each stage's history entries, stage 1 first: [[...], [...]] or [[...]]."""
    parts = [run_dir / "stage1" / "history.json", run_dir / "history.json"]
    return [json.loads(part.read_text())["epochs"] for part in parts if part.exists()]


def _table_row(run_dir, metrics):
    sequence, test = metrics["methods"]["sequence"], run_dir / "metrics_test.json"
    best = json.loads((run_dir / "history.json").read_text())["best"]
    config = json.loads((run_dir / "config.json").read_text())
    row = {"run": run_dir.name, **{k: metrics["run"][k] for k in ("approach", "reader", "temporal", "max_frames")},
           "seed": config["seed"], "pretrained": config.get("pretrained", True),
           "git_commit": config.get("git", {}).get("commit", "")[:7],
           **{k: metrics.get("speed", {}).get(k, "") for k in ("parameters", "weights_mb", "ms_per_sequence_cpu_1_thread")},
           "val_accuracy": sequence["accuracy"], "val_one_digit": sequence["one_digit_accuracy"],
           "val_two_digit": sequence["two_digit_accuracy"], "val_macro_5": sequence["macro_accuracy_5"],
           "anchor_accuracy": metrics["methods"]["anchor"]["accuracy"], "vote_accuracy": metrics["methods"]["vote"]["accuracy"],
           "stable_share": metrics["stability"]["prefix"]["stable_share"],
           "holdout_accuracy": metrics["hold_out"]["accuracy"] if metrics["hold_out"] else "",
           "pair_accuracy": best.get("pair_accuracy", ""),
           "val_accuracy_no_offset": metrics.get("without_count_offset", {}).get("methods", {}).get("sequence", {}).get("accuracy", ""),
           "test_accuracy": "", "test_holdout_accuracy": ""}
    if test.exists():
        found = json.loads(test.read_text())
        row["test_accuracy"] = found["methods"]["sequence"]["accuracy"]
        row["test_holdout_accuracy"] = found["hold_out"]["accuracy"] if found["hold_out"] else ""
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}


# ----- small helpers -----

def _read_rows(path):
    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def _write_rows(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _flips(answer_lists):
    """count_flips on per-sequence answer lists of different lengths."""
    length = max(map(len, answer_lists))
    padded = [answers + answers[-1:] * (length - len(answers)) for answers in answer_lists]
    return count_flips(torch.stack([_digits(a) for a in padded]), torch.tensor([len(a) for a in answer_lists]))


def _digits(numbers):
    return torch.tensor([encode_label(number) for number in numbers])


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else math.nan


def main(argv=None):
    args = parse_args(argv)
    device = pick_device(args.device)
    if args.verify:
        return verify(args, args.verify, device)
    if args.analyse:
        return analyse(args, args.results / "runs" / args.analyse, device)
    for name in args.names:
        evaluate_run(args, args.results / "runs" / name, device)
    compare(args.results)


if __name__ == "__main__":
    main()
