"""Score trained runs on val, or once on test, and compare every evaluated run.
Example: uv run evaluate.py two_step_mean_ex49 fused_ex49        (val)
         uv run evaluate.py two_step_mean_all --split test        (test: once per run)

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
"""

from collections import Counter, defaultdict
from pathlib import Path
import argparse
import csv
import json
import math
import statistics
import time

import torch
from torch.nn import functional as F
from torchvision.transforms import v2

from jersey import plots
from jersey.data import (
    IMAGENET_MEAN, IMAGENET_STD, JerseyDataset, denormalize, encode_label, filter_numbers, number_of, read_splits,
    torso_mask,
)
from jersey.engine import evaluate, make_loader, pick_device
from jersey.metrics import (
    band_accuracy, confusions, count_flips, heldout_error_share, majority_baseline, per_number, scores,
    shot_bands, to_numbers,
)
from jersey.model import load_run, predict_digits

METHODS = ("sequence", "anchor", "vote")
SCORE_COLUMN = {"sequence": "sequence_confidence", "anchor": "anchor_confidence", "vote": "vote_agreement"}
SEPARATOR = "|"  # between per-frame answers in one CSV cell
PREFIX_FRAMES = 5  # accuracy after k frames follows the sequences with at least this many, the same for every k
HOLD_OUT_SPLITS = {"val": ("train", "val"), "test": ("test",)}  # where held-out sequences come from, never pooled
SPEED_PASSES = 50
ATTENTION_FRAMES = 6


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="run folders under results/runs; none: only remake the comparison")
    parser.add_argument("--split", choices=("val", "test"), default="val")
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


@torch.no_grad()
def speed(model, run_dir, max_frames):
    """Parameters, weights size in MB, and the median ms per sequence of max_frames frames on one CPU thread."""
    model = model.to("cpu").eval()
    frames, mask = torch.randn(1, max_frames, 3, 192, 96), torch.ones(1, max_frames, dtype=torch.bool)
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
    return {"parameters": sum(p.numel() for p in model.parameters()),
            "weights_mb": round((run_dir / "best.pt").stat().st_size / 2 ** 20, 2),
            "ms_per_sequence_cpu_1_thread": round(statistics.median(times), 2), "frames": max_frames}


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
    exclude, dropped = tuple(config["exclude"]), set(config["linked_dropped"])
    train_ids = [s for s in filter_numbers(splits["train"], exclude) if s not in dropped]
    train_numbers = [number_of(s) for s in train_ids]

    def loader(ids):
        return make_loader(JerseyDataset(config["data"], ids, load_anchor=True, max_frames=config["max_frames"]),
                           args.batch_size, args.workers, device)

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
    rows, val, histories, answers = [], {}, {}, {}
    for run_dir in runs:
        val[run_dir.name] = metrics = json.loads((run_dir / "metrics_val.json").read_text())
        histories[run_dir.name] = history = _histories(run_dir)
        if metrics["hold_out"]:
            answers[run_dir.name] = metrics["hold_out"]["answer_counts"]
        rows.append(_table_row(run_dir, metrics, history))
        if run_dir.name.endswith("_all") and (run_dir / "stage1").exists():
            switch = json.loads((run_dir / "history.json").read_text())["epochs"][0]["step"]
            plots.curves(history, out / f"curves_{run_dir.name}.png", stage_switch=switch)
    with (out / "table.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plots.compare_curves(histories, out / "curves.png")
    plots.temporal(val, out / "temporal.png")
    if answers:
        plots.holdout(answers, out / "holdout.png")
    print(f"comparison of {len(runs)} runs in {out}")


def _histories(run_dir):
    """Both stages' history entries in order (stage 1 first)."""
    parts = [run_dir / "stage1" / "history.json", run_dir / "history.json"]
    return [entry for part in parts if part.exists() for entry in json.loads(part.read_text())["epochs"]]


def _table_row(run_dir, metrics, history):
    sequence, test = metrics["methods"]["sequence"], run_dir / "metrics_test.json"
    best = json.loads((run_dir / "history.json").read_text())["best"]
    row = {"run": run_dir.name, **{k: metrics["run"][k] for k in ("approach", "reader", "temporal", "max_frames")},
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
    for name in args.names:
        evaluate_run(args, args.results / "runs" / name, device)
    compare(args.results)


if __name__ == "__main__":
    main()
