"""Check whether runs read the digits or lean on colour and context, on val.
Example: uv run check_shortcuts.py averaged_basic averaged_strong

Scores each run's sequence model on val with the input changed, per digit position:
    unchanged      the input as evaluate.py sees it
    grayscale      no colour
    hue shift      every colour turned half way round the colour wheel
    channel swap   RGB -> BRG
    torso covered  the number area greyed out: what stays above "always <number>" is read
                   from context (kit, pitch, person), not from the digits
    only torso     everything but the number area greyed out: should stay near unchanged
and two baselines that cannot read digits: always the most common training number, and a
linear classifier on the colour histogram of the whole crop (kit, pitch and light, not only
the kit). Colour well above the majority baseline means the shortcut is there to take. The
second-digit baselines are high, since most numbers have one digit and so a BLANK second digit.

Read the rows by what the run was trained on. The strong augmentation (data.py) trains on hue
shifts, channel swaps and grayscale, so for those runs these three rows are a sanity check,
not evidence that the model reads digits. Torso covered, and the --exclude hold-out score of
evaluate.py, are the independent tests. A grey box is unusual input, so part of the drop in
the two covered rows comes from the box itself; they are best compared between runs.

Writes to results/runs/<run>/:
    shortcuts_val.json  the scores above
    shortcuts_val.png   val frames unchanged, torso covered and only torso, to check the box by eye
With --count-offset prior or none (two-step runs, see evaluate.py) the files get that suffix.
"""

from collections import Counter
from pathlib import Path
import argparse
import json

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision.utils import make_grid, save_image

from jersey.data import (
    IMAGENET_MEAN, IMAGENET_STD, TORSO_COLUMNS, TORSO_ROWS, JerseyDataset, denormalize, encode_label,
    filter_numbers, run_splits, torso_mask,
)
from jersey.model import load_run, predict_digits
from evaluate import COUNT_OFFSETS, set_count_offset
from train import pick_device

GREY = 0.5
HISTOGRAM_BINS = 8  # per channel, so 512 colours
PICTURE_COUNT = 12
NORMALIZE = v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="+", help="run folders under --runs, e.g. averaged_strong")
    parser.add_argument("--runs", type=Path, default=Path("results/runs"))
    parser.add_argument("--device", default="auto", help="auto picks cuda, then mps, then cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--count-offset", choices=COUNT_OFFSETS, default="saved",
                        help="two-step runs: saved (as trained), prior (from the training numbers) or none")
    return parser.parse_args(argv)


def cover(images, torso):
    """Grey out the torso (torso=True) or everything else: images (N, 3, H, W) in [0, 1]."""
    mask = torso_mask(*images.shape[-2:])
    covered = images.clone()
    covered[..., mask if torso else ~mask] = GREY
    return covered


# Each takes images (N, 3, H, W) in [0, 1] and returns them changed, same shape and range.
PERTURBATIONS = {
    "unchanged": lambda images: images,
    "grayscale": lambda images: v2.functional.rgb_to_grayscale(images, num_output_channels=3),
    "hue shift": lambda images: v2.functional.adjust_hue(images, 0.5),
    "channel swap": lambda images: images[..., [2, 0, 1], :, :],
    "torso covered": lambda images: cover(images, torso=True),
    "only torso": lambda images: cover(images, torso=False),
}


def position_accuracy(predictions, labels):
    """Share right per digit position and for the whole number: predictions and labels (S, 2)."""
    correct = predictions == labels
    return {
        "first_digit": correct[:, 0].float().mean().item(),
        "second_digit": correct[:, 1].float().mean().item(),
        "number": correct.all(dim=1).float().mean().item(),
    }


@torch.no_grad()
def perturbed_predictions(model, loader, device):
    """Return {perturbation: predicted digits (S, 2)} and the labels (S, 2), in loader order.

    Only the real frames are changed, so padding stays zero as in training.
    """
    predictions = {name: [] for name in PERTURBATIONS}
    labels = []
    for batch in loader:
        frames, mask = batch["frames"], batch["mask"]
        real = denormalize(frames[mask])  # (R, 3, H, W) in [0, 1]
        for name, perturb in PERTURBATIONS.items():
            changed = frames.clone()
            changed[mask] = NORMALIZE(perturb(real))
            logits = model(changed.to(device), mask.to(device))["logits"]
            predictions[name].append(predict_digits(logits)[0].cpu())
        labels.append(batch["label"])
    return {name: torch.cat(found) for name, found in predictions.items()}, torch.cat(labels)


def colour_histograms(loader):
    """Colour histogram of each sequence's real frames, summing to 1: (S, HISTOGRAM_BINS ** 3), labels (S, 2)."""
    histograms, labels = [], []
    for batch in loader:
        levels = (denormalize(batch["frames"]) * (HISTOGRAM_BINS - 1)).round().long()  # (B, T, 3, H, W)
        colours = (levels[:, :, 0] * HISTOGRAM_BINS + levels[:, :, 1]) * HISTOGRAM_BINS + levels[:, :, 2]
        for sequence_colours, mask in zip(colours, batch["mask"]):
            counts = torch.bincount(sequence_colours[mask].flatten(), minlength=HISTOGRAM_BINS ** 3).float()
            histograms.append(counts / counts.sum())
        labels.append(batch["label"])
    return torch.stack(histograms), torch.cat(labels)


def colour_baseline(train_loader, val_loader, steps=500, seed=0):
    """Fit a linear classifier on train colour histograms and return its val predictions (S, 2).

    It sees colours only, never shapes, so what it gets right shows how far colour alone goes.
    """
    torch.manual_seed(seed)
    train_x, train_labels = colour_histograms(train_loader)
    val_x, _ = colour_histograms(val_loader)
    classes = sorted({tuple(label) for label in train_labels.tolist()})
    targets = torch.tensor([classes.index(tuple(label)) for label in train_labels.tolist()])
    mean, std = train_x.mean(dim=0), train_x.std(dim=0, correction=0) + 1e-6
    linear = torch.nn.Linear(train_x.shape[1], len(classes))
    optimizer = torch.optim.Adam(linear.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(steps):  # full batch: the histograms are small
        optimizer.zero_grad()
        F.cross_entropy(linear((train_x - mean) / std), targets).backward()
        optimizer.step()
    with torch.no_grad():
        picked = linear((val_x - mean) / std).argmax(dim=1)
    return torch.tensor(classes)[picked]


def save_torso_picture(dataset, path):
    """First frame of a spread of val sequences: unchanged, torso covered, only torso."""
    picks = torch.linspace(0, len(dataset) - 1, min(PICTURE_COUNT, len(dataset))).round().long().tolist()
    images = denormalize(torch.stack([dataset[i]["frames"][0] for i in picks]))
    rows = [images, cover(images, torso=True), cover(images, torso=False)]
    save_image(make_grid(torch.cat(rows), nrow=len(images)), path)


def check_run(args, run_dir, device):
    model, config, _ = load_run(run_dir, device)
    splits = run_splits(run_dir, config["splits"])
    # As trained: without held-out numbers and their linked partners.
    exclude = tuple(config["exclude"])
    dropped = set(config.get("linked_dropped", []))
    train_ids = [sequence_id for sequence_id in filter_numbers(splits["train"], exclude) if sequence_id not in dropped]
    val_ids = filter_numbers(splits["val"], exclude)
    suffix, name_suffix = set_count_offset(model, args.count_offset, [s.split("/")[0] for s in train_ids])

    def loader(ids):
        dataset = JerseyDataset(config["data"], ids, max_frames=config["max_frames"])
        return DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)

    val_loader = loader(val_ids)
    predictions, labels = perturbed_predictions(model, val_loader, device)
    most_common = Counter(sequence_id.split("/")[0] for sequence_id in train_ids).most_common(1)[0][0]
    majority = torch.tensor([encode_label(most_common)] * len(labels))
    report = {
        "run": run_dir.name + name_suffix,
        "count_offset": round(getattr(model, "count_offset", 0.0), 4),
        "split": "val",
        "sequences": len(labels),
        "torso": {"rows": TORSO_ROWS, "columns": TORSO_COLUMNS},
        "baselines": {
            f"always {most_common}": position_accuracy(majority, labels),
            "colour histogram": position_accuracy(colour_baseline(loader(train_ids), val_loader), labels),
        },
        "model": {name: position_accuracy(found, labels) for name, found in predictions.items()},
    }
    (run_dir / f"shortcuts_val{suffix}.json").write_text(json.dumps(report, indent=2))
    save_torso_picture(val_loader.dataset, run_dir / f"shortcuts_val{suffix}.png")
    _print_table(report)


def _print_table(report):
    print(f"\n{report['run']} on val, {report['sequences']} sequences")
    print(f"{'':<26}{'first':>7}{'second':>8}{'number':>8}")
    rows = [*report["baselines"].items(), *((f"model, {name}", row) for name, row in report["model"].items())]
    for name, row in rows:
        print(f"{name:<26}{row['first_digit']:>7.3f}{row['second_digit']:>8.3f}{row['number']:>8.3f}")


def main(argv=None):
    args = parse_args(argv)
    device = pick_device(args.device)
    for name in args.names:
        check_run(args, args.runs / name, device)


if __name__ == "__main__":
    main()
