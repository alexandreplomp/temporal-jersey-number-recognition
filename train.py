"""Train one model and keep the epoch with the lowest val loss.
Example: uv run train.py --variant averaged --epochs 2

Writes results/runs/<run-name>/:
    config.json      every setting, plus device, parameter count and sequence counts
    splits.json      the split this run was trained with, for evaluate.py
    train_batch.png  one training batch as the model sees it, anchor in the last column
    history.json     scores per epoch, and the best epoch. Epoch 0 is before training.
    curves.png       loss, accuracy, two-digit and macro accuracy per epoch
    best.pt          {"model": state_dict, "epoch", "config"} of the best epoch
    per_number.json  val scores of the best epoch: per number, per shot band, top confusions
                     and the always-the-most-common-number baseline
    prediction.png   one val sequence per number: where each digit is read, what covering
                     a cell changes, and how much each frame counts (see jersey/explain.py)
    explain.json     the checks behind prediction.png, per case and averaged

Curves come from a fixed, unaugmented subset of train ("train-check") and from val,
since the running training loss uses sampled, augmented batches.

The best epoch is the one with the lowest val loss, not the best macro accuracy: 64 and
88 have 1-2 val sequences each, so macro accuracy jumps by up to 10 points when one of
them flips, and in practice picked epochs with clearly lower accuracy (see metrics.py).
This rule is fixed and saved in config.json as selected_by. Val loss is dominated by the
common one-digit numbers, so val two-digit accuracy is always reported beside it.
"""

from datetime import datetime
from pathlib import Path
import argparse
import json
import math
import shutil
import time

import matplotlib
matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from jersey.data import AUGMENTS, MAX_FRAMES, JerseyDataset, denormalize, filter_numbers, held_out_partners, make_sampler
from jersey.engine import ANCHOR_WEIGHT, evaluate, train_one_epoch
from jersey.explain import explain_run
from jersey.metrics import (
    band_accuracy, confusions, majority_baseline, per_number, scores, shot_bands,
)
from jersey.model import VARIANTS, JerseyModel

# 7.5% of the steps warm up, as in PARSeq.
WARMUP_SHARE = 0.075
# Loss of a uniform guess: 10 classes for the first digit, 11 for the second.
UNIFORM_LOSS = (math.log(10) + math.log(11)) / 2
# The history key that picks the best epoch, lowest wins. Not a setting: fixed for every run.
SELECTED_BY = "val_loss"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    # 10:1 between the new parts and the pretrained backbone, as in DETR.
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--schedule", choices=("constant", "onecycle"), default="constant",
                        help="constant to debug and compare, onecycle (warmup, then decay) for final runs")
    parser.add_argument("--anchor-weight", type=float, default=ANCHOR_WEIGHT)
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES,
                        help="1 gives the single-frame baseline")
    parser.add_argument("--augment", choices=tuple(AUGMENTS), default="strong",
                        help="basic keeps kit colours, strong scrambles colour and adds blur (see data.py)")
    parser.add_argument("--sampler-power", type=float, default=0.5,
                        help="0 keeps the data as it is, 1 balances numbers fully")
    parser.add_argument("--train-check-size", type=int, default=500)
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="numbers to hold out, e.g. 89; train sequences in their linked groups go too")
    parser.add_argument("--issues", type=Path, default=Path("results/data_issues.json"),
                        help="check_data.py's output, read only with --exclude")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto picks cuda, then mps, then cpu")
    parser.add_argument("--no-pretrained", action="store_true", help="start from random weights")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--splits", type=Path, default=Path("results/splits.json"))
    parser.add_argument("--out", type=Path, default=Path("results/runs"))
    parser.add_argument("--run-name", help="default: <variant>_<date-time>")
    return parser.parse_args(argv)


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_batch_grid(batch, path):
    """Save a batch as the model sees it: one sequence per row, the anchor last, padding black."""
    images = torch.cat([batch["frames"], batch["anchor"][:, None]], dim=1)  # (B, T+1, 3, H, W)
    images = denormalize(images)
    images[:, :-1][~batch["mask"]] = 0
    save_image(make_grid(images.flatten(0, 1), nrow=images.shape[1]), path)
    for row, sequence_id in enumerate(batch["sequence_id"]):
        print(f"row {row}: {sequence_id.split('/')[0]:>2}  {sequence_id}")


def plot_curves(history, best_epoch, path):
    """Plot loss, accuracy, two-digit and macro accuracy per epoch, with the best epoch marked."""
    figure, axes = plt.subplots(1, 4, figsize=(20, 4))
    panels = [("loss", ["train", "train_check", "val"]), ("accuracy", ["train_check", "val"]),
              ("two_digit_accuracy", ["train_check", "val"]), ("macro_accuracy", ["train_check", "val"])]
    for axis, (metric, sources) in zip(axes, panels):
        for source in sources:
            key = "train_loss" if source == "train" else f"{source}_{metric}"
            label = "train (running, augmented)" if source == "train" else source
            # The running train loss has no epoch 0, since nothing was trained yet.
            points = [(entry["epoch"], entry[key]) for entry in history if entry[key] is not None]
            axis.plot(*zip(*points), marker=".", label=label)
        axis.axvline(best_epoch, color="grey", linestyle="--", linewidth=1)
        axis.set_title(metric.replace("_", " "))
        axis.set_xlabel("epoch")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=100)
    plt.close(figure)


def make_loader(dataset, args, device, sampler=None):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
    )


def main(argv=None):
    args = parse_args(argv)
    # Covers model init, sampler, augmentation and the train-check pick (see data.py).
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    splits = json.loads(args.splits.read_text())
    exclude = tuple(args.exclude)
    train_ids = filter_numbers(splits["train"], exclude)
    val_ids = filter_numbers(splits["val"], exclude)
    linked = set()
    if exclude:
        everything = [sequence_id for name in ("train", "val", "test") for sequence_id in splits[name]]
        linked = held_out_partners(json.loads(args.issues.read_text()), everything, exclude) & set(train_ids)
        train_ids = [sequence_id for sequence_id in train_ids if sequence_id not in linked]
        print(f"dropped {len(linked)} train sequences linked to held-out numbers: {sorted(linked)}")
    picked = torch.randperm(len(train_ids), generator=torch.Generator().manual_seed(args.seed))
    check_ids = sorted(train_ids[i] for i in picked[:args.train_check_size].tolist())

    def dataset(ids, **options):
        return JerseyDataset(args.data, ids, max_frames=args.max_frames, **options)

    train_loader = make_loader(dataset(train_ids, augment=True, load_anchor=True, policy=args.augment),
                               args, device, sampler=make_sampler(train_ids, args.sampler_power))
    check_loader = make_loader(dataset(check_ids), args, device)
    val_loader = make_loader(dataset(val_ids), args, device)

    model = JerseyModel(args.variant, max_frames=args.max_frames, pretrained=not args.no_pretrained)
    model.to(device)
    backbone = [p for name, p in model.named_parameters() if name.startswith("backbone.")]
    rest = [p for name, p in model.named_parameters() if not name.startswith("backbone.")]
    # The order of the groups matters: max_lr below and the logged rates follow it.
    optimizer = torch.optim.AdamW(
        [{"params": rest, "lr": args.lr}, {"params": backbone, "lr": args.backbone_lr}],
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.schedule == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[args.lr, args.backbone_lr],
            total_steps=args.epochs * len(train_loader),
            pct_start=WARMUP_SHARE,
            cycle_momentum=False,
        )

    run_dir = args.out / (args.run_name or f"{args.variant}_{datetime.now():%Y%m%d-%H%M%S}")
    run_dir.mkdir(parents=True)  # fails if the run exists, so results are never overwritten
    # The split as trained, so evaluate.py scores the right sequences even after make_splits.py reruns.
    shutil.copy(args.splits, run_dir / "splits.json")
    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "device": str(device),
        # A plain str: torch's own version type would block loading best.pt with weights_only.
        "torch": str(torch.__version__),
        "parameters": sum(p.numel() for p in model.parameters()),
        "sequences": {"train": len(train_ids), "train_check": len(check_ids), "val": len(val_ids)},
        # Train sequences of other numbers left out of a hold-out run, for evaluate.py.
        "linked_dropped": sorted(linked),
        "selected_by": SELECTED_BY,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"run {run_dir}, {config['parameters']:,} parameters on {device}")

    save_batch_grid(next(iter(train_loader)), run_dir / "train_batch.png")

    def score_epoch(epoch, train_loss, start):
        """Return the history entry for an epoch, scoring train-check and val."""
        check = scores(evaluate(model, check_loader, device))
        val = scores(evaluate(model, val_loader, device))
        return {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "backbone_lr": optimizer.param_groups[1]["lr"],
            "train_loss": train_loss,
            **{f"train_check_{key}": value for key, value in check.items()},
            **{f"val_{key}": value for key, value in val.items()},
            "seconds": time.perf_counter() - start,
        }

    # Before training the model should guess about uniformly. A loss far from
    # UNIFORM_LOSS points to a bug in the init, the labels or the loss (Karpathy's recipe).
    history, best = [score_epoch(0, None, time.perf_counter())], None
    print(f"epoch  0/{args.epochs}  val loss {history[0]['val_loss']:.3f}"
          f" (uniform guess {UNIFORM_LOSS:.3f})  two-digit {history[0]['val_two_digit_accuracy']:.3f}"
          f"  val acc {history[0]['val_accuracy']:.3f}")

    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.anchor_weight, scheduler)
        current = score_epoch(epoch, train_loss, start)
        history.append(current)

        improved = best is None or current[SELECTED_BY] < best[SELECTED_BY]
        if improved:
            best = current
            # Weights on the CPU, so they load on any machine. The config says how to
            # rebuild the model (variant, max_frames) without config.json.
            torch.save({
                "model": {key: value.cpu() for key, value in model.state_dict().items()},
                "epoch": epoch,
                "config": config,
            }, run_dir / "best.pt")
        # Rewritten every epoch, so a crash or Ctrl-C keeps everything so far.
        (run_dir / "history.json").write_text(json.dumps({"epochs": history, "best": best}, indent=2))
        plot_curves(history, best["epoch"], run_dir / "curves.png")
        print(
            f"epoch {epoch:>2}/{args.epochs}  train loss {train_loss:.3f}"
            f"  check acc {current['train_check_accuracy']:.3f} macro {current['train_check_macro_accuracy']:.3f}"
            f"  val loss {current['val_loss']:.3f} two-digit {current['val_two_digit_accuracy']:.3f}"
            f" acc {current['val_accuracy']:.3f}"
            f" macro {current['val_macro_accuracy']:.3f} macro5 {current['val_macro_accuracy_5']:.3f}"
            f"  {current['seconds']:.0f}s{'  *' if improved else ''}"
        )

    print(f"best epoch {best['epoch']} by lowest {SELECTED_BY}: val loss {best['val_loss']:.3f},"
          f" two-digit {best['val_two_digit_accuracy']:.3f}, accuracy {best['val_accuracy']:.3f},"
          f" macro {best['val_macro_accuracy']:.3f}, macro5 {best['val_macro_accuracy_5']:.3f}")

    # Score the best weights per number, the detail the history leaves out.
    model.load_state_dict(torch.load(run_dir / "best.pt")["model"])
    result = evaluate(model, val_loader, device)
    train_numbers = [sequence_id.split("/")[0] for sequence_id in train_ids]
    table = per_number(result["predictions"], result["labels"])
    report = {
        "epoch": best["epoch"],
        "split": "val",
        "scores": scores(result),
        "per_number": table,
        "bands": band_accuracy(result["predictions"], result["labels"], shot_bands(train_numbers)),
        "confusions": confusions(result["predictions"], result["labels"]),
        "majority_baseline": majority_baseline(result["labels"], train_numbers),
    }
    (run_dir / "per_number.json").write_text(json.dumps(report, indent=2))
    for number, row in table.items():
        print(f"  {number:>2}: {row['correct']:>3}/{row['count']:<3} {row['accuracy']:.3f}"
              f"  (95% {row['low']:.2f}-{row['high']:.2f})")

    explain_run(model, val_loader.dataset, config, run_dir, device)

if __name__ == "__main__":
    main()
