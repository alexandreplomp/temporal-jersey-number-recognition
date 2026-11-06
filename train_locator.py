"""Train the local digit reader (jersey/local_reader.py) that finds digits for make_pairs.py.
Example: uv run train_locator.py --exclude 49 --run-name locator_ex49

Trains on single-digit training sequences only, one image at a time: every picked frame and
the anchor, labelled with the sequence's digit, with the strong augmentation. The loss is
cross-entropy on the crop score (the mean of the top cells inside the torso). Held-out
numbers and the train sequences linked to them are left out, as in train.py, so the boxes
the locator finds never come from a held-out player.

The best epoch has the lowest val loss. Accuracy is reported per val image (the gate:
make_pairs.py reads one image at a time) and per val sequence (frames' log-probabilities summed).

Writes results/locator/<run-name>/:
    config.json   every setting, plus the held-out numbers and the linked train sequences dropped
    splits.json   the split it was trained with
    history.json  loss and accuracy per epoch, and the best epoch. Epoch 0 is before training.
    best.pt       {"model": state_dict, "epoch", "config"} of the best epoch
"""

from datetime import datetime
from pathlib import Path
import argparse
import json
import math
import shutil
import time

import torch
from torch.nn import functional as F

from jersey.data import JerseyDataset, filter_numbers, make_sampler
from jersey.local_reader import LocalDigitReader, crop_scores
from train import WARMUP_SHARE, make_loader, pick_device, training_ids

GATE = 0.75  # per-image val accuracy the locator needs before make_pairs.py uses it


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32, help="sequences per batch; each gives several images")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--backbone-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-frames", type=int, default=4, help="training frames per sequence, plus the anchor")
    parser.add_argument("--val-frames", type=int, default=4, help="evenly spread val frames per sequence")
    parser.add_argument("--extra-layers", type=int, default=0, help="1 x 1 residual layers beyond the two")
    parser.add_argument("--sampler-power", type=float, default=0.5)
    parser.add_argument("--exclude", nargs="*", default=[], help="numbers to hold out, e.g. 49")
    parser.add_argument("--issues", type=Path, default=Path("results/data_issues.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto picks cuda, then mps, then cpu")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--splits", type=Path, default=Path("results/splits.json"))
    parser.add_argument("--out", type=Path, default=Path("results/locator"))
    parser.add_argument("--run-name", help="default: locator_<date-time>")
    return parser.parse_args(argv)


def single_digit(sequence_ids):
    return [sequence_id for sequence_id in sequence_ids if len(sequence_id.split("/")[0]) == 1]


def images_and_digits(batch, device):
    """A batch of sequences as single images (N, 3, H, W), their digits (N,) and sequence index (N,)."""
    mask = batch["mask"]
    images = [batch["frames"][mask]]
    owner = [torch.arange(len(mask)).repeat_interleave(mask.sum(dim=1))]
    if "anchor" in batch:
        images.append(batch["anchor"])
        owner.append(torch.arange(len(mask)))
    owner = torch.cat(owner)
    digits = batch["label"][:, 0][owner]
    return torch.cat(images).to(device, non_blocking=True), digits.to(device, non_blocking=True), owner


def train_one_epoch(model, loader, optimizer, scheduler, device):
    """One pass over loader: mean loss per image."""
    model.train()
    total, count = 0.0, 0
    for batch in loader:
        images, digits, _ = images_and_digits(batch, device)
        loss = F.cross_entropy(crop_scores(model(images)), digits)
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"loss is {loss.item()}, stopping")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total += loss.item() * len(digits)
        count += len(digits)
    return total / count


@torch.no_grad()
def evaluate_locator(model, loader, device):
    """Loss and accuracy per image, and accuracy per sequence (summed log-probabilities)."""
    model.eval()
    total, right, images_seen, sequences_right, sequences = 0.0, 0, 0, 0, 0
    for batch in loader:
        images, digits, owner = images_and_digits(batch, device)
        scores = crop_scores(model(images))
        total += F.cross_entropy(scores, digits, reduction="sum").item()
        right += (scores.argmax(dim=1) == digits).sum().item()
        images_seen += len(digits)
        summed = torch.zeros(len(batch["mask"]), scores.shape[1], device=device)
        summed.index_add_(0, owner.to(device), scores.log_softmax(dim=1))
        sequences_right += (summed.argmax(dim=1).cpu() == batch["label"][:, 0]).sum().item()
        sequences += len(batch["mask"])
    return {"loss": total / images_seen, "accuracy": right / images_seen, "sequence_accuracy": sequences_right / sequences}


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    splits = json.loads(args.splits.read_text())
    exclude = tuple(args.exclude)
    train_ids, linked = training_ids(splits, exclude, args.issues)
    train_ids = single_digit(train_ids)
    val_ids = single_digit(filter_numbers(splits["val"], exclude))

    train_loader = make_loader(
        JerseyDataset(args.data, train_ids, augment=True, load_anchor=True, max_frames=args.max_frames, policy="strong"),
        args, device, sampler=make_sampler(train_ids, args.sampler_power),
    )
    val_loader = make_loader(JerseyDataset(args.data, val_ids, max_frames=args.val_frames), args, device)

    model = LocalDigitReader(pretrained=not args.no_pretrained, extra_layers=args.extra_layers).to(device)
    backbone = [p for name, p in model.named_parameters() if name.startswith("backbone.")]
    rest = [p for name, p in model.named_parameters() if not name.startswith("backbone.")]
    optimizer = torch.optim.AdamW(
        [{"params": rest, "lr": args.lr}, {"params": backbone, "lr": args.backbone_lr}], weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[args.lr, args.backbone_lr], total_steps=args.epochs * len(train_loader),
        pct_start=WARMUP_SHARE, cycle_momentum=False,
    )

    run_dir = args.out / (args.run_name or f"locator_{datetime.now():%Y%m%d-%H%M%S}")
    run_dir.mkdir(parents=True)  # fails if the run exists, so results are never overwritten
    shutil.copy(args.splits, run_dir / "splits.json")
    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "device": str(device),
        "torch": str(torch.__version__),
        "parameters": sum(p.numel() for p in model.parameters()),
        "sequences": {"train": len(train_ids), "val": len(val_ids)},
        "linked_dropped": sorted(linked),
        "selected_by": "val_loss",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"locator {run_dir}, {config['parameters']:,} parameters on {device}, "
          f"{len(train_ids)} train and {len(val_ids)} val single-digit sequences")

    history, best = [{"epoch": 0, **{f"val_{k}": v for k, v in evaluate_locator(model, val_loader, device).items()}}], None
    print(f"epoch  0/{args.epochs}  val loss {history[0]['val_loss']:.3f} (uniform {math.log(10):.3f})"
          f"  acc {history[0]['val_accuracy']:.3f}")
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val = evaluate_locator(model, val_loader, device)
        current = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val.items()},
                   "seconds": time.perf_counter() - start}
        history.append(current)
        improved = best is None or current["val_loss"] < best["val_loss"]
        if improved:
            best = current
            torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()}, "epoch": epoch, "config": config},
                       run_dir / "best.pt")
        (run_dir / "history.json").write_text(json.dumps({"epochs": history, "best": best}, indent=2))
        print(f"epoch {epoch:>2}/{args.epochs}  train loss {train_loss:.3f}  val loss {val['loss']:.3f}"
              f"  image acc {val['accuracy']:.3f}  sequence acc {val['sequence_accuracy']:.3f}"
              f"  {current['seconds']:.0f}s{'  *' if improved else ''}")

    verdict = "passes" if best["val_accuracy"] >= GATE else "FAILS"
    print(f"best epoch {best['epoch']}: val image accuracy {best['val_accuracy']:.3f} "
          f"(gate {GATE:.2f}: {verdict}), sequence accuracy {best['val_sequence_accuracy']:.3f}")


if __name__ == "__main__":
    main()
