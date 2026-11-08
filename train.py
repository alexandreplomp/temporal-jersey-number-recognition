"""Train one named approach (APPROACHES below) and keep its best epoch.
Example: uv run train.py --approach two_step_mean --exclude 49

Reads results/splits.json and results/data_issues.json (prepare.py). An approach that pastes digits
needs digit boxes for its hold-out setting; if results/boxes/<tag>/ is missing, they are made first
(jersey/pairs.py: the local digit reader is trained, checked against its gate, and boxes are found).
Held-out numbers and the train sequences linked to them are left out of everything.

Writes results/runs/<approach>_<tag>/ (tag "ex49" for --exclude 49, "all" without; "_s<seed>" for
another seed, "_scratch" for --no-pretrained). A two-stage approach writes stage 1 to <run>/stage1/ and
starts stage 2 from it:
    config.json      the approach, its stages, the held-out numbers, the count offset, sizes, and the
                     git commit of the code (with "dirty" if tracked code had uncommitted changes)
    splits.json      the split it was trained with
    history.json     per epoch: train loss, train-check and val scores (+ pair metric), and the best
    curves.png       loss, accuracy, two-digit and macro accuracy per epoch
    train_batch.png  one training batch as the model sees it, anchor last
    best.pt          {"model", "epoch", "config"} of the best epoch
    done.json        written when every stage has finished; a finished run is skipped when asked again
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import json
import math
import shutil
import subprocess
import time

import torch

from jersey import plots
from jersey.data import JerseyDataset, filter_numbers, make_sampler, number_of, read_splits, split_ids
from jersey.engine import evaluate, make_loader, make_optimizer, pick_device, train_one_epoch
from jersey.metrics import scores
from jersey.model import JerseyModel, count_prior
from jersey.pairs import LOCATOR_EPOCHS, StripBank, build_pair_set, make_boxes, pair_metrics, tag_of

UNIFORM_LOSS = (math.log(10) + math.log(11)) / 2  # a uniform guess: 10 classes for position 1, 11 for position 2
TRAIN_CHECK_SIZE = 500  # a fixed, unaugmented subset of train, for train-vs-val curves


@dataclass(frozen=True)
class Stage:
    epochs: int
    lr: float  # new parts
    backbone_lr: float  # the pretrained backbone, 10:1 as in DETR
    pair_share: float  # single-digit samples spliced into pairs
    single_share: float  # single-digit samples given a partner's digit pasted alone
    real_pairs: bool  # train on real two-digit sequences
    selected_by: str  # history key that picks the best epoch
    highest: bool  # whether the highest value wins


@dataclass(frozen=True)
class Approach:
    reader: str  # jersey.model.READERS
    temporal: str  # jersey.model.TEMPORAL
    max_frames: int
    stages: tuple[Stage, ...]


# Stage 1 learns two-digit reading digit by digit from spliced pairs, without real two-digit
# sequences, and is judged on spliced val pairs, since val has too few two-digit numbers. Stage 2
# starts from it with all real data at a tenth of the rates; its start counts as a candidate.
ONE_STAGE = (Stage(30, 3e-4, 3e-5, 0.0, 0.0, True, "val_loss", False),)
TWO_STAGES = (Stage(40, 3e-4, 3e-5, 0.4, 0.4, False, "pair_accuracy", True),
              Stage(20, 3e-5, 3e-6, 0.3, 0.3, True, "val_loss", False))
# A pretrained text reader is fine-tuned gently, on the real (augmented) data only, no pastes.
PARSEQ_STAGE = (Stage(15, 1e-4, 1e-5, 0.0, 0.0, True, "val_loss", False),)

# Each row changes one thing. The final model is the best two_step_* or parseq_gate run by val macro
# accuracy (numbers with at least 5 sequences), retrained on all numbers; never chosen on hold-out or test.
APPROACHES = {
    "single_frame": Approach("two-step", "mean", 1, TWO_STAGES),  # what the temporal component adds
    "fused": Approach("fused", "mean", 10, ONE_STAGE),  # the two-digit structure alone, no pastes
    "two_step_mean": Approach("two-step", "mean", 10, TWO_STAGES),
    "two_step_pool": Approach("two-step", "pool", 10, TWO_STAGES),
    "two_step_full": Approach("two-step", "transformer", 10, TWO_STAGES),
    # PARSeq-Tiny per frame, two position queries, a learned gate per position over up to 10 frames.
    "parseq_gate": Approach("parseq", "pool", 10, PARSEQ_STAGE),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--approach", choices=tuple(APPROACHES), required=True)
    parser.add_argument("--exclude", nargs="*", default=[], help="numbers to hold out, e.g. 49")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, help="override every stage's epochs, e.g. 1 for a smoke run")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto", help="auto picks cuda, then mps, then cpu")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--no-pretrained", action="store_true",
                        help="random start instead of ImageNet / PARSeq weights: measures what pretraining adds")
    return parser.parse_args(argv)


def git_state():
    """{"commit", "dirty"} of the code next to this file; "unknown" outside a git checkout."""
    def git(*command):
        return subprocess.run(["git", *command], cwd=Path(__file__).parent, capture_output=True, text=True,
                              check=True).stdout.strip()
    try:
        changed = git("status", "--porcelain", "--untracked-files=no", "--", "jersey", "*.py", "reproduce.sh")
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(changed)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": None}


def check_bank(bank, exclude, train_ids):
    """Raises ValueError if the train boxes could paste a held-out number or come from outside train."""
    if bank.split != "train":
        raise ValueError(f"the bank holds {bank.split} boxes; only train boxes may feed training")
    if set(exclude) - set(bank.exclude):
        raise ValueError(f"the bank was made without holding out {sorted(set(exclude) - set(bank.exclude))}")
    outside = set(bank.sequence_ids()) - set(train_ids)
    if outside:
        raise ValueError(f"the bank has {len(outside)} sequences outside this run's train split, e.g. {min(outside)}")


def save_box_pictures(data_dir, boxes):
    """strips.png (100 kept train boxes, to count by eye how many sit on the digit), spliced.png and pasted.png."""
    kept = [image for images in json.loads((boxes / "train.json").read_text())["sequences"].values()
            for image in images if image["kept"]]
    bank = StripBank(boxes / "train.json")
    plots.boxes(data_dir, kept, boxes / "strips.png")
    plots.pastes(data_dir, bank, boxes / "spliced.png", single=False)
    plots.pastes(data_dir, bank, boxes / "pasted.png", single=True)
    print(f"boxes in {boxes}: count how many of the boxes on strips.png sit on the digit")


def train_stage(model, stage, run, stage_dir, step0, keep_start):
    """Train one stage into stage_dir; leave the best epoch's weights in model. Returns its history."""
    args, device = run["args"], run["device"]
    train_set = JerseyDataset(args.data, run["train_ids"], augment=True, load_anchor=True,
                              max_frames=run["approach"].max_frames, pastes=run["bank"], pair_share=stage.pair_share,
                              single_share=stage.single_share, real_pairs=stage.real_pairs)
    fit_ids = train_set.sequence_ids  # the sequences actually trained on: sampler and train-check agree
    picked = torch.randperm(len(fit_ids), generator=torch.Generator().manual_seed(args.seed))[:TRAIN_CHECK_SIZE]
    train_loader = make_loader(train_set, args.batch_size, args.workers, device, sampler=make_sampler(fit_ids))
    check_loader = make_loader(JerseyDataset(args.data, sorted(fit_ids[i] for i in picked.tolist()),
                                             max_frames=run["approach"].max_frames), args.batch_size, args.workers, device)
    epochs = args.epochs or stage.epochs
    optimizer, scheduler = make_optimizer(model, stage.lr, stage.backbone_lr, epochs * len(train_loader))
    stage_dir.mkdir(parents=True, exist_ok=True)  # main() refuses a run folder that already exists
    shutil.copy(args.results / "splits.json", stage_dir / "splits.json")
    config = {**run["config"], "stage": asdict(stage), "sequences": {"train": len(fit_ids), "val": len(run["val_ids"])}}
    (stage_dir / "config.json").write_text(json.dumps(config, indent=2))
    plots.batch_grid(next(iter(train_loader)), stage_dir / "train_batch.png")

    def score(epoch, train_loss, start):
        check, val = scores(evaluate(model, check_loader, device)), scores(evaluate(model, run["val_loader"], device))
        return {"epoch": epoch, "step": step0 + epoch, "train_loss": train_loss,
                **{f"train_check_{k}": v for k, v in check.items()}, **{f"val_{k}": v for k, v in val.items()},
                **(pair_metrics(model, run["pair_set"], device) if run["pair_set"] else {}),
                "seconds": time.perf_counter() - start}

    def save(entry):
        state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save({"model": state, "epoch": entry["epoch"], "config": config}, stage_dir / "best.pt")
        return entry, state

    history = [score(0, None, time.perf_counter())]
    print(f"{stage_dir.name} epoch 0  val loss {history[0]['val_loss']:.3f} (uniform guess {UNIFORM_LOSS:.3f})")
    best, best_state = save(history[0]) if keep_start else (None, None)
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        current = score(epoch, train_one_epoch(model, train_loader, optimizer, scheduler, device), start)
        history.append(current)
        value, best_value = current[stage.selected_by], best and best[stage.selected_by]
        if best is None or (value > best_value if stage.highest else value < best_value):
            best, best_state = save(current)
        (stage_dir / "history.json").write_text(json.dumps({"epochs": history, "best": best}, indent=2))
        plots.curves(history, stage_dir / "curves.png")
        pairs = f"  pairs {current['pair_accuracy']:.3f}" if "pair_accuracy" in current else ""
        print(f"{stage_dir.name} epoch {epoch:>2}/{epochs}  train loss {current['train_loss']:.3f}"
              f"  val loss {current['val_loss']:.3f} acc {current['val_accuracy']:.3f}"
              f" two-digit {current['val_two_digit_accuracy']:.3f}{pairs}  {current['seconds']:.0f}s"
              f"{'  *' if best is current else ''}")
    model.load_state_dict(best_state)
    print(f"{stage_dir.name}: best epoch {best['epoch']} by {stage.selected_by} {best[stage.selected_by]:.3f}")
    return history


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    approach, exclude = APPROACHES[args.approach], tuple(args.exclude)
    tag = tag_of(exclude)
    run_dir = args.results / "runs" / (f"{args.approach}_{tag}" + (f"_s{args.seed}" if args.seed else "")
                                       + ("_scratch" if args.no_pretrained else ""))
    if (run_dir / "done.json").exists() and not (run_dir / "best.pt").exists():
        raise SystemExit(f"{run_dir} is a committed record without weights (they are not in Git); to train it again,"
                         f" move the records aside first: mv {args.results / 'runs'} {args.results / 'runs_reported'}")
    if (run_dir / "done.json").exists():
        print(f"{run_dir} is already trained; skipping (results are never overwritten)")
        return
    if run_dir.exists():
        raise SystemExit(f"{run_dir} is an unfinished run (no done.json): delete it to train it again")

    splits = read_splits(args.results / "splits.json")
    issues = json.loads((args.results / "data_issues.json").read_text())
    train_ids, linked = split_ids(splits, "train", exclude, issues)
    val_ids = filter_numbers(splits["val"], exclude)
    print(f"held out {list(exclude)}; dropped {len(linked)} linked train sequences")

    boxes = args.results / "boxes" / tag
    pastes = any(stage.pair_share or stage.single_share for stage in approach.stages)
    if pastes and not (boxes / "train.json").exists():
        make_boxes(args.data, boxes, train_ids, split_ids(splits, "val", exclude, issues)[0], exclude, device,
                   epochs=LOCATOR_EPOCHS, workers=args.workers, pretrained=not args.no_pretrained)
        save_box_pictures(args.data, boxes)
    bank = StripBank(boxes / "train.json") if pastes else None
    if bank is not None:
        check_bank(bank, exclude, train_ids)
    pair_set = build_pair_set(boxes / "val.json", val_ids, exclude, args.data, seed=args.seed) \
        if (boxes / "val.json").exists() else None

    count_offset = count_prior([number_of(s) for s in train_ids]) if approach.reader == "two-step" else 0.0
    model = JerseyModel(approach.reader, approach.temporal, max_frames=approach.max_frames,
                        pretrained=not args.no_pretrained, count_offset=count_offset).to(device)
    config = {"approach": args.approach, "reader": approach.reader, "temporal": approach.temporal,
              "max_frames": approach.max_frames, "exclude": list(exclude), "linked_dropped": sorted(linked),
              "seed": args.seed, "count_offset": count_offset, "data": str(args.data), "device": str(device),
              "parameters": sum(p.numel() for p in model.parameters()), "torch": str(torch.__version__),
              "pretrained": not args.no_pretrained, "git": git_state()}
    run = {"args": args, "device": device, "approach": approach, "train_ids": train_ids, "val_ids": val_ids,
           "bank": bank, "pair_set": pair_set, "config": config,
           "val_loader": make_loader(JerseyDataset(args.data, val_ids, max_frames=approach.max_frames),
                                     args.batch_size, args.workers, device)}
    print(f"run {run_dir}: {config['parameters']:,} parameters on {device}, count offset {count_offset:.2f}")

    step = 0
    for index, stage in enumerate(approach.stages):
        last = index == len(approach.stages) - 1
        history = train_stage(model, stage, run, run_dir if last else run_dir / f"stage{index + 1}", step, keep_start=index > 0)
        step = history[-1]["step"]
    # Written last, so a run stopped halfway is never taken for a finished one.
    (run_dir / "done.json").write_text(json.dumps({"epochs": step}))


if __name__ == "__main__":
    main()
