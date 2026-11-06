"""Find the digit box in every single-digit image of a split, for splicing two-digit sequences.
Example: uv run make_pairs.py --locator results/locator/locator_ex49 --split train --exclude 49 --out results/pairs_ex49

No annotation: each image's digit is known from its folder, and two independent signals
must agree on where it is:
    evidence  the locator's (train_locator.py) strongest torso cell for that digit gives the
              digit centre; its crop score gives how sure it reads the digit
    ink       in a window around the centre, the pixels split into two colour clusters
              (k-means); the smaller cluster is the ink, and its column and row projections
              give a box
A box is kept only if the locator is sure (>= --min-prob), the ink box contains the evidence
centre, and its size fits one digit (BOX_HEIGHT, BOX_WIDTH, in the 192 x 96 input).

--split train boxes feed training (train.py --pairs); --split val boxes feed only the pair
metric (train.py --val-pairs). Held-out numbers' sequences and the sequences linked to them
are never read, and the locator must have held out the same numbers.

Writes to --out:
    strips.json   per image: evidence centre, probability, box, kept, kit colour; and the
                  locator, split and held-out numbers
    summary.json  share kept per digit, usable sequences, the two-digit numbers splicing can make
    strips.png    PICTURE_COUNT kept images with their box, for counting by eye
    rejected.png  rejected images with their box (red), to see what the checks throw away
    spliced.png   spliced frames as training sees them before augmentation (jersey/pairs.py)
"""

from collections import Counter, defaultdict
from pathlib import Path
import argparse
import json

from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from jersey.data import (
    IMAGE_SIZE, MAX_FRAMES, NORMALIZE, _load_image, _pick_frames, denormalize, filter_numbers, find_images,
    held_out_partners, run_splits,
)
from jersey.local_reader import crop_scores, load_locator, torso_cells
from jersey.model import CELL_STRIDE
from jersey.pairs import GAP, StripBank, splice
from train import pick_device

MIN_PROB = 0.9
WINDOW = (48, 32)  # height, width of the ink search around the evidence centre
BOX_HEIGHT = (12, 45)
BOX_WIDTH = (6, 28)
INK_SHARE = 0.15  # a row or column is ink if it has at least this share of the busiest one
KMEANS_STEPS = 10
PICTURE_COUNT = 100


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--locator", type=Path, required=True, help="train_locator.py run folder")
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--exclude", nargs="*", default=[], help="must match the locator's held-out numbers")
    parser.add_argument("--out", type=Path, default=Path("results/pairs"))
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES,
                        help="frames per sequence, evenly spread, plus the anchor")
    parser.add_argument("--min-prob", type=float, default=MIN_PROB,
                        help="keep a box only if the locator reads the image's digit at least this sure")
    parser.add_argument("--device", default="auto", help="auto picks cuda, then mps, then cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args(argv)


class Images(Dataset):
    """(name, digit, is anchor) entries as normalised images."""

    def __init__(self, data_dir, entries):
        self.data_dir = Path(data_dir)
        self.entries = entries
        self.resize = v2.Resize(IMAGE_SIZE, antialias=True)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        name, digit, _ = self.entries[index]
        image = _load_image(self.data_dir / name, self.resize)  # uint8 (3, H, W)
        return {"image": NORMALIZE(image[None])[0], "digit": digit, "index": index}


def split_ids(splits, split, exclude, issues):
    """Single-digit ids of the split, without held-out numbers and the sequences linked to them."""
    everything = [sequence_id for name in ("train", "val", "test") for sequence_id in splits[name]]
    linked = held_out_partners(issues, everything, exclude) if exclude else set()
    return [
        sequence_id for sequence_id in filter_numbers(splits[split], exclude)
        if sequence_id not in linked and len(sequence_id.split("/")[0]) == 1
    ]


def single_digit_images(sequence_ids, data_dir, max_frames):
    """Entries (image name, digit, is anchor): evenly spread frames and the anchor of each sequence."""
    entries = []
    for sequence_id in sequence_ids:
        anchor, frames = find_images(Path(data_dir) / sequence_id)
        picked = _pick_frames(len(frames), max_frames, at_random=False).tolist()
        digit = int(sequence_id.split("/")[0])
        for path in [frames[i] for i in picked] + [anchor]:
            entries.append((f"{sequence_id}/{path.name}", digit, path == anchor))
    return entries


@torch.no_grad()
def evidence(locator, images, digits):
    """Digit centres (N, 2) as (x, y) from the strongest torso cell, and the probability (N,) of the digit."""
    cells = locator(images)  # (N, 10, h, w)
    N, _, h, w = cells.shape
    probability = crop_scores(cells).softmax(dim=1).gather(1, digits[:, None]).squeeze(1)
    own = cells[torch.arange(N, device=cells.device), digits]  # (N, h, w)
    own = own.masked_fill(~torso_cells(h, w).to(cells.device), float("-inf"))
    best = own.flatten(1).argmax(dim=1)
    rows, columns = best // w, best % w
    centres = torch.stack([(columns + 0.5) * CELL_STRIDE, (rows + 0.5) * CELL_STRIDE], dim=1).float()
    return centres, probability


def two_colours(pixels, steps=KMEANS_STEPS):
    """k-means with two clusters on pixels (P, 3): labels (P,) and centres (2, 3).

    Starts from the darkest and the brightest pixel, so it is repeatable.
    """
    brightness = pixels.mean(dim=1)
    centres = torch.stack([pixels[brightness.argmin()], pixels[brightness.argmax()]])
    labels = torch.zeros(len(pixels), dtype=torch.long)
    for _ in range(steps):
        labels = torch.cdist(pixels, centres).argmin(dim=1)
        for k in range(2):
            if (labels == k).any():
                centres[k] = pixels[labels == k].mean(dim=0)
    return labels, centres


def _run(flags, at):
    """The run of consecutive True in flags (L,) containing index at, or the nearest one: (start, end)."""
    found = flags.nonzero().flatten()
    if len(found) == 0:
        return None
    if not flags[at]:
        at = found[(found - at).abs().argmin()].item()
    start = end = at
    while start > 0 and flags[start - 1]:
        start -= 1
    while end < len(flags) - 1 and flags[end + 1]:
        end += 1
    return start, end + 1


def ink_box(image, centre):
    """Box [left, top, right, bottom] of the ink near centre in image (3, H, W) in [0, 1], and the kit colour.

    The window around centre splits into two colour clusters; the smaller is the ink and the
    larger the shirt. Columns, then rows within those columns, count as ink when they hold at
    least INK_SHARE of the busiest one; the run nearest the centre gives the box. None if no ink.
    """
    H, W = image.shape[-2:]
    x, y = centre
    x0 = min(max(round(x - WINDOW[1] / 2), 0), W - WINDOW[1])
    y0 = min(max(round(y - WINDOW[0] / 2), 0), H - WINDOW[0])
    patch = image[:, y0:y0 + WINDOW[0], x0:x0 + WINDOW[1]]
    labels, centres = two_colours(patch.flatten(1).T)
    counts = torch.bincount(labels, minlength=2)
    ink = (labels == counts.argmin()).reshape(WINDOW).float()
    kit = centres[counts.argmax()]

    per_column = ink.sum(dim=0)
    if per_column.max() == 0:
        return None, kit
    columns = _run(per_column >= INK_SHARE * per_column.max(), min(max(round(x) - x0, 0), WINDOW[1] - 1))
    per_row = ink[:, columns[0]:columns[1]].sum(dim=1)
    rows = _run(per_row >= INK_SHARE * per_row.max(), min(max(round(y) - y0, 0), WINDOW[0] - 1))
    return [x0 + columns[0], y0 + rows[0], x0 + columns[1], y0 + rows[1]], kit


def keep(box, centre, probability, min_prob):
    """The three checks: a sure locator, a box containing the evidence centre, a digit-sized box."""
    if box is None or probability < min_prob:
        return False
    left, top, right, bottom = box
    x, y = centre
    return (left <= x < right and top <= y < bottom
            and BOX_HEIGHT[0] <= bottom - top <= BOX_HEIGHT[1] and BOX_WIDTH[0] <= right - left <= BOX_WIDTH[1])


def locate_all(locator, dataset, args, device):
    """Results per entry of dataset: evidence centre, probability, ink box, kit colour and kept."""
    found = [None] * len(dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
    for done, batch in enumerate(loader):
        centres, probability = evidence(locator, batch["image"].to(device), batch["digit"].to(device))
        centres, probability = centres.cpu(), probability.cpu()
        for image, centre, p, index in zip(denormalize(batch["image"]), centres.tolist(), probability.tolist(),
                                           batch["index"].tolist()):
            box, kit = ink_box(image, centre)
            found[index] = {
                "centre": centre,
                "prob": round(p, 4),
                "box": box,
                "colour": [round(v, 4) for v in kit.tolist()],
                "kept": keep(box, centre, p, args.min_prob),
            }
        if done % 20 == 0:
            print(f"{min((done + 1) * args.batch_size, len(dataset))}/{len(dataset)} images", flush=True)
    return found


def summarise(sequences, bank, exclude):
    """Share of images kept per digit, usable sequences, and the numbers splicing can make."""
    per_digit = defaultdict(Counter)
    for sequence_id, images in sequences.items():
        digit = sequence_id.split("/")[0]
        counts = per_digit[digit]
        counts["sequences"] += 1
        counts["usable sequences"] += sequence_id in bank
        for image in images:
            kind = "anchors" if image["anchor"] else "frames"
            counts[kind] += 1
            counts[f"{kind} kept"] += image["kept"]
    digits = sorted({sequence_id.split("/")[0] for sequence_id in bank.sequence_ids()})
    return {
        "per_digit": {
            digit: {**counts, "frames kept share": round(counts["frames kept"] / max(counts["frames"], 1), 3)}
            for digit, counts in sorted(per_digit.items())
        },
        "numbers": [a + b for a in digits for b in digits if a + b not in exclude],
        "never made (held out)": list(exclude),
    }


def _show(axis, image, title):
    axis.imshow(image.permute(1, 2, 0).numpy(), interpolation="nearest")
    axis.set_title(title, fontsize=7)
    axis.axis("off")


def _grid(count, columns=12):
    rows = max(1, -(-count // columns))
    return Figure(figsize=(columns * 1.3, rows * 2.8)), rows, columns


def _spread(items, count):
    return [items[i] for i in torch.linspace(0, len(items) - 1, min(count, len(items))).round().long().tolist()] if items else []


def save_boxes_picture(data_dir, images, path, count=PICTURE_COUNT):
    """A spread of images with their box: green kept, red rejected, and the evidence centre as a dot."""
    picks = _spread(images, count)
    resize = v2.Resize(IMAGE_SIZE, antialias=True)
    figure, rows, columns = _grid(len(picks))
    for i, found in enumerate(picks):
        axis = figure.add_subplot(rows, columns, i + 1)
        _show(axis, _load_image(Path(data_dir) / found["image"], resize),
              f"{i + 1}: {found['image'].split('/')[0]}  p{found['prob']:.2f}")
        if found["box"] is not None:
            left, top, right, bottom = found["box"]
            axis.add_patch(Rectangle((left - 0.5, top - 0.5), right - left, bottom - top, fill=False,
                                     linewidth=1.0, edgecolor="lime" if found["kept"] else "red"))
        axis.plot(*found["centre"], marker=".", color="yellow", markersize=3)
    figure.tight_layout()
    figure.savefig(path, dpi=110)


def save_spliced_picture(data_dir, bank, path, count=48):
    """First spliced frame of a spread of usable sequences, with the number it is labelled and both boxes."""
    resize = v2.Resize(IMAGE_SIZE, antialias=True)
    picks = _spread(bank.sequence_ids(), count)
    figure, rows, columns = _grid(len(picks))
    shown = 0
    for sequence_id in picks:
        found = bank.partner(sequence_id)
        if found is None:
            continue
        partner_id, a_left = found
        (a_name, a_box), (b_name, b_box) = bank.frames[sequence_id][0], bank.frames[partner_id][0]
        gap = torch.randint(GAP[0], GAP[1] + 1, ()).item()
        image, boxes = splice(_load_image(Path(data_dir) / a_name, resize), a_box,
                              _load_image(Path(data_dir) / b_name, resize), b_box, a_left, gap)
        digit, other = sequence_id.split("/")[0], partner_id.split("/")[0]
        shown += 1
        axis = figure.add_subplot(rows, columns, shown)
        _show(axis, image, f"{digit + other if a_left else other + digit}  (from {digit})")
        for (left, top, right, bottom), colour in zip(boxes.tolist(), ("red", "deepskyblue")):
            axis.add_patch(Rectangle((left - 0.5, top - 0.5), right - left, bottom - top, fill=False,
                                     linewidth=0.6, edgecolor=colour))
    figure.tight_layout()
    figure.savefig(path, dpi=110)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(0)  # partner draws for spliced.png
    device = pick_device(args.device)
    locator, config, _ = load_locator(args.locator, device)
    exclude = tuple(args.exclude)
    if sorted(exclude) != sorted(config["exclude"]):
        raise ValueError(f"--exclude {list(exclude)} differs from the locator's {config['exclude']}: "
                         "train a locator that held out the same numbers")
    splits = run_splits(args.locator, config["splits"])
    issues = json.loads(Path(config["issues"]).read_text()) if exclude else {}
    ids = split_ids(splits, args.split, exclude, issues)
    entries = single_digit_images(ids, config["data"], args.max_frames)
    print(f"{len(entries)} images of {len(ids)} single-digit {args.split} sequences, located with {args.locator} on {device}")
    found = locate_all(locator, Images(config["data"], entries), args, device)

    sequences = defaultdict(list)
    for (name, _, is_anchor), result in zip(entries, found):
        sequences[name.rsplit("/", 1)[0]].append({"image": name, "anchor": is_anchor, **result})
    args.out.mkdir(parents=True, exist_ok=True)
    strips = {
        "locator": str(args.locator), "split": args.split, "data": config["data"], "exclude": list(exclude),
        "image_size": list(IMAGE_SIZE), "min_prob": args.min_prob, "box_height": BOX_HEIGHT, "box_width": BOX_WIDTH,
        "sequences": sequences,
    }
    (args.out / "strips.json").write_text(json.dumps(strips))
    bank = StripBank(args.out / "strips.json")
    summary = summarise(sequences, bank, exclude)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    images = [image for images in sequences.values() for image in images]
    save_boxes_picture(config["data"], [image for image in images if image["kept"]], args.out / "strips.png")
    save_boxes_picture(config["data"], [image for image in images if not image["kept"]], args.out / "rejected.png", 48)
    save_spliced_picture(config["data"], bank, args.out / "spliced.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
