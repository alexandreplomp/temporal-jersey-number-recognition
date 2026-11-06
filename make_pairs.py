"""Find the digit in every single-digit training image, for splicing two-digit sequences.
Example: uv run make_pairs.py full_strong_ex49_keypos

Uses a trained run to locate digits; no annotation. Its training split's single-digit
sequences (without its held-out numbers and their linked partners) are read one image at a
time. Each image's digit is known from its folder, so covering the place where it is should
make the run's reading of that digit drop most:
    1. cover a vertical strip of the torso, one digit wide, at each column; the column with the
       largest drop holds the digit
    2. at that column, cover a digit-sized box at each height; the largest drop gives the row
    3. keep the box only if the run reads the image right (base >= --min-read) and covering
       the box takes much of that away (drop >= --min-drop), so the box holds the digit
Covering sets pixels to the ImageNet mean, as explain.py does.

Writes to --out (results/pairs):
    strips.json   per image: centre, base, drop, kept, kit colour; and the run, its exclude
    summary.json  share kept per digit, usable sequences, the two-digit numbers splicing can make
    strips.png    a spread of images with their box: green kept, red rejected
    spliced.png   spliced frames as training sees them before augmentation (jersey/pairs.py)
Train with them: uv run train.py --exclude 49 --synthetic-pairs 0.3
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

from check_shortcuts import TORSO_COLUMNS, TORSO_ROWS
from jersey.data import (
    IMAGE_SIZE, MAX_FRAMES, NORMALIZE, _load_image, _pick_frames, denormalize, filter_numbers, find_images,
    run_splits,
)
from jersey.model import load_run
from jersey.pairs import STRIP_HEIGHT, STRIP_WIDTH, StripBank, cut_strip, splice
from train import pick_device

STEP = 4  # pixels between two covered positions
MIN_READ = 0.5
# Boxes with a smaller drop often sat beside the digit (checked on strips.png).
MIN_DROP = 0.5
PICTURE_COUNT = 48
CHUNK = 512  # covered images per forward pass


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", help="run folder under --runs that locates the digits, e.g. full_strong_ex49_keypos")
    parser.add_argument("--runs", type=Path, default=Path("results/runs"))
    parser.add_argument("--out", type=Path, default=Path("results/pairs"))
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES,
                        help="frames per sequence, evenly spread, plus the anchor")
    parser.add_argument("--min-read", type=float, default=MIN_READ,
                        help="keep a box only if the run reads the image's digit at least this sure")
    parser.add_argument("--min-drop", type=float, default=MIN_DROP,
                        help="keep a box only if covering it lowers that by at least this much")
    parser.add_argument("--device", default="auto", help="auto picks cuda, then mps, then cpu")
    parser.add_argument("--batch-size", type=int, default=32)
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


def torso_box(height, width):
    """Torso rows and columns as (start, end) pairs, as in check_shortcuts.torso_mask."""
    rows = (round(TORSO_ROWS[0] * height), round(TORSO_ROWS[1] * height))
    columns = (round(TORSO_COLUMNS[0] * width), round(TORSO_COLUMNS[1] * width))
    return rows, columns


def single_digit_images(splits, config, data_dir, max_frames):
    """Entries (image name, digit, is anchor) of the run's single-digit training sequences."""
    exclude = tuple(config["exclude"])
    dropped = set(config.get("linked_dropped", []))
    entries = []
    for sequence_id in filter_numbers(splits["train"], exclude):
        if sequence_id in dropped or len(sequence_id.split("/")[0]) != 1:
            continue
        anchor, frames = find_images(Path(data_dir) / sequence_id)
        picked = _pick_frames(len(frames), max_frames, at_random=False).tolist()
        digit = int(sequence_id.split("/")[0])
        for path in [frames[i] for i in picked] + [anchor]:
            entries.append((f"{sequence_id}/{path.name}", digit, path == anchor))
    return entries


def read(model, images, digits):
    """Probability that each image alone reads its digit as the first digit: (N,)."""
    probabilities = model.read_frames(images).softmax(dim=-1)[:, 0]
    return probabilities.gather(1, digits[:, None]).squeeze(1)


def covered_reads(model, images, digits, boxes):
    """read with box (top, bottom, left, right) k of image n covered: boxes (N, K, 4) -> (N, K)."""
    N, K = boxes.shape[:2]
    H, W = images.shape[-2:]
    rows = torch.arange(H, device=images.device)[:, None]
    columns = torch.arange(W, device=images.device)[None, :]
    top, bottom, left, right = (boxes[..., i, None, None] for i in range(4))
    inside = (rows >= top) & (rows < bottom) & (columns >= left) & (columns < right)  # (N, K, H, W)
    results = []
    for part in torch.arange(N * K, device=images.device).split(CHUNK):
        n, k = part // K, part % K
        results.append(read(model, images[n] * ~inside[n, k][:, None], digits[n]))
    return torch.cat(results).reshape(N, K)


@torch.no_grad()
def locate(model, images, digits):
    """Find each image's digit by covering: centres (N, 2) as (x, y), base (N,) and drop (N,)."""
    N, _, H, W = images.shape
    device = images.device
    rows, columns = torso_box(H, W)
    base = read(model, images, digits)

    xs = torch.arange(columns[0] + STRIP_WIDTH // 2, columns[1] - STRIP_WIDTH // 2 + 1, STEP, device=device)
    strips = torch.stack([
        torch.full_like(xs, rows[0]), torch.full_like(xs, rows[1]), xs - STRIP_WIDTH // 2, xs + STRIP_WIDTH - STRIP_WIDTH // 2,
    ], dim=-1).expand(N, -1, -1)
    x = xs[(base[:, None] - covered_reads(model, images, digits, strips)).argmax(dim=1)]  # (N,)

    ys = torch.arange(rows[0] + STRIP_HEIGHT // 2, rows[1] - STRIP_HEIGHT // 2 + 1, STEP, device=device)
    left, right = x - STRIP_WIDTH // 2, x + STRIP_WIDTH - STRIP_WIDTH // 2
    boxes = torch.stack([
        (ys - STRIP_HEIGHT // 2).expand(N, -1), (ys + STRIP_HEIGHT - STRIP_HEIGHT // 2).expand(N, -1),
        left[:, None].expand(-1, len(ys)), right[:, None].expand(-1, len(ys)),
    ], dim=-1)
    drops = base[:, None] - covered_reads(model, images, digits, boxes)
    best = drops.argmax(dim=1)
    y = ys[best]
    return torch.stack([x, y], dim=1).float(), base, drops[torch.arange(N, device=device), best]


def kit_colours(images, centres):
    """Median colour of the digit box of each image, so the kit and not the grass: (N, 3) in [0, 1].

    The median, since the digit's strokes cover less of the box than the shirt does.
    """
    colours = []
    for image, centre in zip(denormalize(images), centres.tolist()):
        strip = cut_strip(image, centre, width=STRIP_WIDTH)
        colours.append(strip.flatten(1).median(dim=1).values)
    return torch.stack(colours)


def locate_all(model, dataset, args, device):
    """Results per entry of dataset: centre, base, drop and kit colour."""
    found = [None] * len(dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
    for done, batch in enumerate(loader):
        centres, base, drop = locate(model, batch["image"].to(device), batch["digit"].to(device))
        colours = kit_colours(batch["image"], centres.cpu())
        for i, index in enumerate(batch["index"].tolist()):
            found[index] = {
                "centre": [round(v, 1) for v in centres[i].tolist()],
                "base": round(base[i].item(), 4),
                "drop": round(drop[i].item(), 4),
                "colour": [round(v, 4) for v in colours[i].tolist()],
            }
        if done % 50 == 0:
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


def save_strips_picture(data_dir, sequences, path):
    """A spread of located images with their box: green kept, red rejected."""
    images = [image for found in sequences.values() for image in found]
    picks = torch.linspace(0, len(images) - 1, min(PICTURE_COUNT, len(images))).round().long().tolist()
    resize = v2.Resize(IMAGE_SIZE, antialias=True)
    columns = 12
    figure = Figure(figsize=(columns * 1.3, -(-len(picks) // columns) * 2.8))
    for i, pick in enumerate(picks):
        found = images[pick]
        axis = figure.add_subplot(-(-len(picks) // columns), columns, i + 1)
        _show(axis, _load_image(Path(data_dir) / found["image"], resize),
              f"{found['image'].split('/')[0]}  r{found['base']:.2f} d{found['drop']:.2f}")
        x, y = found["centre"]
        axis.add_patch(Rectangle((x - STRIP_WIDTH / 2 - 0.5, y - STRIP_HEIGHT / 2 - 0.5), STRIP_WIDTH, STRIP_HEIGHT,
                                 fill=False, linewidth=1.2, edgecolor="lime" if found["kept"] else "red"))
    figure.tight_layout()
    figure.savefig(path, dpi=110)


def save_spliced_picture(data_dir, bank, path):
    """First spliced frame of a spread of usable sequences, with the number it is labelled."""
    ids = bank.sequence_ids()
    picks = torch.linspace(0, len(ids) - 1, min(PICTURE_COUNT, len(ids))).round().long().tolist() if ids else []
    resize = v2.Resize(IMAGE_SIZE, antialias=True)
    columns = 12
    figure = Figure(figsize=(columns * 1.3, max(1, -(-len(picks) // columns)) * 2.8))
    shown = 0
    for pick in picks:
        sequence_id = ids[pick]
        found = bank.partner(sequence_id)
        if found is None:
            continue
        partner_id, a_left = found
        (a_name, a_centre), (b_name, b_centre) = bank.frames[sequence_id][0], bank.frames[partner_id][0]
        image = splice(_load_image(Path(data_dir) / a_name, resize), a_centre,
                       _load_image(Path(data_dir) / b_name, resize), b_centre, a_left)
        digit, other = sequence_id.split("/")[0], partner_id.split("/")[0]
        shown += 1
        _show(figure.add_subplot(max(1, -(-len(picks) // columns)), columns, shown), image,
              f"{digit + other if a_left else other + digit}  (from {digit})")
    figure.tight_layout()
    figure.savefig(path, dpi=110)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(0)  # partner draws for spliced.png
    device = pick_device(args.device)
    run_dir = args.runs / args.run
    model, config, _ = load_run(run_dir, device)
    splits = run_splits(run_dir, config["splits"])
    entries = single_digit_images(splits, config, config["data"], args.max_frames)
    print(f"{len(entries)} images of single-digit training sequences, located with {args.run} on {device}")
    found = locate_all(model, Images(config["data"], entries), args, device)

    sequences = defaultdict(list)
    for (name, _, is_anchor), result in zip(entries, found):
        kept = result["base"] >= args.min_read and result["drop"] >= args.min_drop
        sequences[name.rsplit("/", 1)[0]].append({"image": name, "anchor": is_anchor, "kept": kept, **result})
    args.out.mkdir(parents=True, exist_ok=True)
    strips = {
        "run": args.run, "data": config["data"], "exclude": config["exclude"],
        "image_size": list(IMAGE_SIZE), "strip": [STRIP_WIDTH, STRIP_HEIGHT],
        "min_read": args.min_read, "min_drop": args.min_drop, "sequences": sequences,
    }
    (args.out / "strips.json").write_text(json.dumps(strips))
    bank = StripBank(args.out / "strips.json")
    summary = summarise(sequences, bank, tuple(config["exclude"]))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    save_strips_picture(config["data"], sequences, args.out / "strips.png")
    save_spliced_picture(config["data"], bank, args.out / "spliced.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
