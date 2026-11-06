"""Does a model read a two-digit number digit by digit? A fixed set of spliced val pairs says.

The real two-digit numbers are few and fixed (66, 89, 48, ...), so a model can know them
whole. Here, two different digits from two validation players are spliced side by side
(jersey/pairs.py), in numbers that may never occur in the data. Reading them needs the left
digit in position 1 and the right digit in position 2:
    pair_accuracy     both digits right
    second_is_right   position 2 names the right digit
    second_is_left    position 2 names the left digit: the failure of a reader that takes the
                      same whole-crop content for both positions
    left_right_gap    columns between where the second and the first position look (attention
                      centroids); about 0 when both look at the same place
    singles_accuracy  the unspliced first image of each pair, read as a one-digit number

Only validation players' boxes (make_pairs.py --split val) are used, never a held-out
number is made, and the set is drawn once with a fixed seed, so every epoch and every run
sees the same images.
"""

from pathlib import Path

import torch
from torchvision.transforms import v2

from jersey.data import BLANK, IMAGE_SIZE, NORMALIZE, _load_image
from jersey.model import CELL_STRIDE, predict_digits
from jersey.pairs import GAP, StripBank, splice

PAIR_COUNT = 300
TRIES_PER_PAIR = 20


def build_pair_set(strips_path, val_ids, exclude, data_dir, count=PAIR_COUNT, seed=0):
    """Spliced images of two different digits and their unspliced first images, drawn with seed.

    Returns a dict: images (N, 3, H, W) uint8, labels (N, 2), boxes (N, 2, 4), singles
    (N, 3, H, W) uint8, single_labels (N, 2), and the sequence ids used.

    Raises:
        ValueError: the boxes are not validation boxes, include a sequence outside val_ids,
            or were made without holding out exclude.
    """
    bank = StripBank(strips_path)
    if bank.split != "val":
        raise ValueError(f"{strips_path} holds {bank.split} boxes; the pair metric uses val boxes only")
    outside = set(bank.sequence_ids()) - set(val_ids)
    if outside:
        raise ValueError(f"{strips_path} has {len(outside)} sequences outside val, e.g. {min(outside)}")
    missing = set(exclude) - set(bank.exclude)
    if missing:
        raise ValueError(f"{strips_path} was made without holding out {sorted(missing)}")

    generator = torch.Generator().manual_seed(seed)
    resize = v2.Resize(IMAGE_SIZE, antialias=True)
    ids = bank.sequence_ids()
    found = {"images": [], "labels": [], "boxes": [], "singles": [], "single_labels": [], "sequences": []}
    for _ in range(count * TRIES_PER_PAIR):
        if len(found["images"]) == count or not ids:
            break
        sequence_id = ids[torch.randint(len(ids), (), generator=generator).item()]
        partner = bank.partner(sequence_id, generator=generator, different=True)
        if partner is None:
            continue
        partner_id, a_left = partner
        a_name, a_box = _draw(bank.frames[sequence_id], generator)
        b_name, b_box = _draw(bank.frames[partner_id], generator)
        gap = torch.randint(GAP[0], GAP[1] + 1, (), generator=generator).item()
        a_image = _load_image(Path(data_dir) / a_name, resize)
        image, boxes = splice(a_image, a_box, _load_image(Path(data_dir) / b_name, resize), b_box, a_left, gap)
        digit, other = int(sequence_id.split("/")[0]), int(partner_id.split("/")[0])
        label = (digit, other) if a_left else (other, digit)
        assert f"{label[0]}{label[1]}" not in exclude
        found["images"].append(image)
        found["labels"].append(label)
        found["boxes"].append(boxes)
        found["singles"].append(a_image)
        found["single_labels"].append((digit, BLANK))
        found["sequences"].append((sequence_id, partner_id))
    if not found["images"]:
        raise ValueError(f"{strips_path} gives no pairs of two different digits")
    return {
        "images": torch.stack(found["images"]),
        "labels": torch.tensor(found["labels"]),
        "boxes": torch.stack(found["boxes"]),
        "singles": torch.stack(found["singles"]),
        "single_labels": torch.tensor(found["single_labels"]),
        "sequences": found["sequences"],
    }


def _draw(frames, generator):
    return frames[torch.randint(len(frames), (), generator=generator).item()]


@torch.no_grad()
def _read(model, images, device, batch_size):
    """Each image as a one-frame sequence: predicted digits (N, 2) and position attention (N, 2, C)."""
    digits, attention = [], []
    for part in images.split(batch_size):
        frames = NORMALIZE(part)[:, None].to(device)  # (n, 1, 3, H, W)
        mask = torch.ones(len(part), 1, dtype=torch.bool, device=device)
        digits.append(predict_digits(model(frames, mask)["logits"])[0].cpu())
        # Both readers keep the two digit positions' maps here (two-step: left and right).
        attention.append(model.digits.last_attention.cpu())
    return torch.cat(digits), torch.cat(attention)


def pair_metrics(model, pair_set, device, batch_size=64):
    """The scores in the module docstring, as floats."""
    model.eval()
    digits, attention = _read(model, pair_set["images"], device, batch_size)
    labels = pair_set["labels"]
    columns = IMAGE_SIZE[1] // CELL_STRIDE
    column_of_cell = torch.arange(attention.shape[-1]) % columns
    centroids = (attention * column_of_cell).sum(dim=-1)  # (N, 2)
    singles, _ = _read(model, pair_set["singles"], device, batch_size)
    return {
        "pair_accuracy": (digits == labels).all(dim=1).float().mean().item(),
        "second_is_right": (digits[:, 1] == labels[:, 1]).float().mean().item(),
        "second_is_left": (digits[:, 1] == labels[:, 0]).float().mean().item(),
        "left_right_gap": (centroids[:, 1] - centroids[:, 0]).mean().item(),
        "singles_accuracy": (singles == pair_set["single_labels"]).all(dim=1).float().mean().item(),
    }
