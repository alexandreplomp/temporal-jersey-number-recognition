"""Digit boxes found with no annotation, two-digit sequences spliced from single-digit ones, and the pair metric.

Why: the data shows a digit in a two-digit position only through a few numbers (a first 4 only in
48, 13 sequences). A model cannot learn to read 49 as "a 4, then a 9" from that. Single-digit
sequences are plentiful and their digit is known from the folder.

1. Find the digit (find_boxes). Two independent signals must agree:
     evidence  a local digit reader (LocalDigitReader), trained on single-digit crops with the
               folder's digit as the only label, scores each 8-pixel cell per digit; the strongest
               torso cell for the image's own digit gives the centre, the crop score how sure it is
     ink       around that centre the pixels split into two colour clusters (k-means); the smaller
               is the ink, and its column and row projections give a box
   A box is kept only if the reader is sure (>= MIN_PROB), the box contains the centre, and its
   size fits one digit. make_boxes writes them to results/boxes/<tag>/.
2. Paste (StripBank.paste). Two digits from players with a similar kit, cut as strips around their
   boxes, side by side and centred where the first one was (splice): every digit shows up in both
   positions, and both boxes are known for the attention loss. A pasted patch is itself something a
   network can spot, so a partner's digit is also pasted alone, the same way and as wide, labelled
   as one digit (paste_single), and the blend width varies per sequence for both (Dwibedi et al.
   2017, "Cut, Paste and Learn"). An earlier run without that read a pasted single digit right
   only 44% of the time, against 83% unpasted.
3. Measure (pair_metrics). A fixed set of spliced pairs of two different digits from validation
   players: does the model read left then right?

Pixels and boxes [left, top, right, bottom] are in the resized IMAGE_SIZE input.
"""

from collections import Counter, defaultdict
from pathlib import Path
import json
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from jersey.data import (
    BLANK, IMAGE_SIZE, NORMALIZE, JerseyDataset, denormalize, find_images, load_image, make_sampler, number_of,
    pick_frames, torso_mask,
)
from jersey.engine import make_loader, make_optimizer
from jersey.model import CELL_STRIDE, LOCAL_BLOCKS, LocalHead, make_backbone, predict_digits

DIGITS = 10
TOP_CELLS = 8
MIN_PROB = 0.9
WINDOW = (48, 32)  # height, width of the ink search around the evidence centre
BOX_HEIGHT = (12, 45)
BOX_WIDTH = (6, 28)
INK_SHARE = 0.15  # a row or column is ink if it has at least this share of the busiest one
KMEANS_STEPS = 10
MARGIN = 2  # pixels kept around a digit's box
FEATHER = 3  # default blend width; training draws it from FEATHERS
FEATHERS = (2, 5)  # blend widths, drawn per pasted sequence
GAP = (1, 4)  # pixels between the two digits, drawn per spliced sequence
KIT_NEIGHBOURS = 20  # a partner is drawn from this many sequences with the nearest kit colour
NEIGHBOUR_CHUNK = 512  # rows of the kit distance matrix computed at once, so memory stays bounded
MIN_KEPT = 2  # a sequence needs this many frames with a found digit to be pasted
TRIES = 10  # partner draws before giving up, when every side of a draw makes a held-out number
PAIR_COUNT = 300
LOCATOR_EPOCHS = 20
GATE = 0.75  # val image accuracy the reader needs before its boxes are used
TRAIN_FRAMES = 4  # frames per sequence per locator epoch, plus the anchor
BOX_FRAMES = 10  # evenly spread frames per sequence that get a box, plus the anchor


# ----- the local digit reader -----

def torso_cells(rows, columns, stride=CELL_STRIDE):
    """True for the cells whose centre is in the torso box: (rows, columns)."""
    return torso_mask(rows * stride, columns * stride)[stride // 2::stride, stride // 2::stride]


def crop_scores(cell_scores, top=TOP_CELLS):
    """Cell scores (N, K, h, w) -> crop scores (N, K): mean of the top cells inside the torso."""
    inside = torso_cells(*cell_scores.shape[-2:]).flatten().to(cell_scores.device)
    return cell_scores.flatten(2)[:, :, inside].topk(top, dim=-1).values.mean(dim=-1)


class LocalDigitReader(nn.Module):
    """backbone[:4] -> LocalHead -> 1 x 1 classifier: digit scores per cell (N, 10, H / 8, W / 8)."""

    def __init__(self, pretrained=True):
        super().__init__()
        self.freeze_batchnorm = pretrained
        self.backbone = make_backbone(pretrained)[:LOCAL_BLOCKS]
        self.head = LocalHead()
        self.classifier = nn.Conv2d(self.head.layers[0].out_channels, DIGITS, kernel_size=1)
        self.train()

    def train(self, mode=True):
        """As JerseyModel.train: the pretrained backbone keeps its BatchNorm statistics."""
        super().train(mode)
        if self.freeze_batchnorm:
            for layer in self.backbone.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eval()
        return self

    def forward(self, images):
        return self.classifier(self.head(self.backbone(images)))


def _images_and_digits(batch, device):
    """A batch of single-digit sequences as single images: every real frame and the anchor, with the digit."""
    mask = batch["mask"]
    images = [batch["frames"][mask], batch["anchor"]] if "anchor" in batch else [batch["frames"][mask]]
    owner = torch.arange(len(mask)).repeat_interleave(mask.sum(dim=1))
    if "anchor" in batch:
        owner = torch.cat([owner, torch.arange(len(mask))])
    return torch.cat(images).to(device), batch["label"][:, 0][owner].to(device)


def locator_epoch(model, loader, device, optimizer=None, scheduler=None):
    """One pass: trains if optimizer is given. Returns mean loss and accuracy per image."""
    model.train(optimizer is not None)
    total, right, count = 0.0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            images, digits = _images_and_digits(batch, device)
            scores = crop_scores(model(images))
            loss = F.cross_entropy(scores, digits)
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"locator loss is {loss.item()}, stopping")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
            total += loss.item() * len(digits)
            right += (scores.argmax(dim=1) == digits).sum().item()
            count += len(digits)
    return total / count, right / count


# ----- digit boxes -----

@torch.no_grad()
def evidence(locator, images, digits):
    """Digit centres (N, 2) as (x, y) from the strongest torso cell, and the probability (N,) of the digit."""
    cells = locator(images)  # (N, 10, h, w)
    N, _, h, w = cells.shape
    probability = crop_scores(cells).softmax(dim=1).gather(1, digits[:, None]).squeeze(1)
    own = cells[torch.arange(N, device=cells.device), digits].masked_fill(~torso_cells(h, w).to(cells.device), float("-inf"))
    best = own.flatten(1).argmax(dim=1)
    centres = torch.stack([(best % w + 0.5) * CELL_STRIDE, (best // w + 0.5) * CELL_STRIDE], dim=1).float()
    return centres, probability


def two_colours(pixels, steps=KMEANS_STEPS):
    """k-means with two clusters on pixels (P, 3), from the darkest and brightest pixel: labels (P,), centres (2, 3)."""
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

    The smaller colour cluster in the window is the ink, the larger the shirt. Columns, then rows
    within those columns, count as ink when they hold at least INK_SHARE of the busiest one; the
    run nearest the centre gives the box. None if there is no ink.
    """
    H, W = image.shape[-2:]
    x, y = centre
    x0 = min(max(round(x - WINDOW[1] / 2), 0), W - WINDOW[1])
    y0 = min(max(round(y - WINDOW[0] / 2), 0), H - WINDOW[0])
    labels, centres = two_colours(image[:, y0:y0 + WINDOW[0], x0:x0 + WINDOW[1]].flatten(1).T)
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


def keep(box, centre, probability, min_prob=MIN_PROB):
    """The three checks: a sure reader, a box containing the evidence centre, a digit-sized box."""
    if box is None or probability < min_prob:
        return False
    left, top, right, bottom = box
    return (left <= centre[0] < right and top <= centre[1] < bottom
            and BOX_HEIGHT[0] <= bottom - top <= BOX_HEIGHT[1] and BOX_WIDTH[0] <= right - left <= BOX_WIDTH[1])


def find_boxes(locator, images, digits):
    """Normalised images (N, 3, H, W) of known digits (N,) -> one result dict per image.

    Each has the evidence centre, the probability, the ink box (or None), the kit colour and kept.
    """
    centres, probability = evidence(locator, images, digits)
    results = []
    for image, centre, p in zip(denormalize(images).cpu(), centres.cpu().tolist(), probability.cpu().tolist()):
        box, kit = ink_box(image, centre)
        results.append({"centre": centre, "prob": round(p, 4), "box": box,
                        "colour": [round(v, 4) for v in kit.tolist()], "kept": keep(box, centre, p)})
    return results


# ----- making the boxes for a hold-out setting -----

def tag_of(exclude):
    """The name of a hold-out setting: ("49",) -> "ex49", () -> "all"."""
    return "ex" + "_".join(sorted(exclude)) if exclude else "all"


def train_locator(data_dir, train_ids, val_ids, device, epochs=LOCATOR_EPOCHS, batch_size=32, workers=4,
                  pretrained=True):
    """Train the reader on single-digit sequences; keep the epoch with the lowest val loss. Returns it and its record."""
    model = LocalDigitReader(pretrained=pretrained).to(device)
    train_loader = make_loader(JerseyDataset(data_dir, train_ids, augment=True, load_anchor=True, max_frames=TRAIN_FRAMES),
                               batch_size, workers, device, sampler=make_sampler(train_ids))
    val_loader = make_loader(JerseyDataset(data_dir, val_ids, max_frames=TRAIN_FRAMES), batch_size, workers, device)
    optimizer, scheduler = make_optimizer(model, 3e-4, 3e-5, epochs * len(train_loader), backbone_prefix="backbone.")
    history, best, best_state = [], None, None
    for epoch in range(1, epochs + 1):
        train_loss, _ = locator_epoch(model, train_loader, device, optimizer, scheduler)
        val_loss, val_accuracy = locator_epoch(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_accuracy})
        if best is None or val_loss < best["val_loss"]:
            best, best_state = history[-1], {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print(f"locator epoch {epoch:>2}/{epochs}  val loss {val_loss:.3f}  val image accuracy {val_accuracy:.3f}")
    model.load_state_dict(best_state)
    return model, {"epochs": history, "best": best}


class _Images(Dataset):
    """(name, digit) entries as normalised images."""

    def __init__(self, data_dir, entries):
        self.data_dir, self.entries = Path(data_dir), entries
        self.resize = v2.Resize(IMAGE_SIZE, antialias=True)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        name, digit, _ = self.entries[index]
        return {"image": NORMALIZE(load_image(self.data_dir / name, self.resize)[None])[0], "digit": digit}


@torch.no_grad()
def boxes_for(locator, data_dir, sequence_ids, device, workers=4):
    """{sequence id: [per image: image, anchor, centre, prob, box, colour, kept]} for evenly spread frames and the anchor."""
    locator.eval()
    entries = []
    for sequence_id in sequence_ids:
        anchor, frames = find_images(Path(data_dir) / sequence_id)
        for path in [frames[i] for i in pick_frames(len(frames), BOX_FRAMES, at_random=False).tolist()] + [anchor] * bool(anchor):
            entries.append((f"{sequence_id}/{path.name}", int(number_of(sequence_id)), path == anchor))
    results = []
    for batch in DataLoader(_Images(data_dir, entries), batch_size=128, num_workers=workers):
        results += find_boxes(locator, batch["image"].to(device), batch["digit"].to(device))
    sequences = defaultdict(list)
    for (name, _, is_anchor), found in zip(entries, results):
        sequences[name.rsplit("/", 1)[0]].append({"image": name, "anchor": is_anchor, **found})
    return sequences


def summarise(sequences, bank):
    """Share of frames kept and usable sequences per digit."""
    per_digit = defaultdict(Counter)
    for sequence_id, images in sequences.items():
        counts = per_digit[number_of(sequence_id)]
        counts["sequences"] += 1
        counts["usable sequences"] += sequence_id in bank
        counts["frames"] += sum(not image["anchor"] for image in images)
        counts["frames kept"] += sum(image["kept"] and not image["anchor"] for image in images)
    return {digit: {**counts, "frames kept share": round(counts["frames kept"] / max(counts["frames"], 1), 3)}
            for digit, counts in sorted(per_digit.items())}


def make_boxes(data_dir, out, train_ids, val_ids, exclude, device, epochs=LOCATOR_EPOCHS, workers=4, pretrained=True):
    """Train the reader on single-digit train and val ids (or reuse out/locator.pt) and write the boxes to out.

    Writes locator.pt, locator.json, train.json (feeds training), val.json (feeds only the pair
    metric) and summary.json. The ids must already leave out held-out numbers and their partners.

    Raises:
        SystemExit: the reader reads fewer than GATE of val images right; no boxes are written.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    train_ids = [s for s in train_ids if len(number_of(s)) == 1]
    val_ids = [s for s in val_ids if len(number_of(s)) == 1]
    if (out / "locator.pt").exists():
        locator = LocalDigitReader(pretrained=False).to(device)
        locator.load_state_dict(torch.load(out / "locator.pt", map_location=device))
        record = json.loads((out / "locator.json").read_text())
    else:
        locator, record = train_locator(data_dir, train_ids, val_ids, device, epochs, workers=workers, pretrained=pretrained)
        torch.save({k: v.cpu() for k, v in locator.state_dict().items()}, out / "locator.pt")
        (out / "locator.json").write_text(json.dumps({**record, "exclude": list(exclude)}, indent=2))
    accuracy = record["best"]["val_accuracy"]
    print(f"check: locator val image accuracy {accuracy:.3f} (gate {GATE})")
    if accuracy < GATE:
        raise SystemExit(f"check failed: the locator reads {accuracy:.3f} of val images, below {GATE}; no boxes made")
    summary = {}
    for split, ids in (("train", train_ids), ("val", val_ids)):
        sequences = boxes_for(locator, data_dir, ids, device, workers)
        (out / f"{split}.json").write_text(json.dumps({"split": split, "exclude": list(exclude), "sequences": sequences}))
        summary[split] = summarise(sequences, StripBank(out / f"{split}.json"))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ----- cutting and pasting -----

def _clamp(value, low, high):
    return min(max(value, low), high)


def cut_strip(image, box, margin=MARGIN, side=MARGIN):
    """The box plus margin above and below and side left and right, inside image (3, H, W).

    Returns the strip and the box in the strip's own coordinates.
    """
    H, W = image.shape[-2:]
    left, top, right, bottom = (round(value) for value in box)
    x0, y0 = _clamp(left - side, 0, W), _clamp(top - margin, 0, H)
    x1, y1 = _clamp(right + side, 0, W), _clamp(bottom + margin, 0, H)
    return image[:, y0:y1, x0:x1], [left - x0, top - y0, right - x0, bottom - y0]


def _ramp(length, feather=FEATHER):
    """Weights rising from 1 / (feather + 1) to 1 over feather pixels at both ends: (length,)."""
    steps = torch.arange(length, dtype=torch.float)
    return (torch.minimum(steps + 1, length - steps) / (feather + 1)).clamp(max=1)


def _strip(image, box, height=None, side=MARGIN):
    """The digit's strip as floats, scaled to height if given, and its digit box inside the strip."""
    strip, inner = cut_strip(image.float(), box, side=side)
    if height is None or height == strip.shape[1]:
        return strip, inner
    scale = height / strip.shape[1]
    width = max(1, round(strip.shape[2] * scale))
    strip = F.interpolate(strip[None], size=(height, width), mode="bilinear", align_corners=False, antialias=True)[0]
    return strip, [value * scale for value in inner]


def _blend_in(image_a, box_a, patch, covered, digits, feather):
    """Blend patch (3, h, w) into image_a with its digits (left, right in patch x) centred on A's digit.

    covered (w,) is 0 where the patch has no pixels. Returns the uint8 image and the patch's
    top-left corner (x, y) in the image.
    """
    H, W = image_a.shape[-2:]
    height, width = patch.shape[1:]
    start = _clamp(round((box_a[0] + box_a[2]) / 2 - (digits[0] + digits[1]) / 2), 0, W - width) if width <= W else 0
    width = min(width, W - start)
    top = _clamp(round(box_a[1]) - MARGIN, 0, H)
    alpha = _ramp(height, feather)[:, None] * _ramp(width, feather)[None, :] * covered[None, :width]
    out = image_a.float()
    region = out[:, top:top + height, start:start + width]
    out[:, top:top + height, start:start + width] = alpha * patch[..., :width] + (1 - alpha) * region
    return out.round().clamp(0, 255).to(torch.uint8), (start, top)


def splice(image_a, box_a, image_b, box_b, a_left, gap=2, feather=FEATHER):
    """A's digit and B's digit side by side, centred where A's digit was, covering it.

    Returns the uint8 image (3, H, W) and the two digit boxes (2, 4) in reading order;
    a_left puts A's digit first. feather is the width of both the seam and the patch's edge.
    """
    side = MARGIN + feather  # wide sides, so each strip reaches past the middle of the gap
    strip_a, inner_a = _strip(image_a, box_a, side=side)
    strip_b, inner_b = _strip(image_b, box_b, strip_a.shape[1], side=side)
    height = strip_a.shape[1]
    (left, inner_l), (right, inner_r) = ((strip_a, inner_a), (strip_b, inner_b))[::1 if a_left else -1]

    # Canvas x of each strip: left at 0, right so that its digit starts gap pixels after the left one.
    offset = round(inner_l[2] + gap - inner_r[0])
    width = max(left.shape[2], offset + right.shape[2]) - min(0, offset)
    shift = -min(0, offset)  # only if the right strip would start left of 0
    columns = torch.arange(width, dtype=torch.float) + 0.5
    seam = shift + inner_l[2] + gap / 2
    covers_l = (columns >= shift) & (columns < shift + left.shape[2])
    covers_r = (columns >= shift + offset) & (columns < shift + offset + right.shape[2])
    weight_l = torch.where(covers_r, ((seam + feather / 2 - columns) / feather).clamp(0, 1), 1.0) * covers_l
    weight_r = (1 - weight_l) * covers_r
    pair = torch.zeros(3, height, width)
    pair[..., shift:shift + left.shape[2]] += left * weight_l[shift:shift + left.shape[2]]
    pair[..., shift + offset:shift + offset + right.shape[2]] += right * weight_r[shift + offset:shift + offset + right.shape[2]]
    covered = (weight_l + weight_r).clamp(max=1)  # 0 only in a hole between strips cut off by the image edge

    image, (x, y) = _blend_in(image_a, box_a, pair, covered, (shift + inner_l[0], shift + offset + inner_r[2]), feather)
    boxes = torch.tensor([
        [x + shift + inner_l[0], y + inner_l[1], x + shift + inner_l[2], y + inner_l[3]],
        [x + shift + offset + inner_r[0], y + inner_r[1], x + shift + offset + inner_r[2], y + inner_r[3]],
    ], dtype=torch.float)
    return image, boxes


def paste_single(image_a, box_a, image_b, box_b, gap=2, feather=FEATHER):
    """B's digit alone over A's digit, pasted the way splice pastes a pair: uint8 image and B's box (4,).

    B's strip is scaled to A's height and keeps extra shirt on both sides (half of A's digit
    plus half the gap each), so the patch is about as wide as a pair's and says nothing about
    how many digits it holds.
    """
    height = cut_strip(image_a, box_a)[0].shape[1]
    scale = height / cut_strip(image_b, box_b)[0].shape[1]
    extra = round((box_a[2] - box_a[0] + gap) / 2 / scale)  # in B's own pixels, before scaling
    strip, inner = _strip(image_b, box_b, height, side=MARGIN + feather + extra)
    image, (x, y) = _blend_in(image_a, box_a, strip, torch.ones(strip.shape[2]), (inner[0], inner[2]), feather)
    return image, torch.tensor([x + inner[0], y + inner[1], x + inner[2], y + inner[3]], dtype=torch.float)


# ----- the bank of found digits -----

class StripBank:
    """Where the digit is in each single-digit image, from make_boxes (train.json or val.json).

    Only images whose digit box was kept are used. Image names are relative to the data folder.
    """

    def __init__(self, path):
        found = json.loads(Path(path).read_text())
        self.split = found["split"]
        self.exclude = tuple(found["exclude"])
        self.frames = {}  # sequence id -> [(image name, box)] in time order
        self.anchors = {}  # sequence id -> (image name, box) or None
        colours = {}
        for sequence_id, images in found["sequences"].items():
            kept = [image for image in images if image["kept"]]
            frames = [(image["image"], tuple(image["box"])) for image in kept if not image["anchor"]]
            if len(frames) < MIN_KEPT:
                continue
            self.frames[sequence_id] = frames
            anchors = [(image["image"], tuple(image["box"])) for image in kept if image["anchor"]]
            self.anchors[sequence_id] = anchors[0] if anchors else None
            colours[sequence_id] = torch.tensor([image["colour"] for image in kept]).mean(dim=0)
        self.neighbours = kit_neighbours(colours)

    def __contains__(self, sequence_id):
        return sequence_id in self.frames

    def sequence_ids(self):
        return list(self.frames)

    def single_partner(self, sequence_id, generator=None):
        """A kit-near sequence of another player whose digit, pasted alone, is not held out; or None."""
        options = [other for other in self.neighbours.get(sequence_id, []) if number_of(other) not in self.exclude]
        return options[torch.randint(len(options), (), generator=generator).item()] if options else None

    def partner(self, sequence_id, generator=None, different=False):
        """A kit-near sequence of another player and whether sequence_id's digit goes first, or None.

        Never makes a held-out number; with different, never two equal digits.
        """
        digit = number_of(sequence_id)
        options = [o for o in self.neighbours.get(sequence_id, []) if not different or number_of(o) != digit]
        for _ in range(TRIES if options else 0):
            other = options[torch.randint(len(options), (), generator=generator).item()]
            # Only the sides that make an allowed number: 9 beside 4 with 49 held out gives 94 only.
            sides = [a_left for a_left in (True, False)
                     if (digit + number_of(other) if a_left else number_of(other) + digit) not in self.exclude]
            if sides:
                return other, sides[torch.randint(len(sides), (), generator=generator).item()]
        return None

    def paste(self, sequence_id, single, max_frames, with_anchor, load):
        """A pasted version of a single-digit sequence: (uint8 images (n, 3, H, W), label, boxes or None).

        A spliced pair with its digit boxes (n, 2, 4), or with single a partner's digit pasted
        alone, labelled as one digit, without boxes (the attention loss teaches left and right
        only). load(name) loads an image. None for a sequence without found digits or partner.
        """
        if len(number_of(sequence_id)) != 1 or sequence_id not in self:
            return None
        found = (self.single_partner(sequence_id), None) if single else self.partner(sequence_id)
        if found is None or found[0] is None:
            return None
        partner_id, a_left = found
        own, other = self.frames[sequence_id], self.frames[partner_id]
        picked = pick_frames(len(own), max_frames, at_random=True)
        # The partner's frames in time order too, repeated if it has fewer.
        matched = (pick_frames(len(other), len(picked), at_random=True) if len(other) >= len(picked)
                   else torch.arange(len(picked)) % len(other))
        pairs = [(own[i], other[j]) for i, j in zip(picked.tolist(), matched.tolist())]
        if with_anchor:
            anchors = (self.anchors[sequence_id], self.anchors[partner_id])
            pairs.append(anchors if None not in anchors else pairs[0])  # else the first frame stands in
        # One spacing and one blend width for the whole sequence, drawn the same way for both kinds.
        gap = torch.randint(GAP[0], GAP[1] + 1, ()).item()
        feather = torch.randint(FEATHERS[0], FEATHERS[1] + 1, ()).item()
        partner_digit = int(number_of(partner_id))
        if single:
            images = [paste_single(load(a), a_box, load(b), b_box, gap, feather)[0] for (a, a_box), (b, b_box) in pairs]
            return torch.stack(images), (partner_digit, BLANK), None
        spliced = [splice(load(a), a_box, load(b), b_box, a_left, gap, feather) for (a, a_box), (b, b_box) in pairs]
        digit = int(number_of(sequence_id))
        label = (digit, partner_digit) if a_left else (partner_digit, digit)
        return torch.stack([image for image, _ in spliced]), label, torch.stack([boxes for _, boxes in spliced])


def kit_neighbours(colours):
    """For each sequence, the KIT_NEIGHBOURS sequences of other players nearest in kit colour.

    The distances are computed NEIGHBOUR_CHUNK rows at a time, so memory grows with the number
    of sequences, not with its square.
    """
    ids = list(colours)
    if len(ids) < 2:
        return {}
    points = torch.stack([colours[sequence_id] for sequence_id in ids])
    player_names = [sequence_id.split("/")[1] for sequence_id in ids]
    index_of = {name: i for i, name in enumerate(dict.fromkeys(player_names))}
    players = torch.tensor([index_of[name] for name in player_names])
    count = min(KIT_NEIGHBOURS, len(ids) - 1)
    neighbours = {}
    for start in range(0, len(ids), NEIGHBOUR_CHUNK):
        distances = torch.cdist(points[start:start + NEIGHBOUR_CHUNK], points)
        distances[players[start:start + NEIGHBOUR_CHUNK, None] == players[None, :]] = float("inf")  # never themself
        nearest = distances.topk(count, largest=False)
        for row, (columns, gaps) in enumerate(zip(nearest.indices.tolist(), nearest.values.tolist())):
            neighbours[ids[start + row]] = [ids[c] for c, gap in zip(columns, gaps) if gap != float("inf")]
    return neighbours


# ----- the pair metric -----

def build_pair_set(strips_path, val_ids, exclude, data_dir, count=PAIR_COUNT, seed=0):
    """A fixed set of spliced val images of two different digits, with the unspliced and pasted-single versions.

    Returns images, labels, boxes, singles and single_labels (A's image unspliced), pasted and
    pasted_labels (B's digit pasted alone over A), and the sequence ids used.

    Raises:
        ValueError: the boxes are not val boxes, include a sequence outside val_ids, or were made
            without holding out exclude.
    """
    bank = StripBank(strips_path)
    if bank.split != "val":
        raise ValueError(f"{strips_path} holds {bank.split} boxes; the pair metric uses val boxes only")
    outside = set(bank.sequence_ids()) - set(val_ids)
    if outside:
        raise ValueError(f"{strips_path} has {len(outside)} sequences outside val, e.g. {min(outside)}")
    if set(exclude) - set(bank.exclude):
        raise ValueError(f"{strips_path} was made without holding out {sorted(set(exclude) - set(bank.exclude))}")

    generator = torch.Generator().manual_seed(seed)
    resize = v2.Resize(IMAGE_SIZE, antialias=True)
    ids = bank.sequence_ids()
    keys = ("images", "labels", "boxes", "singles", "single_labels", "pasted", "pasted_labels", "sequences")
    found = {key: [] for key in keys}
    for _ in range(count * 20 if ids else 0):
        if len(found["images"]) == count:
            break
        sequence_id = ids[torch.randint(len(ids), (), generator=generator).item()]
        partner = bank.partner(sequence_id, generator=generator, different=True)
        if partner is None:
            continue
        partner_id, a_left = partner
        frames_a, frames_b = bank.frames[sequence_id], bank.frames[partner_id]
        a_name, a_box = frames_a[torch.randint(len(frames_a), (), generator=generator).item()]
        b_name, b_box = frames_b[torch.randint(len(frames_b), (), generator=generator).item()]
        gap = torch.randint(GAP[0], GAP[1] + 1, (), generator=generator).item()
        a_image, b_image = load_image(Path(data_dir) / a_name, resize), load_image(Path(data_dir) / b_name, resize)
        image, boxes = splice(a_image, a_box, b_image, b_box, a_left, gap)
        digit, other = int(number_of(sequence_id)), int(number_of(partner_id))
        label = (digit, other) if a_left else (other, digit)
        assert f"{label[0]}{label[1]}" not in exclude
        for key, value in zip(keys, (image, label, boxes, a_image, (digit, BLANK),
                                     paste_single(a_image, a_box, b_image, b_box, gap)[0], (other, BLANK),
                                     (sequence_id, partner_id))):
            found[key].append(value)
    if not found["images"]:
        raise ValueError(f"{strips_path} gives no pairs of two different digits")
    return {key: (found[key] if key == "sequences" else
                  torch.stack(found[key]) if torch.is_tensor(found[key][0]) else torch.tensor(found[key]))
            for key in keys}


@torch.no_grad()
def _read(model, images, device, batch_size=64):
    """Each image as a one-frame sequence: predicted digits (N, 2) and the two positions' attention (N, 2, C)."""
    digits, attention = [], []
    for part in images.split(batch_size):
        frames = NORMALIZE(part)[:, None].to(device)
        out = model(frames, torch.ones(len(part), 1, dtype=torch.bool, device=device))
        digits.append(predict_digits(out["logits"])[0].cpu())
        attention.append(out["attention"][:, 0, list(model.positions)].cpu())
    return torch.cat(digits), torch.cat(attention)


def pair_metrics(model, pair_set, device):
    """Reading digit by digit, scored without the model's count offset (the pairs are all two-digit):

        pair_accuracy            both digits right
        second_is_right          position 2 names the right digit
        second_is_left           position 2 names the left digit: the failure of a reader that
                                 takes the same whole-crop content for both positions
        left_right_gap           columns of the reader's grid between where position 2 and position 1 look
        singles_accuracy         the unspliced image read as one digit
        pasted_singles_accuracy  a partner's digit pasted alone read as one digit: low while a
                                 pasted patch is still taken for two digits
    """
    saved, model.count_offset = model.count_offset, 0.0
    model.eval()
    try:
        digits, attention = _read(model, pair_set["images"], device)
        pasted, _ = _read(model, pair_set["pasted"], device)
        singles, _ = _read(model, pair_set["singles"], device)
    finally:
        model.count_offset = saved
    labels = pair_set["labels"]
    column = torch.arange(attention.shape[-1]) % model.reader.attention_grid[1]  # the reader's own grid
    centroids = (attention * column).sum(dim=-1)  # (N, 2)

    def share(flags):
        return flags.float().mean().item()

    return {
        "pair_accuracy": share((digits == labels).all(dim=1)),
        "second_is_right": share(digits[:, 1] == labels[:, 1]),
        "second_is_left": share(digits[:, 1] == labels[:, 0]),
        "left_right_gap": (centroids[:, 1] - centroids[:, 0]).mean().item(),
        "singles_accuracy": share((singles == pair_set["single_labels"]).all(dim=1)),
        "pasted_singles_accuracy": share((pasted == pair_set["pasted_labels"]).all(dim=1)),
    }
