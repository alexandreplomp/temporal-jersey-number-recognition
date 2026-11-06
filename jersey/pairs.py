"""Two-digit training sequences spliced from single-digit ones.

The training data shows a digit in a two-digit position only through five numbers: a first
4 only in 48 (13 sequences), a second 9 only in 89. A model cannot learn to read 49 as
"a 4, then a 9" from that. Single-digit sequences are plentiful and their digit is known
from the folder; make_pairs.py finds a box around it in each image. Here two such digits,
cut as strips around their boxes, are put side by side, so every digit shows up in both
positions, on thousands of players and kits, and both digit boxes are known.

The two sequences come from different players with a similar kit colour, so the pair looks
like one shirt. B's strip is scaled to A's height, the digits sit GAP pixels apart, centred
where A's digit was, and cover it. The strips crossfade over a few pixels in the middle of
the gap, and the patch blends into the frame over as many, so no hard seam marks a spliced
number.

A pasted patch is still something a network can spot, and if only two-digit samples had one,
"pasted" would mean "two digits": two_step_s1_ex49's count head called 98% of spliced val pairs
two digits but only 44% of real two-digit shirts. So paste_single pastes a partner's digit
alone, the same way and as wide, labelled as one digit, and the feather width is drawn per
sequence (FEATHERS) for both, so neither a patch nor its seam says how many digits it holds
(Dwibedi et al. 2017, "Cut, Paste and Learn", blend the same paste in several ways for this).

Pixels, boxes [left, top, right, bottom] and centres (x, y) are in the resized IMAGE_SIZE input.
"""

from collections import defaultdict
from pathlib import Path
import json

import torch
from torch.nn import functional as F

MARGIN = 2  # pixels kept around a digit's box
FEATHER = 3  # default blend width; training draws it from FEATHERS
FEATHERS = (2, 5)  # blend widths, drawn per pasted sequence
GAP = (1, 4)  # pixels between the two digits, drawn per spliced sequence
KIT_NEIGHBOURS = 20  # a partner is drawn from this many sequences with the nearest kit colour
MIN_KEPT = 2  # a sequence needs this many frames with a found digit to be spliced
TRIES = 10  # partner draws before giving up, when every side of a draw makes a held-out number


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
    edge = torch.minimum(steps + 1, length - steps) / (feather + 1)
    return edge.clamp(max=1)


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
    centre = (digits[0] + digits[1]) / 2
    start = _clamp(round((box_a[0] + box_a[2]) / 2 - centre), 0, W - width) if width <= W else 0
    width = min(width, W - start)
    top = _clamp(round(box_a[1]) - MARGIN, 0, H)
    alpha = _ramp(height, feather)[:, None] * _ramp(width, feather)[None, :] * covered[None, :width]
    out = image_a.float()
    region = out[:, top:top + height, start:start + width]
    out[:, top:top + height, start:start + width] = alpha * patch[..., :width] + (1 - alpha) * region
    return out.round().clamp(0, 255).to(torch.uint8), (start, top)


def splice(image_a, box_a, image_b, box_b, a_left, gap=2, feather=FEATHER):
    """A's digit and B's digit side by side, centred where A's digit was.

    Returns the uint8 image (3, H, W) and the two digit boxes (2, 4) in reading order.
    a_left puts A's digit first. The pair is wider than A's digit and centred on it, so A's
    own digit is covered. feather is the width of both the seam and the patch's edge.
    """
    # Wide sides, so each strip reaches past the middle of the gap and the crossfade has room.
    side = MARGIN + feather
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
    weight_l = ((seam + feather / 2 - columns) / feather).clamp(0, 1)
    covers_l = (columns >= shift) & (columns < shift + left.shape[2])
    covers_r = (columns >= shift + offset) & (columns < shift + offset + right.shape[2])
    weight_l = torch.where(covers_r, weight_l, 1.0) * covers_l
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
    """B's digit alone over A's digit, pasted the way splice pastes a pair.

    Returns the uint8 image (3, H, W) and B's digit box (4,). B's strip is scaled to A's
    height and keeps extra shirt on both sides, half of A's digit plus half the gap each, so
    the patch is about as wide as a pair's. Its size, its edges and its kit then say nothing
    about how many digits it holds, and A's own digit is covered as in splice.
    """
    height = cut_strip(image_a, box_a)[0].shape[1]
    scale = height / cut_strip(image_b, box_b)[0].shape[1]
    extra = round((box_a[2] - box_a[0] + gap) / 2 / scale)  # in B's own pixels, before scaling
    strip, inner = _strip(image_b, box_b, height, side=MARGIN + feather + extra)
    covered = torch.ones(strip.shape[2])
    image, (x, y) = _blend_in(image_a, box_a, strip, covered, (inner[0], inner[2]), feather)
    box = torch.tensor([x + inner[0], y + inner[1], x + inner[2], y + inner[3]], dtype=torch.float)
    return image, box


class StripBank:
    """Where the digit is in each single-digit image, from make_pairs.py.

    Only images whose digit box was kept are used. Image names are relative to the data
    folder, as "4/<player>/<sequence>/<file>".
    """

    def __init__(self, path):
        found = json.loads(Path(path).read_text())
        self.locator = found.get("locator")
        self.split = found.get("split", "train")
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
        self.neighbours = _kit_neighbours(colours)

    def __contains__(self, sequence_id):
        return sequence_id in self.frames

    def sequence_ids(self):
        return list(self.frames)

    def single_partner(self, sequence_id, generator=None):
        """A kit-near sequence of another player whose digit, pasted alone, is not a held-out number.

        None if no such partner exists.
        """
        options = [other for other in self.neighbours.get(sequence_id, []) if other.split("/")[0] not in self.exclude]
        if not options:
            return None
        return options[torch.randint(len(options), (), generator=generator).item()]

    def partner(self, sequence_id, generator=None, different=False):
        """A kit-near sequence of another player and whether sequence_id's digit goes first.

        Never makes a held-out number; with different, never two equal digits. Returns None
        if TRIES draws all failed, or no partner exists.
        """
        options = self.neighbours.get(sequence_id, [])
        digit = sequence_id.split("/")[0]
        if different:
            options = [other for other in options if other.split("/")[0] != digit]
        if not options:
            return None
        for _ in range(TRIES):
            other = options[torch.randint(len(options), (), generator=generator).item()]
            other_digit = other.split("/")[0]
            # Only the sides that make an allowed number: 9 beside 4 with 49 held out gives 94 only.
            sides = [a_left for a_left in (True, False)
                     if (digit + other_digit if a_left else other_digit + digit) not in self.exclude]
            if sides:
                return other, sides[torch.randint(len(sides), (), generator=generator).item()]
        return None


def _kit_neighbours(colours):
    """For each sequence, the KIT_NEIGHBOURS sequences of other players nearest in kit colour."""
    ids = list(colours)
    if len(ids) < 2:
        return {}
    points = torch.stack([colours[sequence_id] for sequence_id in ids])
    distances = torch.cdist(points, points)
    players = [sequence_id.split("/")[1] for sequence_id in ids]
    of_player = defaultdict(list)
    for index, player in enumerate(players):
        of_player[player].append(index)
    for indices in of_player.values():  # never a player with themself
        distances[torch.tensor(indices)[:, None], torch.tensor(indices)[None, :]] = float("inf")
    count = min(KIT_NEIGHBOURS, len(ids) - 1)
    nearest = distances.topk(count, largest=False)
    return {
        ids[row]: [ids[column] for column, distance in zip(columns.tolist(), gaps.tolist()) if distance != float("inf")]
        for row, (columns, gaps) in enumerate(zip(nearest.indices, nearest.values))
    }
