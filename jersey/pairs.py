"""Two-digit training sequences spliced from single-digit ones.

The training data shows a digit in a two-digit position only through five numbers: a first
4 only in 48 (13 sequences), a second 9 only in 89. A model cannot learn to read 49 as
"a 4, then a 9" from that. Single-digit sequences are plentiful and their digit is known
from the folder; make_pairs.py finds where it is in each image. Here two such digits, cut as
strips around their centres, are put side by side, so every digit shows up in both positions,
on thousands of players and kits.

The two sequences come from different players with a similar kit colour, so the pair looks
like one shirt. The strips overlap by FEATHER pixels with a crossfade, and the pasted patch
blends into the frame over FEATHER pixels, so no hard seam marks a spliced number.

Pixels and centres (x, y) are in the resized IMAGE_SIZE input.
"""

from collections import defaultdict
from pathlib import Path
import json

import torch

# One digit is about 12 x 25 pixels in the 192 x 96 input: a strip keeps one digit and a margin.
STRIP_WIDTH = 14
STRIP_HEIGHT = 40
FEATHER = 3
KIT_NEIGHBOURS = 20  # a partner is drawn from this many sequences with the nearest kit colour
MIN_KEPT = 2  # a sequence needs this many frames with a found digit to be spliced
TRIES = 10  # partner draws before giving up, when draws keep making a held-out number


def _clamp(value, low, high):
    return min(max(value, low), high)


def cut_strip(image, centre, width=STRIP_WIDTH + FEATHER):
    """The STRIP_HEIGHT x width strip of image (3, H, W) around centre, moved inside the image."""
    H, W = image.shape[-2:]
    top = _clamp(round(centre[1] - STRIP_HEIGHT / 2), 0, H - STRIP_HEIGHT)
    left = _clamp(round(centre[0] - width / 2), 0, W - width)
    return image[:, top:top + STRIP_HEIGHT, left:left + width]


def _ramp(length):
    """Weights rising from 1 / (FEATHER + 1) to 1 over FEATHER pixels at both ends: (length,)."""
    steps = torch.arange(length, dtype=torch.float)
    edge = torch.minimum(steps + 1, length - steps) / (FEATHER + 1)
    return edge.clamp(max=1)


def splice(image_a, centre_a, image_b, centre_b, a_left):
    """A's digit and B's digit side by side, centred where A's digit was: uint8 (3, H, W).

    a_left puts A's digit first. The patch is twice a strip wide, so A's own digit is gone.
    """
    strip_a, strip_b = cut_strip(image_a, centre_a).float(), cut_strip(image_b, centre_b).float()
    left, right = (strip_a, strip_b) if a_left else (strip_b, strip_a)
    width = 2 * STRIP_WIDTH + FEATHER
    # The strips overlap by FEATHER pixels in the middle, where left fades into right.
    fade = torch.ones(width)
    fade[STRIP_WIDTH + FEATHER:] = 0
    fade[STRIP_WIDTH:STRIP_WIDTH + FEATHER] = torch.linspace(1, 0, FEATHER + 2)[1:-1]
    pair = torch.zeros(3, STRIP_HEIGHT, width)
    pair[..., :STRIP_WIDTH + FEATHER] += left * fade[:STRIP_WIDTH + FEATHER]
    pair[..., STRIP_WIDTH:] += right * (1 - fade[STRIP_WIDTH:])

    H, W = image_a.shape[-2:]
    top = _clamp(round(centre_a[1] - STRIP_HEIGHT / 2), 0, H - STRIP_HEIGHT)
    start = _clamp(round(centre_a[0] - width / 2), 0, W - width)
    alpha = _ramp(STRIP_HEIGHT)[:, None] * _ramp(width)[None, :]  # (STRIP_HEIGHT, width)
    out = image_a.float()
    region = out[:, top:top + STRIP_HEIGHT, start:start + width]
    out[:, top:top + STRIP_HEIGHT, start:start + width] = alpha * pair + (1 - alpha) * region
    return out.round().clamp(0, 255).to(torch.uint8)


class StripBank:
    """Where the digit is in each single-digit training image, from make_pairs.py.

    Only images whose digit was found (kept) are used. Image names are relative to the data
    folder, as "4/<player>/<sequence>/<file>".
    """

    def __init__(self, path):
        found = json.loads(Path(path).read_text())
        self.run = found["run"]
        self.exclude = tuple(found["exclude"])
        self.frames = {}  # sequence id -> [(image name, centre)] in time order
        self.anchors = {}  # sequence id -> (image name, centre) or None
        colours = {}
        for sequence_id, images in found["sequences"].items():
            kept = [image for image in images if image["kept"]]
            frames = [(image["image"], tuple(image["centre"])) for image in kept if not image["anchor"]]
            if len(frames) < MIN_KEPT:
                continue
            self.frames[sequence_id] = frames
            anchors = [(image["image"], tuple(image["centre"])) for image in kept if image["anchor"]]
            self.anchors[sequence_id] = anchors[0] if anchors else None
            colours[sequence_id] = torch.tensor([image["colour"] for image in kept]).mean(dim=0)
        self.neighbours = _kit_neighbours(colours)

    def __contains__(self, sequence_id):
        return sequence_id in self.frames

    def sequence_ids(self):
        return list(self.frames)

    def partner(self, sequence_id):
        """A kit-near sequence of another player and whether sequence_id's digit goes first.

        Never makes a held-out number; returns None if TRIES draws all did, or no partner exists.
        """
        options = self.neighbours.get(sequence_id, [])
        if not options:
            return None
        digit = sequence_id.split("/")[0]
        for _ in range(TRIES):
            other = options[torch.randint(len(options), ()).item()]
            a_left = bool(torch.randint(2, ()).item())
            other_digit = other.split("/")[0]
            number = digit + other_digit if a_left else other_digit + digit
            if number not in self.exclude:
                return other, a_left
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

