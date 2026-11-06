"""Load sequences and labels for training and evaluation.

Expected layout: data/<number>/<player>/<sequence>/<image>.jpg|.png, frames in time order
by the number in their name (<prefix>_<frame>_..., else by name), plus an optional
<...>_anchor image. The anchor is the clearest frame. The deployed model never sees it: it is
loaded only as an extra training example and for the anchor-only comparison. A sequence
without one uses its first frame there.

All randomness comes from torch, which DataLoader seeds per worker, so
torch.manual_seed(seed) in the training script is enough to repeat a run.
"""

from collections import Counter, defaultdict
from pathlib import Path
import json

from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import tv_tensors
from torchvision.transforms import v2

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
# Height, width. Crops are about 87 x 38 pixels (median), about 2.3:1, so 2:1 stretches
# the width by about 14%. Harmless, since training and evaluation resize the same way.
# Upscaled because the pretrained CNN expects larger objects than these crops.
IMAGE_SIZE = (192, 96)
MAX_FRAMES = 10
# Digits in reading order, padded with BLANK like text recognition: "4" is (4, BLANK).
BLANK = 10
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
# Where the number is in a player crop, as shares of height and width: below the head,
# above the shorts. Checked by eye on the shortcut check's picture.
TORSO_ROWS = (0.15, 0.55)
TORSO_COLUMNS = (0.1, 0.9)

# Colour predicts the number in this data: a linear classifier on the colours of the whole crop
# gets about 65% of val numbers right (the shortcut checks in evaluate.py). A digit's shape does not depend
# on colour, so every sample gets a random hue from the whole wheel, plus swapped channels, grey
# or inverted, since digits come light on dark and dark on light. Blur, low resolution and JPEG
# helped scene text recognition most (Atienza 2021); they are set at the crops' original scale
# (they are upscaled about 2.5x). Geometry stays mild and never flips: a mirrored digit is a
# different digit. Evaluation keeps the natural images.
AUGMENT = v2.Compose([
    v2.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.5),
    v2.RandomApply([v2.RandomChannelPermutation()], p=0.3),
    v2.RandomGrayscale(p=0.2),
    v2.RandomInvert(p=0.1),
    v2.RandomApply([v2.RandomChoice([  # one at a time: stacking stopped helping in Atienza 2021
        v2.GaussianBlur(kernel_size=21, sigma=(0.1, 3.5)),
        v2.Compose([v2.RandomResize(19, 39, antialias=True), v2.Resize(IMAGE_SIZE, antialias=True)]),
        v2.Compose([
            v2.Resize((IMAGE_SIZE[0] // 2, IMAGE_SIZE[1] // 2), antialias=True),
            v2.JPEG(quality=(40, 90)),
            v2.Resize(IMAGE_SIZE, antialias=True),
        ]),
    ])], p=0.5),
])
# ImageNet statistics, because the CNN starts from ImageNet weights.
NORMALIZE = v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])


def denormalize(images: torch.Tensor) -> torch.Tensor:
    """Undo NORMALIZE for viewing: (..., 3, H, W) -> values in [0, 1]."""
    mean = torch.tensor(IMAGENET_MEAN, device=images.device)[:, None, None]
    std = torch.tensor(IMAGENET_STD, device=images.device)[:, None, None]
    return (images * std + mean).clamp(0, 1)


def torso_mask(height: int, width: int) -> torch.Tensor:
    """True on the number area of a crop: (H, W)."""
    mask = torch.zeros(height, width, dtype=torch.bool)
    rows = slice(round(TORSO_ROWS[0] * height), round(TORSO_ROWS[1] * height))
    columns = slice(round(TORSO_COLUMNS[0] * width), round(TORSO_COLUMNS[1] * width))
    mask[rows, columns] = True
    return mask


# ----- labels -----

def number_of(sequence_id: str) -> str:
    """"49/<player>/<sequence>" -> "49"."""
    return sequence_id.split("/")[0]


def encode_label(number: str) -> tuple[int, int]:
    """Return (first, second) digit classes in reading order: "49" -> (4, 9), "4" -> (4, BLANK).

    Raises:
        ValueError: number is not one or two digits.
    """
    if not (number.isdigit() and 1 <= len(number) <= 2):
        raise ValueError(f"not a jersey number: {number!r}")
    if len(number) == 1:
        return int(number), BLANK
    return int(number[0]), int(number[1])


def decode_label(first: int, second: int) -> str:
    """Inverse of encode_label: (4, 9) -> "49", (4, BLANK) -> "4".

    Raises:
        ValueError: first is BLANK, since every number has a first digit.
    """
    if first == BLANK:
        raise ValueError(f"first digit cannot be BLANK: ({first}, {second})")
    return f"{first}" if second == BLANK else f"{first}{second}"


# ----- sequences and splits -----

def list_sequences(data_dir: str | Path) -> list[str]:
    """Return sorted sequence ids "number/player/sequence" under data_dir."""
    data_dir = Path(data_dir)
    return sorted(path.relative_to(data_dir).as_posix() for path in data_dir.glob("*/*/*") if path.is_dir())


def find_images(sequence_dir: Path) -> tuple[Path | None, list[Path]]:
    """Return the anchor path (None if there is none) and the frame paths in time order.

    Raises:
        ValueError: the sequence has no frames, or more than one anchor.
    """
    images = [path for path in Path(sequence_dir).iterdir() if path.suffix.lower() in IMAGE_SUFFIXES]
    anchors = [path for path in images if path.stem.endswith("_anchor")]
    if len(anchors) > 1:
        raise ValueError(f"{sequence_dir}: expected at most one anchor, found {len(anchors)}")
    frames = sorted((path for path in images if path not in anchors), key=_frame_order)
    if not frames:
        raise ValueError(f"{sequence_dir}: no frames besides the anchor")
    return (anchors[0] if anchors else None), frames


def read_splits(path: str | Path) -> dict[str, list[str]]:
    """Read a splits.json: {"train", "val", "test": sequence ids}. A run keeps its own copy."""
    return json.loads(Path(path).read_text())


def filter_numbers(sequence_ids: list[str], exclude: tuple[str, ...] = ()) -> list[str]:
    """Return the sequence ids whose number is not in exclude, e.g. to hold out "49"."""
    return [sequence_id for sequence_id in sequence_ids if number_of(sequence_id) not in exclude]


def linked_groups(issues: dict, sequence_ids: list[str]) -> list[set[str]]:
    """Return the connected groups of sequences linked by a shared image or a shared player.

    issues is prepare.py's data_issues.json. Links chain: if A shares an image with B, B's
    player has another sequence C, and C shares an image with D, then A, B, C and D are one
    group. Every sequence is in exactly one group, alone if unlinked.
    """
    parent = {sequence_id: sequence_id for sequence_id in sequence_ids}

    def root(sequence_id):
        while parent[sequence_id] != sequence_id:
            parent[sequence_id] = parent[parent[sequence_id]]
            sequence_id = parent[sequence_id]
        return sequence_id

    links = [
        {"/".join(path.split("/")[:3]) for path in group}  # image paths -> sequence ids
        for group in issues["shared_anchors"] + issues["conflicting_frames"]
        + issues["shared_between_players"] + issues["anchor_frame_links"]
    ]
    of_player = defaultdict(set)
    for sequence_id in sequence_ids:
        of_player[sequence_id.split("/")[1]].add(sequence_id)
    for first, *rest in links + list(of_player.values()):
        parent.setdefault(first, first)  # sequences named only in issues still pass links on
        for sequence_id in rest:
            parent.setdefault(sequence_id, sequence_id)
            parent[root(sequence_id)] = root(first)

    groups = defaultdict(set)
    for sequence_id in parent:
        groups[root(sequence_id)].add(sequence_id)
    return list(groups.values())


def held_out_partners(issues: dict, sequence_ids: list[str], exclude: tuple[str, ...]) -> set[str]:
    """Return the sequences of other numbers in the same linked group as a held-out one.

    filter_numbers drops held-out numbers only, so a linked partner left in train would
    still show the held-out player, e.g. a 49 whose player also appears as a 9.
    """
    kept = set(filter_numbers(sequence_ids, exclude))
    held = set(sequence_ids) - kept
    return {partner for group in linked_groups(issues, sequence_ids) if group & held for partner in group & kept}


def split_ids(splits: dict, split: str, exclude: tuple[str, ...], issues: dict) -> tuple[list[str], set[str]]:
    """Ids of one split without held-out numbers and their linked partners, and the partners dropped.

    Every block that trains or locates digits uses this, so all leave out the same sequences.
    """
    ids = filter_numbers(splits[split], exclude)
    if not exclude:
        return ids, set()
    everything = [sequence_id for name in ("train", "val", "test") for sequence_id in splits[name]]
    linked = held_out_partners(issues, everything, exclude) & set(ids)
    return [sequence_id for sequence_id in ids if sequence_id not in linked], linked


def make_sampler(sequence_ids: list[str], power: float = 0.5) -> WeightedRandomSampler:
    """Return a sampler that draws sequences of rare numbers more often. Training only.

    Each sequence is weighted by count(number) ** -power: 0 keeps the data as it is and 1
    balances numbers fully, which would repeat a single rare sequence hundreds of times per epoch.
    """
    numbers = [number_of(sequence_id) for sequence_id in sequence_ids]
    counts = Counter(numbers)
    return WeightedRandomSampler([counts[number] ** -power for number in numbers], len(numbers), replacement=True)


# ----- the dataset -----

class JerseyDataset(Dataset):
    """Sequences as fixed-size tensors, labelled by their number folder.

    augment picks frames at random and augments them, for training; otherwise frames are spread
    evenly, so evaluation is repeatable. load_anchor adds the anchor. pastes (a
    jersey.pairs.StripBank) with augment replaces a share pair_share of the single-digit
    sequences by spliced two-digit ones and a share single_share by a partner's digit pasted
    alone (see jersey/pairs.py); validation and evaluation never paste. real_pairs=False drops
    the real two-digit sequences.

    Each item is a dict with:
        frames       (max_frames, 3, H, W), zero-padded
        mask         (max_frames,), True for real frames
        label        (2,), first and second digit class in reading order
        sequence_id  "number/player/sequence"
        boxes        (max_frames, 2, 4), [left, top, right, bottom] of each digit after augmentation,
                     zero unless has_boxes
        has_boxes    (), True for spliced pairs, whose two digit boxes are known
        anchor       (3, H, W) and anchor_boxes (2, 4), only with load_anchor

    Raises:
        ValueError: a sequence is malformed. Checked here, so it fails before training starts.
    """

    def __init__(self, data_dir, sequence_ids, augment=False, load_anchor=False, max_frames=MAX_FRAMES,
                 pastes=None, pair_share=0.0, single_share=0.0, real_pairs=True):
        if pair_share + single_share > 1:
            raise ValueError(f"pair_share {pair_share} + single_share {single_share} is more than 1")
        self.data_dir = Path(data_dir)
        self.sequence_ids = [s for s in sequence_ids if real_pairs or len(number_of(s)) == 1]
        self.samples = [_index_sequence(self.data_dir, sequence_id) for sequence_id in self.sequence_ids]
        self.augment, self.load_anchor, self.max_frames = augment, load_anchor, max_frames
        self.pastes, self.pair_share, self.single_share = pastes, pair_share, single_share
        self.resize = v2.Resize(IMAGE_SIZE, antialias=True)
        self.transform = AUGMENT

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        pasted = self._pasted(self.sequence_ids[index])
        if pasted is None:
            anchor_path, frame_paths, label = self.samples[index]
            paths = [frame_paths[i] for i in pick_frames(len(frame_paths), self.max_frames, self.augment)]
            if self.load_anchor:
                paths.append(anchor_path or paths[0])
            images, boxes = torch.stack([load_image(path, self.resize) for path in paths]), None
        else:
            images, label, boxes = pasted
        has_boxes = boxes is not None

        if self.augment and boxes is None:
            images = self.transform(images)  # one call: every frame and the anchor get the same change
        elif self.augment:  # the boxes go through the same call, so they move with the geometry
            boxes = tv_tensors.BoundingBoxes(boxes.reshape(-1, 4), format="XYXY", canvas_size=images.shape[-2:])
            images, boxes = self.transform(images, boxes)
            boxes = boxes.as_subclass(torch.Tensor).reshape(len(images), 2, 4)
        images = NORMALIZE(images)
        if boxes is None:
            boxes = torch.zeros(len(images), 2, 4)

        item = {"label": torch.tensor(label), "sequence_id": self.sequence_ids[index],
                "has_boxes": torch.tensor(has_boxes)}
        if self.load_anchor:
            images, item["anchor"] = images[:-1], images[-1]
            boxes, item["anchor_boxes"] = boxes[:-1], boxes[-1]
        item["frames"], item["mask"] = _pad(images, self.max_frames)
        item["boxes"] = _pad(boxes, self.max_frames)[0]
        return item

    def _pasted(self, sequence_id):
        """A pasted version of the sequence, or None: (uint8 images, label, boxes or None)."""
        if not (self.augment and self.pastes is not None):
            return None
        draw = torch.rand(()).item()
        if draw >= self.pair_share + self.single_share:
            return None
        return self.pastes.paste(sequence_id, single=draw >= self.pair_share, max_frames=self.max_frames,
                                 with_anchor=self.load_anchor, load=lambda name: load_image(self.data_dir / name, self.resize))


def _index_sequence(data_dir, sequence_id):
    """Return (anchor path or None, frame paths, label) for one sequence, or raise naming it."""
    anchor_path, frame_paths = find_images(data_dir / sequence_id)
    try:
        label = encode_label(number_of(sequence_id))
    except ValueError as error:
        raise ValueError(f"{sequence_id}: {error}") from error
    return anchor_path, frame_paths, label


def _frame_order(path):
    """Sort key: the frame number in <prefix>_<frame>_..., numerically ("999" before "1000"), else the name."""
    parts = path.stem.split("_")
    if len(parts) > 1 and parts[1].isdigit():
        return 0, int(parts[1]), path.name
    return 1, 0, path.name


def pick_frames(count, max_frames, at_random):
    """Return the indices of at most max_frames frames, in time order: at random, or spread evenly."""
    if count <= max_frames:
        return torch.arange(count)
    if at_random:
        return torch.randperm(count)[:max_frames].sort().values
    return torch.linspace(0, count - 1, max_frames).round().long()


def load_image(path, resize):
    """Return the image as a resized uint8 tensor (3, H, W)."""
    with Image.open(path) as image:
        return resize(v2.functional.pil_to_tensor(image.convert("RGB")))


def _pad(frames, max_frames):
    """Zero-pad frames to max_frames and return them with a mask of the real frames."""
    padded = frames.new_zeros(max_frames, *frames.shape[1:])
    padded[:len(frames)] = frames
    return padded, torch.arange(max_frames) < len(frames)
