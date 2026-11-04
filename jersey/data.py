"""Load sequences and labels for training and evaluation.

Expected layout: data/<number>/<player>/<sequence>/<image>.jpg|.png
Frames are named <prefix>_<frame>_<sequence>, plus one <...>_anchor image.
The anchor is the clearest frame. The deployed model never sees it: it is loaded
only for training and for the anchor-only baseline.

All randomness comes from torch, which DataLoader seeds per worker, so
torch.manual_seed(seed) in the training script is enough to repeat a run.
"""

from collections import Counter, defaultdict
from pathlib import Path
import json

from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision.transforms import v2

IMAGE_SUFFIXES = {".jpg", ".png"}
# Height, width. Crops are about 87 x 38 pixels (median), about 2.3:1, so 2:1 stretches
# the width by about 14%. Harmless, since training and evaluation resize the same way.
# Upscaled because the pretrained CNN expects larger objects than these crops.
IMAGE_SIZE = (192, 96)
MAX_FRAMES = 10
# Digits in reading order, padded with BLANK like text recognition: "4" is (4, BLANK).
BLANK = 10
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# No flips: a mirrored digit is a different or invalid digit.
AUGMENT = v2.Compose([
    v2.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
])
# ImageNet statistics, because the CNN starts from ImageNet weights.
NORMALIZE = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def denormalize(images: torch.Tensor) -> torch.Tensor:
    """Undo NORMALIZE for viewing: (..., 3, H, W) -> values in [0, 1]."""
    mean = torch.tensor(IMAGENET_MEAN, device=images.device)[:, None, None]
    std = torch.tensor(IMAGENET_STD, device=images.device)[:, None, None]
    return (images * std + mean).clamp(0, 1)


def run_splits(run_dir: str | Path, fallback: str | Path) -> dict[str, list[str]]:
    """Return the splits a run was trained with: its own splits.json, else fallback.

    Rerunning make_splits.py reshuffles players, so only the run's own copy is sure to keep
    its training players out of val and test. Older runs have no copy, hence the warning.
    """
    own = Path(run_dir) / "splits.json"
    if own.exists():
        return json.loads(own.read_text())
    print(f"warning: {run_dir} has no splits.json; using {fallback}, which may differ from its training split")
    return json.loads(Path(fallback).read_text())


def list_sequences(data_dir: str | Path) -> list[str]:
    """Return sorted sequence ids "number/player/sequence" under data_dir."""
    data_dir = Path(data_dir)
    return sorted(
        path.relative_to(data_dir).as_posix()
        for path in data_dir.glob("*/*/*")
        if path.is_dir()
    )


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


def find_images(sequence_dir: Path) -> tuple[Path, list[Path]]:
    """Return the anchor path and the frame paths in time order.

    Raises:
        ValueError: the sequence has no frames, not exactly one anchor, or a badly named frame.
    """
    images = [path for path in sequence_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES]
    anchors = [path for path in images if path.stem.endswith("_anchor")]
    if len(anchors) != 1:
        raise ValueError(f"{sequence_dir}: expected one anchor, found {len(anchors)}")
    frames = sorted((path for path in images if path not in anchors), key=_frame_number)
    if not frames:
        raise ValueError(f"{sequence_dir}: no frames besides the anchor")
    return anchors[0], frames


def filter_numbers(sequence_ids: list[str], exclude: tuple[str, ...] = ()) -> list[str]:
    """Return the sequence ids whose number is not in exclude, e.g. to hold out "89".

    Filter the ids before building both the Dataset and the sampler, so they agree.
    """
    return [sequence_id for sequence_id in sequence_ids if sequence_id.split("/")[0] not in exclude]


def linked_groups(issues: dict, sequence_ids: list[str]) -> list[set[str]]:
    """Return the connected groups of sequences linked by a shared image or a shared player.

    issues is check_data.py's output, the links make_splits.py keeps in train. Links chain:
    if A shares an image with B, B's player has another sequence C, and C shares an image
    with D, then A, B, C and D are one group. All sequences of a player are linked, since
    they show the same player. Every sequence is in exactly one group, alone if unlinked.
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
        # Sequences named only in issues still pass links on, so a chain never breaks.
        parent.setdefault(first, first)
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
    still show the held-out player, e.g. a 49 whose player also appears as a 9. The whole
    group goes, not only direct partners, since a chain of links can reach the same player.
    """
    kept = set(filter_numbers(sequence_ids, exclude))
    held = set(sequence_ids) - kept
    return {partner for group in linked_groups(issues, sequence_ids) if group & held for partner in group & kept}


def make_sampler(sequence_ids: list[str], power: float = 0.5) -> WeightedRandomSampler:
    """Return a sampler that draws sequences of rare numbers more often. Training only.

    Each sequence is weighted by count(number) ** -power: 0 keeps the data as it is and
    1 balances numbers fully. Full balance would repeat the single 88 sequence hundreds
    of times per epoch, so the default is the square root.
    """
    numbers = [sequence_id.split("/")[0] for sequence_id in sequence_ids]
    counts = Counter(numbers)
    weights = [counts[number] ** -power for number in numbers]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


class JerseyDataset(Dataset):
    """Sequences as fixed-size tensors, labelled by their number folder.

    augment picks frames at random and augments them, for training.
    load_anchor adds the anchor, for training and the anchor-only baseline.
    Both off gives the repeatable input used for evaluation.

    Each item is a dict with:
        frames      (max_frames, 3, H, W), zero-padded
        mask        (max_frames,), True for real frames
        label       (2,), first and second digit class in reading order
        sequence_id "number/player/sequence", for reviewing errors
        anchor      (3, H, W), only with load_anchor

    Raises:
        ValueError: a sequence is malformed. Checked here, so it fails before training starts.
    """

    def __init__(
        self,
        data_dir: str | Path,
        sequence_ids: list[str],
        augment: bool = False,
        load_anchor: bool = False,
        max_frames: int = MAX_FRAMES,
        image_size: tuple[int, int] = IMAGE_SIZE,
    ):
        data_dir = Path(data_dir)
        self.sequence_ids = list(sequence_ids)
        self.samples = [_index_sequence(data_dir, sequence_id) for sequence_id in self.sequence_ids]
        self.augment = augment
        self.load_anchor = load_anchor
        self.max_frames = max_frames
        self.resize = v2.Resize(image_size, antialias=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        anchor_path, frame_paths, label = self.samples[index]
        picked = _pick_frames(len(frame_paths), self.max_frames, at_random=self.augment)
        paths = [frame_paths[i] for i in picked]
        if self.load_anchor:
            paths.append(anchor_path)

        images = torch.stack([_load_image(path, self.resize) for path in paths])
        if self.augment:
            # One call on the whole stack, so every frame and the anchor get the same augmentation.
            images = AUGMENT(images)
        images = NORMALIZE(images)

        if self.load_anchor:
            images, anchor = images[:-1], images[-1]
        frames, mask = _pad(images, self.max_frames)
        item = {
            "frames": frames,
            "mask": mask,
            "label": torch.tensor(label),
            "sequence_id": self.sequence_ids[index],
        }
        if self.load_anchor:
            item["anchor"] = anchor
        return item


def _index_sequence(data_dir, sequence_id):
    """Return (anchor path, frame paths, label) for one sequence, or raise naming it."""
    anchor_path, frame_paths = find_images(data_dir / sequence_id)
    try:
        label = encode_label(sequence_id.split("/")[0])
    except ValueError as error:
        raise ValueError(f"{sequence_id}: {error}") from error
    return anchor_path, frame_paths, label


def _frame_number(path):
    # Numeric, since frame ids differ in length and "1000" sorts before "999" as text.
    try:
        return int(path.stem.split("_")[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"{path}: frame name is not <prefix>_<frame>_<sequence>") from error


def _pick_frames(count, max_frames, at_random):
    """Return the indices of at most max_frames frames, in time order.

    At random for training variety, otherwise spread evenly so the result is repeatable.
    """
    if count <= max_frames:
        return torch.arange(count)
    if at_random:
        return torch.randperm(count)[:max_frames].sort().values
    return torch.linspace(0, count - 1, max_frames).round().long()


def _load_image(path, resize):
    """Return the image as a resized uint8 tensor (3, H, W)."""
    with Image.open(path) as image:
        return resize(v2.functional.pil_to_tensor(image.convert("RGB")))


def _pad(frames, max_frames):
    """Zero-pad frames to max_frames and return them with a mask of the real frames."""
    padded = frames.new_zeros(max_frames, *frames.shape[1:])
    padded[:len(frames)] = frames
    return padded, torch.arange(max_frames) < len(frames)
