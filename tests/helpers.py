"""Small fake data shared by the tests."""

import json

from PIL import Image
import torch

from jersey.data import IMAGE_SIZE

DIGIT_BOX = [40, 80, 52, 105]  # a digit 12 wide and 25 tall in the 192 x 96 input


def fake_sequence_dir(root, sequence_id, frame_ids, noise=False, anchor=True):
    """A tiny sequence: one frame per id (1_<id>_0.png) plus an anchor, at a size unlike IMAGE_SIZE.

    noise gives the sequence one random texture, so augmentation visibly changes it and no two
    sequences share an image.
    """
    folder = root / sequence_id
    folder.mkdir(parents=True)
    image = Image.effect_noise((22, 35), 64).convert("RGB") if noise else Image.new("RGB", (22, 35), "white")
    for frame_id in frame_ids:
        image.save(folder / f"1_{frame_id}_0.png")
    if anchor:
        image.save(folder / "1_0_0_anchor.jpg")
    return folder


def digit_image(box=DIGIT_BOX, ink=255, shirt=0):
    """A uint8 shirt (3, H, W) with one solid block of ink where box is."""
    image = torch.full((3, *IMAGE_SIZE), shirt, dtype=torch.uint8)
    left, top, right, bottom = box
    image[:, top:bottom, left:right] = ink
    return image


def fake_batch(real_frames=(3, 1), max_frames=4):
    """Random frames for sequences with the given numbers of real frames, zero-padded."""
    frames = torch.zeros(len(real_frames), max_frames, 3, *IMAGE_SIZE)
    mask = torch.zeros(len(real_frames), max_frames, dtype=torch.bool)
    for i, count in enumerate(real_frames):
        frames[i, :count] = torch.randn(count, 3, *IMAGE_SIZE)
        mask[i, :count] = True
    return frames, mask


def fake_strips(path, kept_frames, exclude=(), split="train"):
    """Write boxes as the box step does: each sequence gets kept_frames[sequence] kept frames of 3, and a kept anchor."""
    def image(name, is_anchor, kept):
        return {"image": name, "anchor": is_anchor, "kept": kept, "centre": [46, 92], "prob": 0.95,
                "box": DIGIT_BOX, "colour": [0.5, 0.5, 0.5]}

    sequences = {
        sequence_id: [image(f"{sequence_id}/1_{f}_0.png", False, f <= kept) for f in range(1, 4)]
        + [image(f"{sequence_id}/1_0_0_anchor.jpg", True, True)]
        for sequence_id, kept in kept_frames.items()
    }
    path.write_text(json.dumps({"split": split, "exclude": list(exclude), "sequences": sequences}))
    return path
