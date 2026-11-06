"""Read the jersey number of one player sequence: a folder of crops in, one number out.
Example: uv run predict.py data/4/<player>/<sequence>
         uv run predict.py <folder> --weights results/runs/two_step_mean_all --two-digit-share 0.5

Frames are read in time order (by the number in <prefix>_<frame>_...), at most max_frames of them
spread evenly; any *_anchor image is ignored, since a deployed tracker has no anchor. Prints the
number and its confidence.

--two-digit-share sets the share of two-digit numbers expected where the model is used. The weights
carry this data's mix (about 5% two-digit numbers); a league where about half the numbers have two
digits should pass 0.5 (see jersey/model.count_prior).
"""

from pathlib import Path
import argparse

import torch
from torchvision.transforms import v2

from jersey.data import IMAGE_SIZE, NORMALIZE, decode_label, find_images, load_image, pick_frames
from jersey.engine import pick_device
from jersey.model import load_run, offset_for_share, predict_digits

WEIGHTS = Path("weights/final")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sequence", type=Path, help="folder with the crops of one player")
    parser.add_argument("--weights", type=Path, default=WEIGHTS, help="folder with best.pt (a run or weights/...)")
    parser.add_argument("--two-digit-share", type=float, help="share of two-digit numbers where the model is used")
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


@torch.no_grad()
def read_number(model, sequence_dir, max_frames, device):
    """The number read from a folder of crops and its confidence; the anchor is never loaded."""
    _, frame_paths = find_images(sequence_dir)
    resize = v2.Resize(IMAGE_SIZE, antialias=True)
    picked = pick_frames(len(frame_paths), max_frames, at_random=False).tolist()
    frames = NORMALIZE(torch.stack([load_image(frame_paths[i], resize) for i in picked]))[None].to(device)
    digits, confidence = predict_digits(model(frames, torch.ones(1, len(picked), dtype=torch.bool, device=device))["logits"])
    return decode_label(*digits[0].tolist()), confidence.item()


def main(argv=None):
    args = parse_args(argv)
    device = pick_device(args.device)
    model, config, _ = load_run(args.weights, device)
    if args.two_digit_share is not None and config["reader"] == "two-step":
        model.count_offset = offset_for_share(args.two_digit_share)
    number, confidence = read_number(model, args.sequence, config["max_frames"], device)
    print(f"{number}  (confidence {confidence:.2f})")
    return number


if __name__ == "__main__":
    main()
