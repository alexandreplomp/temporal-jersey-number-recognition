"""A local digit reader: digit evidence per cell, for finding where the digit is.

    image (N, 3, H, W)
      -> backbone[:4]            local features, 24 channels at 1/8 of the input (24 x 12 cells)
      -> 3 x 3 conv + 1 x 1 residual layers (LocalHead)
      -> 1 x 1 classifier        cell scores (N, 10, h, w)

Each cell sees about one digit (block 3's 31-pixel view plus one cell each side), so a
cell can only score high for a digit that is near it. Trained on single-digit crops with
the crop's digit as the only label (train_locator.py): the crop score per digit is the mean
of its TOP_CELLS best cells inside the torso, so the loss rewards a few cells that see the
digit, and the strongest cell then marks where the digit is (make_pairs.py).
"""

from pathlib import Path

import torch
from torch import nn

from jersey.data import torso_mask
from jersey.model import CELL_STRIDE, LOCAL_BLOCKS, LocalHead, make_backbone

DIGITS = 10
TOP_CELLS = 8
HEAD_LAYERS = 2  # 1 x 1 residual layers; extra_layers adds to this


def torso_cells(rows, columns, stride=CELL_STRIDE):
    """True for the cells whose centre is in the torso box: (rows, columns)."""
    mask = torso_mask(rows * stride, columns * stride)
    return mask[stride // 2::stride, stride // 2::stride]


def crop_scores(cell_scores, top=TOP_CELLS):
    """Cell scores (N, K, h, w) -> crop scores (N, K): mean of the top cells inside the torso."""
    _, _, h, w = cell_scores.shape
    inside = torso_cells(h, w).flatten().to(cell_scores.device)
    return cell_scores.flatten(2)[:, :, inside].topk(top, dim=-1).values.mean(dim=-1)


class LocalDigitReader(nn.Module):
    """Per-cell digit scores from local features only. See the module docstring."""

    def __init__(self, pretrained=True, extra_layers=0, dim=128):
        super().__init__()
        self.freeze_batchnorm = pretrained
        self.backbone = make_backbone(pretrained)[:LOCAL_BLOCKS]
        self.head = LocalHead(dim, HEAD_LAYERS + extra_layers)
        self.classifier = nn.Conv2d(dim, DIGITS, kernel_size=1)
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
        """Images (N, 3, H, W) -> cell scores (N, 10, H / 8, W / 8)."""
        return self.classifier(self.head(self.backbone(images)))


def load_locator(run_dir, device="cpu"):
    """Rebuild a trained locator from run_dir/best.pt: (model in eval mode, config, epoch)."""
    checkpoint = torch.load(Path(run_dir) / "best.pt", map_location="cpu")
    config = checkpoint["config"]
    model = LocalDigitReader(pretrained=False, extra_layers=config["extra_layers"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), config, checkpoint["epoch"]
