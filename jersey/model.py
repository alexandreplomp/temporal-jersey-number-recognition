"""The network: reads a jersey number as two digits from a sequence of player crops.

    frames (B, T, 3, H, W)
      -> CNN on each frame          -> a grid of cells per frame
      -> two digit queries          -> frame tokens (B, T, 2, D): first and second digit
      -> combine the frames         -> (B, 2, D)
      -> shared classifier          -> logits (B, 2, 11): digits 0-9 plus BLANK

The variants differ only in how the frames are combined, so the report can show
what each part adds:
    averaged  plain mean over the real frames
    pooled    learned weighted mean (attention pooling)
    full      a transformer lets frames see each other, then attention pooling

Shapes: B batch, T frames, N images, C cells per frame, D feature size.
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from jersey.data import BLANK, IMAGE_SIZE, MAX_FRAMES

VARIANTS = ("averaged", "pooled", "full")
NUM_CLASSES = 11  # digits 0-9 plus BLANK
# The first 9 MobileNetV3-Small blocks: 48 channels at 1/16 of the input size.
# 0.19M parameters instead of 0.93M for all blocks.
BACKBONE_BLOCKS = 9
BACKBONE_CHANNELS = 48
BACKBONE_STRIDE = 16
# Each deep cell summarises the whole crop (squeeze-excitation, wide view), so two digit
# queries get the same content and read the same digit twice. Block 3's output, at 1/8 of
# the input with a 31-pixel view (about one digit), is added to the deep cells upsampled to
# its grid, FPN-style: each cell keeps the deep reading power and gains its own local
# detail, so the queries on either side see different digits. 192 x 96 gives 24 x 12 cells.
LOCAL_BLOCKS = 4
LOCAL_CHANNELS = 24
CELL_STRIDE = 8
HEADS = 4
# Each digit query starts on its own side: a soft bias on its attention scores that
# falls off with the distance from a point left (first digit) or right (second digit)
# of the middle. Fixed, so the start cannot drift back to the middle. In pixels of the
# input, so the bias sits in the same place on the digits whatever the grid.
SIDE_OFFSET = 8
SIDE_WIDTH = 12


def make_backbone(pretrained):
    """Return the first MobileNetV3-Small blocks, with ImageNet weights if pretrained."""
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    return mobilenet_v3_small(weights=weights).features[:BACKBONE_BLOCKS]


class Residual1x1(nn.Module):
    """x + ReLU(BN(conv1x1(x))): mixes channels within a cell, never across cells."""

    def __init__(self, channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(),
        )

    def forward(self, x):
        return x + self.layer(x)


class LocalHead(nn.Module):
    """Local features (N, LOCAL_CHANNELS, h, w) -> (N, dim, h, w), seeing about one digit per cell.

    One 3 x 3 convolution, then 1 x 1 residual layers. No squeeze-excitation or pooling, so a
    cell's output depends only on the pixels near it: block 3's 31-pixel view plus one cell
    each side. (Block 1 of the backbone has squeeze-excitation, a global channel gate, so the
    input is not strictly local: a far change moves a cell slightly, not much.)
    """

    def __init__(self, dim=128, layers=2, in_channels=LOCAL_CHANNELS):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(dim), nn.ReLU(),
            *(Residual1x1(dim) for _ in range(layers)),
        )

    def forward(self, local):
        return self.layers(local)


class DigitQueries(nn.Module):
    """Two learned queries search a frame's cells: one for the first digit, one for the second."""

    def __init__(self, dim, grid):
        super().__init__()
        rows, columns = grid
        self.queries = nn.Parameter(torch.randn(1, 2, dim) * 0.02)
        # Attention treats the cells as an unordered set. A learned position per cell
        # is what lets the queries tell left from right. It goes into the keys only, as in
        # DETR: it decides where a query looks, but is kept out of what the token carries.
        self.cell_positions = nn.Parameter(torch.randn(1, rows * columns, dim) * 0.02)
        self.register_buffer("side_bias", side_bias(rows, columns), persistent=False)
        self.attention = nn.MultiheadAttention(dim, HEADS, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, cells):
        queries = self.queries.expand(len(cells), -1, -1)  # (N, 2, D)
        # (N, 2, D), (N, 2, C); cells (N, C, D). A float attn_mask is added to the scores.
        digits, attention = self.attention(
            queries, cells + self.cell_positions, cells, attn_mask=self.side_bias,
        )
        # Kept to show where each query looks, averaged over the heads.
        self.last_attention = attention.detach()
        return self.norm(digits)


def side_bias(rows, columns, stride=CELL_STRIDE):
    """Bias on the attention scores of the two digit queries: (2, rows * columns).

    0 at the query's own centre, -(distance / SIDE_WIDTH)^2 / 2 elsewhere, with the
    distance in pixels from the middle of each column of cells.
    """
    middle = columns * stride / 2
    centres = torch.tensor([middle - SIDE_OFFSET, middle + SIDE_OFFSET])  # first left, second right
    x = (torch.arange(columns, dtype=torch.float) + 0.5) * stride  # (columns,) cell centres
    bias = -((x - centres[:, None]) / SIDE_WIDTH) ** 2 / 2  # (2, columns)
    return bias[:, None, :].expand(2, rows, columns).reshape(2, rows * columns)


def masked_mean(tokens, mask):
    """Mean over the real frames: (B, T, 2, D) -> (B, 2, D)."""
    weights = mask[:, :, None, None].float()  # (B, T, 1, 1), 1 for real frames
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1)


class AttentionPool(nn.Module):
    """Weighted mean over the real frames, with a learned score per frame (Ilse et al. 2018)."""

    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))

    def forward(self, tokens, mask):
        """Return the pooled tokens (B, 2, D) and the frame weights (B, T, 2)."""
        scores = self.score(tokens).squeeze(-1)  # (B, T, 2)
        scores = scores.masked_fill(~mask[:, :, None], float("-inf"))
        weights = scores.softmax(dim=1)  # (B, T, 2), sums to 1 over the real frames
        return (weights[..., None] * tokens).sum(dim=1), weights


class TemporalEncoder(nn.Module):
    """One transformer layer: each frame's digit token sees the same digit in the other frames."""

    def __init__(self, dim, max_frames, dropout):
        super().__init__()
        # Unlike the mean and the pooling, the transformer can use frame order.
        self.frame_positions = nn.Parameter(torch.randn(1, max_frames, dim) * 0.02)
        self.layer = nn.TransformerEncoderLayer(
            dim, HEADS, dim_feedforward=2 * dim, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        # A pre-norm layer leaves its output unnormalised, so normalise once at the end.
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens, mask):
        B, T, _, D = tokens.shape
        # Each digit position becomes its own sequence of T frames.
        x = tokens.transpose(1, 2).reshape(B * 2, T, D)  # (B*2, T, D)
        x = x + self.frame_positions[:, :T]
        padding = (~mask).repeat_interleave(2, dim=0)  # (B*2, T), True where padded
        x = self.norm(self.layer(x, src_key_padding_mask=padding))
        return x.reshape(B, 2, T, D).transpose(1, 2)  # (B, T, 2, D)


class JerseyModel(nn.Module):
    """Reads a jersey number from a sequence of crops. See the module docstring."""

    def __init__(
        self,
        variant="full",
        dim=128,
        image_size=IMAGE_SIZE,
        max_frames=MAX_FRAMES,
        pretrained=True,
        dropout=0.1,
    ):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}, expected one of {VARIANTS}")
        # Otherwise the CNN grid and the cell positions disagree in size.
        if image_size[0] % BACKBONE_STRIDE or image_size[1] % BACKBONE_STRIDE:
            raise ValueError(f"image_size must be divisible by {BACKBONE_STRIDE}, got {image_size}")
        self.variant = variant
        # With pretrained weights, keep the backbone's ImageNet BatchNorm statistics fixed,
        # so training and evaluation compute the same thing.
        self.freeze_batchnorm = pretrained
        self.backbone = make_backbone(pretrained)
        self.project = nn.Conv2d(BACKBONE_CHANNELS, dim, kernel_size=1)
        self.project_local = nn.Conv2d(LOCAL_CHANNELS, dim, kernel_size=1)
        grid = (image_size[0] // CELL_STRIDE, image_size[1] // CELL_STRIDE)
        self.digits = DigitQueries(dim, grid)
        # Only build what this variant uses, so parameter counts compare fairly.
        self.temporal = TemporalEncoder(dim, max_frames, dropout) if variant == "full" else None
        self.pool = AttentionPool(dim) if variant in ("pooled", "full") else None
        # One classifier for both positions: a 9 learned as second digit also helps read a first 9.
        self.classifier = nn.Linear(dim, NUM_CLASSES)
        # Every number has a first digit, so BLANK is never allowed in position 1.
        not_allowed = torch.zeros(2, NUM_CLASSES)
        not_allowed[0, BLANK] = float("-inf")
        self.register_buffer("not_allowed", not_allowed, persistent=False)
        self.train()  # applies the BatchNorm freeze from the start, not only after .train()

    def train(self, mode=True):
        """Switch train/eval mode, but keep the backbone's BatchNorm statistics fixed if frozen."""
        super().train(mode)
        if self.freeze_batchnorm:
            for layer in self.backbone.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eval()
        return self

    def classify(self, tokens):
        """Digit tokens (..., 2, D) -> logits (..., 2, 11)."""
        return self.classifier(tokens) + self.not_allowed

    def read_tokens(self, images):
        """Single images (N, 3, H, W) -> digit tokens (N, 2, D)."""
        local = self.backbone[:LOCAL_BLOCKS](images)  # (N, LOCAL_CHANNELS, h, w), h = H / CELL_STRIDE
        deep = self.backbone[LOCAL_BLOCKS:](local)  # (N, BACKBONE_CHANNELS, h / 2, w / 2)
        deep = F.interpolate(self.project(deep), size=local.shape[-2:], mode="bilinear", align_corners=False)
        cells = deep + self.project_local(local)  # (N, D, h, w)
        cells = cells.flatten(2).transpose(1, 2)  # (N, C, D), C = h * w
        return self.digits(cells)

    def read_frames(self, images):
        """Single images (N, 3, H, W) -> logits (N, 2, 11). Used for the anchor."""
        return self.classify(self.read_tokens(images))

    def forward(self, frames, mask):
        """Sequences (B, T, 3, H, W) with mask (B, T) -> dict of logits.

        logits (B, 2, 11), frame_logits (B, T, 2, 11), and frame_weights (B, T, 2),
        which is None for the averaged variant.
        """
        B, T = mask.shape
        # Run the CNN on the real frames only, so padding never reaches BatchNorm.
        real = self.read_tokens(frames[mask])  # (R, 2, D), R real frames in the batch
        tokens = real.new_zeros(B, T, 2, real.shape[-1])  # (B, T, 2, D)
        tokens[mask] = real
        frame_logits = self.classify(tokens)  # (B, T, 2, 11)

        frame_weights = None  # the averaged variant weighs all real frames equally
        if self.variant == "averaged":
            summary = masked_mean(tokens, mask)  # (B, 2, D)
        else:
            if self.temporal is not None:
                tokens = self.temporal(tokens, mask)  # (B, T, 2, D)
            summary, frame_weights = self.pool(tokens, mask)  # (B, 2, D), (B, T, 2)
        return {
            "logits": self.classify(summary),
            "frame_logits": frame_logits,
            "frame_weights": frame_weights,
        }


def load_run(run_dir, device="cpu"):
    """Rebuild a trained model from run_dir/best.pt: returns (model in eval mode, config, epoch)."""
    checkpoint = torch.load(Path(run_dir) / "best.pt", map_location="cpu")
    config = checkpoint["config"]
    # No download: the trained weights replace the pretrained ones.
    model = JerseyModel(config["variant"], max_frames=config["max_frames"], pretrained=False)
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), config, checkpoint["epoch"]


def predict_digits(logits):
    """Logits (..., 2, 11) -> digits (..., 2) and one confidence per number (...).

    The two positions are scored independently, so the most likely digit at each
    position gives the most likely number, and its confidence is their product.
    """
    probabilities, digits = logits.softmax(dim=-1).max(dim=-1)
    return digits, probabilities.prod(dim=-1)
