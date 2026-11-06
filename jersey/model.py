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

Two readers turn a frame into digit tokens:
    fused     two queries over cells that mix deep and local features (the first reader)
    two-step  deep features decide where to look, local features alone what is there:

        images -> backbone[:4] -> local (24 x 12 cells, each about one digit wide)
        local -> backbone[4:] -> deep -> upsampled to 24 x 12 -> keys (+ positions, side bias)
        deep -> global average -> count logits (one digit or two)
        local -> LocalHead -> values
        three queries (single, left, right), one head: keys choose cells, values give content
        -> tokens (N, 3, D) -> frame combining -> one shared 10-class classifier

      No deep feature reaches the values or the classifier, so a token can only carry what
      the cells it attends to show: the left query cannot read the right digit without
      looking at it. The three readings and the count make the same (2, 11) output as the
      fused reader (compose), so evaluation, explanation and metrics work for both.

Shapes: B batch, T frames, Q queries, N images, C cells per frame, D feature size.
"""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from jersey.data import BLANK, IMAGE_SIZE, MAX_FRAMES

VARIANTS = ("averaged", "pooled", "full")
READERS = ("fused", "two-step")
NUM_CLASSES = 11  # digits 0-9 plus BLANK
DIGITS = 10
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
# The two-step reader's queries: single starts at the centre, left and right THREE_SIDE_OFFSET
# pixels to either side, about half a digit plus half the gap. Narrower and further out than
# SIDE_OFFSET / SIDE_WIDTH, so left and right start on different digits. Tuned once.
THREE_SIDE_OFFSET = 14
THREE_SIDE_WIDTH = 8
VALUE_CHANNELS = 128


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


def side_bias(rows, columns, stride=CELL_STRIDE, offsets=(-SIDE_OFFSET, SIDE_OFFSET), width=SIDE_WIDTH):
    """Bias on the attention scores of the digit queries: (len(offsets), rows * columns).

    0 at each query's own centre (the middle plus its offset, in pixels), and
    -(distance / width)^2 / 2 elsewhere, with the distance in pixels from the middle of
    each column of cells. The default is the fused reader's: first left, second right.
    """
    middle = columns * stride / 2
    centres = middle + torch.tensor(offsets, dtype=torch.float)
    x = (torch.arange(columns, dtype=torch.float) + 0.5) * stride  # (columns,) cell centres
    bias = -((x - centres[:, None]) / width) ** 2 / 2  # (Q, columns)
    return bias[:, None, :].expand(len(offsets), rows, columns).reshape(len(offsets), rows * columns)


class ThreeQueryReader(nn.Module):
    """Single, left and right queries: keys from deep cells choose where, values from local cells give what.

    One attention head, so each query's map is exactly the weights its token is made of.
    """

    QUERIES = ("single", "left", "right")

    def __init__(self, dim, grid, value_channels=VALUE_CHANNELS):
        super().__init__()
        rows, columns = grid
        self.queries = nn.Parameter(torch.randn(1, 3, dim) * 0.02)
        # In the keys only, as in DigitQueries: positions decide where a query looks.
        self.cell_positions = nn.Parameter(torch.randn(1, rows * columns, dim) * 0.02)
        bias = side_bias(rows, columns, offsets=(0, -THREE_SIDE_OFFSET, THREE_SIDE_OFFSET), width=THREE_SIDE_WIDTH)
        self.register_buffer("side_bias", bias, persistent=False)
        self.attention = nn.MultiheadAttention(dim, 1, batch_first=True, kdim=dim, vdim=value_channels)
        self.norm = nn.LayerNorm(dim)

    def forward(self, keys, values):
        """Keys (N, C, D) and values (N, C, value_channels) -> tokens (N, 3, D) and attention (N, 3, C)."""
        queries = self.queries.expand(len(keys), -1, -1)
        tokens, attention = self.attention(queries, keys + self.cell_positions, values, attn_mask=self.side_bias)
        # Kept for explain.py: left and right as the fused reader's two digits, and all three.
        self.last_attention = attention[:, 1:].detach()
        self.all_attention = attention.detach()
        return self.norm(tokens), attention


def compose(digit_logits, count_logits):
    """The two-step readings as the fused reader's output: log-probabilities (..., 2, 11).

    digit_logits (..., 3, 10) for single, left and right; count_logits (..., 2) for one and
    two digits. Each position sums to 1:
        position 1, digit d   P(one) P_single(d) + P(two) P_left(d)
        position 1, BLANK     0
        position 2, digit d   P(two) P_right(d)
        position 2, BLANK     P(one)
    """
    digits = digit_logits.log_softmax(dim=-1)
    count = count_logits.log_softmax(dim=-1)
    one, two = count[..., :1], count[..., 1:]
    first = torch.logaddexp(one + digits[..., 0, :], two + digits[..., 1, :])
    first = torch.cat([first, torch.full_like(one, float("-inf"))], dim=-1)
    second = torch.cat([two + digits[..., 2, :], one], dim=-1)
    return torch.stack([first, second], dim=-2)


def masked_mean(values, mask):
    """Mean over the real frames: (B, T, ...) -> (B, ...), e.g. tokens (B, T, Q, D) -> (B, Q, D)."""
    weights = mask.reshape(*mask.shape, *[1] * (values.dim() - 2)).float()  # 1 for real frames
    return (values * weights).sum(dim=1) / weights.sum(dim=1)


class AttentionPool(nn.Module):
    """Weighted mean over the real frames, with a learned score per frame (Ilse et al. 2018)."""

    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))

    def forward(self, tokens, mask):
        """Return the pooled tokens (B, Q, D) and the frame weights (B, T, Q)."""
        scores = self.score(tokens).squeeze(-1)  # (B, T, Q)
        scores = scores.masked_fill(~mask[:, :, None], float("-inf"))
        weights = scores.softmax(dim=1)  # (B, T, Q), sums to 1 over the real frames
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
        B, T, Q, D = tokens.shape
        # Each query becomes its own sequence of T frames.
        x = tokens.transpose(1, 2).reshape(B * Q, T, D)  # (B*Q, T, D)
        x = x + self.frame_positions[:, :T]
        padding = (~mask).repeat_interleave(Q, dim=0)  # (B*Q, T), True where padded
        x = self.norm(self.layer(x, src_key_padding_mask=padding))
        return x.reshape(B, Q, T, D).transpose(1, 2)  # (B, T, Q, D)


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
        reader="fused",
    ):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}, expected one of {VARIANTS}")
        if reader not in READERS:
            raise ValueError(f"unknown reader {reader!r}, expected one of {READERS}")
        # Otherwise the CNN grid and the cell positions disagree in size.
        if image_size[0] % BACKBONE_STRIDE or image_size[1] % BACKBONE_STRIDE:
            raise ValueError(f"image_size must be divisible by {BACKBONE_STRIDE}, got {image_size}")
        self.variant = variant
        self.reader = reader
        # With pretrained weights, keep the backbone's ImageNet BatchNorm statistics fixed,
        # so training and evaluation compute the same thing.
        self.freeze_batchnorm = pretrained
        self.backbone = make_backbone(pretrained)
        grid = (image_size[0] // CELL_STRIDE, image_size[1] // CELL_STRIDE)
        if reader == "fused":
            self.project = nn.Conv2d(BACKBONE_CHANNELS, dim, kernel_size=1)
            self.project_local = nn.Conv2d(LOCAL_CHANNELS, dim, kernel_size=1)
            self.digits = DigitQueries(dim, grid)
        else:
            self.project_keys = nn.Conv2d(BACKBONE_CHANNELS, dim, kernel_size=1)
            self.values = LocalHead(VALUE_CHANNELS)
            self.count_head = nn.Linear(BACKBONE_CHANNELS, 2)
            self.digits = ThreeQueryReader(dim, grid)
        # Only build what this variant uses, so parameter counts compare fairly.
        self.temporal = TemporalEncoder(dim, max_frames, dropout) if variant == "full" else None
        self.pool = AttentionPool(dim) if variant in ("pooled", "full") else None
        # One classifier for every position: a 9 learned as second digit also helps read a first 9.
        self.classifier = nn.Linear(dim, NUM_CLASSES if reader == "fused" else DIGITS)
        # Every number has a first digit, so BLANK is never allowed in position 1 (compose does
        # the same for the two-step reader).
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

    def classify(self, tokens, count_logits=None):
        """Digit tokens -> log-probabilities or logits (..., 2, 11).

        Fused: tokens (..., 2, D). Two-step: tokens (..., 3, D) and count_logits (..., 2).
        """
        if self.reader == "fused":
            return self.classifier(tokens) + self.not_allowed
        return compose(self.classifier(tokens), count_logits)

    def read(self, images):
        """Single images (N, 3, H, W) -> tokens (N, Q, D), count logits (N, 2) and attention (N, Q, C).

        Count logits and attention are None for the fused reader.
        """
        local = self.backbone[:LOCAL_BLOCKS](images)  # (N, LOCAL_CHANNELS, h, w), h = H / CELL_STRIDE
        deep = self.backbone[LOCAL_BLOCKS:](local)  # (N, BACKBONE_CHANNELS, h / 2, w / 2)
        if self.reader == "fused":
            deep = F.interpolate(self.project(deep), size=local.shape[-2:], mode="bilinear", align_corners=False)
            cells = deep + self.project_local(local)  # (N, D, h, w)
            cells = cells.flatten(2).transpose(1, 2)  # (N, C, D), C = h * w
            return self.digits(cells), None, None
        keys = F.interpolate(self.project_keys(deep), size=local.shape[-2:], mode="bilinear", align_corners=False)
        values = self.values(local)  # (N, VALUE_CHANNELS, h, w), from local features only
        tokens, attention = self.digits(keys.flatten(2).transpose(1, 2), values.flatten(2).transpose(1, 2))
        count_logits = self.count_head(deep.mean(dim=(2, 3)))  # (N, 2)
        return tokens, count_logits, attention

    def read_tokens(self, images):
        """Single images (N, 3, H, W) -> digit tokens (N, Q, D): Q = 2 fused, 3 two-step."""
        return self.read(images)[0]

    def read_frames_full(self, images):
        """Single images (N, 3, H, W) read one at a time: dict as forward's, without frame keys.

        logits (N, 2, 11); for the two-step reader also digit_logits (N, 3, 10), count_logits
        (N, 2) and attention (N, 3, C). Used for the anchor.
        """
        tokens, count_logits, attention = self.read(images)
        out = {"logits": self.classify(tokens, count_logits), "tokens": tokens}
        if self.reader == "two-step":
            out.update(digit_logits=self.classifier(tokens), count_logits=count_logits, attention=attention)
        return out

    def read_frames(self, images):
        """Single images (N, 3, H, W) -> logits (N, 2, 11). Used for the anchor."""
        return self.read_frames_full(images)["logits"]

    def forward(self, frames, mask):
        """Sequences (B, T, 3, H, W) with mask (B, T) -> dict.

        logits (B, 2, 11), frame_logits (B, T, 2, 11), and frame_weights (B, T, 2), which
        is None for the averaged variant. The two-step reader's logits are log-probabilities
        (compose), and it adds:
            tokens              (B, 3, D) combined over frames: single, left, right
            digit_logits        (B, 3, 10)
            count_logits        (B, 2), the mean of the frames' count logits over real frames
            frame_count_logits  (B, T, 2)
            attention           (B, T, 3, C), zero for padded frames
            all_frame_weights   (B, T, 3) or None; frame_weights keeps left and right
        """
        B, T = mask.shape
        # Run the CNN on the real frames only, so padding never reaches BatchNorm.
        real, real_count, real_attention = self.read(frames[mask])  # (R, Q, D), R real frames in the batch
        tokens = real.new_zeros(B, T, *real.shape[1:])  # (B, T, Q, D)
        tokens[mask] = real
        count = None
        if self.reader == "two-step":
            count = real_count.new_zeros(B, T, 2)
            count[mask] = real_count
        frame_logits = self.classify(tokens, count)  # (B, T, 2, 11)

        frame_weights = None  # the averaged variant weighs all real frames equally
        if self.variant == "averaged":
            summary = masked_mean(tokens, mask)  # (B, Q, D)
        else:
            if self.temporal is not None:
                tokens = self.temporal(tokens, mask)  # (B, T, Q, D)
            summary, frame_weights = self.pool(tokens, mask)  # (B, Q, D), (B, T, Q)
        if self.reader == "fused":
            return {"logits": self.classify(summary), "frame_logits": frame_logits, "frame_weights": frame_weights}

        attention = real_attention.new_zeros(B, T, *real_attention.shape[1:])
        attention[mask] = real_attention
        count_logits = masked_mean(count, mask)  # (B, 2)
        return {
            "logits": self.classify(summary, count_logits),
            "frame_logits": frame_logits,
            "frame_weights": None if frame_weights is None else frame_weights[..., 1:],
            "all_frame_weights": frame_weights,
            "tokens": summary,
            "digit_logits": self.classifier(summary),
            "count_logits": count_logits,
            "frame_count_logits": count,
            "attention": attention,
        }


def load_run(run_dir, device="cpu"):
    """Rebuild a trained model from run_dir/best.pt: returns (model in eval mode, config, epoch)."""
    checkpoint = torch.load(Path(run_dir) / "best.pt", map_location="cpu")
    config = checkpoint["config"]
    # No download: the trained weights replace the pretrained ones.
    model = JerseyModel(config["variant"], max_frames=config["max_frames"], pretrained=False,
                        reader=config.get("reader", "fused"))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), config, checkpoint["epoch"]


def predict_digits(logits):
    """Logits (..., 2, 11) -> digits (..., 2) and one confidence per number (...).

    The two positions are scored independently, so the most likely digit at each
    position gives the most likely number, and its confidence is their product.
    """
    probabilities, digits = logits.softmax(dim=-1).max(dim=-1)
    return digits, probabilities.prod(dim=-1)
