"""The network: reads a jersey number as two digits from a sequence of player crops.

    frames (B, T, 3, H, W)
      -> backbone on each real frame     local cells (1/8 of the input) and deep cells (1/16)
      -> reader                          digit tokens per frame (B, T, Q, D)
      -> frame combiner                  one token per query for the sequence (B, Q, D)
      -> the reader's classifier         log-probabilities (B, 2, 11): digits 0-9 plus BLANK

Every number is two positions, each a digit or BLANK ("4" is 4, BLANK), read by one shared digit
classifier. A digit learned in one position or number counts for every other one, so 94 can be
read after training on 4, 9 and 49: nothing is a 100-class label.

Readers (READERS), each with its own pretrained part (backbone), forward(images) -> tokens (N, Q, D),
attention (N, Q, C), count logits (N, 2) or None, and logits(tokens, count, count_offset) -> (..., 2, 11):
    fused     two queries (first, second digit) over cells that mix deep and local features.
              Simple, but each cell sees the whole crop, so it cannot learn left from right
              from this data: the baseline that shows the two-digit structure alone is not enough.
    two-step  deep features decide where to look, local features alone what is there:
                local -> LocalHead -> values; deep -> keys (+ cell positions, side bias)
                deep -> count logits (one digit or two)
                three queries (single, left, right), one head -> tokens (N, 3, D)
              No deep feature reaches the values or the classifier, so the left query cannot
              read the right digit without looking at it. compose turns the three readings and
              the count into the same (2, 11) output.
    parseq    PARSeq-Tiny, a scene text reader pretrained on millions of text images
              (its network is the PARSeq section below), on the number area of the frame stretched to its 32 x 128
              line: two position queries read digit 1 and digit 2 (or "end") from the image
              alone. It learned every digit in every position before seeing this data. With the
              pool combiner, a learned gate per position weighs the frames; the head is linear,
              so the weighted mean of features is a weighted mean of the frames' scores.

Frame combiners (TEMPORAL), each forward(tokens (B, T, Q, D), mask (B, T)) -> (B, Q, D), weights:
    mean         plain mean over the real frames
    pool         learned weighted mean (attention pooling)
    transformer  one transformer layer lets frames see each other, then attention pooling

Shapes: B batch, T frames, Q queries, N images, C cells per frame, D feature size.
"""

from pathlib import Path
import math

import torch
from torch import nn
from torch.nn import functional as F
from timm.models.vision_transformer import PatchEmbed, VisionTransformer
from torch.nn.modules import transformer
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from jersey.data import BLANK, IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE, MAX_FRAMES

NUM_CLASSES = 11  # digits 0-9 plus BLANK
DIGITS = 10
# The first 9 MobileNetV3-Small blocks: 48 channels at 1/16 of the input. 0.19M parameters
# instead of 0.93M for all blocks. Block 3's output, at 1/8 of the input with a 31-pixel view
# (about one digit), gives the local cells: 192 x 96 gives 24 x 12 cells.
BACKBONE_BLOCKS = 9
BACKBONE_CHANNELS = 48
BACKBONE_STRIDE = 16
LOCAL_BLOCKS = 4
LOCAL_CHANNELS = 24
CELL_STRIDE = 8
HEADS = 4
VALUE_CHANNELS = 128
# Each digit query starts on its own side: a fixed soft bias on its attention scores that falls
# off with the distance (in input pixels) from its starting column. Fused: first and second
# digit 8 px either side of the middle. Two-step: single at the middle, left and right 14 px out,
# about half a digit plus half the gap, narrower so left and right start on different digits.
FUSED_SIDES, FUSED_WIDTH = (-8, 8), 12
TWO_STEP_SIDES, TWO_STEP_WIDTH = (0, -14, 14), 8
# The number area PARSeq reads, as shares of the crop: tighter than data.TORSO_ROWS/COLUMNS, so the
# digits fill more of its text line; it holds 94% of the digit boxes found in train whole.
PARSEQ_ROWS, PARSEQ_COLUMNS = (0.17, 0.48), (0.15, 0.85)
CELL_GRID = (IMAGE_SIZE[0] // CELL_STRIDE, IMAGE_SIZE[1] // CELL_STRIDE)


def make_backbone(pretrained):
    """Return the first MobileNetV3-Small blocks, with ImageNet weights if pretrained."""
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    return mobilenet_v3_small(weights=weights).features[:BACKBONE_BLOCKS]


def side_bias(rows, columns, offsets, width, stride=CELL_STRIDE):
    """Bias on the attention scores of the queries: (len(offsets), rows * columns).

    0 at each query's own column (the middle plus its offset, in pixels), -(distance / width)^2 / 2 elsewhere.
    """
    centres = columns * stride / 2 + torch.tensor(offsets, dtype=torch.float)
    x = (torch.arange(columns, dtype=torch.float) + 0.5) * stride  # cell centres
    bias = -((x - centres[:, None]) / width) ** 2 / 2  # (Q, columns)
    return bias[:, None, :].expand(len(offsets), rows, columns).reshape(len(offsets), rows * columns)


class Residual1x1(nn.Module):
    """x + ReLU(BN(conv1x1(x))): mixes channels within a cell, never across cells."""

    def __init__(self, channels):
        super().__init__()
        self.layer = nn.Sequential(nn.Conv2d(channels, channels, 1, bias=False), nn.BatchNorm2d(channels), nn.ReLU())

    def forward(self, x):
        return x + self.layer(x)


class LocalHead(nn.Module):
    """Local features (N, LOCAL_CHANNELS, h, w) -> (N, dim, h, w), seeing about one digit per cell.

    One 3 x 3 convolution, then 1 x 1 residual layers: no pooling, so a cell depends only on the
    pixels near it. (Block 1 of the backbone has a squeeze-excitation channel gate, which saturates
    with the pretrained weights: a far change moves a cell not at all, or very slightly.)
    """

    def __init__(self, dim=128, layers=2, in_channels=LOCAL_CHANNELS):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(dim), nn.ReLU(),
            *(Residual1x1(dim) for _ in range(layers)),
        )

    def forward(self, local):
        return self.layers(local)


def compose(digit_logits, count_logits):
    """The two-step readings as the (2, 11) output: log-probabilities (..., 2, 11).

    digit_logits (..., 3, 10) for single, left and right; count_logits (..., 2) for one and two
    digits. Each position sums to 1:
        position 1, digit d   P(one) P_single(d) + P(two) P_left(d)
        position 1, BLANK     0
        position 2, digit d   P(two) P_right(d)
        position 2, BLANK     P(one)
    """
    digits, count = digit_logits.log_softmax(dim=-1), count_logits.log_softmax(dim=-1)
    one, two = count[..., :1], count[..., 1:]
    first = torch.cat([torch.logaddexp(one + digits[..., 0, :], two + digits[..., 1, :]),
                       torch.full_like(one, float("-inf"))], dim=-1)
    second = torch.cat([two + digits[..., 2, :], one], dim=-1)
    return torch.stack([first, second], dim=-2)


def count_prior(numbers):
    """log(one-digit / two-digit numbers): the count offset for a count head trained with both weighed equally.

    The count loss weighs one and two digits equally, so the count head answers as if half of all
    numbers had two digits. Adding this to the "one digit" score corrects that for the real mix
    (logit adjustment, Menon et al. 2021): with 4.5% two-digit numbers it is log(0.955 / 0.045) =
    3.05. Computed from training numbers only. 0 if either kind is missing.
    """
    two = sum(len(number) == 2 for number in numbers)
    one = len(numbers) - two
    return math.log(one / two) if one and two else 0.0


def offset_for_share(two_digit_share):
    """The count offset for a league where a share two_digit_share of numbers has two digits."""
    return math.log((1 - two_digit_share) / two_digit_share)


# ----- readers -----

class FusedReader(nn.Module):
    """Two queries (first and second digit) over cells that add deep and local features."""

    positions = (0, 1)  # the queries of digit position 1 and 2
    attention_grid = CELL_GRID

    def __init__(self, dim, pretrained):
        super().__init__()
        rows, columns = CELL_GRID
        self.token_dim = dim
        self.backbone = make_backbone(pretrained)
        self.project = nn.Conv2d(BACKBONE_CHANNELS, dim, kernel_size=1)
        self.project_local = nn.Conv2d(LOCAL_CHANNELS, dim, kernel_size=1)
        self.queries = nn.Parameter(torch.randn(1, 2, dim) * 0.02)
        # A learned position per cell, in the keys only (as in DETR): it decides where a query
        # looks, but is kept out of what the token carries.
        self.cell_positions = nn.Parameter(torch.randn(1, rows * columns, dim) * 0.02)
        self.register_buffer("side_bias", side_bias(rows, columns, FUSED_SIDES, FUSED_WIDTH), persistent=False)
        self.attention = nn.MultiheadAttention(dim, HEADS, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, NUM_CLASSES)
        not_allowed = torch.zeros(2, NUM_CLASSES)
        not_allowed[0, BLANK] = float("-inf")  # every number has a first digit
        self.register_buffer("not_allowed", not_allowed, persistent=False)

    def forward(self, images):
        local = self.backbone[:LOCAL_BLOCKS](images)
        deep = self.backbone[LOCAL_BLOCKS:](local)
        deep = F.interpolate(self.project(deep), size=local.shape[-2:], mode="bilinear", align_corners=False)
        cells = (deep + self.project_local(local)).flatten(2).transpose(1, 2)  # (N, C, D)
        queries = self.queries.expand(len(cells), -1, -1)
        tokens, attention = self.attention(queries, cells + self.cell_positions, cells, attn_mask=self.side_bias)
        return self.norm(tokens), attention, None

    def logits(self, tokens, count_logits=None, count_offset=0.0):
        return (self.classifier(tokens) + self.not_allowed).log_softmax(dim=-1)

    def digit_logits(self, tokens):
        return None


class TwoStepReader(nn.Module):
    """Single, left and right queries: deep keys choose where, local values give what. See the module docstring."""

    positions = (1, 2)  # left and right are digit position 1 and 2 of a two-digit number
    attention_grid = CELL_GRID

    def __init__(self, dim, pretrained):
        super().__init__()
        rows, columns = CELL_GRID
        self.token_dim = dim
        self.backbone = make_backbone(pretrained)
        self.project_keys = nn.Conv2d(BACKBONE_CHANNELS, dim, kernel_size=1)
        self.values = LocalHead(VALUE_CHANNELS)
        self.count_head = nn.Linear(BACKBONE_CHANNELS, 2)
        self.queries = nn.Parameter(torch.randn(1, 3, dim) * 0.02)
        self.cell_positions = nn.Parameter(torch.randn(1, rows * columns, dim) * 0.02)
        self.register_buffer("side_bias", side_bias(rows, columns, TWO_STEP_SIDES, TWO_STEP_WIDTH), persistent=False)
        # One head, so each query's map is exactly the weights its token is made of.
        self.attention = nn.MultiheadAttention(dim, 1, batch_first=True, kdim=dim, vdim=VALUE_CHANNELS)
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, DIGITS)  # shared by single, left and right

    def forward(self, images):
        local = self.backbone[:LOCAL_BLOCKS](images)
        deep = self.backbone[LOCAL_BLOCKS:](local)
        keys = F.interpolate(self.project_keys(deep), size=local.shape[-2:], mode="bilinear", align_corners=False)
        values = self.values(local)  # local features only
        queries = self.queries.expand(len(local), -1, -1)
        tokens, attention = self.attention(queries, keys.flatten(2).transpose(1, 2) + self.cell_positions,
                                           values.flatten(2).transpose(1, 2), attn_mask=self.side_bias)
        return self.norm(tokens), attention, self.count_head(deep.mean(dim=(2, 3)))

    def logits(self, tokens, count_logits, count_offset=0.0):
        """The count offset is added to the "one digit" score here only, never to the count loss."""
        return compose(self.classifier(tokens), count_logits + count_logits.new_tensor([count_offset, 0.0]))

    def digit_logits(self, tokens):
        return self.classifier(tokens)


# ----- PARSeq-Tiny (the parseq reader's network) -----
# The encoder (a small ViT over 4 x 8 patches of a 32 x 128 text line) and the decoder are from
# https://github.com/baudm/parseq, Copyright 2022 Darwin Bautista, Apache License 2.0 (LICENSE-parseq,
# NOTICE), kept as written there except: the tokenizer-driven decoding loop is replaced by
# ParseqDecoder.read() (every position read from the image alone, never from the digits read before
# it), and named_apply is defined here.
# The released weights read 94 characters, "0123456789" first; parseq_digit_weights keeps the rows of the
# end token and the ten digits. Tokens here: 0 end of number, 1-10 the digits 0-9, 11 start, 12 padding.

PARSEQ_TEXT_SIZE = (32, 128)  # height, width of PARSeq's input
PARSEQ_PATCH_SIZE = (4, 8)
PARSEQ_DIM, PARSEQ_ENCODER_HEADS, PARSEQ_DECODER_HEADS = 192, 3, 6  # the Tiny size
PARSEQ_POSITIONS = 3  # digit 1, digit 2 or end, end after two digits
PARSEQ_EOS, PARSEQ_BOS, PARSEQ_PAD, PARSEQ_TOKENS = 0, 11, 12, 13
PARSEQ_WEIGHTS_URL = "https://github.com/baudm/parseq/releases/download/v1.0.0/parseq_tiny-e7a21b54.pt"
PARSEQ_RELEASED_ROWS = list(range(11)) + [95, 96]  # end, the digits, start, padding among the 97 released tokens


# ----- from baudm/parseq (strhub/models/parseq/modules.py), Apache License 2.0 -----

class DecoderLayer(nn.Module):
    """A Transformer decoder layer supporting two-stream attention (XLNet)
    This implements a pre-LN decoder, as opposed to the post-LN default in PyTorch."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='gelu', layer_norm_eps=1e-5):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_q = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm_c = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = transformer._get_activation_fn(activation)

    def forward_stream(self, tgt, tgt_norm, tgt_kv, memory, tgt_mask, tgt_key_padding_mask):
        tgt2, sa_weights = self.self_attn(tgt_norm, tgt_kv, tgt_kv, attn_mask=tgt_mask,
                                          key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt2, ca_weights = self.cross_attn(self.norm1(tgt), memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(tgt)))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt, sa_weights, ca_weights

    def forward(self, query, content, memory, query_mask=None, content_mask=None, content_key_padding_mask=None,
                update_content=True):
        query_norm = self.norm_q(query)
        content_norm = self.norm_c(content)
        query = self.forward_stream(query, query_norm, content_norm, memory, query_mask, content_key_padding_mask)[0]
        if update_content:
            content = self.forward_stream(content, content_norm, content_norm, memory, content_mask,
                                          content_key_padding_mask)[0]
        return query, content


class Decoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm):
        super().__init__()
        self.layers = transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, query, content, memory, query_mask=None, content_mask=None, content_key_padding_mask=None):
        for i, mod in enumerate(self.layers):
            last = i == len(self.layers) - 1
            query, content = mod(query, content, memory, query_mask, content_mask, content_key_padding_mask,
                                 update_content=not last)
        return self.norm(query)


class Encoder(VisionTransformer):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
                 embed_layer=PatchEmbed):
        super().__init__(img_size, patch_size, in_chans, embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                         drop_path_rate=drop_path_rate, embed_layer=embed_layer,
                         num_classes=0, global_pool='', class_token=False)  # these disable the classifier head

    def forward(self, x):
        return self.forward_features(x)  # all tokens


class TokenEmbedding(nn.Module):

    def __init__(self, charset_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(charset_size, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, tokens):
        return math.sqrt(self.embed_dim) * self.embedding(tokens)


def init_weights(module, name='', exclude=()):
    """Initialize the weights using the typical initialization schemes used in SOTA models."""
    if any(map(name.startswith, exclude)):
        return
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


# ----- PARSeq, this project's part -----

def named_apply(fn, module, name=''):
    """fn(module=..., name=...) on every submodule, children first (as timm's named_apply)."""
    for child_name, child in module.named_children():
        named_apply(fn, child, f"{name}.{child_name}" if name else child_name)
    fn(module=module, name=name)


def make_parseq_encoder():
    return Encoder(PARSEQ_TEXT_SIZE, PARSEQ_PATCH_SIZE, embed_dim=PARSEQ_DIM, depth=12, num_heads=PARSEQ_ENCODER_HEADS, mlp_ratio=4)


class ParseqDecoder(nn.Module):
    """PARSeq's decoder, head and position queries; read() queries every position at once, from the image alone."""

    def __init__(self, dropout=0.1):
        super().__init__()
        self.decoder = Decoder(DecoderLayer(PARSEQ_DIM, PARSEQ_DECODER_HEADS, PARSEQ_DIM * 4, dropout), num_layers=1,
                               norm=nn.LayerNorm(PARSEQ_DIM))
        self.head = nn.Linear(PARSEQ_DIM, PARSEQ_TOKENS - 2)  # predicts the end token and the ten digits
        self.text_embed = TokenEmbedding(PARSEQ_TOKENS, PARSEQ_DIM)
        self.pos_queries = nn.Parameter(torch.Tensor(1, PARSEQ_POSITIONS, PARSEQ_DIM))
        self.dropout = nn.Dropout(p=dropout)
        named_apply(init_weights, self)
        nn.init.trunc_normal_(self.pos_queries, std=0.02)
        self.decoder.layers[-1].cross_attn.register_forward_hook(self._keep_attention)

    def _keep_attention(self, module, inputs, output):
        self.attention = output[1]  # (N, positions, patches), averaged over heads

    def read(self, memory):
        """Encoded text lines (N, patches, D) -> position features (N, 3, D) and their attention (N, 3, patches)."""
        start = self.text_embed(torch.full((len(memory), 1), PARSEQ_BOS, dtype=torch.long, device=memory.device))
        queries = self.dropout(self.pos_queries.expand(len(memory), -1, -1))
        return self.decoder(queries, self.dropout(start), memory), self.attention


def parseq_digit_weights():
    """The released PARSeq-Tiny weights, cut to the digits, split into the encoder's and the decoder's state dicts."""
    released = torch.hub.load_state_dict_from_url(PARSEQ_WEIGHTS_URL, map_location="cpu", check_hash=True)
    released = {key.removeprefix("model."): value for key, value in released.items()}
    released["head.weight"], released["head.bias"] = released["head.weight"][:PARSEQ_TOKENS - 2], released["head.bias"][:PARSEQ_TOKENS - 2]
    released["text_embed.embedding.weight"] = released["text_embed.embedding.weight"][PARSEQ_RELEASED_ROWS]
    released["pos_queries"] = released["pos_queries"][:, :PARSEQ_POSITIONS]
    encoder = {k.removeprefix("encoder."): v for k, v in released.items() if k.startswith("encoder.")}
    decoder = {k: v for k, v in released.items() if not k.startswith("encoder.")}
    return encoder, decoder


def parseq_text_lines(images, rows, columns, mean, std):
    """ImageNet-normalised crops (N, 3, H, W) -> PARSeq's text lines (N, 3, 32, 128) of the number area.

    rows and columns are (start, end) pixels; the area is stretched to 32 x 128 (stretching read
    47.7% of 300 val sequences zero-shot, keeping the aspect ratio 3-15%).
    """
    area = (images[..., rows[0]:rows[1], columns[0]:columns[1]] * std + mean).clamp(0, 1)
    return (F.interpolate(area, size=PARSEQ_TEXT_SIZE, mode="bilinear", align_corners=False, antialias=True) - 0.5) / 0.5


class ParseqReader(nn.Module):
    """PARSeq-Tiny on the stretched number area: digit 1 and digit 2 (or end) per frame. See the module docstring."""

    positions = (0, 1)
    attention_grid = (PARSEQ_TEXT_SIZE[0] // PARSEQ_PATCH_SIZE[0], PARSEQ_TEXT_SIZE[1] // PARSEQ_PATCH_SIZE[1])
    token_dim = PARSEQ_DIM

    def __init__(self, dim, pretrained):
        super().__init__()
        self.backbone = make_parseq_encoder()
        self.text = ParseqDecoder()
        if pretrained:
            encoder, decoder = parseq_digit_weights()
            self.backbone.load_state_dict(encoder)
            self.text.load_state_dict(decoder)
        self.rows = (round(PARSEQ_ROWS[0] * IMAGE_SIZE[0]), round(PARSEQ_ROWS[1] * IMAGE_SIZE[0]))
        self.columns = (round(PARSEQ_COLUMNS[0] * IMAGE_SIZE[1]), round(PARSEQ_COLUMNS[1] * IMAGE_SIZE[1]))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN)[:, None, None], persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD)[:, None, None], persistent=False)

    def forward(self, images):
        lines = parseq_text_lines(images, self.rows, self.columns, self.mean, self.std)
        features, attention = self.text.read(self.backbone(lines))
        return features[:, :2], attention[:, :2], None  # digit 1, digit 2 or end

    def logits(self, tokens, count_logits=None, count_offset=0.0):
        """PARSeq's (end, digits) scores per position -> (..., 2, 11) over (digits, BLANK); end in position 2 is BLANK."""
        scores = self.text.head(tokens)
        first = torch.cat([scores[..., 0, 1:].log_softmax(dim=-1), torch.full_like(scores[..., 0, :1], float("-inf"))], dim=-1)
        second = torch.cat([scores[..., 1, 1:], scores[..., 1, :1]], dim=-1).log_softmax(dim=-1)
        return torch.stack([first, second], dim=-2)

    def digit_logits(self, tokens):
        return None


READERS = {"fused": FusedReader, "two-step": TwoStepReader, "parseq": ParseqReader}


# ----- frame combiners -----

def masked_mean(values, mask):
    """Mean over the real frames: (B, T, ...) -> (B, ...)."""
    weights = mask.reshape(*mask.shape, *[1] * (values.dim() - 2)).float()
    return (values * weights).sum(dim=1) / weights.sum(dim=1)


class FrameMean(nn.Module):
    """Plain mean over the real frames."""

    def __init__(self, dim, max_frames, dropout):
        super().__init__()

    def forward(self, tokens, mask):
        return masked_mean(tokens, mask), None


class FramePool(nn.Module):
    """Weighted mean over the real frames, with a learned score per frame and query (Ilse et al. 2018)."""

    def __init__(self, dim, max_frames=MAX_FRAMES, dropout=0.0):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))

    def forward(self, tokens, mask):
        scores = self.score(tokens).squeeze(-1).masked_fill(~mask[:, :, None], float("-inf"))  # (B, T, Q)
        weights = scores.softmax(dim=1)
        return (weights[..., None] * tokens).sum(dim=1), weights


class FrameTransformer(nn.Module):
    """One transformer layer: each frame's token sees the same query in the other frames, then pooling."""

    def __init__(self, dim, max_frames, dropout):
        super().__init__()
        self.frame_positions = nn.Parameter(torch.randn(1, max_frames, dim) * 0.02)  # frame order
        self.layer = nn.TransformerEncoderLayer(dim, HEADS, dim_feedforward=2 * dim, dropout=dropout,
                                                batch_first=True, norm_first=True)
        self.norm = nn.LayerNorm(dim)  # a pre-norm layer leaves its output unnormalised
        self.pool = FramePool(dim)

    def forward(self, tokens, mask):
        B, T, Q, D = tokens.shape
        x = tokens.transpose(1, 2).reshape(B * Q, T, D) + self.frame_positions[:, :T]  # each query its own sequence
        x = self.norm(self.layer(x, src_key_padding_mask=(~mask).repeat_interleave(Q, dim=0)))
        return self.pool(x.reshape(B, Q, T, D).transpose(1, 2), mask)


TEMPORAL = {"mean": FrameMean, "pool": FramePool, "transformer": FrameTransformer}


# ----- the pipeline -----

class JerseyModel(nn.Module):
    """Backbone -> reader -> frame combiner -> (2, 11) log-probabilities. See the module docstring.

    count_offset is added to the two-step reader's "one digit" score (see count_prior). It is not a
    weight: train.py saves it in config.json and load_run restores it.
    """

    def __init__(self, reader="two-step", temporal="mean", dim=128, max_frames=MAX_FRAMES, pretrained=True,
                 dropout=0.1, count_offset=0.0):
        super().__init__()
        if reader not in READERS or temporal not in TEMPORAL:
            raise ValueError(f"unknown reader {reader!r} or temporal {temporal!r}")
        if IMAGE_SIZE[0] % BACKBONE_STRIDE or IMAGE_SIZE[1] % BACKBONE_STRIDE:
            raise ValueError(f"IMAGE_SIZE {IMAGE_SIZE} must be divisible by {BACKBONE_STRIDE}")
        self.reader_name, self.temporal_name, self.count_offset = reader, temporal, count_offset
        self.freeze_batchnorm = pretrained  # keep the pretrained BatchNorm statistics fixed
        self.reader = READERS[reader](dim, pretrained)
        self.frames = TEMPORAL[temporal](self.reader.token_dim, max_frames, dropout)
        self.positions = self.reader.positions
        self.train()

    def train(self, mode=True):
        """Switch train/eval mode, but keep the backbone's BatchNorm statistics fixed if pretrained."""
        super().train(mode)
        if self.freeze_batchnorm:
            for layer in self.reader.backbone.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eval()
        return self

    def _encode(self, images):
        return self.reader(images)

    def _outputs(self, tokens, count_logits):
        return {"logits": self.reader.logits(tokens, count_logits, self.count_offset),
                "digit_logits": self.reader.digit_logits(tokens), "count_logits": count_logits}

    def read(self, images):
        """Single images (N, 3, H, W), e.g. the anchor -> dict: logits (N, 2, 11), digit_logits, count_logits, attention."""
        tokens, attention, count_logits = self._encode(images)
        return {**self._outputs(tokens, count_logits), "attention": attention}

    def forward(self, frames, mask):
        """Sequences (B, T, 3, H, W) with mask (B, T) -> dict.

        logits (B, 2, 11), frame_logits (B, T, 2, 11), frame_weights (B, T, Q) or None (mean),
        attention (B, T, Q, C), and for the two-step reader digit_logits (B, 3, 10), count_logits
        (B, 2), the mean of the frames' count logits, and frame_count_logits (B, T, 2); else None.
        """
        tokens, attention, count = (_scatter(part, mask) for part in self._encode(frames[mask]))  # real frames only
        summary, weights = self.frames(tokens, mask)
        out = self._outputs(summary, None if count is None else masked_mean(count, mask))
        out.update(frame_logits=self.reader.logits(tokens, count, self.count_offset), frame_weights=weights,
                   attention=attention, frame_count_logits=count)
        return out


def _scatter(values, mask):
    """Values of the real frames (R, ...) -> (B, T, ...), zero for padding; None stays None."""
    if values is None:
        return None
    out = values.new_zeros(*mask.shape, *values.shape[1:])
    out[mask] = values
    return out


def build_model(config, pretrained=False):
    """The model a run's config describes."""
    return JerseyModel(config["reader"], config["temporal"], max_frames=config["max_frames"], pretrained=pretrained,
                       count_offset=config.get("count_offset", 0.0))


def load_run(run_dir, device="cpu"):
    """Rebuild a trained model from run_dir/best.pt: returns (model in eval mode, config, epoch)."""
    checkpoint = torch.load(Path(run_dir) / "best.pt", map_location="cpu")
    model = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint["config"], checkpoint["epoch"]


def predict_digits(logits):
    """Log-probabilities (..., 2, 11) -> digits (..., 2) and one confidence per number (...).

    The two positions are scored independently, so the most likely digit at each position gives
    the most likely number, and its confidence is their product.
    """
    probabilities, digits = logits.softmax(dim=-1).max(dim=-1)
    return digits, probabilities.prod(dim=-1)
