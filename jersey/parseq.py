"""PARSeq-Tiny, a pretrained scene text reader, cut down to the ten digits: the network of the "parseq" reader.

The encoder (a small ViT over 4 x 8 patches of a 32 x 128 text line) and the decoder are from
https://github.com/baudm/parseq, Copyright 2022 Darwin Bautista, Apache License 2.0, kept as written
there except: the tokenizer-driven decoding loop is replaced by read() (every position read from the
image alone, never from the digits read before it), and named_apply is defined here.

The released weights read 94 characters, "0123456789" first; digit_weights keeps the rows of the end
token and the ten digits. Tokens here: 0 end of number, 1-10 the digits 0-9, 11 start, 12 padding.
"""

from functools import partial
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.modules import transformer
from timm.models.vision_transformer import PatchEmbed, VisionTransformer

TEXT_SIZE = (32, 128)  # height, width of PARSeq's input
PATCH_SIZE = (4, 8)
EMBED_DIM, ENCODER_HEADS, DECODER_HEADS = 192, 3, 6  # the Tiny size
POSITIONS = 3  # digit 1, digit 2 or end, end after two digits
EOS, BOS, PAD, TOKENS = 0, 11, 12, 13
WEIGHTS_URL = "https://github.com/baudm/parseq/releases/download/v1.0.0/parseq_tiny-e7a21b54.pt"
RELEASED_ROWS = list(range(11)) + [95, 96]  # end, the digits, start, padding among the 97 released tokens


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


# ----- this project -----

def named_apply(fn, module, name=''):
    """fn(module=..., name=...) on every submodule, children first (as timm's named_apply)."""
    for child_name, child in module.named_children():
        named_apply(fn, child, f"{name}.{child_name}" if name else child_name)
    fn(module=module, name=name)


def make_encoder():
    return Encoder(TEXT_SIZE, PATCH_SIZE, embed_dim=EMBED_DIM, depth=12, num_heads=ENCODER_HEADS, mlp_ratio=4)


class TextDecoder(nn.Module):
    """PARSeq's decoder, head and position queries; read() queries every position at once, from the image alone."""

    def __init__(self, dropout=0.1):
        super().__init__()
        self.decoder = Decoder(DecoderLayer(EMBED_DIM, DECODER_HEADS, EMBED_DIM * 4, dropout), num_layers=1,
                               norm=nn.LayerNorm(EMBED_DIM))
        self.head = nn.Linear(EMBED_DIM, TOKENS - 2)  # predicts the end token and the ten digits
        self.text_embed = TokenEmbedding(TOKENS, EMBED_DIM)
        self.pos_queries = nn.Parameter(torch.Tensor(1, POSITIONS, EMBED_DIM))
        self.dropout = nn.Dropout(p=dropout)
        named_apply(init_weights, self)
        nn.init.trunc_normal_(self.pos_queries, std=0.02)
        self.decoder.layers[-1].cross_attn.register_forward_hook(self._keep_attention)

    def _keep_attention(self, module, inputs, output):
        self.attention = output[1]  # (N, positions, patches), averaged over heads

    def read(self, memory):
        """Encoded text lines (N, patches, D) -> position features (N, 3, D) and their attention (N, 3, patches)."""
        start = self.text_embed(torch.full((len(memory), 1), BOS, dtype=torch.long, device=memory.device))
        queries = self.dropout(self.pos_queries.expand(len(memory), -1, -1))
        return self.decoder(queries, self.dropout(start), memory), self.attention


def digit_weights():
    """The released PARSeq-Tiny weights, cut to the digits, split into the encoder's and the decoder's state dicts."""
    released = torch.hub.load_state_dict_from_url(WEIGHTS_URL, map_location="cpu", check_hash=True)
    released = {key.removeprefix("model."): value for key, value in released.items()}
    released["head.weight"], released["head.bias"] = released["head.weight"][:TOKENS - 2], released["head.bias"][:TOKENS - 2]
    released["text_embed.embedding.weight"] = released["text_embed.embedding.weight"][RELEASED_ROWS]
    released["pos_queries"] = released["pos_queries"][:, :POSITIONS]
    encoder = {k.removeprefix("encoder."): v for k, v in released.items() if k.startswith("encoder.")}
    decoder = {k: v for k, v in released.items() if not k.startswith("encoder.")}
    return encoder, decoder


def text_lines(images, rows, columns, mean, std):
    """ImageNet-normalised crops (N, 3, H, W) -> PARSeq's text lines (N, 3, 32, 128) of the number area.

    rows and columns are (start, end) pixels; the area is stretched to 32 x 128 (stretching read
    47.7% of 300 val sequences zero-shot, keeping the aspect ratio 3-15%).
    """
    area = (images[..., rows[0]:rows[1], columns[0]:columns[1]] * std + mean).clamp(0, 1)
    return (F.interpolate(area, size=TEXT_SIZE, mode="bilinear", align_corners=False, antialias=True) - 0.5) / 0.5
