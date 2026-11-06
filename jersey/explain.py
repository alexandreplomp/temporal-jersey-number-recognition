"""Show what the model bases its answer on, and check that the pictures are faithful.

For one sequence at a time:
    query_maps          where each digit query looks in each frame (attention)
    occlusion_map       how much covering each cell lowers a frame's own reading
    attention_check     covering the attended cells vs the same shape elsewhere
    frame_weights       how much each frame counts per digit
    temporal_attention  which frames each frame draws on (full only)
    frame_removal       how much the answer drops without each frame
    map_similarity      agreement between two maps

Attention alone is not an explanation (Jain & Wallace 2019): a model can be accurate
while attending off the digits. So each attention map is set against occlusion (Zeiler
& Fergus 2014), which changes the input and measures the effect, and against the maps of
an untrained model (Adebayo et al. 2018): maps that look alike for both show the image,
not the model. Frame removal is small when other frames can compensate, so read it next
to the frame weights, not alone.

Maps use the model's grid of CELL_STRIDE-pixel cells (24 x 12 for a 192 x 96 crop), so
attention and occlusion compare cell for cell. Occlusion covers a cell in one frame and
reads that frame alone, since in a sequence the other frames would hide the effect.
Functions take one sequence, frames (T, 3, H, W) and mask (T,), use its R real frames,
and expect a model in eval mode. explain_run puts it together for train.py and explain.py.
"""

from collections import defaultdict
import json
import math

from matplotlib.figure import Figure
import torch

from jersey.data import BLANK, decode_label, denormalize
from jersey.model import CELL_STRIDE, JerseyModel, predict_digits

CELL = CELL_STRIDE  # pixels per side of one grid cell
QUERY_COLOURS = ((1.0, 0.15, 0.1), (0.1, 0.45, 1.0))  # digit 1 red, digit 2 blue


@torch.no_grad()
def query_maps(model, frames, mask):
    """Attention of each digit query over the cells of each real frame: (R, 2, h, w), each sums to 1."""
    real = frames[mask]
    model.read_tokens(real)
    return model.digits.last_attention.reshape(len(real), 2, *_grid(real))


@torch.no_grad()
def occlusion_map(model, frames, mask, chunk=256):
    """Drop in each frame's own probability for its own reading when one cell is covered: (R, 2, h, w).

    Positive where covering the cell hurts the reading of that digit.
    """
    real = frames[mask]
    h, w = _grid(real)
    digits = model.read_frames(real).argmax(dim=-1)  # (R, 2), the frame's own reading
    base = _digit_probability(model, real, digits)  # (R, 2)
    cells = torch.eye(h * w, dtype=torch.bool, device=real.device).reshape(h * w, h, w)
    # Every real frame with every cell covered, in chunks to bound memory.
    covered = []
    for part in torch.arange(len(real) * h * w, device=real.device).split(chunk):
        frame, cell = part // (h * w), part % (h * w)
        covered.append(_digit_probability(model, _cover(real[frame], cells[cell]), digits[frame]))
    covered = torch.cat(covered).reshape(len(real), h * w, 2)
    return (base[:, None] - covered).transpose(1, 2).reshape(len(real), 2, h, w)


@torch.no_grad()
def attention_check(model, frames, mask, maps, k=4, draws=5, seed=0):
    """Cover the k most attended cells, or the same shape shifted elsewhere, and read each frame again.

    Returns (R, 2) probabilities of each frame's own reading: base (nothing covered),
    attended (the top-k cells of that digit's query covered) and elsewhere (mean over
    draws of the same cell pattern shifted to where it does not overlap). If attention
    shows what is read, attended should fall well below elsewhere.
    """
    real = frames[mask]
    R = len(real)
    h, w = _grid(real)
    digits = model.read_frames(real).argmax(dim=-1)
    generator = torch.Generator().manual_seed(seed)
    attended = torch.zeros(R, 2, h, w, dtype=torch.bool)
    for r in range(R):
        for p in range(2):
            attended[r, p].view(-1)[maps[r, p].flatten().topk(k).indices.cpu()] = True
    elsewhere = torch.stack([
        torch.stack([_shifted(attended[r, p], draws, generator) for p in range(2)])
        for r in range(R)
    ])  # (R, 2, draws, h, w)

    result = {"base": _digit_probability(model, real, digits)}
    result["attended"] = torch.stack([
        _digit_probability(model, _cover(real, attended[:, p].to(real.device)), digits)[:, p]
        for p in range(2)
    ], dim=1)
    result["elsewhere"] = torch.stack([
        torch.stack([
            _digit_probability(model, _cover(real, elsewhere[:, p, d].to(real.device)), digits)[:, p]
            for d in range(draws)
        ]).mean(dim=0)
        for p in range(2)
    ], dim=1)
    return {key: value.cpu() for key, value in result.items()}


@torch.no_grad()
def frame_weights(model, frames, mask):
    """How much each real frame counts per digit: (R, 2). Equal for the averaged variant."""
    real = frames[mask]
    weights = model(real[None], _all_real(real))["frame_weights"]
    return torch.full((len(real), 2), 1 / len(real)) if weights is None else weights[0].cpu()


@torch.no_grad()
def temporal_attention(model, frames, mask):
    """Frame-to-frame attention per digit, mean over heads: (2, R, R), row i is what frame i draws on.

    None unless the model is the full variant.
    """
    if model.temporal is None:
        return None
    real = frames[mask]
    # The two-step reader's last two queries are the two digit positions (left, right).
    return temporal_pass(model, model.read_tokens(real)[None])[1][-2:].cpu()


def temporal_pass(model, tokens):
    """Repeat model.temporal on tokens (1, R, Q, D) of real frames, keeping the attention.

    nn.TransformerEncoderLayer never returns its attention weights, so its pre-norm steps
    are redone here from its own modules. Returns the output (1, R, 2, D), equal to
    model.temporal(tokens, mask), and the attention (2, R, R).
    """
    encoder, (_, R, _, D) = model.temporal, tokens.shape
    Q = tokens.shape[2]
    x = tokens.transpose(1, 2).reshape(Q, R, D) + encoder.frame_positions[:, :R]
    layer = encoder.layer
    normed = layer.norm1(x)
    attended, weights = layer.self_attn(normed, normed, normed, need_weights=True)
    x = x + attended
    x = x + layer._ff_block(layer.norm2(x))  # the layer's own feed-forward step; dropout is off in eval
    return encoder.norm(x).reshape(1, Q, R, D).transpose(1, 2), weights


@torch.no_grad()
def frame_removal(model, frames, mask):
    """Drop in the sequence's probability for its answer, per digit, when one frame is left out: (R, 2).

    nan for a single frame, which cannot be left out.
    """
    real = frames[mask]
    R = len(real)
    if R < 2:
        return torch.full((R, 2), math.nan)
    logits = model(real[None], _all_real(real))["logits"]
    digits = logits.argmax(dim=-1)  # (1, 2)
    base = logits.softmax(dim=-1).gather(-1, digits[..., None]).squeeze(-1)  # (1, 2)
    leave_out = ~torch.eye(R, dtype=torch.bool, device=real.device)  # row r leaves out frame r
    probabilities = model(real[None].expand(R, -1, -1, -1, -1), leave_out)["logits"].softmax(dim=-1)
    return (base - probabilities.gather(-1, digits.expand(R, 2)[..., None]).squeeze(-1)).cpu()


def map_similarity(a, b):
    """Pearson correlation of two maps of any shape; nan if one is flat."""
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    norm = a.norm() * b.norm()
    return (a @ b / norm).item() if norm > 0 else math.nan


def explain_sequence(model, random_model, item, device):
    """Run every view and check on one dataset item and return them as one case (on the CPU)."""
    frames, mask = item["frames"].to(device), item["mask"].to(device)
    real = frames[mask]
    with torch.no_grad():
        digits, confidence = predict_digits(model(real[None], _all_real(real))["logits"])
        frame_digits = model.read_frames(real).argmax(dim=-1).cpu()
    maps = query_maps(model, frames, mask)
    random_maps = query_maps(random_model, frames, mask)
    occlusion = occlusion_map(model, frames, mask)
    true = decode_label(*item["label"].tolist())
    predicted = decode_label(*digits[0].tolist())
    return {
        "sequence_id": item["sequence_id"],
        "true": true,
        "predicted": predicted,
        "confidence": confidence.item(),
        "label": item["label"],
        "images": denormalize(real).cpu(),
        "frame_digits": frame_digits,
        "maps": maps.cpu(),
        "occlusion": occlusion.cpu(),
        "check": attention_check(model, frames, mask, maps),
        "weights": frame_weights(model, frames, mask),
        "removal": frame_removal(model, frames, mask),
        "temporal": temporal_attention(model, frames, mask),
        "similarity": {
            "occlusion": [_mean([map_similarity(maps[r, p], occlusion[r, p]) for r in range(len(real))])
                          for p in range(2)],
            "random_model": [_mean([map_similarity(maps[r, p], random_maps[r, p]) for r in range(len(real))])
                             for p in range(2)],
        },
    }


def pick_cases(sequence_ids, seed):
    """Return one dataset index per number, chosen at random with seed, in numeric order."""
    by_number = defaultdict(list)
    for index, sequence_id in enumerate(sequence_ids):
        by_number[sequence_id.split("/")[0]].append(index)
    generator = torch.Generator().manual_seed(seed)
    return [
        by_number[number][torch.randint(len(by_number[number]), (1,), generator=generator).item()]
        for number in sorted(by_number, key=int)
    ]


def summarize(cases):
    """Means over all cases, per digit: the numbers behind the figure, for explain.json."""
    def per_digit(values_of):
        return [_mean([value for case in cases for value in values_of(case, p)]) for p in range(2)]

    def own_reading_right(case, p):
        return case["frame_digits"][:, p] == case["label"][p]

    return {
        "cases": len(cases),
        "right": sum(case["true"] == case["predicted"] for case in cases),
        "attention_vs_occlusion": per_digit(lambda case, p: [case["similarity"]["occlusion"][p]]),
        "attention_vs_random_model": per_digit(lambda case, p: [case["similarity"]["random_model"][p]]),
        "probability": {
            key: per_digit(lambda case, p, key=key: case["check"][key][:, p].tolist())
            for key in ("base", "attended", "elsewhere")
        },
        # Does weighting favour frames that read the digit right on their own?
        "frame_weight": {
            "reads_right": per_digit(lambda case, p: case["weights"][own_reading_right(case, p), p].tolist()),
            "reads_wrong": per_digit(lambda case, p: case["weights"][~own_reading_right(case, p), p].tolist()),
        },
        "frame_removal": per_digit(lambda case, p: case["removal"][:, p].tolist()),
    }


def save_prediction_figure(cases, path):
    """One block of three rows per case: query attention, then covering each cell for digit 1 and 2."""
    columns = max(len(case["images"]) for case in cases) + 1
    figure = Figure(figsize=(1.25 * columns, 2.0 * 3 * len(cases)))
    axes = figure.subplots(3 * len(cases), columns, squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for block, case in enumerate(cases):
        _draw_case(axes[3 * block:3 * block + 3], case)
    figure.subplots_adjust(left=0.04, right=0.98, top=1 - 0.5 / (2.0 * 3 * len(cases)), bottom=0.01,
                           hspace=0.45, wspace=0.08)
    figure.savefig(path, dpi=110)


def explain_run(model, dataset, config, run_dir, device):
    """Explain one sequence per number of dataset: write prediction.png and explain.json, return the summary."""
    model.eval()
    # A fixed untrained model for the randomization check, without touching the training RNG.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config["seed"])
        random_model = JerseyModel(config["variant"], max_frames=config["max_frames"], pretrained=False,
                                   reader=config.get("reader", "fused"))
    random_model.to(device).eval()
    cases = [
        explain_sequence(model, random_model, dataset[index], device)
        for index in pick_cases(dataset.sequence_ids, config["seed"])
    ]
    save_prediction_figure(cases, run_dir / "prediction.png")
    summary = summarize(cases)
    records = [_record(case) for case in cases]
    (run_dir / "explain.json").write_text(json.dumps({"summary": summary, "cases": records}, indent=2))
    return summary


def _grid(images):
    return images.shape[-2] // CELL, images.shape[-1] // CELL


def _all_real(real):
    return torch.ones(1, len(real), dtype=torch.bool, device=real.device)


def _cover(images, cells):
    """Images (N, 3, H, W) with the cells marked True in cells (N, h, w) set to 0, the ImageNet mean."""
    pixels = cells.repeat_interleave(CELL, dim=1).repeat_interleave(CELL, dim=2)
    return images * ~pixels[:, None]


def _digit_probability(model, images, digits):
    """Probability that each image alone reads the given digits (N, 2): (N, 2)."""
    probabilities = model.read_frames(images).softmax(dim=-1)
    return probabilities.gather(-1, digits[..., None]).squeeze(-1)


def _shifted(cells, draws, generator):
    """draws copies of the cell pattern (h, w) moved to where it does not overlap itself.

    Shifts wrap around the grid. If no shift avoids overlap, the least overlapping ones are used.
    """
    h, w = cells.shape
    shifts = [(dy, dx) for dy in range(h) for dx in range(w) if (dy, dx) != (0, 0)]
    moved = torch.stack([cells.roll((dy, dx), dims=(0, 1)) for dy, dx in shifts])
    overlap = (moved & cells).flatten(1).sum(dim=1)
    candidates = moved[overlap == overlap.min()]
    picked = torch.randperm(len(candidates), generator=generator)[:draws]
    return candidates[picked.repeat(draws)[:draws]]


def _mean(values):
    values = [value for value in values if not math.isnan(value)]
    return sum(values) / len(values) if values else math.nan


def _reading(digits):
    """A frame's own reading as text; a frame can read BLANK in position 2 only."""
    return decode_label(*digits.tolist()) if digits[0] != BLANK else "?"


def _overlay(maps, size, colours):
    """Maps (k, h, w) as one RGBA image (H, W, 4): each map in its colour, opacity by strength."""
    H, W = size
    strength = maps.clamp(min=0)
    strength = strength / strength.flatten(1).amax(dim=1).clamp(min=1e-8)[:, None, None]
    strength = strength.repeat_interleave(H // maps.shape[1], dim=1).repeat_interleave(W // maps.shape[2], dim=2)
    colours = torch.tensor(colours)[:, :, None, None]  # (k, 3, 1, 1)
    rgb = (colours * strength[:, None]).sum(dim=0) / strength.sum(dim=0).clamp(min=1e-8)
    alpha = 0.75 * strength.amax(dim=0)
    return torch.cat([rgb, alpha[None]]).permute(1, 2, 0).numpy()


def _draw_case(rows, case):
    images, H, W = case["images"], *case["images"].shape[-2:]
    right = case["true"] == case["predicted"]
    lines = [f"{case['true']} -> {case['predicted']} ({case['confidence']:.2f})"
             f" {'right' if right else 'WRONG'}   {case['sequence_id']}"]
    for p in range(2):
        check = {key: value[:, p].mean().item() for key, value in case["check"].items()}
        lines.append(
            f"digit {p + 1}: p {check['base']:.2f}, attended covered {check['attended']:.2f},"
            f" elsewhere {check['elsewhere']:.2f} | attention~occlusion"
            f" {case['similarity']['occlusion'][p]:+.2f}, ~untrained {case['similarity']['random_model'][p]:+.2f}"
        )
    rows[0][0].text(0, 1.08, "\n".join(lines), transform=rows[0][0].transAxes,
                    fontsize=7, va="bottom", ha="left", color="black" if right else "crimson")
    for row, label in zip(rows, ("attention", "cover: digit 1", "cover: digit 2")):
        row[0].text(-0.08, 0.5, label, transform=row[0].transAxes, rotation=90, fontsize=6,
                    ha="right", va="center")

    for r, image in enumerate(images.permute(0, 2, 3, 1).numpy()):
        weights, removal = case["weights"][r], case["removal"][r]
        layers = (
            (case["maps"][r], QUERY_COLOURS),
            (case["occlusion"][r, :1], QUERY_COLOURS[:1]),
            (case["occlusion"][r, 1:], QUERY_COLOURS[1:]),
        )
        for row, (maps, colours) in zip(rows, layers):
            row[r].imshow(image)
            row[r].imshow(_overlay(maps, (H, W), colours))
        rows[0][r].text(
            0.5, -0.03,
            f"reads {_reading(case['frame_digits'][r])}\nw {weights[0]:.2f}/{weights[1]:.2f}"
            f"\nout {removal[0]:+.2f}/{removal[1]:+.2f}",
            transform=rows[0][r].transAxes, fontsize=6, ha="center", va="top",
        )

    # Last column: how frames relate (full) or how much each counts (averaged, pooled).
    rows[0][-1].text(0, 0.5, "red: digit 1\nblue: digit 2\nw: frame weight\nout: drop if left out",
                     fontsize=6, va="center")
    for p in range(2):
        axis = rows[p + 1][-1]
        axis.axis("on")
        axis.tick_params(labelsize=5)
        if case["temporal"] is not None:
            axis.imshow(case["temporal"][p].numpy(), vmin=0, vmax=1, cmap="viridis")
            axis.set_title(f"frame attention d{p + 1}", fontsize=6)
        else:
            axis.barh(range(len(images)), case["weights"][:, p].numpy(), color=QUERY_COLOURS[p])
            axis.invert_yaxis()
            axis.set_title(f"frame weight d{p + 1}", fontsize=6)


def _record(case):
    """The JSON-friendly part of a case."""
    return {
        "sequence_id": case["sequence_id"],
        "true": case["true"],
        "predicted": case["predicted"],
        "confidence": case["confidence"],
        "frames": len(case["images"]),
        "frame_readings": [_reading(digits) for digits in case["frame_digits"]],
        "similarity": case["similarity"],
        "check": {key: value.mean(dim=0).tolist() for key, value in case["check"].items()},
        "frame_weights": case["weights"].tolist(),
        "frame_removal": case["removal"].tolist(),
    }
