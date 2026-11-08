"""Every figure of the project. Each function takes data and a path and writes one PNG; no computation."""

from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import torch
from torchvision.transforms import v2
from torchvision.utils import make_grid, save_image

from jersey.data import IMAGE_SIZE, denormalize, load_image, number_of
from jersey.pairs import FEATHERS, GAP, paste_single, splice

PANELS = ("loss", "accuracy", "two_digit_accuracy", "macro_accuracy")


def _save(figure, path):
    figure.tight_layout()
    figure.savefig(path, dpi=110)


def curves(history, path, stage_switch=None):
    """Train (running), train-check and val loss, accuracy, two-digit and macro accuracy per epoch.

    history is a list of history.json entries; stage_switch marks the first epoch of stage 2.
    """
    figure = Figure(figsize=(20, 4))
    for axis, metric in zip(figure.subplots(1, 4), PANELS):
        sources = (["train"] if metric == "loss" else []) + ["train_check", "val"]
        for source in sources:
            key = "train_loss" if source == "train" else f"{source}_{metric}"
            points = [(entry["step"], entry[key]) for entry in history if entry.get(key) is not None]
            if points:
                axis.plot(*zip(*points), marker=".", label="train (running, augmented)" if source == "train" else source)
        if stage_switch is not None:
            axis.axvline(stage_switch, color="grey", linestyle="--", linewidth=1, label="stage 2")
        axis.set_title(metric.replace("_", " "))
        axis.set_xlabel("epoch")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.legend()
    _save(figure, path)


def compare_curves(segments, path):
    """Val loss and val accuracy per epoch, one colour per run: segments {run name: [stage 1 entries, stage 2 entries]}.

    Stage 2 starts from stage 1's best epoch, not its last, so the stages are drawn as separate lines
    (stage 2 dashed) with the switch marked, never joined.
    """
    figure = Figure(figsize=(12, 4))
    switches = sorted({stages[-1][0]["step"] for stages in segments.values() if len(stages) > 1})
    for axis, key in zip(figure.subplots(1, 2), ("val_loss", "val_accuracy")):
        for index, (name, stages) in enumerate(segments.items()):
            colour = f"C{index}"
            for number, stage in enumerate(stages):
                axis.plot([entry["step"] for entry in stage], [entry[key] for entry in stage], color=colour,
                          linestyle="--" if number else "-", label=name if number == 0 else None)
        for switch in switches:
            axis.axvline(switch, color="grey", linestyle=":", linewidth=1)
        axis.set_title(key.replace("_", " ") + (" (dotted line: stage 2 starts from stage 1's best epoch)" if switches else ""),
                       fontsize=9)
        axis.set_xlabel("epoch")
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.legend(fontsize=7)
    _save(figure, path)


def temporal(metrics, path):
    """Left: accuracy after the first 1-5 frames per run. Right: sequence vs anchor vs vote accuracy per run."""
    figure = Figure(figsize=(12, 4))
    left, right = figure.subplots(1, 2)
    for name, found in metrics.items():
        by_frames = found["stability"]["accuracy_by_frames"]
        left.plot([int(k) for k in by_frames], [v["accuracy"] for v in by_frames.values()], marker="o", label=name)
    left.set(title="accuracy after the first k frames", xlabel="frames", ylabel="accuracy")
    left.xaxis.set_major_locator(MaxNLocator(integer=True))
    left.legend(fontsize=7)
    names = list(metrics)
    for i, method in enumerate(("sequence", "anchor", "vote")):
        right.bar([x + 0.27 * i for x in range(len(names))], [metrics[n]["methods"][method]["accuracy"] for n in names],
                  width=0.27, label=method)
    right.set_xticks([x + 0.27 for x in range(len(names))], names, rotation=20, fontsize=8)
    right.set(title="sequence vs anchor alone vs frame vote", ylabel="val accuracy", ylim=(0, 1.15), yticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    right.legend(loc="upper center", ncol=3, fontsize=8)
    _save(figure, path)


def holdout(answers, path, held="49", named=("99", "89")):
    """What the held-out number is read as, per run: answers {run name: {"counts": {answer: count}, "sequences": n}}.

    Bars are shares with the count written on each part; the common wrong answers in named get their own part.
    """
    def kind_of(answer):
        if answer == held:
            return "right"
        if answer in named:
            return answer
        return "one digit" if len(answer) == 1 else "other two-digit"

    kinds = ("right", *named, "other two-digit", "one digit")
    figure = Figure(figsize=(9, 4.5))
    axis = figure.subplots()
    labels = [f"{name.removesuffix('_ex' + held)}\n(n={found['sequences']})" for name, found in answers.items()]
    bottoms = [0.0] * len(answers)
    for kind in kinds:
        counts = [sum(c for a, c in found["counts"].items() if kind_of(a) == kind) for found in answers.values()]
        heights = [c / found["sequences"] for c, found in zip(counts, answers.values())]
        bars = axis.bar(labels, heights, bottom=bottoms, label=kind if kind == "right" else f"read as {kind}")
        axis.bar_label(bars, labels=[str(c) if c else "" for c in counts], label_type="center", fontsize=7)
        bottoms = [b + h for b, h in zip(bottoms, heights)]
    axis.set(title=f"held-out {held} read as (development hold-out: train- and val-split players)",
             ylabel="share of sequences", ylim=(0, 1))
    axis.tick_params(axis="x", labelsize=7)
    axis.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    _save(figure, path)


# The final model: (box text, tensor shape on the arrow into the next box).
ARCHITECTURE = (
    ("sequence of T crops\n(T ≤ 10, 192 × 96)", "B, T, 3, 192, 96"),
    ("number area:\nrows 17-48%,\ncolumns 15-85%,\nstretched to 32 × 128", "N real frames,\n3, 32, 128"),
    ("PARSeq-Tiny encoder\n(ViT, 12 layers, 5.38M)\nshared over frames", "N, 128 patches,\n192"),
    ("PARSeq decoder,\n1 layer (0.60M):\nquery 1 → digit 1,\nquery 2 → digit 2 or end", "B, T, 2, 192"),
    ("frame gate per position\n(learned weight\nper frame, 0.02M)", "B, 2, 192"),
    ("linear head:\nposition 1: digit 0-9\nposition 2: digit 0-9\nor BLANK", "B, 2, 11"),
    ("one number\n+ confidence", None),
)


def architecture(path, boxes=ARCHITECTURE):
    """The final model's blocks left to right, with the tensor shape on each arrow (N = real frames <= B x T)."""
    figure = Figure(figsize=(2.3 * len(boxes), 1.9))
    axis = figure.subplots()
    width, gap = 1.6, 0.7
    for i, (text, shape) in enumerate(boxes):
        left = i * (width + gap)
        axis.add_patch(Rectangle((left, 0.1), width, 1.2, facecolor="#eef3fb", edgecolor="#3b5b92", linewidth=1.2))
        axis.text(left + width / 2, 0.7, text, ha="center", va="center", fontsize=8)
        if shape is not None:
            axis.annotate("", xy=(left + width + gap, 0.7), xytext=(left + width, 0.7),
                          arrowprops={"arrowstyle": "->", "color": "#3b5b92"})
            axis.text(left + width + gap / 2, 0.8, shape, ha="center", va="bottom", fontsize=6.5, color="#7a2e2e")
    axis.set(xlim=(-0.1, len(boxes) * (width + gap) - gap + 0.1), ylim=(0.05, 1.35))
    axis.axis("off")
    _save(figure, path)


def attention(images, maps, titles, path):
    """Each query's attention over a few frames: images (N, 3, H, W) normalised, maps (N, Q, rows, columns).

    For the parseq reader the map covers its number area only (the stretched text line), drawn over the whole crop.
    """
    colours = ((1.0, 0.8, 0.1), (1.0, 0.15, 0.1), (0.1, 0.45, 1.0))  # single / first, left / second, right
    figure = Figure(figsize=(1.6 * len(images), 3.2))
    for i, axis in enumerate(figure.subplots(1, len(images), squeeze=False)[0]):
        picture = denormalize(images[i]).permute(1, 2, 0)
        overlay = torch.zeros(*IMAGE_SIZE, 3)
        for query, colour in enumerate(colours[-maps.shape[1]:]):
            heat = torch.nn.functional.interpolate(maps[i, query][None, None], size=IMAGE_SIZE, mode="nearest")[0, 0]
            overlay += (heat / heat.max().clamp(min=1e-9))[..., None] * torch.tensor(colour)
        axis.imshow((0.5 * picture + 0.5 * overlay.clamp(0, 1)).numpy())
        axis.set_title(titles[i], fontsize=8)
        axis.axis("off")
    _save(figure, path)


def batch_grid(batch, path):
    """A training batch as the model sees it: one sequence per row, the anchor last, padding black."""
    images = torch.cat([batch["frames"], batch["anchor"][:, None]], dim=1) if "anchor" in batch else batch["frames"]
    images = denormalize(images)
    images[:, :batch["mask"].shape[1]][~batch["mask"]] = 0
    save_image(make_grid(images.flatten(0, 1), nrow=images.shape[1]), path)


def boxes(data_dir, images, path, count=100):
    """A spread of located images with their box (green kept, red rejected) and the evidence centre."""
    picks = _spread(images, count)
    figure, rows, columns = _grid(len(picks))
    resize = v2.Resize(IMAGE_SIZE, antialias=True)
    for i, found in enumerate(picks):
        axis = figure.add_subplot(rows, columns, i + 1)
        _show(axis, load_image(Path(data_dir) / found["image"], resize), f"{i + 1}: {number_of(found['image'])} p{found['prob']:.2f}")
        if found["box"] is not None:
            _box(axis, found["box"], "lime" if found["kept"] else "red")
        axis.plot(*found["centre"], marker=".", color="yellow", markersize=3)
    _save(figure, path)


def pastes(data_dir, bank, path, single, count=48):
    """First frame of a spread of usable sequences, spliced with a partner (or its digit pasted alone)."""
    resize = v2.Resize(IMAGE_SIZE, antialias=True)
    picks = _spread(bank.sequence_ids(), count)
    figure, rows, columns = _grid(len(picks))
    shown = 0
    for sequence_id in picks:
        found = (bank.single_partner(sequence_id), None) if single else bank.partner(sequence_id)
        if found is None or found[0] is None:
            continue
        partner_id, a_left = found
        (a, a_box), (b, b_box) = bank.frames[sequence_id][0], bank.frames[partner_id][0]
        a_image, b_image = load_image(Path(data_dir) / a, resize), load_image(Path(data_dir) / b, resize)
        gap, feather = torch.randint(GAP[0], GAP[1] + 1, ()).item(), torch.randint(FEATHERS[0], FEATHERS[1] + 1, ()).item()
        digit, other = number_of(sequence_id), number_of(partner_id)
        shown += 1
        axis = figure.add_subplot(rows, columns, shown)
        if single:
            image, box = paste_single(a_image, a_box, b_image, b_box, gap, feather)
            _show(axis, image, f"{other}  (on {digit})")
            _box(axis, box.tolist(), "red")
        else:
            image, found_boxes = splice(a_image, a_box, b_image, b_box, a_left, gap, feather)
            _show(axis, image, f"{digit + other if a_left else other + digit}  (from {digit})")
            for found_box, colour in zip(found_boxes.tolist(), ("red", "deepskyblue")):
                _box(axis, found_box, colour)
    _save(figure, path)


def _show(axis, image, title):
    axis.imshow(image.permute(1, 2, 0).numpy(), interpolation="nearest")
    axis.set_title(title, fontsize=7)
    axis.axis("off")


def _box(axis, box, colour):
    left, top, right, bottom = box
    axis.add_patch(Rectangle((left - 0.5, top - 0.5), right - left, bottom - top, fill=False, linewidth=0.8, edgecolor=colour))


def _grid(count, columns=12):
    rows = max(1, -(-count // columns))
    return Figure(figsize=(columns * 1.3, rows * 2.8)), rows, columns


def _spread(items, count):
    return [items[i] for i in torch.linspace(0, len(items) - 1, min(count, len(items))).round().long().tolist()] if items else []


def cell_grid(maps, grid):
    """Attention (N, Q, C) over a reader's grid of cells or patches -> (N, Q, rows, columns)."""
    return maps.reshape(*maps.shape[:2], *grid)
