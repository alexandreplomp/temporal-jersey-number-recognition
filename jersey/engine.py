"""One pass over a loader: train for an epoch, or evaluate. Saving files is left to train.py.

Loss: cross-entropy over both digit positions of the sequence logits, plus the same
loss on the anchor read on its own through read_frames, weighted by anchor_weight.
The anchor is the clearest frame, so it gives the CNN and the digit queries a clean
example. The deployed model never sees it, so evaluation leaves it out, except for
the anchor-only comparison (use_anchor).

frame_logits are not trained yet. To add a per-frame loss later, use
frame_logits[mask], so padded frames add nothing.

The two-step reader trains on its parts instead (two_step_loss): each query on the digit
it should read, the count, and, where the digit boxes are known (spliced sequences), the
left and right attention on their boxes. Its validation loss stays digit_loss on the
composed output, so val losses compare between readers.
"""

import math

import torch
from torch.nn import functional as F

from jersey.data import BLANK, IMAGE_SIZE
from jersey.model import CELL_STRIDE, predict_digits

ANCHOR_WEIGHT = 0.5  # a starting guess, to tune on val
ATTENTION_WEIGHT = 0.5
COUNT_WEIGHT = 1.0


def digit_loss(logits, labels):
    """Mean cross-entropy over both positions: logits (..., 2, 11), labels (..., 2).

    The model fixes BLANK in position 1 at -inf, and no label puts BLANK there, so the
    loss stays finite. Label smoothing would put weight on that class and make it inf.
    """
    return F.cross_entropy(logits.flatten(0, -2), labels.flatten())


def box_cells(boxes, grid=(IMAGE_SIZE[0] // CELL_STRIDE, IMAGE_SIZE[1] // CELL_STRIDE), stride=CELL_STRIDE):
    """True for the cells a box overlaps: boxes (..., 4) as [left, top, right, bottom] -> (..., rows * columns).

    Overlap rather than the cell centre, since a thin digit (6 pixels) can fall between centres.
    """
    rows, columns = grid
    x0 = torch.arange(columns, device=boxes.device) * stride  # (columns,) left edge of each cell
    y0 = torch.arange(rows, device=boxes.device) * stride
    left, top, right, bottom = (boxes[..., i, None] for i in range(4))
    across = (x0 < right) & (x0 + stride > left)  # (..., columns)
    down = (y0 < bottom) & (y0 + stride > top)  # (..., rows)
    return (down[..., :, None] & across[..., None, :]).flatten(-2)


def two_step_loss(out, labels, boxes=None, has_boxes=None, frame_mask=None,
                  attention_weight=ATTENTION_WEIGHT, count_weight=COUNT_WEIGHT):
    """Loss of the two-step reader on a batch: the total and its parts (floats, for logging).

    out is forward's dict (or read_frames_full's, with frame_mask None for single images):
        digits     one-digit samples: cross-entropy of the single query on the digit;
                   two-digit samples: the mean of left on the first and right on the second
        count      cross-entropy on one or two digits, the two classes weighed equally
        attention  where boxes are known (has_boxes): -log(attention inside the digit's box)
                   for left and right, over the real frames
    boxes (B, T, 2, 4) or (B, 2, 4) for single images; has_boxes (B,).
    """
    two = labels[:, 1] != BLANK
    digit_logits = out["digit_logits"]
    single = F.cross_entropy(digit_logits[:, 0], labels[:, 0], reduction="none")
    left = F.cross_entropy(digit_logits[:, 1], labels[:, 0], reduction="none")
    right = F.cross_entropy(digit_logits[:, 2], labels[:, 1].clamp(max=BLANK - 1), reduction="none")
    digits = torch.where(two, (left + right) / 2, single).mean()

    count_each = F.cross_entropy(out["count_logits"], two.long(), reduction="none")
    count = torch.stack([count_each[two == kind].mean() for kind in (False, True) if (two == kind).any()]).mean()

    total = digits + count_weight * count
    parts = {"digits": digits.item(), "count": count.item(), "attention": 0.0}
    if boxes is not None and has_boxes is not None and has_boxes.any() and attention_weight > 0:
        attention = out["attention"]  # (B, T, 3, C), or (B, 3, C) for single images
        if frame_mask is None:
            attention, boxes = attention[:, None], boxes[:, None]
            frame_mask = torch.ones(attention.shape[:2], dtype=torch.bool, device=attention.device)
        inside = box_cells(boxes)  # (B, T, 2, C)
        found = (attention[:, :, 1:] * inside).sum(dim=-1)  # (B, T, 2): left on box 1, right on box 2
        valid = (frame_mask & has_boxes[:, None])[..., None] & inside.any(dim=-1)  # (B, T, 2)
        if valid.any():
            attended = -torch.log(found[valid] + 1e-6).mean()
            total = total + attention_weight * attended
            parts["attention"] = attended.item()
    return total, parts


def train_one_epoch(model, loader, optimizer, device, anchor_weight=ANCHOR_WEIGHT, scheduler=None,
                    attention_weight=ATTENTION_WEIGHT, count_weight=COUNT_WEIGHT):
    """Train for one pass over loader and return the mean sequence loss per sequence.

    The returned loss is for monitoring only. It comes from sampled, augmented batches
    with dropout on, so it does not compare with evaluate. For train-vs-val curves, run
    evaluate on a fixed, unaugmented subset of train.

    scheduler, if given, steps after every batch, as OneCycleLR expects. attention_weight and
    count_weight are used by the two-step reader only (two_step_loss); its returned loss is
    still digit_loss on the composed output, so it compares with the fused reader's.

    Raises:
        RuntimeError: the loss is not finite, before the update spreads it to the weights.
    """
    model.train()
    total, count = 0.0, 0
    for batch in loader:
        frames = batch["frames"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        out = model(frames, mask)
        sequence_loss = digit_loss(out["logits"], labels)
        if model.reader == "fused":
            loss = sequence_loss
            if "anchor" in batch:
                anchor = batch["anchor"].to(device, non_blocking=True)
                loss = loss + anchor_weight * digit_loss(model.read_frames(anchor), labels)
        else:
            has_boxes = batch["has_boxes"].to(device, non_blocking=True) if "has_boxes" in batch else None
            boxes = batch["boxes"].to(device, non_blocking=True) if "boxes" in batch else None
            loss, _ = two_step_loss(out, labels, boxes, has_boxes, mask, attention_weight, count_weight)
            if "anchor" in batch:
                anchor = batch["anchor"].to(device, non_blocking=True)
                anchor_boxes = batch["anchor_boxes"].to(device, non_blocking=True) if "anchor_boxes" in batch else None
                anchor_loss, _ = two_step_loss(model.read_frames_full(anchor), labels, anchor_boxes, has_boxes, None,
                                               attention_weight, count_weight)
                loss = loss + anchor_weight * anchor_loss
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"loss is {loss.item()}, stopping")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Weighted by batch size, since the last batch can be smaller.
        total += sequence_loss.item() * len(labels)
        count += len(labels)
    return total / count


@torch.no_grad()
def evaluate(model, loader, device, use_anchor=False):
    """Run the model over loader without gradients and return a dict with:

        loss          mean loss per sequence
        predictions   (S, 2) predicted digits, S sequences in loader order
        confidences   (S,) probability of the predicted number, for reviewing errors
        labels        (S, 2) true digits
        sequence_ids  list of S ids, for reviewing errors

    use_anchor reads only the anchor, for the anchor-only comparison. The loader must
    then come from a dataset with load_anchor.

    Accuracy is left to the caller, so train.py can pick the best epoch by macro accuracy.
    """
    model.eval()
    total, predictions, confidences, labels, sequence_ids = 0.0, [], [], [], []
    for batch in loader:
        batch_labels = batch["label"].to(device, non_blocking=True)
        if use_anchor:
            logits = model.read_frames(batch["anchor"].to(device, non_blocking=True))
        else:
            frames = batch["frames"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            logits = model(frames, mask)["logits"]

        total += digit_loss(logits, batch_labels).item() * len(batch_labels)
        digits, confidence = predict_digits(logits)
        predictions.append(digits.cpu())
        confidences.append(confidence.cpu())
        labels.append(batch_labels.cpu())
        sequence_ids.extend(batch["sequence_id"])
    labels = torch.cat(labels)
    return {
        "loss": total / len(labels),
        "predictions": torch.cat(predictions),
        "confidences": torch.cat(confidences),
        "labels": labels,
        "sequence_ids": sequence_ids,
    }
