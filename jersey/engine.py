"""Losses, one training epoch, evaluation, and the helpers every block shares. No files here.

Loss per reader (LOSSES), the same call for the sequence and for the anchor read alone:
    fused     cross-entropy over both digit positions (also parseq)
    two-step  each query on the digit it should read, the count (one digit or two, both weighed
              equally), and where digit boxes are known (spliced pairs) left and right attention
              on their box
The anchor, the clearest frame, adds the same loss weighted by ANCHOR_WEIGHT; the deployed model
never sees it. The reported and validation loss is digit_loss on the (2, 11) output, for every
reader, so losses compare.
"""

import math

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from jersey.data import BLANK, IMAGE_SIZE
from jersey.model import CELL_STRIDE, predict_digits

ANCHOR_WEIGHT = 0.5
ATTENTION_WEIGHT = 0.5
COUNT_WEIGHT = 1.0
WEIGHT_DECAY = 1e-4
WARMUP_SHARE = 0.075  # of the steps, as in PARSeq


def digit_loss(logits, labels):
    """Mean cross-entropy over both positions: log-probabilities (..., 2, 11), labels (..., 2).

    BLANK is never a first digit, in the model or the labels, so the loss stays finite.
    """
    return F.cross_entropy(logits.flatten(0, -2), labels.flatten())


def fused_loss(out, labels, boxes=None, has_boxes=None, frame_mask=None):
    return digit_loss(out["logits"], labels)


def box_cells(boxes, grid=(IMAGE_SIZE[0] // CELL_STRIDE, IMAGE_SIZE[1] // CELL_STRIDE), stride=CELL_STRIDE):
    """True for the cells a box overlaps: boxes (..., 4) -> (..., rows * columns).

    Overlap rather than the cell centre, since a thin digit (6 pixels) can fall between centres.
    """
    rows, columns = grid
    x0 = torch.arange(columns, device=boxes.device) * stride
    y0 = torch.arange(rows, device=boxes.device) * stride
    left, top, right, bottom = (boxes[..., i, None] for i in range(4))
    across = (x0 < right) & (x0 + stride > left)
    down = (y0 < bottom) & (y0 + stride > top)
    return (down[..., :, None] & across[..., None, :]).flatten(-2)


def two_step_loss(out, labels, boxes=None, has_boxes=None, frame_mask=None):
    """Loss of the two-step reader on a batch.

    out is JerseyModel.forward's dict (frame_mask the frame mask), or JerseyModel.read's for
    single images (frame_mask None). boxes are (B, T, 2, 4), or (B, 2, 4) for single images.
    """
    two = labels[:, 1] != BLANK
    digit_logits = out["digit_logits"]
    single = F.cross_entropy(digit_logits[:, 0], labels[:, 0], reduction="none")
    left = F.cross_entropy(digit_logits[:, 1], labels[:, 0], reduction="none")
    right = F.cross_entropy(digit_logits[:, 2], labels[:, 1].clamp(max=BLANK - 1), reduction="none")
    loss = torch.where(two, (left + right) / 2, single).mean()

    each = F.cross_entropy(out["count_logits"], two.long(), reduction="none")
    balanced = torch.stack([each[two == kind].mean() for kind in (False, True) if (two == kind).any()]).mean()
    loss = loss + COUNT_WEIGHT * balanced

    if boxes is not None and has_boxes is not None and has_boxes.any():
        attention = out["attention"]  # (B, T, 3, C), or (B, 3, C) for single images
        if frame_mask is None:
            attention, boxes = attention[:, None], boxes[:, None]
            frame_mask = torch.ones(attention.shape[:2], dtype=torch.bool, device=attention.device)
        inside = box_cells(boxes)  # (B, T, 2, C)
        found = (attention[:, :, 1:] * inside).sum(dim=-1)  # left on box 1, right on box 2
        valid = (frame_mask & has_boxes[:, None])[..., None] & inside.any(dim=-1)
        if valid.any():
            loss = loss + ATTENTION_WEIGHT * -torch.log(found[valid] + 1e-6).mean()
    return loss


LOSSES = {"fused": fused_loss, "two-step": two_step_loss, "parseq": fused_loss}


def _to(batch, key, device):
    return batch[key].to(device, non_blocking=True) if key in batch else None


def train_one_epoch(model, loader, optimizer, scheduler, device):
    """Train for one pass over loader (the scheduler steps per batch); return the mean digit loss per sequence.

    For monitoring only: sampled, augmented batches with dropout on.

    Raises:
        RuntimeError: the loss is not finite, before the update spreads it to the weights.
    """
    model.train()
    loss_of = LOSSES[model.reader_name]
    total, count = 0.0, 0
    for batch in loader:
        frames, mask, labels = _to(batch, "frames", device), _to(batch, "mask", device), _to(batch, "label", device)
        has_boxes = _to(batch, "has_boxes", device)
        out = model(frames, mask)
        loss = loss_of(out, labels, _to(batch, "boxes", device), has_boxes, mask)
        if "anchor" in batch:
            anchor_out = model.read(_to(batch, "anchor", device))
            loss = loss + ANCHOR_WEIGHT * loss_of(anchor_out, labels, _to(batch, "anchor_boxes", device), has_boxes)
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"loss is {loss.item()}, stopping")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total += digit_loss(out["logits"].detach(), labels).item() * len(labels)
        count += len(labels)
    return total / count


@torch.no_grad()
def evaluate(model, loader, device, use_anchor=False):
    """Run the model over loader and return a dict, in loader order:

        loss          mean digit loss per sequence
        predictions   (S, 2) predicted digits
        confidences   (S,) probability of the predicted number
        labels        (S, 2) true digits
        sequence_ids  list of S ids

    use_anchor reads only the anchor, for the anchor-only comparison (dataset with load_anchor).
    """
    model.eval()
    total, predictions, confidences, labels, sequence_ids = 0.0, [], [], [], []
    for batch in loader:
        batch_labels = _to(batch, "label", device)
        if use_anchor:
            logits = model.read(_to(batch, "anchor", device))["logits"]
        else:
            logits = model(_to(batch, "frames", device), _to(batch, "mask", device))["logits"]
        total += digit_loss(logits, batch_labels).item() * len(batch_labels)
        digits, confidence = predict_digits(logits)
        predictions.append(digits.cpu())
        confidences.append(confidence.cpu())
        labels.append(batch_labels.cpu())
        sequence_ids.extend(batch["sequence_id"])
    labels = torch.cat(labels)
    return {"loss": total / len(labels), "predictions": torch.cat(predictions),
            "confidences": torch.cat(confidences), "labels": labels, "sequence_ids": sequence_ids}


# ----- shared helpers -----

def pick_device(name="auto"):
    """The device to use: name, or for "auto" cuda, then mps, then cpu."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(dataset, batch_size, workers, device, sampler=None):
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=workers,
                      pin_memory=torch.device(device).type == "cuda", persistent_workers=workers > 0)


def make_optimizer(model, lr, backbone_lr, total_steps, backbone_prefix="reader.backbone."):
    """AdamW with the pretrained backbone at backbone_lr, and a OneCycle schedule (warmup, then decay)."""
    backbone = [p for name, p in model.named_parameters() if name.startswith(backbone_prefix)]
    rest = [p for name, p in model.named_parameters() if not name.startswith(backbone_prefix)]
    optimizer = torch.optim.AdamW([{"params": rest, "lr": lr}, {"params": backbone, "lr": backbone_lr}],
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[lr, backbone_lr], total_steps=total_steps,
                                                    pct_start=WARMUP_SHARE, cycle_momentum=False)
    return optimizer, scheduler
