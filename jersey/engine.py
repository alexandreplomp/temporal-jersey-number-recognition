"""One pass over a loader: train for an epoch, or evaluate. Saving files is left to train.py.

Loss: cross-entropy over both digit positions of the sequence logits, plus the same
loss on the anchor read on its own through read_frames, weighted by anchor_weight.
The anchor is the clearest frame, so it gives the CNN and the digit queries a clean
example. The deployed model never sees it, so evaluation leaves it out, except for
the anchor-only comparison (use_anchor).

frame_logits are not trained yet. To add a per-frame loss later, use
frame_logits[mask], so padded frames add nothing.
"""

import math

import torch
from torch.nn import functional as F

from jersey.model import predict_digits

ANCHOR_WEIGHT = 0.5  # a starting guess, to tune on val


def digit_loss(logits, labels):
    """Mean cross-entropy over both positions: logits (..., 2, 11), labels (..., 2).

    The model fixes BLANK in position 1 at -inf, and no label puts BLANK there, so the
    loss stays finite. Label smoothing would put weight on that class and make it inf.
    """
    return F.cross_entropy(logits.flatten(0, -2), labels.flatten())


def train_one_epoch(model, loader, optimizer, device, anchor_weight=ANCHOR_WEIGHT, scheduler=None):
    """Train for one pass over loader and return the mean sequence loss per sequence.

    The returned loss is for monitoring only. It comes from sampled, augmented batches
    with dropout on, so it does not compare with evaluate. For train-vs-val curves, run
    evaluate on a fixed, unaugmented subset of train.

    scheduler, if given, steps after every batch, as OneCycleLR expects.

    Raises:
        RuntimeError: the loss is not finite, before the update spreads it to the weights.
    """
    model.train()
    total, count = 0.0, 0
    for batch in loader:
        frames = batch["frames"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        sequence_loss = digit_loss(model(frames, mask)["logits"], labels)
        loss = sequence_loss
        if "anchor" in batch:
            anchor = batch["anchor"].to(device, non_blocking=True)
            loss = loss + anchor_weight * digit_loss(model.read_frames(anchor), labels)
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
