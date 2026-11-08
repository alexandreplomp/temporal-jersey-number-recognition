"""Scores for jersey number predictions: predictions and labels in, plain numbers out.

No model and no files here: engine.evaluate makes the predictions, and train.py and
evaluate.py report them. Predictions and labels are (S, 2) digit classes, as in data.py,
and a number counts as read only if both digits are right.

Macro accuracy averages the per-number accuracies, so rare numbers count as much as
common ones. Only numbers present in the labels count; scoring absent ones as 0 would
lower every score. But a number with 1-3 sequences (64 and 88 in val and test) moves
macro accuracy by 5-10 points per sequence, so:
    macro_accuracy_5  keeps only numbers with at least MIN_COUNT sequences
    per_number        gives each score a Wilson interval, which stays sensible for tiny
                      counts (Brown, Cai & DasGupta 2001). Sequences of one player are
                      correlated, so the true uncertainty is larger than these show.
The same noise is why train.py picks the best epoch by val loss, not macro accuracy.

Scores over an empty group are nan.
"""

from collections import Counter, defaultdict
import math

import torch

from jersey.data import BLANK, decode_label, encode_label

MIN_COUNT = 5
# Bands by training sequences, as in ImageNet-LT (Liu et al. 2019): many is over 100,
# medium 20-100, few under 20. Each pair is (band, minimum count).
SHOT_BANDS = (("many", 101), ("medium", 20), ("few", 0))
UNSEEN = "unseen"  # numbers held out of training with --exclude


def to_numbers(digits):
    """Digit classes (S, 2) -> numbers as strings: [[4, 9], [7, BLANK]] -> ["49", "7"]."""
    return [decode_label(*pair) for pair in digits.tolist()]


def wilson_interval(correct, count, z=1.96):
    """Return the 95% Wilson interval (low, high) for correct out of count.

    Unlike the usual p +- z*sqrt(p(1-p)/n), it stays inside [0, 1] and is not zero
    width at 0/n or n/n: 1 out of 1 gives about (0.21, 1.0), not (1.0, 1.0).
    """
    p = correct / count
    shrink = 1 + z * z / count
    centre = (p + z * z / (2 * count)) / shrink
    half = z / shrink * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count))
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(only_first, only_second):
    """Two-sided exact McNemar p-value: of the sequences where exactly one method is right, is the split 50/50?

    only_first and only_second count the sequences only the first or only the second method gets right.
    """
    discordant = only_first + only_second
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(only_first, only_second) + 1)) / 2 ** discordant
    return min(1.0, 2 * tail)


def group_bootstrap(correct, groups, resamples=2000, seed=0):
    """95% interval of accuracy when whole groups (e.g. tracks) are resampled, not single sequences.

    correct and groups are one entry per sequence. Sequences of one group are correlated, so this
    interval is the honest one when a group has several sequences; it is wider than Wilson's.
    """
    by_group = defaultdict(list)
    for right, group in zip(correct, groups):
        by_group[group].append(float(right))
    totals = torch.tensor([[sum(v), len(v)] for v in by_group.values()])
    generator = torch.Generator().manual_seed(seed)
    picks = torch.randint(len(totals), (resamples, len(totals)), generator=generator)
    sums = totals[picks].sum(dim=1)
    accuracies = (sums[:, 0] / sums[:, 1]).sort().values
    return accuracies[int(0.025 * resamples)].item(), accuracies[int(0.975 * resamples) - 1].item()


def track_of(sequence_id):
    """"49/<match>_<track>/<sequence>" -> "<match>_<track>": the player folder, one track of one player."""
    return sequence_id.split("/")[1]


def match_of(sequence_id):
    """"49/<match>_<track>/<sequence>" -> "<match>". The first part of a player folder is shared across
    numbers and players, so it most likely names the match or video (not documented in the data)."""
    return track_of(sequence_id).split("_")[0]


def per_number(predictions, labels):
    """Return {number: count, correct, accuracy, low, high} for each number in labels."""
    return _summaries(to_numbers(labels), to_numbers(labels), to_numbers(predictions))


def macro(table, min_count=1):
    """Mean accuracy over the numbers in a per_number table with at least min_count sequences."""
    kept = [row["accuracy"] for row in table.values() if row["count"] >= min_count]
    return sum(kept) / len(kept) if kept else math.nan


def scores(result):
    """Per-epoch summary of an engine.evaluate result."""
    table = per_number(result["predictions"], result["labels"])
    correct = (result["predictions"] == result["labels"]).all(dim=1).tolist()
    two_digit = (result["labels"][:, 1] != BLANK).tolist()
    return {
        "loss": result["loss"],
        "accuracy": _share(correct),
        "macro_accuracy": macro(table),
        "macro_accuracy_5": macro(table, MIN_COUNT),
        "one_digit_accuracy": _share([right for right, two in zip(correct, two_digit) if not two]),
        "two_digit_accuracy": _share([right for right, two in zip(correct, two_digit) if two]),
    }


def shot_bands(train_numbers):
    """Return {number: "many" | "medium" | "few"} from the numbers of the training sequences."""
    return {
        number: next(band for band, minimum in SHOT_BANDS if count >= minimum)
        for number, count in Counter(train_numbers).items()
    }


def band_accuracy(predictions, labels, bands):
    """Return {band: count, correct, accuracy, low, high}, pooled over the sequences of each band.

    Numbers missing from bands, because they were held out of training, go to "unseen".
    """
    true = to_numbers(labels)
    groups = [bands.get(number, UNSEEN) for number in true]
    table = _summaries(groups, true, to_numbers(predictions))
    order = [band for band, _ in SHOT_BANDS] + [UNSEEN]
    return {band: table[band] for band in order if band in table}


def confusions(predictions, labels, top=10):
    """Return the most common mistakes as [(true, predicted, count)]."""
    mistakes = Counter(
        (true, predicted)
        for true, predicted in zip(to_numbers(labels), to_numbers(predictions))
        if true != predicted
    )
    return [(true, predicted, count) for (true, predicted), count in mistakes.most_common(top)]


def majority_baseline(labels, train_numbers):
    """Scores of always answering the most common training number, to put real scores in context."""
    guess = Counter(train_numbers).most_common(1)[0][0]
    predictions = torch.tensor([encode_label(guess)] * len(labels))
    table = per_number(predictions, labels)
    correct = sum(row["correct"] for row in table.values())
    return {"number": guess, "accuracy": correct / len(labels), "macro_accuracy": macro(table)}


def count_flips(prefix_predictions, lengths):
    """Measure how often the answer changes as frames arrive.

    prefix_predictions (S, T, 2) holds the answer after the first 1, 2, ..., T frames,
    lengths (S,) the number of real frames. Only sequences with at least 2 frames count,
    since one frame cannot flip. Returns the number of such sequences, the mean number
    of changes per sequence, and the share of sequences whose answer never changes.
    """
    changed = (prefix_predictions[:, 1:] != prefix_predictions[:, :-1]).any(dim=-1)  # (S, T-1)
    # Step k compares k+1 frames with k+2 frames, which needs k+2 real frames.
    real = torch.arange(1, prefix_predictions.shape[1])[None] < lengths[:, None]
    flips = (changed & real).sum(dim=1)[lengths >= 2].tolist()
    return {
        "sequences": len(flips),
        "flips": sum(flips) / len(flips) if flips else math.nan,
        "stable_share": _share([count == 0 for count in flips]),
    }


def heldout_error_share(predictions, labels, train_numbers):
    """Among wrong answers, the share that is a training number, e.g. a held-out 94 read as 49.

    A high share means the model recalls numbers it trained on instead of reading digits.
    """
    known = set(train_numbers)
    wrong = [
        predicted
        for true, predicted in zip(to_numbers(labels), to_numbers(predictions))
        if true != predicted
    ]
    return _share([predicted in known for predicted in wrong])


def _summaries(groups, true, predicted):
    """Return {group: count, correct, accuracy, low, high}, groups in numeric or first-seen order."""
    hits = defaultdict(list)
    for group, t, p in zip(groups, true, predicted):
        hits[group].append(t == p)
    order = sorted(hits, key=int) if all(group.isdigit() for group in hits) else list(hits)
    return {group: _summary(sum(hits[group]), len(hits[group])) for group in order}


def _summary(correct, count):
    low, high = wilson_interval(correct, count)
    return {"count": count, "correct": correct, "accuracy": correct / count, "low": low, "high": high}


def _share(flags):
    return sum(flags) / len(flags) if flags else math.nan
