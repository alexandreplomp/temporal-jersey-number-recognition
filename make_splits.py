"""Split the data by player into train, val and test (70/15/15) per number.
Example: uv run make_splits.py   (run check_data.py first)

Players flagged in results/data_issues.json always go to train to avoid leakage.
Every number gets at least one player in each split.
"""

from collections import Counter, defaultdict
from pathlib import Path
import json
import random

from jersey.data import list_sequences

DATA = Path("data")
ISSUES = Path("results/data_issues.json")
OUTPUT = Path("results/splits.json")
SPLITS = ("train", "val", "test")
VAL_SHARE = 0.15
TEST_SHARE = 0.15  # train gets the rest, about 70%
SEED = 1234


def flagged_players(issues):
    """Return players that share labels or images with other players.

    Putting them in val or test would leak training data into evaluation.
    """
    flagged = set(issues["players_in_multiple_numbers"])
    for group in (
        issues["shared_anchors"]
        + issues["conflicting_frames"]
        + issues["shared_between_players"]
        + issues["anchor_frame_links"]
    ):
        flagged.update(path.split("/")[1] for path in group)
    return flagged


def assign_splits(players_of_number, flagged, seed):
    """Return {player: split}, splitting each number separately so every class keeps the same ratio."""
    rng = random.Random(seed)
    split_of = dict.fromkeys(flagged, "train")
    for number in sorted(players_of_number, key=int):
        # Sort before shuffling: set order changes between runs, the seed alone should decide.
        rest = sorted(players_of_number[number] - flagged)
        rng.shuffle(rest)
        # Size from all players, so flagged players don't shrink val and test.
        # max(1, ...) keeps rare numbers like 88 in every split.
        n_val = max(1, round(VAL_SHARE * len(players_of_number[number])))
        n_test = max(1, round(TEST_SHARE * len(players_of_number[number])))
        for player in rest[:n_val]:
            split_of[player] = "val"
        for player in rest[n_val:n_val + n_test]:
            split_of[player] = "test"
        for player in rest[n_val + n_test:]:
            split_of[player] = "train"

    for number, players in players_of_number.items():
        missing = set(SPLITS) - {split_of[player] for player in players}
        if missing:
            raise ValueError(f"number {number} has no players in {sorted(missing)}")
    return split_of


def make_splits(sequences, issues, seed=SEED):
    """Return {"train", "val", "test": sorted sequence ids}, with each player in one split."""
    # A player can appear under several numbers, so players are grouped per number here
    # but assigned to a split once.
    players_of_number = defaultdict(set)
    for sequence in sequences:
        number, player, _ = sequence.split("/")
        players_of_number[number].add(player)
    split_of = assign_splits(players_of_number, flagged_players(issues), seed)

    splits = {split: [] for split in SPLITS}
    for sequence in sequences:
        splits[split_of[sequence.split("/")[1]]].append(sequence)
    return {split: sorted(found) for split, found in splits.items()}


def print_summary(splits):
    """Print players/sequences per number and split."""
    players = defaultdict(set)
    counts = Counter()
    for split, sequences in splits.items():
        for sequence in sequences:
            number, player, _ = sequence.split("/")
            players[number, split].add(player)
            counts[number, split] += 1

    print("players/sequences per split")
    print(f"{'number':>6}" + "".join(f"{split:>12}" for split in splits))
    for number in sorted({number for number, _ in counts}, key=int):
        cells = [f"{len(players[number, split])}/{counts[number, split]}" for split in splits]
        print(f"{number:>6}" + "".join(f"{cell:>12}" for cell in cells))
    totals = [
        f"{len({sequence.split('/')[1] for sequence in found})}/{len(found)}"
        for found in splits.values()
    ]
    print(f"{'total':>6}" + "".join(f"{cell:>12}" for cell in totals))


def main():
    issues = json.loads(ISSUES.read_text())
    splits = make_splits(list_sequences(DATA), issues)
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(splits, indent=2))
    print_summary(splits)


if __name__ == "__main__":
    main()
