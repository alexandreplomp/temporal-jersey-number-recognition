"""Block 1: check the data and split it by player into train, val and test.
Example: uv run prepare.py

The raw data is never changed. Writes:
    results/data_issues.json  sequences without an anchor or frames, players under several
                              numbers, and identical images shared between players or numbers
    results/splits.json       {"train", "val", "test": sequence ids}, 70/15/15 per number, by player

Players linked to others (same player under two numbers, identical images) always go to train, so
no val or test image was seen in training. Every number gets a player in each split if it has at
least three; a rarer number goes to train only, with a warning, so it is learned but not scored.
Prints the counts per number and split, and checks that no player is in two splits.
"""

from collections import Counter, defaultdict
from hashlib import file_digest
from pathlib import Path
import argparse
import json
import random

from jersey.data import IMAGE_SUFFIXES, list_sequences

SPLITS = ("train", "val", "test")
VAL_SHARE = 0.15
TEST_SHARE = 0.15  # train gets the rest, about 70%
SEED = 1234


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    return parser.parse_args(argv)


# ----- data checks -----

def _hash_file(path):
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def _conflicts(groups, part=0):
    """Identical files whose paths differ in one part (0 = number, 1 = player)."""
    return sorted(sorted(paths) for paths in groups.values() if len({path.split("/")[part] for path in paths}) > 1)


def find_anchor_frame_links(images):
    """Identical images where an anchor matches a frame of another player or number."""
    links = []
    for paths in images.values():
        owners = {"anchor": set(), "frame": set()}
        for path in paths:
            owners["anchor" if Path(path).stem.endswith("_anchor") else "frame"].add(tuple(path.split("/")[:2]))
        if any(anchor != frame for anchor in owners["anchor"] for frame in owners["frame"]):
            links.append(sorted(paths))
    return sorted(links)


def find_issues(data_dir):
    """Sequences missing an anchor or frames, and players linked by label or by identical images."""
    sequences = list_sequences(data_dir)
    anchors, frames, images_by_hash = defaultdict(set), defaultdict(set), defaultdict(set)
    missing_anchor, no_frames = [], []
    for sequence in sequences:
        images = sorted(p for p in (Path(data_dir) / sequence).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        is_anchor = {image: image.stem.endswith("_anchor") for image in images}
        if not any(is_anchor.values()):
            missing_anchor.append(sequence)
        if all(is_anchor.values()):
            no_frames.append(sequence)
        for image in images:
            digest = _hash_file(image)
            images_by_hash[digest].add(f"{sequence}/{image.name}")
            if is_anchor[image]:
                anchors[digest].add(sequence)
            else:
                frames[digest].add(f"{sequence}/{image.name}")
    numbers = defaultdict(set)
    for sequence in sequences:
        numbers[sequence.split("/")[1]].add(sequence.split("/")[0])
    return {
        "missing_anchor": missing_anchor,
        "no_frames": no_frames,
        "players_in_multiple_numbers": {p: sorted(found) for p, found in sorted(numbers.items()) if len(found) > 1},
        "shared_anchors": _conflicts(anchors),
        "conflicting_frames": _conflicts(frames),
        "shared_between_players": _conflicts(anchors, 1) + _conflicts(frames, 1),
        "anchor_frame_links": find_anchor_frame_links(images_by_hash),
    }


# ----- splits -----

def flagged_players(issues):
    """Players that share labels or images with other players: in val or test they would leak."""
    flagged = set(issues["players_in_multiple_numbers"])
    for group in (issues["shared_anchors"] + issues["conflicting_frames"]
                  + issues["shared_between_players"] + issues["anchor_frame_links"]):
        flagged.update(path.split("/")[1] for path in group)
    return flagged


def assign_splits(players_of_number, flagged, seed=SEED):
    """{player: split}, splitting each number separately so every class keeps the same ratio.

    A number with fewer than 3 players goes to train only. Returns the mapping and those numbers.
    The random draws are the same as without that rule when every number has 3 players or more.
    """
    rng = random.Random(seed)
    split_of = dict.fromkeys(flagged, "train")
    train_only = []
    for number in sorted(players_of_number, key=int):
        rest = sorted(players_of_number[number] - flagged)  # sorted first: only the seed decides
        rng.shuffle(rest)
        if len(players_of_number[number]) < len(SPLITS):
            train_only.append(number)
            split_of.update(dict.fromkeys(rest, "train"))
            continue
        # Sized from all players, so flagged players do not shrink val and test.
        n_val = max(1, round(VAL_SHARE * len(players_of_number[number])))
        n_test = max(1, round(TEST_SHARE * len(players_of_number[number])))
        for player in rest[:n_val]:
            split_of[player] = "val"
        for player in rest[n_val:n_val + n_test]:
            split_of[player] = "test"
        for player in rest[n_val + n_test:]:
            split_of[player] = "train"
    return split_of, train_only


def make_splits(sequences, issues, seed=SEED):
    """{"train", "val", "test": sorted sequence ids} with each player in one split, and the numbers not in every split."""
    players_of_number = defaultdict(set)
    for sequence in sequences:
        number, player, _ = sequence.split("/")
        players_of_number[number].add(player)
    split_of, train_only = assign_splits(players_of_number, flagged_players(issues), seed)
    splits = {split: [] for split in SPLITS}
    for sequence in sequences:
        splits[split_of[sequence.split("/")[1]]].append(sequence)
    # Also numbers whose players are mostly linked to others, so val or test got none of them.
    unscored = sorted({number for number, players in players_of_number.items()
                       if {split_of[player] for player in players} != set(SPLITS)}, key=int)
    return {split: sorted(found) for split, found in splits.items()}, unscored


def players_in_two_splits(splits):
    """Players found in more than one split: must be empty."""
    seen = defaultdict(set)
    for split, sequences in splits.items():
        for sequence in sequences:
            seen[sequence.split("/")[1]].add(split)
    return sorted(player for player, found in seen.items() if len(found) > 1)


def print_counts(splits):
    players, counts = defaultdict(set), Counter()
    for split, sequences in splits.items():
        for sequence in sequences:
            number, player, _ = sequence.split("/")
            players[number, split].add(player)
            counts[number, split] += 1
    print("players/sequences per split")
    print(f"{'number':>6}" + "".join(f"{split:>12}" for split in splits))
    for number in sorted({number for number, _ in counts}, key=int):
        print(f"{number:>6}" + "".join(f"{len(players[number, s])}/{counts[number, s]}".rjust(12) for s in splits))


def main(argv=None):
    args = parse_args(argv)
    issues = find_issues(args.data)
    splits, unscored = make_splits(list_sequences(args.data), issues)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "data_issues.json").write_text(json.dumps(issues, indent=2))
    (args.out / "splits.json").write_text(json.dumps(splits, indent=2))
    print({name: len(found) for name, found in issues.items()})
    print_counts(splits)
    if unscored:
        print(f"warning: numbers missing from val or test (too few unlinked players), not fully scored: {unscored}")
    leaked = players_in_two_splits(splits)
    if leaked:
        raise SystemExit(f"check failed: players in two splits: {leaked[:5]}")
    print("check: no player is in two splits")


if __name__ == "__main__":
    main()
