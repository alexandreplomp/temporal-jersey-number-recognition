"""Report data problems and count players and sequences per number.
Example: uv run check_data.py

Writes results/data_issues.json. The raw data is never changed.
"""

from collections import Counter, defaultdict
from hashlib import file_digest
from pathlib import Path
import json

from jersey.data import IMAGE_SUFFIXES, list_sequences

DATA = Path("data")
OUTPUT = Path("results/data_issues.json")


def hash_file(path):
    with path.open("rb") as file:
        return file_digest(file, "sha256").hexdigest()


def find_conflicts(groups, part=0):
    """Return identical files whose paths differ in one part (0 = number, 1 = player)."""
    return sorted(
        sorted(paths)
        for paths in groups.values()
        if len({path.split("/")[part] for path in paths}) > 1
    )


def find_anchor_frame_links(images):
    """Return identical images where an anchor matches a frame of another player or number.

    Anchors and frames are compared within their own kind above, so one player's anchor
    saved as another player's frame shows up only here.
    """
    links = []
    for paths in images.values():
        owners = {"anchor": set(), "frame": set()}
        for path in paths:
            kind = "anchor" if Path(path).stem.endswith("_anchor") else "frame"
            owners[kind].add(tuple(path.split("/")[:2]))  # (number, player)
        if any(anchor != frame for anchor in owners["anchor"] for frame in owners["frame"]):
            links.append(sorted(paths))
    return sorted(links)


def numbers_per_player(sequences):
    numbers = defaultdict(set)
    for sequence in sequences:
        number, player, _ = sequence.split("/")
        numbers[player].add(number)
    return numbers


def find_issues(data_dir):
    """Return sequences missing an anchor or frames, and players linked by label or by identical images.

    Identical images are compared anchor to anchor, frame to frame and anchor to frame.
    """
    sequences = list_sequences(data_dir)
    anchors = defaultdict(set)
    frames = defaultdict(set)
    images_by_hash = defaultdict(set)  # anchors and frames together
    missing_anchor = []
    no_frames = []
    for sequence in sequences:
        # Sorted so the output does not depend on the filesystem's listing order.
        images = sorted(path for path in (data_dir / sequence).iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        is_anchor = {image: image.stem.endswith("_anchor") for image in images}
        if not any(is_anchor.values()):
            missing_anchor.append(sequence)
        if all(is_anchor.values()):
            no_frames.append(sequence)
        for image in images:
            digest = hash_file(image)
            images_by_hash[digest].add(f"{sequence}/{image.name}")
            if is_anchor[image]:
                anchors[digest].add(sequence)
            else:
                frames[digest].add(f"{sequence}/{image.name}")

    numbers = numbers_per_player(sequences)
    return {
        "missing_anchor": missing_anchor,
        "no_frames": no_frames,
        "players_in_multiple_numbers": {
            player: sorted(found) for player, found in sorted(numbers.items()) if len(found) > 1
        },
        "shared_anchors": find_conflicts(anchors),
        "conflicting_frames": find_conflicts(frames),
        "shared_between_players": find_conflicts(anchors, 1) + find_conflicts(frames, 1),
        "anchor_frame_links": find_anchor_frame_links(images_by_hash),
    }


def count_per_number(sequences):
    """Return {number: {"players": ..., "sequences": ...}} in numeric order."""
    players = defaultdict(set)
    counts = Counter()
    for sequence in sequences:
        number, player, _ = sequence.split("/")
        players[number].add(player)
        counts[number] += 1
    return {
        number: {"players": len(players[number]), "sequences": counts[number]}
        for number in sorted(counts, key=int)
    }


def print_counts(sequences):
    print(f"{'number':>6}{'players':>9}{'sequences':>11}")
    for number, found in count_per_number(sequences).items():
        print(f"{number:>6}{found['players']:>9}{found['sequences']:>11}")
    # Counted from the ids, since a player can appear under several numbers.
    total_players = len({sequence.split("/")[1] for sequence in sequences})
    print(f"{'total':>6}{total_players:>9}{len(sequences):>11}")


def main():
    issues = find_issues(DATA)
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(issues, indent=2))
    print({name: len(found) for name, found in issues.items()})
    print_counts(list_sequences(DATA))


if __name__ == "__main__":
    main()
