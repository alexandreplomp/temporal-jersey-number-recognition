"""Core tests on small fake data. Run from the project root: uv run pytest"""

from collections import Counter, defaultdict
from pathlib import Path
import csv
import json
import math

from PIL import Image
import pytest
import torch

from check_data import find_anchor_frame_links, find_issues
from check_shortcuts import PERTURBATIONS, colour_baseline, cover, main as shortcuts_main, position_accuracy
from jersey.data import (
    AUGMENTS, BLANK, denormalize, IMAGE_SIZE, MAX_FRAMES, JerseyDataset, decode_label, encode_label, find_images,
    filter_numbers, held_out_partners, linked_groups, make_sampler,
)
from jersey.engine import box_cells, digit_loss, evaluate, train_one_epoch, two_step_loss
from jersey.explain import (
    attention_check, frame_removal, frame_weights, map_similarity, occlusion_map, query_maps,
    temporal_attention, temporal_pass,
)
from jersey.pair_check import build_pair_set, pair_metrics
from jersey.local_reader import TOP_CELLS, LocalDigitReader, crop_scores, load_locator, torso_cells
from jersey.metrics import (
    band_accuracy, confusions, count_flips, heldout_error_share, macro, majority_baseline,
    per_number, scores, shot_bands, wilson_interval,
)
from jersey.model import (
    BACKBONE_STRIDE, CELL_STRIDE, LOCAL_BLOCKS, READERS, VARIANTS, JerseyModel, compose, count_prior, load_run,
    predict_digits,
    side_bias,
)
from jersey.pairs import FEATHER, MIN_KEPT, StripBank, paste_single, splice
from make_pairs import MIN_PROB, ink_box, keep, main as pairs_main
from make_splits import SPLITS, flagged_players, make_splits
from evaluate import main as evaluate_main, prefix_answers, score, vote
from train import main as train_main
from train_locator import main as locator_main

RESULTS = Path(__file__).parent.parent / "results"
NO_ISSUES = {
    "players_in_multiple_numbers": {},
    "shared_anchors": [],
    "conflicting_frames": [],
    "shared_between_players": [],
    "anchor_frame_links": [],
}


def fake_sequences(numbers=("4", "8"), players=20, sequences=2):
    """Return ids like "4/p4_0/1": every player has several sequences."""
    return [
        f"{number}/p{number}_{player}/{sequence}"
        for number in numbers
        for player in range(players)
        for sequence in range(sequences)
    ]


def fake_sequence_dir(root, sequence_id, frame_ids, noise=False):
    """Write a tiny sequence: one frame per id plus an anchor, at a size unlike IMAGE_SIZE.

    noise gives every frame the same random texture, so augmentation visibly changes it.
    """
    folder = root / sequence_id
    folder.mkdir(parents=True)
    image = Image.effect_noise((22, 35), 64).convert("RGB") if noise else Image.new("RGB", (22, 35), "white")
    for frame_id in frame_ids:
        image.save(folder / f"1_{frame_id}_0.png")
    image.save(folder / "1_0_0_anchor.jpg")
    return folder


def splits_per_player(splits):
    found = defaultdict(set)
    for split in SPLITS:
        for sequence in splits[split]:
            found[sequence.split("/")[1]].add(split)
    return found


def test_every_sequence_in_exactly_one_split():
    sequences = fake_sequences()

    splits = make_splits(sequences, NO_ISSUES)

    assert sorted(sum(splits.values(), [])) == sorted(sequences)


def test_no_player_in_two_splits():
    sequences = fake_sequences()

    splits = make_splits(sequences, NO_ISSUES)

    assert [player for player, found in splits_per_player(splits).items() if len(found) > 1] == []


def test_linked_players_go_to_train():
    sequences = fake_sequences() + ["8/p4_0/9"]
    issues = {
        **NO_ISSUES,
        "players_in_multiple_numbers": {"p4_0": ["4", "8"]},
        "shared_between_players": [["4/p4_1/0/a.jpg", "8/p8_1/0/a.jpg"]],
    }

    found = splits_per_player(make_splits(sequences, issues))

    assert found["p4_0"] == found["p4_1"] == found["p8_1"] == {"train"}


def test_rare_number_gets_one_player_per_split():
    sequences = fake_sequences() + fake_sequences(numbers=("88",), players=3, sequences=1)

    splits = make_splits(sequences, NO_ISSUES)

    assert [sum(s.startswith("88/") for s in splits[split]) for split in SPLITS] == [1, 1, 1]


def test_number_too_small_to_split_raises():
    sequences = fake_sequences() + fake_sequences(numbers=("88",), players=2, sequences=1)

    with pytest.raises(ValueError, match="number 88"):
        make_splits(sequences, NO_ISSUES)


def test_identical_anchors_link_players(tmp_path):
    for sequence in ("4/a/0", "9/b/0"):
        folder = tmp_path / sequence
        folder.mkdir(parents=True)
        (folder / "1_anchor.jpg").write_bytes(b"same image")
        (folder / "2.jpg").write_bytes(sequence.encode())

    issues = find_issues(tmp_path)

    assert issues["shared_anchors"] == [["4/a/0", "9/b/0"]]


def test_anchor_matching_another_players_frame_links_players(tmp_path):
    for sequence, anchor, frame in (("4/a/0", b"same image", b"a frame"), ("4/b/0", b"b anchor", b"same image")):
        folder = tmp_path / sequence
        folder.mkdir(parents=True)
        (folder / "1_anchor.jpg").write_bytes(anchor)
        (folder / "1_2_0.jpg").write_bytes(frame)

    issues = find_issues(tmp_path)

    assert issues["shared_anchors"] == issues["conflicting_frames"] == issues["shared_between_players"] == []
    assert issues["anchor_frame_links"] == [["4/a/0/1_anchor.jpg", "4/b/0/1_2_0.jpg"]]
    assert flagged_players(issues) == {"a", "b"}
    # An anchor that is also a frame of its own player is normal and not reported.
    assert find_anchor_frame_links({"same": {"4/a/0/1_anchor.jpg", "4/a/0/1_3_0.jpg", "4/a/1/1_5_1.jpg"}}) == []


def test_saved_splits_have_no_player_in_two_splits():
    splits = json.loads((RESULTS / "splits.json").read_text())

    found = splits_per_player(splits)

    assert [player for player, splits in found.items() if len(splits) > 1] == []


def test_saved_splits_keep_flagged_players_in_train():
    splits = json.loads((RESULTS / "splits.json").read_text())
    issues = json.loads((RESULTS / "data_issues.json").read_text())

    found = splits_per_player(splits)

    assert [player for player in flagged_players(issues) if found[player] - {"train"}] == []


@pytest.mark.parametrize("number, expected", [
    ("4", (4, BLANK)),
    ("49", (4, 9)),
    ("94", (9, 4)),
    ("0", (0, BLANK)),
    ("00", (0, 0)),
])
def test_encode_label(number, expected):
    assert encode_label(number) == expected


@pytest.mark.parametrize("number", ["4", "49", "94", "0", "00"])
def test_decode_label_inverts_encode(number):
    assert decode_label(*encode_label(number)) == number


@pytest.mark.parametrize("number", ["", "100", "4a"])
def test_encode_label_rejects_invalid(number):
    with pytest.raises(ValueError):
        encode_label(number)


def test_frames_sorted_by_frame_number(tmp_path):
    folder = fake_sequence_dir(tmp_path, "4/p/0", [1000, 999, 5])

    _, frames = find_images(folder)

    assert [frame.name for frame in frames] == ["1_5_0.png", "1_999_0.png", "1_1000_0.png"]


def test_eval_item_shapes_and_no_anchor(tmp_path):
    fake_sequence_dir(tmp_path, "49/p/0", [1, 2, 3])

    item = JerseyDataset(tmp_path, ["49/p/0"])[0]

    assert item["frames"].shape == (MAX_FRAMES, 3, *IMAGE_SIZE)
    assert item["mask"].tolist() == [True] * 3 + [False] * (MAX_FRAMES - 3)
    assert item["label"].tolist() == [4, 9]
    assert item["sequence_id"] == "49/p/0"
    assert "anchor" not in item


def test_train_item_has_anchor(tmp_path):
    fake_sequence_dir(tmp_path, "4/p/0", [1, 2, 3])

    item = JerseyDataset(tmp_path, ["4/p/0"], augment=True, load_anchor=True)[0]

    assert item["anchor"].shape == (3, *IMAGE_SIZE)


def test_anchor_without_augment_is_repeatable(tmp_path):
    # Needed to overfit one batch and for the anchor-only baseline.
    fake_sequence_dir(tmp_path, "4/p/0", range(MAX_FRAMES + 5))
    dataset = JerseyDataset(str(tmp_path), ["4/p/0"], load_anchor=True)

    first, second = dataset[0], dataset[0]

    assert torch.equal(first["frames"], second["frames"])
    assert torch.equal(first["anchor"], second["anchor"])


def test_sampler_weights_rare_numbers_up():
    sequence_ids = [f"4/p{i}/0" for i in range(4)] + ["49/q/0"]

    sampler = make_sampler(sequence_ids, power=0.5)

    assert sampler.weights.tolist() == [0.5] * 4 + [1.0]


def test_long_sequence_capped_at_max_frames(tmp_path):
    fake_sequence_dir(tmp_path, "4/p/0", range(MAX_FRAMES + 10))

    item = JerseyDataset(tmp_path, ["4/p/0"])[0]

    assert item["mask"].all()


@pytest.mark.parametrize("second", [4, BLANK])
def test_decode_label_rejects_blank_first_digit(second):
    with pytest.raises(ValueError, match="BLANK"):
        decode_label(BLANK, second)


@pytest.mark.parametrize("policy", AUGMENTS)
def test_augment_is_same_for_every_frame(tmp_path, policy):
    # Identical input frames must stay identical, or augmentation adds flicker real tracks lack.
    fake_sequence_dir(tmp_path, "4/p/0", [1, 2, 3], noise=True)
    dataset = JerseyDataset(tmp_path, ["4/p/0"], augment=True, load_anchor=True, policy=policy)

    item = dataset[0]

    frames = item["frames"][item["mask"]]
    assert all(torch.equal(frame, frames[0]) for frame in frames)


def test_missing_anchor_fails_at_construction(tmp_path):
    folder = fake_sequence_dir(tmp_path, "4/p/0", [1, 2])
    (folder / "1_0_0_anchor.jpg").unlink()

    with pytest.raises(ValueError, match="expected one anchor"):
        JerseyDataset(tmp_path, ["4/p/0"])


def test_bad_frame_name_error_names_file(tmp_path):
    folder = fake_sequence_dir(tmp_path, "4/p/0", [1])
    Image.new("RGB", (22, 35)).save(folder / "frame.png")

    with pytest.raises(ValueError, match="frame.png"):
        JerseyDataset(tmp_path, ["4/p/0"])


def test_custom_max_frames_and_image_size(tmp_path):
    fake_sequence_dir(tmp_path, "4/p/0", range(6))

    item = JerseyDataset(tmp_path, ["4/p/0"], max_frames=4, image_size=(16, 8))[0]

    assert item["frames"].shape == (4, 3, 16, 8)
    assert item["mask"].all()


def fake_batch(real_frames=(3, 1), max_frames=4):
    """Random frames for sequences with the given numbers of real frames, zero-padded."""
    frames = torch.zeros(len(real_frames), max_frames, 3, *IMAGE_SIZE)
    mask = torch.zeros(len(real_frames), max_frames, dtype=torch.bool)
    for i, count in enumerate(real_frames):
        frames[i, :count] = torch.randn(count, 3, *IMAGE_SIZE)
        mask[i, :count] = True
    return frames, mask


@pytest.mark.parametrize("reader", READERS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_model_output_shapes(variant, reader):
    model = JerseyModel(variant, max_frames=4, pretrained=False, reader=reader)
    frames, mask = fake_batch()

    output = model(frames, mask)

    assert output["logits"].shape == (2, 2, 11)
    assert output["frame_logits"].shape == (2, 4, 2, 11)


@pytest.mark.parametrize("reader", READERS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_padded_frames_do_not_change_output(variant, reader):
    # Same 3 real frames, once padded to 4 with noise and once without padding:
    # a missing mask in the mean, pooling or transformer would make these differ.
    model = JerseyModel(variant, max_frames=4, pretrained=False, reader=reader).eval()
    frames, mask = fake_batch(real_frames=(3,))
    frames[~mask] = torch.randn_like(frames[~mask])

    with torch.no_grad():
        padded = model(frames, mask)["logits"]
        unpadded = model(frames[:, :3], mask[:, :3])["logits"]

    assert torch.allclose(padded, unpadded, atol=1e-5)


@pytest.mark.parametrize("reader", READERS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_single_real_frame_gives_finite_logits(variant, reader):
    model = JerseyModel(variant, max_frames=4, pretrained=False, reader=reader)
    frames, mask = fake_batch(real_frames=(1,))

    logits = model(frames, mask)["logits"]

    assert logits[:, :, :BLANK].isfinite().all()


@pytest.mark.parametrize("reader", READERS)
def test_first_digit_is_never_blank(reader):
    model = JerseyModel("averaged", max_frames=4, pretrained=False, reader=reader)
    frames, mask = fake_batch()

    logits = model(frames, mask)["logits"]

    assert (logits[:, 0, BLANK] == float("-inf")).all()


def test_side_bias_starts_first_query_left_and_second_right():
    rows, columns = 3, 12
    bias = side_bias(rows, columns).reshape(2, rows, columns)
    middle = (columns - 1) / 2

    assert (bias <= 0).all()
    assert torch.equal(bias[:, 0], bias[:, -1])  # the same for every row
    first, second = bias[:, 0].argmax(dim=-1).tolist()
    assert first < middle < second
    assert torch.allclose(bias[0, 0], bias[1, 0].flip(0))  # mirror images


def test_read_frames_shape():
    model = JerseyModel("averaged", pretrained=False)

    logits = model.read_frames(torch.randn(5, 3, *IMAGE_SIZE))

    assert logits.shape == (5, 2, 11)


def test_unknown_variant_raises():
    with pytest.raises(ValueError, match="unknown variant"):
        JerseyModel("baseline", pretrained=False)


def test_sequence_without_frames_fails_at_construction(tmp_path):
    fake_sequence_dir(tmp_path, "4/p/0", [])

    with pytest.raises(ValueError, match="no frames"):
        JerseyDataset(tmp_path, ["4/p/0"])


def test_backbone_batchnorm_stays_frozen_in_train_mode():
    model = JerseyModel("full", pretrained=False)
    model.freeze_batchnorm = True

    model.train()

    batchnorms = [m for m in model.backbone.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert batchnorms and not any(layer.training for layer in batchnorms)
    assert model.training and model.temporal.training


def test_attention_maps_are_kept():
    model = JerseyModel("averaged", pretrained=False)
    frames, mask = fake_batch()

    model(frames, mask)

    cells = (IMAGE_SIZE[0] // CELL_STRIDE) * (IMAGE_SIZE[1] // CELL_STRIDE)
    assert model.digits.last_attention.shape == (int(mask.sum()), 2, cells)


def test_saved_weights_load_back(tmp_path):
    model = JerseyModel("full", max_frames=4, pretrained=False).eval()
    torch.save(model.state_dict(), tmp_path / "weights.pt")
    loaded = JerseyModel("full", max_frames=4, pretrained=False).eval()
    frames, mask = fake_batch()

    loaded.load_state_dict(torch.load(tmp_path / "weights.pt"))

    with torch.no_grad():
        assert torch.equal(model(frames, mask)["logits"], loaded(frames, mask)["logits"])


@pytest.mark.parametrize("exclude, expected", [
    (("89",), ["8/a/0", "9/b/0"]),
    ((), ["8/a/0", "89/c/0", "9/b/0"]),
])
def test_filter_numbers_drops_held_out(exclude, expected):
    assert filter_numbers(["8/a/0", "89/c/0", "9/b/0"], exclude) == expected


@pytest.mark.parametrize("reader", READERS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_frame_weights_cover_only_real_frames(variant, reader):
    model = JerseyModel(variant, max_frames=4, pretrained=False, reader=reader)
    frames, mask = fake_batch()

    weights = model(frames, mask)["frame_weights"]

    if variant == "averaged":
        assert weights is None
    else:
        assert weights.shape == (2, 4, 2)
        assert torch.allclose(weights.sum(dim=1), torch.ones(2, 2))
        assert (weights[~mask] == 0).all()


def test_predict_digits_picks_most_likely_digit_per_position():
    logits = torch.zeros(1, 2, 11)
    logits[0, 0, 1] = 5.0
    logits[0, 1, BLANK] = 5.0

    digits, confidence = predict_digits(logits)

    assert digits.tolist() == [[1, BLANK]]
    assert 0 < confidence.item() <= 1


def test_predict_digits_never_gives_blank_first():
    model = JerseyModel("averaged", max_frames=4, pretrained=False)
    frames, mask = fake_batch()

    digits, _ = predict_digits(model(frames, mask)["logits"])

    assert (digits[:, 0] != BLANK).all()


def test_image_size_not_divisible_by_stride_raises():
    with pytest.raises(ValueError, match=f"divisible by {BACKBONE_STRIDE}"):
        JerseyModel("averaged", image_size=(200, 100), pretrained=False)


def fake_loader(with_anchor=True):
    """One batch of two sequences, numbered 49 and 7, in the format JerseyDataset gives."""
    frames, mask = fake_batch()
    batch = {
        "frames": frames,
        "mask": mask,
        "label": torch.tensor([[4, 9], [7, BLANK]]),
        "sequence_id": ["49/p/0", "7/q/0"],
    }
    if with_anchor:
        batch["anchor"] = torch.randn(2, 3, *IMAGE_SIZE)
    return [batch]


def test_digit_loss_is_finite_with_blank_second_digit():
    model = JerseyModel("averaged", max_frames=4, pretrained=False)
    frames, mask = fake_batch()

    loss = digit_loss(model(frames, mask)["logits"], torch.tensor([[4, 9], [7, BLANK]]))

    assert loss.isfinite()


@pytest.mark.parametrize("with_anchor", [True, False])
def test_train_one_epoch_updates_weights(with_anchor):
    model = JerseyModel("full", max_frames=4, pretrained=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = model.classifier.weight.clone()

    loss = train_one_epoch(model, fake_loader(with_anchor), optimizer, "cpu")

    assert loss > 0
    assert not torch.equal(before, model.classifier.weight)


def test_evaluate_returns_predictions_in_loader_order():
    model = JerseyModel("pooled", max_frames=4, pretrained=False)

    result = evaluate(model, fake_loader(), "cpu")

    assert result["loss"] > 0
    assert result["predictions"].shape == result["labels"].shape == (2, 2)
    assert result["labels"].tolist() == [[4, 9], [7, BLANK]]
    assert result["sequence_ids"] == ["49/p/0", "7/q/0"]
    assert result["confidences"].shape == (2,)
    assert not model.training


def test_evaluate_with_anchor_reads_only_the_anchor():
    model = JerseyModel("full", max_frames=4, pretrained=False).eval()
    loader = fake_loader()

    result = evaluate(model, loader, "cpu", use_anchor=True)

    expected, _ = predict_digits(model.read_frames(loader[0]["anchor"]))
    assert torch.equal(result["predictions"], expected)


def test_non_finite_loss_stops_training():
    model = JerseyModel("averaged", max_frames=4, pretrained=False)
    with torch.no_grad():
        model.classifier.bias.fill_(float("nan"))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(RuntimeError, match="loss is nan"):
        train_one_epoch(model, fake_loader(), optimizer, "cpu")


def test_training_overfits_one_batch():
    # The basic check that the loss, the backward pass and the update fit together.
    # Checked on the training loss: without pretrained weights the backbone BatchNorm is
    # not frozen, and its running statistics from two random sequences make eval mode
    # differ from train mode, which is not what this tests.
    torch.manual_seed(0)
    model = JerseyModel("full", max_frames=4, pretrained=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = fake_loader()

    losses = [train_one_epoch(model, loader, optimizer, "cpu") for _ in range(30)]

    assert losses[-1] < 0.1 * losses[0]


def test_train_one_epoch_steps_scheduler_per_batch():
    model = JerseyModel("averaged", max_frames=4, pretrained=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    loader = fake_loader() * 3

    train_one_epoch(model, loader, optimizer, "cpu", scheduler=scheduler)

    assert scheduler.last_epoch == 3


def test_macro_accuracy_weighs_numbers_equally():
    # 3 of 3 right on "4", 0 of 1 right on "49".
    result = {
        "loss": 1.0,
        "predictions": torch.tensor([[4, BLANK]] * 3 + [[4, 8]]),
        "labels": torch.tensor([[4, BLANK]] * 3 + [[4, 9]]),
    }

    found = scores(result)

    assert found["accuracy"] == 0.75
    assert found["macro_accuracy"] == 0.5


@pytest.fixture(scope="module")
def tiny_runs(tmp_path_factory):
    """Two runs trained on fake data, shared by the script tests: "smoke", and "holdout" without 49.

    Train sequence 4/f/0 shares an image with 49/b/0, so the hold-out run drops it too.
    """
    root = tmp_path_factory.mktemp("tiny")
    data = root / "data"
    sequences = {"train": ["4/a/0", "4/f/0", "49/b/0"], "val": ["4/c/0", "49/d/0"], "test": ["4/e/0", "49/g/0"]}
    for sequence_id in sum(sequences.values(), []):
        fake_sequence_dir(data, sequence_id, [1, 2, 3], noise=True)
    # A key besides the splits, as a real splits file may carry, must be ignored.
    (root / "splits.json").write_text(json.dumps({**sequences, "seed": 1234}))
    issues = {**NO_ISSUES, "shared_between_players": [["4/f/0/1_1_0.png", "49/b/0/1_1_0.png"]]}
    (root / "data_issues.json").write_text(json.dumps(issues))
    for name, extra in (("smoke", []), ("holdout", ["--exclude", "49"])):
        train_main([
            "--variant", "full", "--epochs", "2", "--batch-size", "2", "--max-frames", "4",
            "--train-check-size", "2", "--no-pretrained", "--workers", "0", "--device", "cpu",
            "--schedule", "onecycle", "--data", str(data), "--splits", str(root / "splits.json"),
            "--issues", str(root / "data_issues.json"), "--out", str(root / "runs"), "--run-name", name, *extra,
        ])
    return root / "runs"


def test_train_script_writes_run(tiny_runs):
    run = tiny_runs / "smoke"

    assert {path.name for path in run.iterdir()} >= {
        "config.json", "splits.json", "train_batch.png", "history.json", "curves.png", "best.pt",
        "per_number.json", "prediction.png", "explain.json",
    }
    history = json.loads((run / "history.json").read_text())
    assert [entry["epoch"] for entry in history["epochs"]] == [0, 1, 2]
    lowest = min(history["epochs"][1:], key=lambda entry: entry["val_loss"])
    assert history["best"]["epoch"] == lowest["epoch"]
    checkpoint = torch.load(run / "best.pt")
    assert checkpoint["epoch"] == lowest["epoch"] and checkpoint["config"]["variant"] == "full"
    assert checkpoint["config"]["selected_by"] == "val_loss"
    assert "val_two_digit_accuracy" in history["best"]
    report = json.loads((run / "per_number.json").read_text())
    assert set(report["per_number"]) == {"4", "49"}
    explained = json.loads((run / "explain.json").read_text())
    assert [case["true"] for case in explained["cases"]] == ["4", "49"]
    model = JerseyModel("full", max_frames=4, pretrained=False)
    model.load_state_dict(checkpoint["model"])


def test_evaluate_writes_predictions_metrics_and_one_results_row(tiny_runs):
    arguments = ["smoke", "--runs", str(tiny_runs), "--device", "cpu", "--workers", "0"]

    evaluate_main(arguments)
    evaluate_main(arguments)  # a re-run replaces the run's results row

    run = tiny_runs / "smoke"
    metrics = json.loads((run / "metrics_val.json").read_text())
    assert set(metrics) == {"run", "sequences", "baseline", "methods", "stability", "hold_out"}
    assert set(metrics["methods"]) == {"sequence", "anchor", "vote"}
    assert metrics["sequences"] == 2 and metrics["hold_out"] is None
    with (run / "predictions_val.csv").open() as file:
        assert [row["sequence_id"] for row in csv.DictReader(file)] == ["4/c/0", "49/d/0"]
    with (tiny_runs.parent / "results_val.csv").open() as file:
        assert [row["name"] for row in csv.DictReader(file)] == ["smoke"]


def test_test_split_is_scored_once(tiny_runs):
    arguments = ["smoke", "--split", "test", "--runs", str(tiny_runs), "--device", "cpu", "--workers", "0"]

    evaluate_main(arguments)

    with pytest.raises(FileExistsError, match="only once"):
        evaluate_main(arguments)


def test_evaluate_scores_held_out_numbers_separately(tiny_runs):
    evaluate_main(["holdout", "--runs", str(tiny_runs), "--device", "cpu", "--workers", "0"])

    metrics = json.loads((tiny_runs / "holdout" / "metrics_val.json").read_text())
    assert set(metrics["methods"]["sequence"]["per_number"]) == {"4"}
    # The train and val 49s, since the run never trained on 49, but not the test one.
    with (tiny_runs / "holdout" / "predictions_holdout_val.csv").open() as file:
        assert [row["sequence_id"] for row in csv.DictReader(file)] == ["49/b/0", "49/d/0"]
    assert set(metrics["hold_out"]["per_number"]) == {"49"}
    assert metrics["hold_out"]["splits"] == ["train", "val"]


def test_test_hold_out_scores_only_test_sequences(tiny_runs):
    evaluate_main(["holdout", "--split", "test", "--runs", str(tiny_runs), "--device", "cpu", "--workers", "0"])

    metrics = json.loads((tiny_runs / "holdout" / "metrics_test.json").read_text())
    # The train and val 49s were seen while developing, so the test score leaves them out.
    with (tiny_runs / "holdout" / "predictions_holdout_test.csv").open() as file:
        assert [row["sequence_id"] for row in csv.DictReader(file)] == ["49/g/0"]
    assert metrics["hold_out"]["splits"] == ["test"] and metrics["hold_out"]["sequences"] == 1


def test_holdout_run_drops_train_sequences_linked_to_held_out_ones(tiny_runs):
    config = json.loads((tiny_runs / "holdout" / "config.json").read_text())

    assert config["linked_dropped"] == ["4/f/0"]
    assert config["sequences"]["train"] == 1
    assert json.loads((tiny_runs / "smoke" / "config.json").read_text())["sequences"]["train"] == 3


def test_linked_groups_by_image_and_by_player():
    issues = {
        **NO_ISSUES,
        "players_in_multiple_numbers": {"p": ["49", "9"]},
        "shared_anchors": [["4/a/0", "8/b/0"]],
        "conflicting_frames": [["8/b/0/x.jpg", "9/c/1/x.jpg"]],
    }
    sequence_ids = ["4/a/0", "8/b/0", "9/c/1", "49/p/0", "9/p/0", "9/p/1", "4/z/0"]

    groups = sorted(sorted(group) for group in linked_groups(issues, sequence_ids))

    assert groups == [["4/a/0", "8/b/0", "9/c/1"], ["4/z/0"], ["49/p/0", "9/p/0", "9/p/1"]]
    assert held_out_partners(issues, sequence_ids, ("49",)) == {"9/p/0", "9/p/1"}
    # A link between two held-out sequences drops nothing.
    assert held_out_partners(issues, sequence_ids, ("4", "8")) == {"9/c/1"}


def test_held_out_partners_follow_the_whole_chain():
    # 49/a/0 shares an image with 9/b/0; player b's other sequence shares one with 4/c/0,
    # whose player has another sequence. Direct partners alone would keep 9/b/1 and player c.
    issues = {
        **NO_ISSUES,
        "shared_between_players": [["49/a/0/x.jpg", "9/b/0/x.jpg"]],
        "anchor_frame_links": [["9/b/1/1_anchor.jpg", "4/c/0/y.jpg"]],
    }
    sequence_ids = ["49/a/0", "9/b/0", "9/b/1", "4/c/0", "4/c/1", "4/d/0"]

    assert held_out_partners(issues, sequence_ids, ("49",)) == {"9/b/0", "9/b/1", "4/c/0", "4/c/1"}


def test_vote_weighs_frames_by_confidence():
    assert vote(["8", "9", "9"], [0.9, 0.3, 0.3]) == ("8", pytest.approx(0.6))
    assert vote(["8", "9", "9"], [0.5, 0.3, 0.3])[0] == "9"


def test_prefix_answers_end_with_the_full_answer():
    model = JerseyModel("full", max_frames=4, pretrained=False).eval()
    frames, mask = fake_batch(real_frames=(3, 1))

    prefixes = prefix_answers(model, frames, mask)
    with torch.no_grad():
        full, _ = predict_digits(model(frames, mask)["logits"])

    assert prefixes.shape == (2, 4, 2)
    assert torch.equal(prefixes[:, -1], full) and torch.equal(prefixes[0, 2], full[0])


def test_score_rows_the_same_way_for_every_method():
    def row(true, sequence, anchor, voted, prefix):
        return {
            "true": true, "sequence": sequence, "anchor": anchor, "vote": voted,
            "sequence_confidence": 0.9, "anchor_confidence": 0.8, "vote_agreement": 0.7,
            "readings": prefix, "prefix": prefix,
        }
    rows = [row("8", "8", "9", "8", "8|8|9|8|8"), row("49", "49", "49", "9", "49|49")]

    found = score(rows, train_numbers=["8", "8", "49"])

    assert [found["methods"][m]["accuracy"] for m in ("sequence", "anchor", "vote")] == [1.0, 0.5, 0.5]
    # The vote's agreement is not a probability, so it is not reported as a confidence.
    assert "mean_confidence" not in found["methods"]["vote"]
    assert found["methods"]["vote"]["mean_agreement"] == pytest.approx(0.7)
    assert found["stability"]["prefix"] == {"sequences": 2, "flips": 1.0, "stable_share": 0.5}
    # Only the sequence with at least PREFIX_FRAMES frames, at every k: the short 49 never counts.
    by_frames = found["stability"]["accuracy_by_frames"]
    assert list(by_frames) == [1, 2, 3, 4, 5]
    assert [entry["count"] for entry in by_frames.values()] == [1] * 5
    assert by_frames[1]["accuracy"] == 1.0 and by_frames[3]["accuracy"] == 0.0
    assert found["baseline"]["number"] == "8"


def digits(*numbers):
    """Numbers as strings -> digit classes (S, 2), for the metric tests."""
    return torch.tensor([encode_label(number) for number in numbers])


def test_wilson_interval_for_tiny_counts():
    low, high = wilson_interval(1, 1)

    assert low == pytest.approx(0.2065, abs=1e-3) and high == 1.0
    assert wilson_interval(0, 1)[0] == 0.0


def test_per_number_leaves_out_absent_numbers_and_macro_drops_small_ones():
    labels = digits(*["4"] * 5, "88")
    predictions = digits(*["4"] * 4, "6", "66")

    table = per_number(predictions, labels)

    assert list(table) == ["4", "88"]
    assert table["4"]["accuracy"] == 0.8 and table["88"]["accuracy"] == 0.0
    assert macro(table) == pytest.approx(0.4)
    assert macro(table, min_count=5) == pytest.approx(0.8)


def test_scores_split_one_and_two_digit_numbers():
    result = {"loss": 1.0, "labels": digits("4", "4", "49", "89"), "predictions": digits("4", "6", "49", "9")}

    found = scores(result)

    assert found["one_digit_accuracy"] == 0.5 and found["two_digit_accuracy"] == 0.5


def test_bands_and_baseline_come_from_training_counts():
    train_numbers = ["8"] * 150 + ["49"] * 30 + ["64"] * 2
    labels = digits("8", "8", "49", "64", "94")

    bands = shot_bands(train_numbers)
    found = band_accuracy(digits("8", "9", "49", "6", "49"), labels, bands)
    baseline = majority_baseline(labels, train_numbers)

    assert bands == {"8": "many", "49": "medium", "64": "few"}
    assert {band: row["accuracy"] for band, row in found.items()} == {
        "many": 0.5, "medium": 1.0, "few": 0.0, "unseen": 0.0,
    }
    assert baseline["number"] == "8" and baseline["accuracy"] == 0.4


def test_confusions_most_common_first():
    found = confusions(digits("8", "8", "6", "4"), digits("9", "9", "4", "4"))

    assert found == [("9", "8", 2), ("4", "6", 1)]


def test_count_flips_counts_changes_within_real_frames():
    # After 1, 2, 3 frames: 8 -> 9 -> 9 flips once; 4 -> 4 is stable; a 1-frame sequence is skipped.
    # The fourth answers are padding and must not count.
    prefixes = torch.stack([
        digits("8", "9", "9", "6"),
        digits("4", "4", "6", "6"),
        digits("6", "9", "9", "9"),
    ])

    found = count_flips(prefixes, torch.tensor([3, 2, 1]))

    assert found == {"sequences": 2, "flips": 0.5, "stable_share": 0.5}


def test_heldout_error_share_counts_answers_from_training_numbers():
    labels = digits("94", "94", "94")
    predictions = digits("49", "9", "94")

    assert heldout_error_share(predictions, labels, train_numbers=["49", "4"]) == 0.5


def one_sequence(real_frames=3, max_frames=4):
    frames, mask = fake_batch(real_frames=(real_frames,), max_frames=max_frames)
    return frames[0], mask[0]


GRID = (IMAGE_SIZE[0] // CELL_STRIDE, IMAGE_SIZE[1] // CELL_STRIDE)


def test_query_maps_sum_to_one_per_frame_and_digit():
    model = JerseyModel("averaged", max_frames=4, pretrained=False).eval()
    frames, mask = one_sequence()

    maps = query_maps(model, frames, mask)

    assert maps.shape == (3, 2, *GRID)
    assert torch.allclose(maps.sum(dim=(2, 3)), torch.ones(3, 2), atol=1e-5)


def test_occlusion_and_attention_check_shapes():
    model = JerseyModel("averaged", max_frames=4, pretrained=False).eval()
    frames, mask = one_sequence(real_frames=2)

    occlusion = occlusion_map(model, frames, mask)
    check = attention_check(model, frames, mask, query_maps(model, frames, mask), draws=2)

    assert occlusion.shape == (2, 2, *GRID)
    assert {key: value.shape for key, value in check.items()} == {
        "base": (2, 2), "attended": (2, 2), "elsewhere": (2, 2),
    }


@pytest.mark.parametrize("variant", VARIANTS)
def test_frame_weights_sum_to_one(variant):
    model = JerseyModel(variant, max_frames=4, pretrained=False).eval()
    frames, mask = one_sequence()

    weights = frame_weights(model, frames, mask)

    assert weights.shape == (3, 2)
    assert torch.allclose(weights.sum(dim=0), torch.ones(2), atol=1e-5)


@pytest.mark.parametrize("reader, queries", [("fused", 2), ("two-step", 3)])
def test_temporal_pass_rebuilds_the_encoder_exactly(reader, queries):
    model = JerseyModel("full", max_frames=4, pretrained=False, reader=reader).eval()
    frames, mask = one_sequence()

    with torch.no_grad():
        tokens = model.read_tokens(frames[mask])[None]
        rebuilt, attention = temporal_pass(model, tokens)
        expected = model.temporal(tokens, mask[mask][None])

    assert tokens.shape[2] == queries
    assert torch.allclose(rebuilt, expected, atol=1e-5)
    assert attention.shape == (queries, 3, 3)
    assert torch.allclose(attention.sum(dim=-1), torch.ones(queries, 3), atol=1e-5)


def test_temporal_attention_only_for_full():
    frames, mask = one_sequence()

    assert temporal_attention(JerseyModel("averaged", max_frames=4, pretrained=False).eval(), frames, mask) is None
    assert temporal_attention(JerseyModel("full", max_frames=4, pretrained=False).eval(), frames, mask).shape == (2, 3, 3)


def test_frame_removal_one_value_per_real_frame():
    model = JerseyModel("pooled", max_frames=4, pretrained=False).eval()

    removal = frame_removal(model, *one_sequence())
    single = frame_removal(model, *one_sequence(real_frames=1))

    assert removal.shape == (3, 2) and removal.isfinite().all()
    assert single.shape == (1, 2) and single.isnan().all()


def test_map_similarity():
    a = torch.rand(12, 6)

    assert map_similarity(a, a) == pytest.approx(1.0)
    assert map_similarity(a, -a) == pytest.approx(-1.0)
    assert math.isnan(map_similarity(a, torch.ones(12, 6)))


@pytest.mark.parametrize("name", PERTURBATIONS)
def test_perturbations_keep_shape_and_range(name):
    images = torch.rand(2, 3, *IMAGE_SIZE)

    changed = PERTURBATIONS[name](images)

    assert changed.shape == images.shape
    assert changed.min() >= 0 and changed.max() <= 1


def test_torso_cover_and_only_torso_split_the_crop():
    images = torch.rand(1, 3, *IMAGE_SIZE)

    covered, kept = cover(images, torso=True), cover(images, torso=False)

    # Every pixel is greyed out in exactly one of the two, and kept in the other.
    assert torch.equal(torch.where(covered == images, covered, kept), images)
    assert (covered != images).any() and (kept != images).any()


def test_position_accuracy_scores_each_digit_and_the_number():
    labels = torch.tensor([[4, BLANK], [4, 9], [8, 9], [6, BLANK]])
    predictions = torch.tensor([[4, BLANK], [4, 8], [6, 9], [9, 6]])

    found = position_accuracy(predictions, labels)

    assert found == {"first_digit": 0.5, "second_digit": 0.5, "number": 0.25}


def test_colour_baseline_finds_numbers_told_apart_by_colour(tmp_path):
    for number, colour in (("4", "red"), ("8", "blue")):
        for player in range(3):
            folder = tmp_path / number / f"p{player}" / "0"
            folder.mkdir(parents=True)
            for name in ("1_1_0.png", "1_2_0.png", "1_0_0_anchor.png"):
                Image.new("RGB", (22, 35), colour).save(folder / name)
    ids = [f"{number}/p{player}/0" for number in ("4", "8") for player in range(3)]
    loader = torch.utils.data.DataLoader(JerseyDataset(tmp_path, ids), batch_size=4)

    predictions = colour_baseline(loader, loader, steps=50)

    assert torch.equal(predictions, torch.tensor([[4, BLANK]] * 3 + [[8, BLANK]] * 3))


def test_check_shortcuts_writes_scores_and_picture(tiny_runs):
    shortcuts_main(["holdout", "--runs", str(tiny_runs), "--device", "cpu", "--workers", "0"])

    report = json.loads((tiny_runs / "holdout" / "shortcuts_val.json").read_text())
    assert report["sequences"] == 1  # val without the held-out 49
    assert set(report["model"]) == set(PERTURBATIONS)
    assert (tiny_runs / "holdout" / "shortcuts_val.png").exists()


DIGIT_BOX = [40, 80, 52, 105]  # a digit 12 wide and 25 tall in the 192 x 96 input


def digit_image(box=DIGIT_BOX, ink=255, shirt=0, size=IMAGE_SIZE):
    """A uint8 shirt (3, H, W) with one solid block of ink where box is."""
    image = torch.full((3, *size), shirt, dtype=torch.uint8)
    left, top, right, bottom = box
    image[:, top:bottom, left:right] = ink
    return image


def test_splice_boxes_sit_side_by_side_where_a_was_and_cover_it():
    a = digit_image()  # white digit on black
    b = digit_image([30, 60, 40, 80], ink=200, shirt=100)  # a smaller grey digit on grey
    a_centre = (DIGIT_BOX[0] + DIGIT_BOX[2]) / 2

    for a_left in (True, False):
        image, boxes = splice(a, DIGIT_BOX, b, [30, 60, 40, 80], a_left, gap=3)
        first, second = boxes.tolist()
        own, other = (first, second) if a_left else (second, first)

        assert image.shape == a.shape and image.dtype == torch.uint8 and boxes.shape == (2, 4)
        assert second[0] - first[2] == pytest.approx(3, abs=1)  # the gap, in reading order
        assert other[3] - other[1] == pytest.approx(DIGIT_BOX[3] - DIGIT_BOX[1], abs=1)  # scaled to A's height
        assert (first[0] + second[2]) / 2 == pytest.approx(a_centre, abs=1)  # centred where A's digit was
        assert own[1] == DIGIT_BOX[1] and own[3] == DIGIT_BOX[3]
        l, t, r, b_ = (round(v) for v in own)
        assert image[:, t + 2:b_ - 2, l + 2:r - 2].eq(255).all()  # A's digit inside its box
        l, t, r, b_ = (round(v) for v in other)
        assert image[:, t + 2:b_ - 2, l + 2:r - 2].sub(200).abs().le(2).all()  # B's digit inside its box
        white = (image[0] == 255).nonzero()
        assert white[:, 1].min() >= own[0] - 1 and white[:, 1].max() < own[2] + 1  # A's old digit is covered
        assert image[:, :70].eq(0).all() and image[:, :, :10].eq(0).all()  # outside the patch: A as it was
        steps = image[0, DIGIT_BOX[1] - 1].int().diff().abs()  # a row above the digits, inside the patch
        assert steps.max() <= 100 / FEATHER + 2  # A's shirt fades into B's, and the patch into the frame


def fake_strips(path, kept_frames, exclude=(), split="train"):
    """Write a make_pairs.py output: each sequence gets kept_frames[sequence] kept frames and a kept anchor."""
    sequences = {
        sequence_id: [
            {"image": f"{sequence_id}/1_{frame}_0.png", "anchor": False, "kept": frame <= kept,
             "centre": [46, 92], "prob": 0.95, "box": DIGIT_BOX, "colour": [0.5, 0.5, 0.5]}
            for frame in range(1, 4)
        ] + [{"image": f"{sequence_id}/1_0_0_anchor.jpg", "anchor": True, "kept": True,
              "centre": [46, 92], "prob": 0.95, "box": DIGIT_BOX, "colour": [0.5, 0.5, 0.5]}]
        for sequence_id, kept in kept_frames.items()
    }
    path.write_text(json.dumps({"locator": "fake", "split": split, "exclude": list(exclude), "sequences": sequences}))
    return path


def test_partner_never_makes_a_held_out_number_or_pairs_a_player_with_themself(tmp_path):
    kept = {"4/a/0": 3, "4/a/1": 3, "4/b/0": 3, "9/c/0": 3, "9/d/0": 3, "8/e/0": MIN_KEPT - 1}
    bank = StripBank(fake_strips(tmp_path / "strips.json", kept, exclude=("49",)))
    torch.manual_seed(0)

    made = Counter()
    for sequence_id in bank.sequence_ids():
        for _ in range(50):
            partner_id, a_left = bank.partner(sequence_id)
            assert partner_id.split("/")[1] != sequence_id.split("/")[1]
            digit, other = sequence_id.split("/")[0], partner_id.split("/")[0]
            made[digit + other if a_left else other + digit] += 1

    assert "8/e/0" not in bank  # too few frames with a found digit
    assert "49" not in made and {"44", "94", "99"} <= set(made)


def test_dataset_splices_only_when_augmenting(tmp_path):
    kept = {"4/a/0": 3, "9/c/0": 3}
    for sequence_id in kept:
        fake_sequence_dir(tmp_path / "data", sequence_id, [1, 2, 3], noise=True)
    bank = StripBank(fake_strips(tmp_path / "strips.json", kept, exclude=("49",)))
    torch.manual_seed(0)

    def items(augment):
        dataset = JerseyDataset(tmp_path / "data", list(kept), augment=augment, load_anchor=True,
                                max_frames=4, pairs=bank, pair_share=1.0)
        return [dataset[i] for i in range(len(dataset)) for _ in range(5)]

    spliced, plain = items(augment=True), items(augment=False)

    assert {decode_label(*item["label"].tolist()) for item in spliced} == {"94"}
    assert all(item["mask"].sum() == 3 and item["frames"].shape == (4, 3, *IMAGE_SIZE) for item in spliced)
    assert all(item["anchor"].shape == (3, *IMAGE_SIZE) for item in spliced)
    assert all(item["has_boxes"] and item["boxes"].shape == (4, 2, 4) for item in spliced)
    assert all(item["boxes"][:3].abs().sum() > 0 and item["anchor_boxes"].shape == (2, 4) for item in spliced)
    assert {decode_label(*item["label"].tolist()) for item in plain} == {"4", "9"}
    assert not any(item["has_boxes"] or item["boxes"].any() for item in plain)


@pytest.fixture(scope="module")
def tiny_locator(tiny_runs):
    """A locator trained one epoch on the tiny data, holding out 49 as the "holdout" run does."""
    root = tiny_runs.parent
    locator_main([
        "--epochs", "1", "--batch-size", "2", "--no-pretrained", "--workers", "0", "--device", "cpu",
        "--data", str(root / "data"), "--splits", str(root / "splits.json"), "--issues", str(root / "data_issues.json"),
        "--out", str(root / "locator"), "--run-name", "loc_ex49", "--exclude", "49",
    ])
    return root / "locator" / "loc_ex49"


@pytest.mark.parametrize("split, expected", [("train", {"4/a/0"}), ("val", {"4/c/0"})])
def test_make_pairs_reads_only_the_split_without_held_out_partners(tiny_locator, tmp_path, split, expected):
    out = tmp_path / "pairs"
    pairs_main(["--locator", str(tiny_locator), "--split", split, "--exclude", "49", "--out", str(out),
                "--device", "cpu", "--workers", "0", "--min-prob", "0"])

    strips = json.loads((out / "strips.json").read_text())
    assert set(strips["sequences"]) == expected  # 4/f/0 is linked to a 49, so never read
    assert strips["split"] == split and strips["exclude"] == ["49"]
    image = strips["sequences"][min(expected)][0]
    assert {"centre", "prob", "box", "colour", "kept"} <= set(image)
    assert json.loads((out / "summary.json").read_text())["never made (held out)"] == ["49"]
    assert all((out / name).exists() for name in ("strips.png", "rejected.png", "spliced.png"))


def test_make_pairs_refuses_a_locator_with_other_held_out_numbers(tiny_locator, tmp_path):
    with pytest.raises(ValueError, match="differs from the locator"):
        pairs_main(["--locator", str(tiny_locator), "--exclude", "89", "--out", str(tmp_path), "--device", "cpu"])


def test_ink_box_finds_a_digit_on_a_plain_shirt():
    image = digit_image(ink=230, shirt=40).float() / 255
    centre = ((DIGIT_BOX[0] + DIGIT_BOX[2]) / 2, (DIGIT_BOX[1] + DIGIT_BOX[3]) / 2)

    box, kit = ink_box(image, centre)

    assert box == DIGIT_BOX
    assert kit.tolist() == pytest.approx([40 / 255] * 3, abs=1e-3)
    assert keep(box, centre, 0.95, MIN_PROB)
    assert not keep(box, centre, 0.5, MIN_PROB)  # the locator is not sure
    assert not keep(box, (10, 10), 0.95, MIN_PROB)  # the box misses the evidence centre
    assert not keep([40, 80, 52, 90], (45, 85), 0.95, MIN_PROB)  # too short for a digit


def test_spliced_boxes_follow_the_augmentation(tmp_path):
    data = tmp_path / "data"
    kept = {"4/a/0": 3, "9/c/0": 3}
    for sequence_id in kept:
        folder = data / sequence_id
        folder.mkdir(parents=True)
        image = Image.fromarray(digit_image().permute(1, 2, 0).numpy())
        for name in ("1_1_0.png", "1_2_0.png", "1_3_0.png", "1_0_0_anchor.jpg"):
            image.save(folder / name)
    bank = StripBank(fake_strips(tmp_path / "strips.json", kept))
    dataset = JerseyDataset(data, list(kept), augment=True, load_anchor=True, max_frames=4, pairs=bank, pair_share=1.0)
    torch.manual_seed(0)

    checked = 0
    for _ in range(30):
        item = dataset[0]
        # Ink is far from the shirt's colour whatever the colour augmentation did.
        frame = denormalize(item["frames"][0]).mean(dim=0)
        ink = ((frame - frame.median()).abs() > 0.25).nonzero()
        if len(ink) < 50:  # blurred or squeezed out of sight: nothing to compare
            continue
        boxes = item["boxes"][0]
        assert ink[:, 1].min() >= boxes[0, 0] - 3 and ink[:, 1].max() < boxes[1, 2] + 3
        assert ink[:, 0].min() >= boxes[:, 1].min() - 3 and ink[:, 0].max() < boxes[:, 3].max() + 3
        checked += 1
    assert checked > 10


def test_real_pairs_false_drops_real_two_digit_sequences(tmp_path):
    for sequence_id in ("4/a/0", "49/b/0"):
        fake_sequence_dir(tmp_path, sequence_id, [1, 2])

    assert JerseyDataset(tmp_path, ["4/a/0", "49/b/0"], real_pairs=False).sequence_ids == ["4/a/0"]
    assert len(JerseyDataset(tmp_path, ["4/a/0", "49/b/0"])) == 2


def test_train_refuses_a_bank_that_could_splice_a_held_out_number(tiny_runs, tmp_path):
    bank = fake_strips(tmp_path / "strips.json", {"4/a/0": 3}, exclude=())
    root = tiny_runs.parent

    with pytest.raises(ValueError, match="without holding out"):
        train_main([
            "--epochs", "1", "--no-pretrained", "--workers", "0", "--device", "cpu", "--max-frames", "4",
            "--data", str(root / "data"), "--splits", str(root / "splits.json"),
            "--issues", str(root / "data_issues.json"), "--out", str(tmp_path / "runs"),
            "--exclude", "49", "--synthetic-pairs", "0.5", "--pairs", str(bank),
        ])


def test_train_with_spliced_pairs(tiny_runs, tmp_path):
    bank = fake_strips(tmp_path / "strips.json", {"4/a/0": 3, "4/f/0": 3})
    root = tiny_runs.parent

    train_main([
        "--variant", "averaged", "--epochs", "1", "--batch-size", "2", "--max-frames", "4",
        "--train-check-size", "2", "--no-pretrained", "--workers", "0", "--device", "cpu",
        "--data", str(root / "data"), "--splits", str(root / "splits.json"), "--out", str(tmp_path / "runs"),
        "--run-name", "pairs", "--synthetic-pairs", "1", "--pairs", str(bank),
    ])

    config = json.loads((tmp_path / "runs" / "pairs" / "config.json").read_text())
    assert config["synthetic_pairs"] == 1.0


# --- Local digit reader (locator) ---------------------------------------------------------------

def test_locator_gives_ten_scores_per_cell():
    model = LocalDigitReader(pretrained=False).eval()
    scores = model(torch.randn(2, 3, *IMAGE_SIZE))

    assert scores.shape == (2, 10, IMAGE_SIZE[0] // CELL_STRIDE, IMAGE_SIZE[1] // CELL_STRIDE)


def test_crop_score_is_mean_of_top_cells_inside_the_torso():
    rows, columns = IMAGE_SIZE[0] // CELL_STRIDE, IMAGE_SIZE[1] // CELL_STRIDE
    inside = torso_cells(rows, columns)
    cells = torch.zeros(1, 10, rows, columns)
    cells[0, 3][~inside] = 100.0  # strong scores outside the torso must not count
    torso = inside.nonzero()
    for r, c in torso[:TOP_CELLS].tolist():
        cells[0, 3, r, c] = 2.0
    cells[0, 3, torso[TOP_CELLS, 0], torso[TOP_CELLS, 1]] = 1.0  # the ninth best is left out

    scores = crop_scores(cells)

    assert inside.any() and not inside.all()
    assert scores[0, 3].item() == pytest.approx(2.0)
    assert scores[0, 0].item() == 0


def test_train_locator_leaves_out_held_out_partners_and_saves_a_run(tiny_runs, tmp_path):
    root = tiny_runs.parent
    locator_main([
        "--epochs", "1", "--batch-size", "2", "--no-pretrained", "--workers", "0", "--device", "cpu",
        "--data", str(root / "data"), "--splits", str(root / "splits.json"), "--issues", str(root / "data_issues.json"),
        "--out", str(tmp_path / "locator"), "--run-name", "loc", "--exclude", "49",
    ])
    run = tmp_path / "locator" / "loc"

    config = json.loads((run / "config.json").read_text())
    assert config["linked_dropped"] == ["4/f/0"] and config["sequences"] == {"train": 1, "val": 1}
    history = json.loads((run / "history.json").read_text())
    assert [entry["epoch"] for entry in history["epochs"]] == [0, 1]
    model, config, epoch = load_locator(run)
    assert epoch == 1 and not model.training
    with pytest.raises(FileExistsError):  # never overwrites a run
        locator_main(["--epochs", "1", "--no-pretrained", "--workers", "0", "--device", "cpu",
                      "--data", str(root / "data"), "--splits", str(root / "splits.json"),
                      "--out", str(tmp_path / "locator"), "--run-name", "loc"])


# --- Two-step reader --------------------------------------------------------------------------

def test_compose_gives_probabilities_that_sum_to_one_per_position():
    torch.manual_seed(0)
    digit_logits, count_logits = torch.randn(5, 7, 3, 10) * 3, torch.randn(5, 7, 2) * 3

    log_probs = compose(digit_logits, count_logits)

    assert log_probs.shape == (5, 7, 2, 11)
    assert torch.allclose(log_probs.exp().sum(dim=-1), torch.ones(5, 7, 2), atol=1e-5)
    assert (log_probs[..., 0, BLANK] == float("-inf")).all()
    one = count_logits.softmax(dim=-1)[..., 0]
    assert torch.allclose(log_probs[..., 1, BLANK].exp(), one, atol=1e-6)  # second BLANK is P(one digit)


def test_compose_reads_a_sure_two_digit_count_as_left_then_right():
    digit_logits = torch.full((1, 3, 10), -20.0)
    digit_logits[0, 0, 8] = digit_logits[0, 1, 4] = digit_logits[0, 2, 9] = 20.0  # single 8, left 4, right 9

    assert predict_digits(compose(digit_logits, torch.tensor([[-20.0, 20.0]])))[0].tolist() == [[4, 9]]
    assert predict_digits(compose(digit_logits, torch.tensor([[20.0, -20.0]])))[0].tolist() == [[8, BLANK]]


@pytest.mark.parametrize("reader", READERS)
def test_load_run_rebuilds_either_reader(tmp_path, reader):
    model = JerseyModel("averaged", max_frames=4, pretrained=False, reader=reader).eval()
    config = {"variant": "averaged", "max_frames": 4}
    if reader != "fused":  # older runs have no reader key: they are fused
        config["reader"] = reader
    torch.save({"model": model.state_dict(), "epoch": 3, "config": config}, tmp_path / "best.pt")

    loaded, _, epoch = load_run(tmp_path)
    frames, mask = fake_batch()

    assert loaded.reader == reader and epoch == 3
    with torch.no_grad():
        assert torch.equal(model(frames, mask)["logits"], loaded(frames, mask)["logits"])


def test_two_step_values_get_no_deep_features():
    model = JerseyModel("averaged", pretrained=False, reader="two-step").eval()
    images = torch.randn(2, 3, *IMAGE_SIZE)
    seen = []
    hook = model.digits.attention.register_forward_pre_hook(lambda module, args: seen.append(args[2].clone()))

    with torch.no_grad():
        first = model.read(images)
        for parameter in model.backbone[LOCAL_BLOCKS:].parameters():  # change only the deep path
            parameter.add_(torch.randn_like(parameter))
        second = model.read(images)
    hook.remove()

    assert torch.equal(seen[0], seen[1])  # the values are the same
    assert not torch.allclose(first[1], second[1])  # while the count (deep) did change
    assert not torch.allclose(first[2], second[2])  # and so did where the queries look


def test_two_step_values_are_local():
    torch.manual_seed(0)
    model = JerseyModel("averaged", pretrained=False, reader="two-step").eval()
    images = torch.randn(2, 3, *IMAGE_SIZE)
    changed = images.clone()
    changed[..., 72:] = torch.randn_like(changed[..., 72:])  # the far right only

    with torch.no_grad():
        values = [model.values(model.backbone[:LOCAL_BLOCKS](x)) for x in (images, changed)]
    moved = (values[0] - values[1]).abs()

    assert moved[..., 9:].mean() > 0
    assert moved[..., :2].mean() < 0.05 * moved[..., 9:].mean()  # the far left barely moves


def test_two_step_keeps_left_and_right_attention_for_explain():
    model = JerseyModel("averaged", pretrained=False, reader="two-step")
    frames, mask = fake_batch()

    out = model(frames, mask)

    cells = (IMAGE_SIZE[0] // CELL_STRIDE) * (IMAGE_SIZE[1] // CELL_STRIDE)
    assert model.digits.last_attention.shape == (int(mask.sum()), 2, cells)
    assert model.digits.all_attention.shape == (int(mask.sum()), 3, cells)
    assert torch.equal(model.digits.all_attention[:, 1:], model.digits.last_attention)
    assert out["attention"].shape == (2, 4, 3, cells) and out["attention"][~mask].eq(0).all()
    assert torch.allclose(out["attention"][mask].sum(dim=-1), torch.ones(int(mask.sum()), 3), atol=1e-5)


def test_three_query_side_bias_starts_single_centre_left_left_and_right_right():
    bias = JerseyModel("averaged", pretrained=False, reader="two-step").digits.side_bias
    columns = IMAGE_SIZE[1] // CELL_STRIDE
    per_column = bias.reshape(3, -1, columns)[:, 0]  # (3, columns)
    peaks = per_column.argmax(dim=1).tolist()

    assert peaks[1] < peaks[0] < peaks[2]
    assert peaks[0] in (columns // 2 - 1, columns // 2)


def test_box_cells_include_a_thin_box_between_cell_centres():
    inside = box_cells(torch.tensor([[9.0, 20.0, 14.0, 40.0]]), grid=(6, 4)).reshape(6, 4)

    assert inside[2:5, 1].all() and inside.sum() == 3  # x 9-14 is cell column 1; y 20-40 rows 2-4


def test_two_step_loss_rewards_attention_inside_the_boxes():
    model = JerseyModel("averaged", max_frames=2, pretrained=False, reader="two-step")
    frames, mask = fake_batch(real_frames=(2, 1), max_frames=2)
    labels = torch.tensor([[4, 9], [8, BLANK]])
    boxes = torch.zeros(2, 2, 2, 4)
    boxes[0, :, 0] = torch.tensor([20.0, 40, 40, 70])
    boxes[0, :, 1] = torch.tensor([50.0, 40, 70, 70])
    has_boxes = torch.tensor([True, False])
    out = model(frames, mask)

    total, parts = two_step_loss(out, labels, boxes, has_boxes, mask)
    without, _ = two_step_loss(out, labels, boxes, has_boxes, mask, attention_weight=0)
    on_box = dict(out, attention=torch.where(box_cells(boxes)[:, :, [0, 0, 1]], 1.0, 0.0) / 12)

    assert total.isfinite() and parts["attention"] > 0 and total > without
    assert two_step_loss(on_box, labels, boxes, has_boxes, mask)[1]["attention"] < parts["attention"]
    total.backward()  # reaches the weights


def test_two_step_training_overfits_one_batch():
    torch.manual_seed(0)
    model = JerseyModel("averaged", max_frames=2, pretrained=False, reader="two-step")
    frames, mask = fake_batch(real_frames=(2, 1, 2, 1), max_frames=2)
    labels = torch.tensor([[4, 9], [8, BLANK], [6, 6], [9, BLANK]])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(60):
        loss, _ = two_step_loss(model(frames, mask), labels, frame_mask=mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # In train mode: without pretrained weights BatchNorm's running statistics come from
    # random frames, as in test_training_overfits_one_batch.
    with torch.no_grad():
        assert torch.equal(predict_digits(model(frames, mask)["logits"])[0], labels)


@pytest.fixture(scope="module")
def two_step_runs(tmp_path_factory):
    """Stage 1 and stage 2 two-step runs on fake data holding out 49, with train and val digit boxes."""
    root = tmp_path_factory.mktemp("two_step")
    sequences = {"train": ["4/a/0", "9/b/0", "8/e/0", "89/w/0", "49/x/0"], "val": ["4/c/0", "9/d/0", "49/y/0"],
                 "test": ["4/g/0", "49/z/0"]}
    for sequence_id in sum(sequences.values(), []):
        fake_sequence_dir(root / "data", sequence_id, [1, 2, 3], noise=True)
    (root / "splits.json").write_text(json.dumps(sequences))
    (root / "data_issues.json").write_text(json.dumps(NO_ISSUES))
    train_strips = fake_strips(root / "train_strips.json", {"4/a/0": 3, "9/b/0": 3, "8/e/0": 3}, exclude=("49",))
    val_strips = fake_strips(root / "val_strips.json", {"4/c/0": 3, "9/d/0": 3}, exclude=("49",), split="val")
    common = [
        "--reader", "two-step", "--variant", "averaged", "--epochs", "2", "--batch-size", "2", "--max-frames", "4",
        "--train-check-size", "2", "--no-pretrained", "--workers", "0", "--device", "cpu",
        "--data", str(root / "data"), "--splits", str(root / "splits.json"), "--issues", str(root / "data_issues.json"),
        "--out", str(root / "runs"), "--exclude", "49", "--pairs", str(train_strips), "--val-pairs", str(val_strips),
    ]
    train_main([*common, "--run-name", "s1", "--no-real-pairs", "--synthetic-pairs", "0.4", "--synthetic-singles", "0.4"])
    train_main([*common, "--run-name", "s2", "--init-from", "s1", "--synthetic-pairs", "0.3"])
    return root


def test_stage_one_trains_without_real_pairs_and_picks_by_pair_accuracy(two_step_runs):
    run = two_step_runs / "runs" / "s1"
    config = json.loads((run / "config.json").read_text())
    history = json.loads((run / "history.json").read_text())

    assert config["reader"] == "two-step" and config["selected_by"] == "pair_accuracy" and config["selected_highest"]
    assert config["sequences"]["train"] == 3  # 4, 9 and 8: not the real 89, nor the held-out 49
    assert config["synthetic_singles"] == 0.4
    assert config["count_offset"] == pytest.approx(math.log(3))  # train 4, 9, 8 and 89; the 49 is held out
    assert load_run(run)[0].count_offset == pytest.approx(math.log(3))
    assert all({"pair_accuracy", "second_is_right", "second_is_left", "left_right_gap", "singles_accuracy"} <= set(entry)
               for entry in history["epochs"])
    highest = max(history["epochs"][1:], key=lambda entry: entry["pair_accuracy"])
    assert history["best"]["pair_accuracy"] == highest["pair_accuracy"]
    model, _, _ = load_run(run)
    assert model.reader == "two-step"


def test_stage_two_starts_from_stage_one_and_picks_by_val_loss(two_step_runs):
    config = json.loads((two_step_runs / "runs" / "s2" / "config.json").read_text())

    assert config["init_from"] == "s1" and config["selected_by"] == "val_loss" and not config["selected_highest"]
    assert config["sequences"]["train"] == 4  # the real 89 is back; the held-out 49 is not


def test_init_from_refuses_a_run_that_did_not_hold_out_the_same_numbers(two_step_runs, tmp_path):
    root = two_step_runs
    with pytest.raises(ValueError, match="did not hold out"):
        train_main([
            "--reader", "two-step", "--variant", "averaged", "--epochs", "1", "--max-frames", "4", "--no-pretrained",
            "--workers", "0", "--device", "cpu", "--data", str(root / "data"), "--splits", str(root / "splits.json"),
            "--issues", str(root / "data_issues.json"), "--out", str(root / "runs"), "--run-name", "bad",
            "--exclude", "49", "89", "--init-from", "s1",
        ])
    with pytest.raises(ValueError, match="is a two-step averaged run"):
        train_main([
            "--variant", "averaged", "--epochs", "1", "--max-frames", "4", "--no-pretrained", "--workers", "0",
            "--device", "cpu", "--data", str(root / "data"), "--splits", str(root / "splits.json"),
            "--issues", str(root / "data_issues.json"), "--out", str(root / "runs"), "--run-name", "bad2",
            "--exclude", "49", "--init-from", "s1",
        ])


def test_evaluate_reports_how_held_out_answers_fail(two_step_runs):
    evaluate_main(["s2", "--runs", str(two_step_runs / "runs"), "--device", "cpu", "--workers", "0"])

    metrics = json.loads((two_step_runs / "runs" / "s2" / "metrics_val.json").read_text())
    held = metrics["hold_out"]
    assert metrics["run"]["reader"] == "two-step"
    assert held["sequences"] == 2 and sum(held["answer_counts"].values()) == 2
    assert 0 <= held["one_digit_share"] <= 1 and 0 <= held["first_digit_hits"] <= 1
    assert 0 <= held["second_digit_hits"] <= 1


def test_check_shortcuts_runs_on_a_two_step_run(two_step_runs):
    shortcuts_main(["s2", "--runs", str(two_step_runs / "runs"), "--device", "cpu", "--workers", "0"])

    assert (two_step_runs / "runs" / "s2" / "shortcuts_val.json").exists()


def test_pair_set_uses_validation_players_only_and_never_a_held_out_number(two_step_runs):
    root = two_step_runs
    val_ids = ["4/c/0", "9/d/0", "49/y/0"]

    found = build_pair_set(root / "val_strips.json", val_ids, ("49",), root / "data", count=20, seed=1)
    again = build_pair_set(root / "val_strips.json", val_ids, ("49",), root / "data", count=20, seed=1)

    assert {decode_label(*label) for label in found["labels"].tolist()} == {"94"}
    assert {sequence for pair in found["sequences"] for sequence in pair} <= set(val_ids)
    assert torch.equal(found["images"], again["images"])  # a fixed set
    assert found["single_labels"][:, 1].eq(BLANK).all()
    assert found["pasted"].shape == found["images"].shape and found["pasted_labels"][:, 1].eq(BLANK).all()
    assert torch.equal(found["pasted_labels"][:, 0], torch.tensor([int(b.split("/")[0]) for _, b in found["sequences"]]))
    with pytest.raises(ValueError, match="val boxes only"):
        build_pair_set(root / "train_strips.json", val_ids, ("49",), root / "data")
    with pytest.raises(ValueError, match="outside val"):
        build_pair_set(root / "val_strips.json", ["4/c/0"], ("49",), root / "data")
    with pytest.raises(ValueError, match="without holding out"):
        build_pair_set(root / "val_strips.json", val_ids, ("49", "89"), root / "data")


@pytest.mark.parametrize("reader", READERS)
def test_pair_metrics_for_either_reader(two_step_runs, reader):
    root = two_step_runs
    pair_set = build_pair_set(root / "val_strips.json", ["4/c/0", "9/d/0"], ("49",), root / "data", count=6)
    model = JerseyModel("averaged", pretrained=False, reader=reader)

    found = pair_metrics(model, pair_set, "cpu")

    assert set(found) == {"pair_accuracy", "second_is_right", "second_is_left", "left_right_gap", "singles_accuracy",
                          "pasted_singles_accuracy"}
    assert found["second_is_right"] + found["second_is_left"] <= 1  # the two digits differ
    assert math.isfinite(found["left_right_gap"])


def test_train_refuses_val_boxes_for_training(tiny_runs, tmp_path):
    bank = fake_strips(tmp_path / "strips.json", {"4/a/0": 3}, split="val")
    root = tiny_runs.parent

    with pytest.raises(ValueError, match="only train boxes"):
        train_main([
            "--epochs", "1", "--no-pretrained", "--workers", "0", "--device", "cpu", "--max-frames", "4",
            "--data", str(root / "data"), "--splits", str(root / "splits.json"), "--out", str(tmp_path / "runs"),
            "--synthetic-pairs", "0.5", "--pairs", str(bank),
        ])


# --- Pasted singles: a paste must not mean two digits ---------------------------------------------

def test_paste_single_covers_a_with_b_alone_and_is_as_wide_as_a_pair():
    a = digit_image()  # white digit on black
    b_box = [30, 60, 40, 80]
    b = digit_image(b_box, ink=200, shirt=100)
    a_centre = (DIGIT_BOX[0] + DIGIT_BOX[2]) / 2

    image, box = paste_single(a, DIGIT_BOX, b, b_box, gap=3)

    assert image.shape == a.shape and image.dtype == torch.uint8 and box.shape == (4,)
    assert not image.eq(255).all(dim=0).any()  # A's white digit is gone
    assert (box[0] + box[2]) / 2 == pytest.approx(a_centre, abs=1)  # B's digit where A's was
    assert box[3] - box[1] == pytest.approx(DIGIT_BOX[3] - DIGIT_BOX[1], abs=1)  # scaled to A's height
    l, t, r, b_ = (round(v) for v in box.tolist())
    assert image[:, t + 2:b_ - 2, l + 2:r - 2].sub(200).abs().le(2).all()

    # On a shirt shaded across, any moved or pasted strip changes pixels, so the changed columns
    # are the whole patch (on a flat shirt, a pair's own strip would look unchanged).
    shaded = a.clone()
    shaded[:, :, :] = (torch.arange(IMAGE_SIZE[1]) * 2).to(torch.uint8)
    shaded[:, DIGIT_BOX[1]:DIGIT_BOX[3], DIGIT_BOX[0]:DIGIT_BOX[2]] = 255

    def patch_width(found):
        columns = (found.int() - shaded.int()).abs().sum(dim=0).gt(0).any(dim=0).nonzero()
        return columns.max().item() - columns.min().item() + 1

    single_width = patch_width(paste_single(shaded, DIGIT_BOX, b, b_box, gap=3)[0])
    pair_width = patch_width(splice(shaded, DIGIT_BOX, b, b_box, True, gap=3)[0])
    assert single_width == pytest.approx(pair_width, abs=3)  # its size gives no count away


def test_blend_width_changes_the_seam_but_hardly_the_boxes():
    a, b = digit_image(), digit_image([30, 60, 40, 80], ink=200, shirt=100)

    narrow, narrow_boxes = splice(a, DIGIT_BOX, b, [30, 60, 40, 80], True, gap=3, feather=2)
    wide, wide_boxes = splice(a, DIGIT_BOX, b, [30, 60, 40, 80], True, gap=3, feather=5)

    assert torch.allclose(narrow_boxes, wide_boxes, atol=1) and not torch.equal(narrow, wide)  # rounding: 1 px


def test_dataset_pastes_the_partners_digit_alone_without_boxes(tmp_path):
    kept = {"4/a/0": 3, "9/c/0": 3}
    for sequence_id in kept:
        fake_sequence_dir(tmp_path / "data", sequence_id, [1, 2, 3], noise=True)
    bank = StripBank(fake_strips(tmp_path / "strips.json", kept, exclude=("49",)))
    dataset = JerseyDataset(tmp_path / "data", list(kept), augment=True, load_anchor=True, max_frames=4,
                            pairs=bank, single_share=1.0)
    torch.manual_seed(0)

    items = [(sequence_id, dataset[i]) for i, sequence_id in enumerate(dataset.sequence_ids) for _ in range(4)]

    assert all(decode_label(*item["label"].tolist()) == ("9" if sequence_id[0] == "4" else "4") for sequence_id, item in items)
    assert not any(item["has_boxes"] for _, item in items)  # the attention loss teaches left and right only
    with pytest.raises(ValueError, match="more than 1"):
        JerseyDataset(tmp_path / "data", list(kept), pairs=bank, pair_share=0.6, single_share=0.6)


def test_single_partner_never_pastes_a_held_out_number(tmp_path):
    bank = StripBank(fake_strips(tmp_path / "strips.json", {"4/a/0": 3, "9/c/0": 3, "8/e/0": 3}, exclude=("9",)))
    torch.manual_seed(0)

    partners = {bank.single_partner("4/a/0").split("/")[0] for _ in range(30)}

    assert partners == {"8"}  # a 9 alone is held out here


# --- Count prior: the count head is trained with one and two digits weighed equally ---------------

def test_count_prior_is_the_log_ratio_of_one_to_two_digit_numbers():
    assert count_prior(["4", "8", "9", "48"]) == pytest.approx(math.log(3))
    assert count_prior(["4", "8"]) == 0  # no two-digit number: no correction


def test_count_offset_moves_answers_towards_one_digit_and_zero_changes_nothing():
    torch.manual_seed(0)
    model = JerseyModel("averaged", max_frames=4, pretrained=False, reader="two-step").eval()
    frames, mask = fake_batch()

    with torch.no_grad():
        plain = model(frames, mask)
        model.count_offset = 3.0
        shifted = model(frames, mask)

    assert (shifted["logits"][:, 1, BLANK] > plain["logits"][:, 1, BLANK]).all()  # P(one digit) rises
    assert torch.allclose(shifted["logits"].exp().sum(dim=-1), torch.ones(2, 2), atol=1e-5)
    assert torch.equal(shifted["count_logits"], plain["count_logits"])  # the count loss never sees it
    assert torch.equal(shifted["digit_logits"], plain["digit_logits"])


def test_pair_metrics_ignore_the_count_offset(two_step_runs):
    root = two_step_runs
    pair_set = build_pair_set(root / "val_strips.json", ["4/c/0", "9/d/0"], ("49",), root / "data", count=6)
    model = JerseyModel("averaged", pretrained=False, reader="two-step")

    plain = pair_metrics(model, pair_set, "cpu")
    model.count_offset = 10.0
    shifted = pair_metrics(model, pair_set, "cpu")

    assert plain == shifted and model.count_offset == 10.0  # restored afterwards


def test_evaluate_and_shortcuts_score_a_run_with_and_without_the_prior(two_step_runs):
    runs = two_step_runs / "runs"
    for choice in ("prior", "none"):
        evaluate_main(["s2", "--runs", str(runs), "--device", "cpu", "--workers", "0", "--count-offset", choice])
    shortcuts_main(["s2", "--runs", str(runs), "--device", "cpu", "--workers", "0", "--count-offset", "none"])

    prior = json.loads((runs / "s2" / "metrics_val_prior.json").read_text())
    none = json.loads((runs / "s2" / "metrics_val_none.json").read_text())
    assert prior["run"]["name"] == "s2 (prior)" and prior["run"]["count_offset"] == pytest.approx(math.log(3), abs=1e-4)
    assert none["run"]["count_offset"] == 0
    assert (runs / "s2" / "predictions_holdout_val_prior.csv").exists()
    assert json.loads((runs / "s2" / "shortcuts_val_none.json").read_text())["count_offset"] == 0
    with (runs.parent / "results_val.csv").open() as file:
        names = {row["name"] for row in csv.DictReader(file)}
    assert {"s2 (prior)", "s2 (none)"} <= names
