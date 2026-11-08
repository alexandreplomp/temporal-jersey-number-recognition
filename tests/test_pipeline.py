"""The pipeline end to end on fake data: prepare -> train -> evaluate -> predict. Run: uv run pytest"""

import csv
import json
import math
import shutil

import pytest

from evaluate import main as evaluate_main
from jersey import pairs
from jersey.data import encode_label, number_of, read_splits, split_ids
from predict import main as predict_main
from prepare import main as prepare_main
from tests.helpers import fake_sequence_dir, fake_strips
from train import main as train_main

NUMBERS = ("4", "9", "8", "89", "49")


def fake_data(root, numbers=NUMBERS, players=3):
    for number in numbers:
        for player in range(players):
            fake_sequence_dir(root, f"{number}/p{number}_{player}/0", [1, 2, 3], noise=True)
    return root


def common(root):
    return ["--results", str(root / "results"), "--device", "cpu", "--workers", "0"]


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Fake data, split by prepare.py, fake boxes for ex49, and two approaches trained one epoch per stage."""
    root = tmp_path_factory.mktemp("pipeline")
    data = fake_data(root / "data")
    prepare_main(["--data", str(data), "--out", str(root / "results")])
    splits = read_splits(root / "results" / "splits.json")
    boxes = root / "results" / "boxes" / "ex49"
    boxes.mkdir(parents=True)
    for split in ("train", "val"):
        singles = {s: 3 for s in splits[split] if len(number_of(s)) == 1}
        fake_strips(boxes / f"{split}.json", singles, exclude=("49",), split=split)
    for approach in ("fused", "two_step_mean", "parseq_gate"):
        train_main(["--approach", approach, "--exclude", "49", "--epochs", "1", "--batch-size", "2",
                    "--no-pretrained", "--data", str(data), *common(root)])
    return root


def test_prepare_splits_by_player_and_sends_rare_numbers_to_train(tmp_path):
    data = fake_data(tmp_path / "data", numbers=("4", "9"))
    fake_data(tmp_path / "data", numbers=("88",), players=2)
    prepare_main(["--data", str(data), "--out", str(tmp_path / "results")])

    splits = read_splits(tmp_path / "results" / "splits.json")
    players = {split: {s.split("/")[1] for s in ids} for split, ids in splits.items()}
    assert not (players["train"] & players["val"] or players["train"] & players["test"] or players["val"] & players["test"])
    assert {number_of(s) for s in splits["val"]} == {"4", "9"} and {number_of(s) for s in splits["train"]} >= {"88"}


def test_two_stage_training_writes_both_stages_skips_finished_runs_and_refuses_unfinished_ones(pipeline):
    run = pipeline / "results" / "runs" / "two_step_mean_ex49_scratch"
    config = json.loads((run / "config.json").read_text())
    history = json.loads((run / "history.json").read_text())

    assert {"best.pt", "history.json", "curves.png", "train_batch.png", "splits.json", "done.json"} <= {p.name for p in run.iterdir()}
    assert (run / "stage1" / "best.pt").exists() and config["stage"]["selected_by"] == "val_loss"
    assert config["count_offset"] == pytest.approx(math.log(3))  # train has 4, 9, 8 and 89; 49 is held out
    assert history["epochs"][0]["step"] == 1 and "pair_accuracy" in history["epochs"][0]
    assert config["pretrained"] is False and set(config["git"]) == {"commit", "dirty"}
    args = ["--approach", "two_step_mean", "--exclude", "49", "--epochs", "1", "--no-pretrained",
            "--data", str(pipeline / "data"), *common(pipeline)]
    before = (run / "best.pt").stat().st_mtime
    train_main(args)  # finished: skipped, never overwritten
    assert (run / "best.pt").stat().st_mtime == before
    (run / "done.json").rename(run / "done.json.bak")
    try:
        with pytest.raises(SystemExit, match="unfinished"):
            train_main(args)
    finally:
        (run / "done.json.bak").rename(run / "done.json")


def test_training_refuses_boxes_that_could_paste_a_held_out_number(pipeline, tmp_path):
    results = tmp_path / "results"
    (results / "boxes" / "ex49").mkdir(parents=True)
    for name in ("splits.json", "data_issues.json"):
        shutil.copy(pipeline / "results" / name, results / name)
    fake_strips(results / "boxes" / "ex49" / "train.json", {"4/p4_0/0": 3}, exclude=())
    with pytest.raises(ValueError, match="without holding out"):
        train_main(["--approach", "two_step_mean", "--exclude", "49", "--epochs", "1", "--no-pretrained",
                    "--data", str(pipeline / "data"), "--results", str(results), "--device", "cpu", "--workers", "0"])


def test_boxes_come_only_from_single_digit_train_and_val_sequences_without_held_out_ones(pipeline, tmp_path, monkeypatch):
    monkeypatch.setattr(pairs, "GATE", 0.0)  # a reader trained one epoch on noise cannot pass the real gate
    splits = read_splits(pipeline / "results" / "splits.json")
    issues = json.loads((pipeline / "results" / "data_issues.json").read_text())
    train_ids, val_ids = (split_ids(splits, split, ("49",), issues)[0] for split in ("train", "val"))

    pairs.make_boxes(pipeline / "data", tmp_path, train_ids, val_ids, ("49",), "cpu", epochs=1, workers=0, pretrained=False)

    for split, ids in (("train", train_ids), ("val", val_ids)):
        found = json.loads((tmp_path / f"{split}.json").read_text())
        assert set(found["sequences"]) == {s for s in ids if len(number_of(s)) == 1} and found["exclude"] == ["49"]
    assert {"locator.pt", "locator.json", "summary.json"} <= {p.name for p in tmp_path.iterdir()}


def test_evaluate_scores_val_compares_runs_and_scores_test_once(pipeline):
    evaluate_main(["fused_ex49_scratch", "two_step_mean_ex49_scratch", *common(pipeline)])

    metrics = json.loads((pipeline / "results" / "runs" / "two_step_mean_ex49_scratch" / "metrics_val.json").read_text())
    assert set(metrics["methods"]) == {"sequence", "anchor", "vote"}
    assert metrics["hold_out"]["sequences"] == 2  # the 49s of train and val players
    assert {"shortcuts", "speed", "without_count_offset"} <= set(metrics)
    compare = pipeline / "results" / "compare"
    with (compare / "table.csv").open() as file:
        assert {row["run"] for row in csv.DictReader(file)} == {"fused_ex49_scratch", "two_step_mean_ex49_scratch"}
    evaluate_main(["parseq_gate_ex49_scratch", *common(pipeline)])  # the third reader through the same evaluation
    assert (pipeline / "results" / "runs" / "parseq_gate_ex49_scratch" / "attention.png").exists()
    assert all((compare / name).exists() for name in ("curves.png", "temporal.png", "holdout.png"))

    evaluate_main(["two_step_mean_ex49_scratch", "--split", "test", *common(pipeline)])
    with pytest.raises(FileExistsError, match="only once"):
        evaluate_main(["two_step_mean_ex49_scratch", "--split", "test", *common(pipeline)])


def test_verify_matches_the_recorded_test_answers_and_analyse_writes_its_evidence(pipeline):
    run = pipeline / "results" / "runs" / "two_step_mean_ex49_scratch"  # scored on test above
    before = (run / "metrics_test.json").read_text()
    found = evaluate_main(["--verify", str(run), *common(pipeline)])
    assert found["same_answers"] == found["sequences"] > 0
    assert (run / "verify_test.json").exists() and (run / "metrics_test.json").read_text() == before

    analysis = evaluate_main(["--analyse", "two_step_mean_ex49_scratch", *common(pipeline)])
    assert {"match_overlap", "paired_tests", "track_bootstrap", "hold_out", "few_frames", "int8"} <= set(analysis)
    assert analysis["match_overlap"]["all"]["count"] == found["sequences"] and analysis["gate"] is None
    assert {"train players", "val players"} <= set(analysis["hold_out"]["two_step_mean_ex49_scratch"])
    assert (pipeline / "results" / "compare" / "analysis_two_step_mean_ex49_scratch.json").exists()


def test_predict_reads_one_number_and_never_loads_the_anchor(pipeline, tmp_path):
    sequence = shutil.copytree(pipeline / "data" / "4" / "p4_0" / "0", tmp_path / "sequence")
    (sequence / "1_0_0_anchor.jpg").write_bytes(b"not an image")  # loading it would fail
    run = pipeline / "results" / "runs" / "two_step_mean_ex49_scratch"

    for extra in ([], ["--two-digit-share", "0.5"]):
        number = predict_main([str(sequence), "--weights", str(run), "--device", "cpu", *extra])
        encode_label(number)  # a valid one- or two-digit number


def test_predict_warns_about_a_flag_it_cannot_use_and_explains_a_bad_folder(pipeline, tmp_path, capsys):
    sequence = pipeline / "data" / "4" / "p4_0" / "0"
    run = pipeline / "results" / "runs" / "parseq_gate_ex49_scratch"
    predict_main([str(sequence), "--weights", str(run), "--device", "cpu", "--two-digit-share", "0.5"])
    assert "only applies to two-step" in capsys.readouterr().err
    for folder in (tmp_path / "missing", tmp_path):  # no folder; a folder without frames
        with pytest.raises(SystemExit, match="cannot read"):
            predict_main([str(folder), "--weights", str(run), "--device", "cpu"])
