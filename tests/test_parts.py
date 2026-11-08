"""Tests of the pieces: data, digit finding and pasting, model, losses, metrics. Run: uv run pytest"""

import math

from PIL import Image
import pytest
import torch

from jersey import pairs
from jersey.data import (
    BLANK, IMAGE_SIZE, JerseyDataset, decode_label, denormalize, encode_label, find_images, make_sampler,
    split_ids,
)
from jersey.engine import box_cells, two_step_loss
from jersey.metrics import count_flips, group_bootstrap, macro, match_of, mcnemar_exact, per_number, track_of, wilson_interval
from jersey.model import (
    CELL_STRIDE, LOCAL_BLOCKS, READERS, TEMPORAL, JerseyModel, compose, count_prior, load_run, offset_for_share,
    predict_digits,
)
from jersey.pairs import StripBank, crop_scores, ink_box, keep, kit_neighbours, paste_single, splice, torso_cells
from tests.helpers import DIGIT_BOX, digit_image, fake_batch, fake_sequence_dir, fake_strips

NO_ISSUES = {"shared_anchors": [], "conflicting_frames": [], "shared_between_players": [], "anchor_frame_links": []}


# ----- labels and data -----

@pytest.mark.parametrize("number, expected", [("4", (4, BLANK)), ("49", (4, 9)), ("00", (0, 0)), ("0", (0, BLANK))])
def test_labels_are_two_positions_in_reading_order(number, expected):
    assert encode_label(number) == expected and decode_label(*expected) == number


@pytest.mark.parametrize("number", ["", "123", "4a"])
def test_invalid_number_raises(number):
    with pytest.raises(ValueError):
        encode_label(number)


def test_frames_in_time_order_and_the_anchor_is_optional(tmp_path):
    folder = fake_sequence_dir(tmp_path, "4/a/0", [10, 9, 100])
    anchor, frames = find_images(folder)
    assert anchor.name == "1_0_0_anchor.jpg" and [p.name for p in frames] == ["1_9_0.png", "1_10_0.png", "1_100_0.png"]

    other = fake_sequence_dir(tmp_path, "4/b/0", [1], anchor=False)
    (other / "zz.png").write_bytes((other / "1_1_0.png").read_bytes())  # a name without a frame number: last
    anchor, frames = find_images(other)
    assert anchor is None and [p.name for p in frames] == ["1_1_0.png", "zz.png"]


def test_items_keep_the_anchor_apart_from_the_frames(tmp_path):
    fake_sequence_dir(tmp_path, "4/a/0", [1, 2, 3])
    fake_sequence_dir(tmp_path, "9/b/0", [1, 2], anchor=False)
    item, no_anchor = JerseyDataset(tmp_path, ["4/a/0", "9/b/0"], load_anchor=True, max_frames=4)

    assert item["frames"].shape == (4, 3, *IMAGE_SIZE) and item["mask"].tolist() == [True] * 3 + [False]
    assert item["anchor"].shape == (3, *IMAGE_SIZE) and item["label"].tolist() == [4, BLANK]
    assert torch.equal(no_anchor["anchor"], no_anchor["frames"][0])  # without an anchor the first frame stands in
    assert "anchor" not in JerseyDataset(tmp_path, ["4/a/0"])[0]


def test_augmentation_is_the_same_for_every_frame(tmp_path):
    fake_sequence_dir(tmp_path, "4/a/0", [1, 2, 3], noise=True)
    torch.manual_seed(0)
    item = JerseyDataset(tmp_path, ["4/a/0"], augment=True, load_anchor=True, max_frames=4)[0]

    assert torch.equal(item["frames"][0], item["frames"][1]) and torch.equal(item["frames"][0], item["frames"][2])


def test_held_out_numbers_and_their_whole_linked_group_are_left_out():
    splits = {"train": ["4/a/0", "9/b/0", "9/b/1", "8/c/0"], "val": ["4/d/0"], "test": ["49/x/0"]}
    # 49/x shares an image with 9/b/0; the player b also has 9/b/1, so both go.
    issues = {**NO_ISSUES, "shared_between_players": [["49/x/0/1.png", "9/b/0/1.png"]]}

    kept, linked = split_ids(splits, "train", ("49",), issues)

    assert kept == ["4/a/0", "8/c/0"] and linked == {"9/b/0", "9/b/1"}


def test_sampler_draws_rare_numbers_more_often():
    weights = make_sampler(["4/a/0", "4/b/0", "4/c/0", "4/d/0", "48/e/0"]).weights
    assert weights[4] == pytest.approx(2 * weights[0])


# ----- digit boxes and pastes -----

def test_crop_score_is_the_mean_of_the_top_cells_inside_the_torso():
    rows, columns = IMAGE_SIZE[0] // CELL_STRIDE, IMAGE_SIZE[1] // CELL_STRIDE
    inside = torso_cells(rows, columns)
    cells = torch.zeros(1, 10, rows, columns)
    cells[0, 3][~inside] = 100.0  # outside the torso: never counts
    for r, c in inside.nonzero()[:pairs.TOP_CELLS].tolist():
        cells[0, 3, r, c] = 2.0

    assert crop_scores(cells)[0, 3].item() == pytest.approx(2.0)


def test_ink_box_finds_a_digit_and_keep_applies_the_three_checks():
    image = digit_image(ink=230, shirt=40).float() / 255
    centre = ((DIGIT_BOX[0] + DIGIT_BOX[2]) / 2, (DIGIT_BOX[1] + DIGIT_BOX[3]) / 2)

    box, kit = ink_box(image, centre)

    assert box == DIGIT_BOX and kit.tolist() == pytest.approx([40 / 255] * 3, abs=1e-3)
    assert keep(box, centre, 0.95) and not keep(box, centre, 0.5)  # the reader must be sure
    assert not keep(box, (10, 10), 0.95) and not keep([40, 80, 52, 90], (45, 85), 0.95)  # off-centre, too short


def test_splice_puts_two_digits_side_by_side_where_a_was_and_covers_it():
    a, b_box = digit_image(), [30, 60, 40, 80]
    b = digit_image(b_box, ink=200, shirt=100)
    for a_left in (True, False):
        image, boxes = splice(a, DIGIT_BOX, b, b_box, a_left, gap=3)
        first, second = boxes.tolist()
        own = first if a_left else second
        assert second[0] - first[2] == pytest.approx(3, abs=1)  # the gap, in reading order
        assert (first[0] + second[2]) / 2 == pytest.approx((DIGIT_BOX[0] + DIGIT_BOX[2]) / 2, abs=1)
        white = (image[0] == 255).nonzero()
        assert white[:, 1].min() >= own[0] - 1 and white[:, 1].max() < own[2] + 1  # A's old digit covered
        assert image[:, :70].eq(0).all()  # outside the patch: A as it was


def test_a_single_paste_is_as_wide_as_a_pair():
    shaded = digit_image()
    shaded[:, :, :] = (torch.arange(IMAGE_SIZE[1]) * 2).to(torch.uint8)  # moved strips always change pixels
    shaded[:, DIGIT_BOX[1]:DIGIT_BOX[3], DIGIT_BOX[0]:DIGIT_BOX[2]] = 255
    b_box = [30, 60, 40, 80]
    b = digit_image(b_box, ink=200, shirt=100)

    def width(image):
        columns = (image.int() - shaded.int()).abs().sum(dim=0).gt(0).any(dim=0).nonzero()
        return columns.max().item() - columns.min().item() + 1

    image, box = paste_single(shaded, DIGIT_BOX, b, b_box, gap=3)
    assert width(image) == pytest.approx(width(splice(shaded, DIGIT_BOX, b, b_box, True, gap=3)[0]), abs=3)
    assert box[3] - box[1] == pytest.approx(DIGIT_BOX[3] - DIGIT_BOX[1], abs=1)  # scaled to A's height


def test_pastes_never_make_a_held_out_number_or_pair_a_player_with_themself(tmp_path):
    kept = {"4/a/0": 3, "4/a/1": 3, "4/b/0": 3, "9/c/0": 3, "9/d/0": 3, "8/f/0": 3, "8/e/0": pairs.MIN_KEPT - 1}
    bank = StripBank(fake_strips(tmp_path / "strips.json", kept, exclude=("49", "8")))
    torch.manual_seed(0)
    made = set()
    for sequence_id in bank.sequence_ids():
        for _ in range(30):
            other, a_left = bank.partner(sequence_id)
            assert other.split("/")[1] != sequence_id.split("/")[1]
            made.add(sequence_id[0] + other[0] if a_left else other[0] + sequence_id[0])
            assert bank.single_partner(sequence_id)[0] != "8"  # a lone 8 is held out here
    assert "8/e/0" not in bank and "49" not in made and {"94", "44", "99"} <= made


def test_kit_neighbours_in_chunks_match_one_full_pass(monkeypatch):
    colours = {f"{n}/p{i}/0": torch.rand(3, generator=torch.Generator().manual_seed(i)) for i, n in enumerate("4896" * 3)}
    full = kit_neighbours(colours)
    monkeypatch.setattr(pairs, "NEIGHBOUR_CHUNK", 2)
    assert kit_neighbours(colours) == full


def test_dataset_pastes_pairs_with_boxes_that_follow_the_augmentation(tmp_path):
    kept = {"4/a/0": 3, "9/c/0": 3}
    for sequence_id in kept:
        folder = tmp_path / "data" / sequence_id
        folder.mkdir(parents=True)
        image = Image.fromarray(digit_image().permute(1, 2, 0).numpy())
        for name in ("1_1_0.png", "1_2_0.png", "1_3_0.png", "1_0_0_anchor.jpg"):
            image.save(folder / name)
    bank = StripBank(fake_strips(tmp_path / "strips.json", kept, exclude=("49",)))
    dataset = JerseyDataset(tmp_path / "data", list(kept), augment=True, load_anchor=True, max_frames=4,
                            pastes=bank, pair_share=1.0)
    torch.manual_seed(0)
    checked = 0
    for _ in range(30):
        item = dataset[0]
        assert decode_label(*item["label"].tolist()) == "94" and item["has_boxes"]  # never 49
        frame = denormalize(item["frames"][0]).mean(dim=0)
        ink = ((frame - frame.median()).abs() > 0.25).nonzero()
        if len(ink) < 50:  # blurred or squeezed out of sight
            continue
        boxes = item["boxes"][0]
        assert ink[:, 1].min() >= boxes[0, 0] - 3 and ink[:, 1].max() < boxes[1, 2] + 3
        checked += 1
    assert checked > 10
    singles = JerseyDataset(tmp_path / "data", list(kept), augment=True, max_frames=4, pastes=bank, single_share=1.0)
    assert {decode_label(*singles[i]["label"].tolist()) for i in range(2)} == {"9", "4"}
    assert not singles[0]["has_boxes"]
    plain = JerseyDataset(tmp_path / "data", list(kept), max_frames=4, pastes=bank, pair_share=1.0)
    assert decode_label(*plain[0]["label"].tolist()) == "4"  # evaluation never pastes


def test_pair_set_uses_val_boxes_of_val_players_only(tmp_path):
    for sequence_id in ("4/c/0", "9/d/0"):
        fake_sequence_dir(tmp_path / "data", sequence_id, [1, 2, 3], noise=True)
    val = fake_strips(tmp_path / "val.json", {"4/c/0": 3, "9/d/0": 3}, exclude=("49",), split="val")

    found = pairs.build_pair_set(val, ["4/c/0", "9/d/0"], ("49",), tmp_path / "data", count=10, seed=1)

    assert {decode_label(*label) for label in found["labels"].tolist()} == {"94"}
    with pytest.raises(ValueError, match="val boxes only"):
        pairs.build_pair_set(fake_strips(tmp_path / "train.json", {"4/c/0": 3}), ["4/c/0"], (), tmp_path / "data")
    with pytest.raises(ValueError, match="outside val"):
        pairs.build_pair_set(val, ["4/c/0"], ("49",), tmp_path / "data")


# ----- model -----

@pytest.mark.parametrize("temporal", TEMPORAL)
@pytest.mark.parametrize("reader", READERS)
def test_every_reader_and_frame_combiner_gives_one_answer_that_padding_does_not_change(reader, temporal):
    torch.manual_seed(0)
    model = JerseyModel(reader, temporal, max_frames=4, pretrained=False).eval()
    frames, mask = fake_batch(real_frames=(3,))
    frames[~mask] = torch.randn_like(frames[~mask])

    with torch.no_grad():
        out = model(frames, mask)
        unpadded = model(frames[:, :3], mask[:, :3])["logits"]

    assert out["logits"].shape == (1, 2, 11) and out["frame_logits"].shape == (1, 4, 2, 11)
    assert torch.allclose(out["logits"].exp().sum(dim=-1), torch.ones(1, 2), atol=1e-5)
    assert (out["logits"][:, 0, BLANK] == float("-inf")).all()  # every number has a first digit
    assert torch.allclose(out["logits"], unpadded, atol=1e-5)
    assert model.read(frames[0, :2])["logits"].shape == (2, 2, 11)


def test_compose_reads_a_sure_count_as_single_or_left_then_right():
    digit_logits = torch.full((1, 3, 10), -20.0)
    digit_logits[0, 0, 8] = digit_logits[0, 1, 4] = digit_logits[0, 2, 9] = 20.0  # single 8, left 4, right 9

    assert predict_digits(compose(digit_logits, torch.tensor([[-20.0, 20.0]])))[0].tolist() == [[4, 9]]
    assert predict_digits(compose(digit_logits, torch.tensor([[20.0, -20.0]])))[0].tolist() == [[8, BLANK]]
    random = compose(torch.randn(5, 3, 10), torch.randn(5, 2))
    assert torch.allclose(random.exp().sum(dim=-1), torch.ones(5, 2), atol=1e-5)


def test_count_offset_moves_answers_towards_one_digit():
    assert count_prior(["4", "8", "9", "48"]) == pytest.approx(math.log(3)) and count_prior(["4"]) == 0
    assert offset_for_share(0.5) == 0
    model = JerseyModel("two-step", max_frames=4, pretrained=False).eval()
    frames, mask = fake_batch()
    with torch.no_grad():
        plain = model(frames, mask)
        model.count_offset = 3.0
        shifted = model(frames, mask)
    assert (shifted["logits"][:, 1, BLANK] > plain["logits"][:, 1, BLANK]).all()
    assert torch.equal(shifted["count_logits"], plain["count_logits"])  # the count loss never sees it


def test_two_step_values_are_local_and_get_no_deep_features():
    torch.manual_seed(0)
    model = JerseyModel("two-step", pretrained=False).eval()
    images = torch.randn(2, 3, *IMAGE_SIZE)
    changed = images.clone()
    changed[..., 72:] = torch.randn_like(changed[..., 72:])  # the far right only
    with torch.no_grad():
        backbone = model.reader.backbone
        values = [model.reader.values(backbone[:LOCAL_BLOCKS](x)) for x in (images, changed)]
        moved = (values[0] - values[1]).abs()
        before = model.reader.values(backbone[:LOCAL_BLOCKS](images))
        for parameter in backbone[LOCAL_BLOCKS:].parameters():  # change only the deep path
            parameter.add_(1.0)
        after = model.reader.values(backbone[:LOCAL_BLOCKS](images))
    assert moved[..., :2].mean() < 0.05 * moved[..., 9:].mean()  # the far left barely moves
    assert torch.equal(before, after)


@pytest.mark.parametrize("reader", READERS)
def test_saved_runs_load_back(tmp_path, reader):
    model = JerseyModel(reader, "pool", max_frames=4, pretrained=False, count_offset=1.5).eval()
    config = {"reader": reader, "temporal": "pool", "max_frames": 4, "count_offset": 1.5}
    torch.save({"model": model.state_dict(), "epoch": 3, "config": config}, tmp_path / "best.pt")
    loaded, _, epoch = load_run(tmp_path)
    frames, mask = fake_batch()
    with torch.no_grad():
        assert torch.equal(model(frames, mask)["logits"], loaded(frames, mask)["logits"])
    assert epoch == 3 and loaded.count_offset == 1.5


# ----- losses and metrics -----

def test_box_cells_include_a_thin_box_between_cell_centres():
    inside = box_cells(torch.tensor([[9.0, 20.0, 14.0, 40.0]]), grid=(6, 4)).reshape(6, 4)
    assert inside[2:5, 1].all() and inside.sum() == 3


def test_two_step_loss_rewards_attention_inside_the_boxes():
    model = JerseyModel("two-step", max_frames=2, pretrained=False)
    frames, mask = fake_batch(real_frames=(2, 1), max_frames=2)
    labels = torch.tensor([[4, 9], [8, BLANK]])
    boxes = torch.zeros(2, 2, 2, 4)
    boxes[0, :, 0], boxes[0, :, 1] = torch.tensor([20.0, 40, 40, 70]), torch.tensor([50.0, 40, 70, 70])
    has_boxes = torch.tensor([True, False])
    out = model(frames, mask)
    on_box = dict(out, attention=torch.where(box_cells(boxes)[:, :, [0, 0, 1]], 1.0, 0.0) / 12)

    loss = two_step_loss(out, labels, boxes, has_boxes, mask)
    assert loss.isfinite() and two_step_loss(on_box, labels, boxes, has_boxes, mask) < loss
    loss.backward()


def test_metrics_on_a_hand_computed_example():
    predictions = torch.tensor([[4, BLANK], [4, BLANK], [8, BLANK], [4, 9]])
    labels = torch.tensor([[4, BLANK], [8, BLANK], [8, BLANK], [4, 9]])
    table = per_number(predictions, labels)
    assert table["4"]["accuracy"] == 1 and table["8"]["accuracy"] == 0.5 and macro(table) == pytest.approx(2.5 / 3)
    low, high = wilson_interval(1, 1)
    assert low == pytest.approx(0.21, abs=0.01) and high == 1
    prefix = torch.tensor([[[4, BLANK], [8, BLANK], [8, BLANK]]])
    assert count_flips(prefix, torch.tensor([3]))["flips"] == 1


def test_paired_test_track_bootstrap_and_match_of():
    assert mcnemar_exact(0, 0) == 1.0 and mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 2 ** 10)  # all ten disagreements one way
    assert mcnemar_exact(3, 1) == mcnemar_exact(1, 3) == pytest.approx(2 * 5 / 16)
    correct = [1, 1, 1, 1, 0, 0, 1, 0]
    groups = ["a", "a", "a", "a", "b", "b", "c", "c"]  # whole tracks are resampled
    low, high = group_bootstrap(correct, groups, resamples=500)
    assert 0 <= low <= sum(correct) / len(correct) <= high <= 1 and low < high
    assert group_bootstrap([1, 1], ["a", "b"]) == (1.0, 1.0)
    assert track_of("49/1003_15408/2") == "1003_15408" and match_of("49/1003_15408/2") == "1003"


def test_parseq_reads_only_the_number_area_and_end_in_position_2_is_blank():
    model = JerseyModel("parseq", "pool", max_frames=4, pretrained=False).eval()
    images = torch.randn(2, 3, *IMAGE_SIZE)
    changed = images.clone()
    changed[..., :20, :] = torch.randn_like(changed[..., :20, :])  # the head, above the number area
    with torch.no_grad():
        assert torch.allclose(model.read(images)["logits"], model.read(changed)["logits"], atol=1e-5)
        tokens = torch.zeros(1, 2, model.reader.token_dim)
        model.reader.text.head.bias.zero_()
        model.reader.text.head.bias[0] = 10.0  # the end token wins everywhere
        logits = model.reader.logits(tokens)
    assert predict_digits(logits)[0][0, 1].item() == BLANK and logits[0, 0, BLANK] == float("-inf")
