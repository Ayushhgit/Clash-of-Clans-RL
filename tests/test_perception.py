import numpy as np
import torch

from ai.perception.dataset import _centernet_targets
from ai.perception.detector import (
    Detector,
    average_precision,
    box_iou,
    decode,
    detector_loss,
)
from ai.perception.pipeline import NAME_TO_SIM, HUDRegions, PerceptionConfig, Perceiver
from ai.perception.ui import DigitOCR, UIClassifier, parse_number, segment_glyphs
from annotation.app import CLASSES
from simulator.env import N_GLOBALS, N_TOKENS, TOKEN_DIM, XY_BINS


def test_detector_output_shapes():
    d = Detector(width=8)
    out = d(torch.rand(2, 3, 128, 128))
    assert out["hm"].shape == (2, len(CLASSES), 32, 32)
    assert out["wh"].shape == (2, 2, 32, 32)
    assert out["off"].shape == (2, 2, 32, 32)


def test_centernet_targets_place_peaks_at_centres():
    boxes = np.array([[40, 40, 80, 80]], dtype=np.float32)
    labels = np.array([3])
    hm, wh, off, mask, ind = _centernet_targets(boxes, labels, out=64, stride=4,
                                               n_classes=len(CLASSES))
    peak = np.unravel_index(int(hm[3].argmax()), hm[3].shape)
    assert peak == (15, 15)                 # centre (60, 60) / stride 4
    assert mask[0] == 1.0
    assert np.allclose(wh[0].numpy(), [10.0, 10.0])


def test_decode_recovers_a_planted_box():
    """A hand-built heatmap must decode back to the box that created it."""
    n_c, out_hw, stride = len(CLASSES), 32, 4
    hm = torch.full((1, n_c, out_hw, out_hw), -10.0)
    hm[0, 2, 10, 12] = 10.0                  # logit -> sigmoid ~1
    wh = torch.zeros(1, 2, out_hw, out_hw)
    wh[0, :, 10, 12] = 6.0
    off = torch.zeros(1, 2, out_hw, out_hw)

    det = decode({"hm": hm, "wh": wh, "off": off}, k=5, threshold=0.5, stride=stride)[0]
    assert len(det["boxes"]) == 1
    assert det["labels"][0] == 2
    x1, y1, x2, y2 = det["boxes"][0]
    assert np.allclose([(x1 + x2) / 2, (y1 + y2) / 2], [12 * stride, 10 * stride])
    assert np.allclose([x2 - x1, y2 - y1], [6 * stride, 6 * stride])


def test_detector_loss_is_finite_and_decreases_on_overfit():
    torch.manual_seed(0)
    model = Detector(width=8)
    x = torch.rand(1, 3, 128, 128)
    boxes = np.array([[20, 20, 60, 60]], dtype=np.float32)
    hm, wh, off, mask, ind = _centernet_targets(boxes, np.array([1]), 32, 4, len(CLASSES))
    batch = (hm[None], wh[None], off[None], mask[None], ind[None])

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    first = last = None
    for i in range(25):
        loss, _ = detector_loss(model(x), *batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if i == 0:
            first = float(loss.detach())
        last = float(loss.detach())
    assert np.isfinite(first) and last < first, "detector must overfit one image"


def test_average_precision_is_one_for_perfect_predictions():
    boxes = np.array([[0, 0, 10, 10], [20, 20, 40, 40]], dtype=np.float32)
    labels = np.array([0, 1])
    preds = [{"boxes": boxes, "labels": labels, "scores": np.array([0.9, 0.8])}]
    gts = [{"boxes": boxes, "labels": labels}]
    assert average_precision(preds, gts)["mAP"] > 0.99


def test_average_precision_is_zero_when_classes_are_wrong():
    boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    preds = [{"boxes": boxes, "labels": np.array([5]), "scores": np.array([0.9])}]
    gts = [{"boxes": boxes, "labels": np.array([0])}]
    assert average_precision(preds, gts)["mAP"] == 0.0


def test_box_iou_identity_and_disjoint():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=np.float32)
    iou = box_iou(a, b)
    assert np.isclose(iou[0, 0], 1.0) and iou[0, 1] == 0.0


def test_ui_classifier_predicts_a_known_state():
    ui = UIClassifier(width=8)
    names, conf = ui.predict(torch.rand(3, 3, 224, 224))
    assert len(names) == 3 and all(0.0 <= c <= 1.0 for c in conf)


def test_parse_number_handles_game_formats():
    assert parse_number("1,250,000") == 1_250_000
    assert parse_number("900000") == 900_000
    assert parse_number("1.2M") == 1_200_000
    assert parse_number("35K") == 35_000
    assert parse_number("") is None and parse_number("--") is None


def test_glyph_segmentation_splits_digits():
    strip = np.zeros((16, 40), dtype=np.float32)
    for x0 in (2, 12, 22):                    # three 6px "glyphs" with gaps
        strip[4:12, x0:x0 + 6] = 1.0
    glyphs = segment_glyphs(strip)
    assert len(glyphs) == 3


def test_ocr_reads_a_string_of_the_right_length():
    strip = np.zeros((24, 40), dtype=np.float32)
    for x0 in (2, 12, 22):
        strip[4:20, x0:x0 + 6] = 1.0
    ocr = DigitOCR()
    text = ocr.read(strip)
    assert isinstance(text, str) and len(text.strip()) <= 3


def test_pipeline_emits_the_simulator_observation_schema():
    p = Perceiver(detector=Detector(width=8), ui_classifier=UIClassifier(width=8),
                  cfg=PerceptionConfig(img_size=128,
                                       hud=HUDRegions(battle_area=(0, 0, 320, 320))))
    obs = p.perceive(np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8))
    assert obs["tokens"].shape == (N_TOKENS, TOKEN_DIM)
    assert obs["class_ids"].shape == (N_TOKENS,)
    assert obs["globals"].shape == (N_GLOBALS,)
    assert obs["action_mask"]["xy"].shape == (XY_BINS, XY_BINS)
    assert obs["action_mask"]["xy"].any(), "there must always be a legal deploy cell"


def test_pipeline_tokens_are_in_range_and_padded_with_zeros():
    p = Perceiver(detector=Detector(width=8), cfg=PerceptionConfig(img_size=128))
    obs = p.perceive(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    tok, mask = obs["tokens"], obs["token_mask"]
    assert np.all(tok[mask][:, 0:2] >= 0.0) and np.all(tok[mask][:, 0:2] <= 1.0)
    assert np.all(tok[~mask] == 0.0)


def test_annotation_classes_map_into_the_simulator_vocabulary():
    """Every detector class is either a battlefield entity with a simulator
    equivalent, or explicitly declared to be UI chrome. Nothing may be
    silently dropped between perception and the policy."""
    from ai.perception.pipeline import UI_CLASS_NAMES

    unmapped = {c for c in CLASSES if c not in NAME_TO_SIM and c not in UI_CLASS_NAMES}
    assert not unmapped, f"perception classes with no simulator meaning: {unmapped}"
    overlap = set(NAME_TO_SIM) & UI_CLASS_NAMES
    assert not overlap, f"class is both an entity and UI: {overlap}"
