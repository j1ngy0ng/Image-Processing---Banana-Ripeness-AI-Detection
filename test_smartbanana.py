import re
from pathlib import Path

import cv2
import numpy as np
import pytest

from interactive_overlay import _thumbnail_images
from smartbanana import (
    AnalysisError,
    assess_image,
    classify_ripeness,
    create_pdf,
    create_pdf_report,
    grade_quality,
    load_history,
    mask_from_drawing,
    save_record,
)


def banana(colour, defect=None):
    image = np.full((400, 600, 3), 245, np.uint8)
    cv2.ellipse(image, (300, 210), (210, 75), -8, 15, 345, colour, 70)
    if defect:
        cv2.circle(image, (315, 235), defect[0], defect[1], -1)
    return image


def test_ripeness_and_evidence():
    unripe = assess_image(banana((45, 165, 90)), "green.jpg")
    ripe = assess_image(banana((0, 215, 245)), "yellow.jpg")
    spoiled = assess_image(banana((20, 65, 105)), "brown.jpg")
    assert unripe.record["ripeness_stage"] == "Unripe"
    assert ripe.record["ripeness_stage"] == "Ripe"
    assert spoiled.record["ripeness_stage"] == "Spoiled"
    assert cv2.countNonZero(ripe.banana_mask) > 10_000
    assert ripe.overlay.shape == ripe.original.shape
    assert ripe.region_map.shape == ripe.banana_mask.shape
    assert set(np.unique(ripe.region_map)).issubset({0, 1, 2, 3})


def test_hover_evidence_is_a_compact_thumbnail():
    overlay = np.zeros((800, 400, 3), np.uint8)
    region_map = np.tile(np.arange(4, dtype=np.uint8), (800, 100))

    thumbnail, thumbnail_map = _thumbnail_images(overlay, region_map)

    assert thumbnail.shape == (240, 120, 3)
    assert thumbnail_map.shape == (240, 120)
    assert set(np.unique(thumbnail_map)) == {0, 1, 2, 3}


def test_grade_boundaries():
    assert grade_quality(19.99) == "A"
    assert grade_quality(20) == "B"
    assert grade_quality(50) == "B"
    assert grade_quality(50.01) == "C"


def test_published_bbch_ripeness_boundaries():
    assert classify_ripeness(51, 49, 0) == "Unripe"
    assert classify_ripeness(49, 51, 0) == "Ripe"
    assert classify_ripeness(0, 80, 20) == "Overripe"
    assert classify_ripeness(0, 50, 50) == "Overripe"
    assert classify_ripeness(0, 49.99, 50.01) == "Spoiled"


@pytest.mark.parametrize("colour", [(0, 0, 255), (0, 255, 255), (0, 255, 0)])
def test_rejects_full_frame_colour_as_a_banana(colour):
    image = np.full((300, 400, 3), colour, np.uint8)

    with pytest.raises(AnalysisError, match="No clear banana was detected"):
        assess_image(image)


def test_rejects_simple_coloured_shapes_as_bananas():
    image = np.full((400, 600, 3), 245, np.uint8)
    cv2.rectangle(image, (100, 100), (500, 300), (0, 215, 245), -1)

    with pytest.raises(AnalysisError, match="No clear banana was detected"):
        assess_image(image)


def test_history_and_pdf(tmp_path: Path):
    result = assess_image(banana((0, 215, 245), (30, (10, 40, 60))), "sample.jpg")
    database = tmp_path / "history.db"
    save_record(result.record, database)
    history = load_history(database)
    assert len(history) == 1
    assert history[0]["file_name"] == "sample.jpg"
    pdf = create_pdf(result)
    assert pdf.startswith(b"%PDF") and len(pdf) > 5_000
    assert len(re.findall(rb"/Type\s*/Page\b", pdf)) == 1
    combined_pdf = create_pdf_report([result, result], "Batch Test Report")
    assert len(re.findall(rb"/Type\s*/Page\b", combined_pdf)) == 3


def test_manual_boundary_overrides_automatic_segmentation():
    image = np.full((400, 600, 3), 245, np.uint8)
    cv2.ellipse(image, (205, 155), (170, 65), -8, 15, 345, (45, 165, 90), 70)
    cv2.ellipse(image, (430, 280), (125, 55), 0, 0, 360, (0, 215, 245), -1)
    automatic = assess_image(image)

    drawing = np.zeros((200, 300, 4), np.uint8)
    alpha = np.zeros((200, 300), np.uint8)
    cv2.ellipse(alpha, (215, 140), (63, 28), 0, 0, 360, 255, -1)
    drawing[:, :, 3] = alpha
    manual_mask = mask_from_drawing(drawing, image.shape[:2])
    corrected = assess_image(image, manual_mask=manual_mask)

    assert automatic.record["ripeness_stage"] == "Unripe"
    assert corrected.record["ripeness_stage"] == "Ripe"
    assert corrected.record["fruit_area"] < automatic.record["fruit_area"]
