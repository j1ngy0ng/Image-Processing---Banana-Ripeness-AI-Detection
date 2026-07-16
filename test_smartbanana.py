from pathlib import Path

import cv2
import numpy as np

from smartbanana import assess_image, create_pdf, grade_quality, load_history, mask_from_drawing, save_record


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


def test_grade_boundaries():
    assert grade_quality(4.99) == "A"
    assert grade_quality(5) == "B"
    assert grade_quality(9.99) == "B"
    assert grade_quality(10) == "C"


def test_history_and_pdf(tmp_path: Path):
    result = assess_image(banana((0, 215, 245), (30, (10, 40, 60))), "sample.jpg")
    database = tmp_path / "history.db"
    save_record(result.record, database)
    history = load_history(database)
    assert len(history) == 1
    assert history[0]["file_name"] == "sample.jpg"
    pdf = create_pdf(result)
    assert pdf.startswith(b"%PDF") and len(pdf) > 5_000


def test_manual_boundary_overrides_automatic_segmentation():
    image = np.full((400, 600, 3), 245, np.uint8)
    cv2.rectangle(image, (20, 20), (300, 190), (45, 165, 90), -1)
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
