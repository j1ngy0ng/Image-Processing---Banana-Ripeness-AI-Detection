"""Classical image-processing pipeline for the SmartBanana system."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class AnalysisError(ValueError):
    """Raised when an input cannot be analysed reliably."""


@dataclass
class Analysis:
    record: dict[str, Any]
    original: np.ndarray
    enhanced: np.ndarray
    banana_mask: np.ndarray
    defect_mask: np.ndarray
    overlay: np.ndarray


def decode_image(data: bytes) -> np.ndarray:
    if not data:
        raise AnalysisError("The uploaded file is empty.")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3:
        raise AnalysisError("The file is not a valid JPG, JPEG, or PNG image.")
    return image


def _resize(image: np.ndarray, maximum: int = 1024) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, maximum / max(height, width))
    if scale == 1:
        return image.copy()
    return cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


def _preprocess(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
    return cv2.cvtColor(cv2.merge((lightness, a, b)), cv2.COLOR_LAB2BGR)


def _largest_mask(mask: np.ndarray) -> np.ndarray:
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise AnalysisError("No usable banana boundary was found.")
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < mask.shape[0] * mask.shape[1] * 0.02:
        raise AnalysisError("The banana boundary is too small. Draw it again or move the banana closer.")
    final_mask = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(final_mask, [contour], -1, 255, cv2.FILLED)
    return final_mask


def _segment(hsv: np.ndarray) -> np.ndarray:
    # Banana peel spans brown/orange/yellow/green; saturation rejects white/grey/black backgrounds.
    mask = cv2.inRange(hsv, (0, 35, 25), (95, 255, 255))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    return _largest_mask(mask)


def mask_from_drawing(drawing: np.ndarray | None, target_shape: tuple[int, int]) -> np.ndarray:
    """Convert the drawable canvas's transparent RGBA layer into a binary mask."""
    if drawing is None or drawing.ndim != 3 or drawing.shape[2] < 4:
        raise AnalysisError("Draw and close a boundary before applying it.")
    mask = (drawing[:, :, 3] > 10).astype(np.uint8) * 255
    mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return _largest_mask(mask)


def _percentage(binary: np.ndarray, fruit_area: int) -> float:
    return round(100 * cv2.countNonZero(binary) / fruit_area, 2)


def grade_quality(defect_percentage: float) -> str:
    if defect_percentage < 5:
        return "A"
    if defect_percentage < 10:
        return "B"
    return "C"


def assess_image(
    image: np.ndarray,
    file_name: str = "image.jpg",
    input_type: str = "image",
    manual_mask: np.ndarray | None = None,
) -> Analysis:
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise AnalysisError("A three-channel colour image is required.")

    started = time.perf_counter()
    original = _resize(image)
    enhanced = _preprocess(original)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    if manual_mask is None:
        banana_mask = _segment(hsv)
    else:
        manual_mask = cv2.resize(np.asarray(manual_mask), (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
        banana_mask = _largest_mask(manual_mask)
    fruit_area = cv2.countNonZero(banana_mask)

    h, s, v = cv2.split(hsv)
    inside = banana_mask > 0
    green = ((h >= 36) & (h <= 95) & (s >= 40) & (v >= 30) & inside).astype(np.uint8) * 255
    brown_pixels = ((h < 18) & (s >= 40) & (v >= 20) & (v <= 140) & inside) | ((v < 55) & inside)
    yellow_pixels = (h >= 10) & (h <= 35) & (s >= 40) & (v >= 80) & inside & ~brown_pixels
    yellow = yellow_pixels.astype(np.uint8) * 255
    brown = brown_pixels.astype(np.uint8) * 255

    green_pct = _percentage(green, fruit_area)
    yellow_pct = _percentage(yellow, fruit_area)
    brown_pct = _percentage(brown, fruit_area)

    dark = ((((v < 105) & (s >= 25)) | ((h < 18) & (s >= 55) & (v < 170))) & inside).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    defect_mask = np.zeros_like(dark)
    minimum_area = max(20, round(fruit_area * 0.0005))
    defect_count = 0
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum_area:
            defect_mask[labels == label] = 255
            defect_count += 1
    defect_area = cv2.countNonZero(defect_mask)
    defect_pct = round(100 * defect_area / fruit_area, 2)

    # Calibrated on clean, source-grouped dataset labels; augmented copies stay together.
    if green_pct >= 20:
        ripeness = "Unripe"
    elif defect_pct >= 60 or brown_pct >= 50 or (defect_pct >= 35 and brown_pct >= 35):
        ripeness = "Spoiled"
    elif yellow_pct >= 65 and defect_pct < 25:
        ripeness = "Ripe"
    elif defect_pct >= 25 or brown_pct >= 15:
        ripeness = "Overripe"
    else:
        ripeness = "Ripe"

    grade = grade_quality(defect_pct)
    shelf_life = max(0, {"Unripe": 6, "Ripe": 3, "Overripe": 1, "Spoiled": 0}[ripeness] - {"A": 0, "B": 1, "C": 2}[grade])
    if ripeness == "Spoiled" or grade == "C":
        recommendation = "Use for food processing or composting; discard if mouldy or unsafe."
    elif grade == "B" or ripeness == "Overripe":
        recommendation = "Consume early or offer for quick, discounted sale."
    elif ripeness == "Unripe":
        recommendation = "Store at room temperature until ripe; keep away from direct heat."
    else:
        recommendation = "Ready for normal sale or consumption."

    scores = sorted((green_pct, yellow_pct, brown_pct), reverse=True)
    coverage = min(100.0, sum(scores))
    confidence = round(min(99.0, 55 + 0.3 * (scores[0] - scores[1]) + 0.15 * coverage), 1)

    overlay = original.copy()
    contours, _ = cv2.findContours(banana_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 255), 3)
    overlay[defect_mask > 0] = (30, 30, 230)

    now = datetime.now().astimezone()
    record = {
        "record_id": now.strftime("%Y%m%d%H%M%S%f"),
        "file_name": file_name,
        "input_type": input_type,
        "date_time": now.isoformat(timespec="seconds"),
        "image_width": original.shape[1],
        "image_height": original.shape[0],
        "fruit_area": fruit_area,
        "green_pixels": cv2.countNonZero(green),
        "yellow_pixels": cv2.countNonZero(yellow),
        "brown_pixels": cv2.countNonZero(brown),
        "green_percentage": green_pct,
        "yellow_percentage": yellow_pct,
        "brown_percentage": brown_pct,
        "ripeness_stage": ripeness,
        "defect_count": defect_count,
        "defect_area": defect_area,
        "defect_percentage": defect_pct,
        "quality_grade": grade,
        "shelf_life_days": shelf_life,
        "recommendation": recommendation,
        "confidence": confidence,
        "processing_time": round(time.perf_counter() - started, 3),
        "status": "success",
    }
    return Analysis(record, original, enhanced, banana_mask, defect_mask, overlay)


def analyse_bytes(data: bytes, file_name: str = "image.jpg", input_type: str = "image") -> Analysis:
    return assess_image(decode_image(data), file_name, input_type)


_FIELDS = (
    "record_id file_name input_type date_time image_width image_height fruit_area green_pixels "
    "yellow_pixels brown_pixels green_percentage yellow_percentage brown_percentage ripeness_stage "
    "defect_count defect_area defect_percentage quality_grade shelf_life_days recommendation confidence "
    "processing_time status"
).split()


def _connect(db_path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(db_path)
    columns = ", ".join(f"{name} TEXT" for name in _FIELDS)
    connection.execute(f"CREATE TABLE IF NOT EXISTS analyses ({columns}, PRIMARY KEY (record_id))")
    return connection


def save_record(record: dict[str, Any], db_path: str | Path = "smartbanana.db") -> None:
    with _connect(db_path) as connection:
        names = ", ".join(_FIELDS)
        placeholders = ", ".join("?" for _ in _FIELDS)
        connection.execute(
            f"INSERT OR REPLACE INTO analyses ({names}) VALUES ({placeholders})",
            [record.get(name, "") for name in _FIELDS],
        )


def load_history(db_path: str | Path = "smartbanana.db") -> list[dict[str, Any]]:
    with _connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        return [dict(row) for row in connection.execute("SELECT * FROM analyses ORDER BY date_time DESC")]


def clear_history(db_path: str | Path = "smartbanana.db") -> None:
    with _connect(db_path) as connection:
        connection.execute("DELETE FROM analyses")


def _png(image: np.ndarray) -> BytesIO:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise AnalysisError("A report image could not be encoded.")
    return BytesIO(encoded.tobytes())


def create_pdf(analysis: Analysis) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image as ReportImage
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    output = BytesIO()
    document = SimpleDocTemplate(output, pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("SmartTitle", parent=styles["Title"], textColor=colors.HexColor("#365314"), alignment=TA_CENTER)
    heading = ParagraphStyle("SmartHeading", parent=styles["Heading2"], textColor=colors.HexColor("#4d7c0f"))
    cell = ParagraphStyle("Cell", parent=styles["BodyText"], fontSize=8, leading=9, wordWrap="CJK")
    label = ParagraphStyle("Label", parent=cell, fontName="Helvetica-Bold")
    r = analysis.record
    story = [Paragraph("SmartBanana Inspection Report", title), Spacer(1, 4 * mm), Paragraph(f"Generated {r['date_time']}", styles["BodyText"]), Spacer(1, 5 * mm)]
    def row(left_label, left_value, right_label, right_value):
        return [Paragraph(str(left_label), label), Paragraph(str(left_value), cell), Paragraph(str(right_label), label), Paragraph(str(right_value), cell)]

    summary = [
        row("File", r["file_name"], "Status", r["ripeness_stage"]),
        row("Quality grade", r["quality_grade"], "Confidence", f"{r['confidence']}%"),
        row("Defect area", f"{r['defect_percentage']}%", "Shelf life", f"{r['shelf_life_days']} day(s)"),
        row("Processing time", f"{r['processing_time']} s", "Defects", r["defect_count"]),
    ]
    table = Table(summary, colWidths=[27 * mm, 58 * mm, 27 * mm, 47 * mm])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ecfccb")), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#a3a3a3")), ("FONTNAME", (0, 0), (-1, -1), "Helvetica"), ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("PADDING", (0, 0), (-1, -1), 6)]))
    story += [table, Spacer(1, 5 * mm), Paragraph("Recommendation", heading), Paragraph(r["recommendation"], styles["BodyText"]), Spacer(1, 5 * mm)]
    colour_rows = [["Peel colour", "Percentage"], ["Green", f"{r['green_percentage']}%"], ["Yellow", f"{r['yellow_percentage']}%"], ["Brown / dark", f"{r['brown_percentage']}%"]]
    colours_table = Table(colour_rows, colWidths=[80 * mm, 50 * mm])
    colours_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#365314")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("ALIGN", (1, 1), (1, -1), "RIGHT"), ("PADDING", (0, 0), (-1, -1), 6)]))
    story += [colours_table, PageBreak(), Paragraph("Processing evidence", heading)]
    images = [("Original image", analysis.original), ("CLAHE enhanced image", analysis.enhanced), ("Banana segmentation mask", analysis.banana_mask), ("Detected defects", analysis.overlay)]
    image_rows = []
    for index in range(0, len(images), 2):
        row = []
        for caption, image in images[index:index + 2]:
            picture = ReportImage(_png(image), width=78 * mm, height=58 * mm, kind="proportional")
            row.append([picture, Paragraph(caption, ParagraphStyle("Caption", parent=styles["BodyText"], alignment=TA_CENTER, fontSize=8))])
        image_rows.append(row)
    evidence = Table(image_rows, colWidths=[84 * mm, 84 * mm])
    evidence.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("PADDING", (0, 0), (-1, -1), 4)]))
    story += [evidence, Spacer(1, 5 * mm), Paragraph("Method", heading), Paragraph("The result uses Gaussian filtering, CLAHE enhancement, HSV thresholding, morphological opening and closing, largest-contour isolation, colour percentage analysis, and connected-component defect measurement. No machine-learning model is used.", styles["BodyText"])]
    document.build(story)
    return output.getvalue()
