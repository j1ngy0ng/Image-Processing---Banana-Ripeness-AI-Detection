"""Classical image-processing pipeline for the SmartBanana system."""

from __future__ import annotations

import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from statistics import fmean
from typing import Any, Sequence
from xml.sax.saxutils import escape

import cv2
import numpy as np


_CLEAR_BANANA_MESSAGE = (
    "No clear banana was detected. Upload or capture a clearer image with one "
    "banana on a plain, contrasting background."
)
# ponytail: geometry keeps validation dependency-free; use a trained detector only
# if evaluation data shows these conservative limits no longer generalise.
_MIN_BOUNDARY_RATIO = 0.02
_MAX_BOUNDARY_RATIO = 0.90
_MAX_CIRCULARITY = 0.82
_MAX_RECTANGULARITY = 0.95

# Published BBCH thresholds for edible Musaceae (Gonzales et al., 2002):
# stage 8 has 20-50% brown surface; stage 9 has more than 50% brown surface.
# Source: https://www.openagrar.de/servlets/MCRFileNodeServlet/openagrar_derivate_00010428/BBCH-Skala_en.pdf
_BROWN_OVER_RIPE_MIN = 20.0
_BROWN_SPOILED_MIN = 50.0

# FAO ambient postharvest-life ranges. The cited guidance does not provide a
# defensible numeric remaining-life value after the fruit becomes overripe.
# Source: https://www.fao.org/4/t0073e/T0073E07.htm
_SHELF_LIFE = {
    "Unripe": "4-10 days",
    "Ripe": "2-4 days",
    "Overripe": "Consume immediately",
    "Spoiled": "Not suitable for storage",
}


class AnalysisError(ValueError):
    """Raised when an input cannot be analysed reliably."""


@dataclass
class Analysis:
    """Complete output required by the interface, database, and PDF report."""

    record: dict[str, Any]
    original: np.ndarray
    enhanced: np.ndarray
    banana_mask: np.ndarray
    defect_mask: np.ndarray
    overlay: np.ndarray
    region_map: np.ndarray


# Image preparation and banana segmentation


def decode_image(data: bytes) -> np.ndarray:
    """Decode an uploaded image and reject empty or unsupported content."""
    if not data:
        raise AnalysisError("The uploaded file is empty.")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3:
        raise AnalysisError("The file is not a valid JPG, JPEG, or PNG image.")
    return image


def _resize(image: np.ndarray, maximum: int = 1024) -> np.ndarray:
    """Downscale large images while preserving aspect ratio and smaller inputs."""
    height, width = image.shape[:2]
    scale = min(1.0, maximum / max(height, width))
    if scale == 1:
        return image.copy()
    return cv2.resize(
        image,
        (round(width * scale), round(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def _preprocess(image: np.ndarray) -> np.ndarray:
    """Reduce noise and improve local contrast without distorting peel colour."""
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
    return cv2.cvtColor(cv2.merge((lightness, a, b)), cv2.COLOR_LAB2BGR)


def _is_plausible_banana(contour: np.ndarray, image_area: int) -> bool:
    """Reject full-frame backgrounds and simple circular or rectangular objects."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    rectangle_width, rectangle_height = cv2.minAreaRect(contour)[1]
    rectangle_area = rectangle_width * rectangle_height

    circularity = 4 * np.pi * area / max(perimeter**2, 1)
    rectangularity = area / max(rectangle_area, 1)
    return (
        area / image_area <= _MAX_BOUNDARY_RATIO
        and circularity <= _MAX_CIRCULARITY
        and rectangularity <= _MAX_RECTANGULARITY
    )


def _largest_mask(mask: np.ndarray, *, validate_shape: bool = False) -> np.ndarray:
    """Return the largest valid filled contour from a binary candidate mask."""
    mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise AnalysisError(_CLEAR_BANANA_MESSAGE)

    contour = max(contours, key=cv2.contourArea)
    image_area = mask.shape[0] * mask.shape[1]
    if cv2.contourArea(contour) < image_area * _MIN_BOUNDARY_RATIO:
        raise AnalysisError(
            "The banana boundary is too small. Draw it again or move the banana closer."
        )
    if validate_shape and not _is_plausible_banana(contour, image_area):
        raise AnalysisError(_CLEAR_BANANA_MESSAGE)

    final_mask = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(final_mask, [contour], -1, 255, cv2.FILLED)
    return final_mask


def _segment(hsv: np.ndarray) -> np.ndarray:
    """Create and clean the automatic banana-colour mask in HSV space."""
    # Banana peel spans brown/orange/yellow/green; saturation rejects white/grey/black backgrounds.
    mask = cv2.inRange(hsv, (0, 35, 25), (95, 255, 255))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    return _largest_mask(mask, validate_shape=True)


def mask_from_drawing(
    drawing: np.ndarray | None, target_shape: tuple[int, int]
) -> np.ndarray:
    """Convert the drawable canvas's transparent RGBA layer into a binary mask."""
    if drawing is None or drawing.ndim != 3 or drawing.shape[2] < 4:
        raise AnalysisError("Draw and close a boundary before applying it.")
    mask = (drawing[:, :, 3] > 10).astype(np.uint8) * 255
    mask = cv2.resize(
        mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST
    )
    return _largest_mask(mask)


# Feature measurement and decision rules


def _percentage(binary: np.ndarray, fruit_area: int) -> float:
    return round(100 * cv2.countNonZero(binary) / fruit_area, 2)


def grade_quality(defect_percentage: float) -> str:
    """Map visible defects to project grades using BBCH surface boundaries."""
    if defect_percentage < _BROWN_OVER_RIPE_MIN:
        return "A"
    if defect_percentage <= _BROWN_SPOILED_MIN:
        return "B"
    return "C"


def classify_ripeness(
    green_percentage: float, yellow_percentage: float, brown_percentage: float
) -> str:
    """Collapse the published nine-stage BBCH surface scale into four outcomes."""
    if brown_percentage > _BROWN_SPOILED_MIN:
        return "Spoiled"
    if brown_percentage >= _BROWN_OVER_RIPE_MIN:
        return "Overripe"
    return "Unripe" if green_percentage > yellow_percentage else "Ripe"


def assess_image(
    image: np.ndarray,
    file_name: str = "image.jpg",
    input_type: str = "image",
    manual_mask: np.ndarray | None = None,
) -> Analysis:
    """Run the complete explainable assessment pipeline on one BGR image."""
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise AnalysisError("A three-channel colour image is required.")

    started = time.perf_counter()
    original = _resize(image)
    enhanced = _preprocess(original)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    # Manual masks intentionally bypass automatic shape checks: the user has
    # explicitly identified the region, but minimum-area validation still applies.
    if manual_mask is None:
        banana_mask = _segment(hsv)
    else:
        manual_mask = cv2.resize(
            np.asarray(manual_mask),
            (original.shape[1], original.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        banana_mask = _largest_mask(manual_mask)
    fruit_area = cv2.countNonZero(banana_mask)

    h, s, v = cv2.split(hsv)
    inside = banana_mask > 0
    # Every colour measurement is restricted to the selected banana region.
    green = ((h >= 36) & (h <= 95) & (s >= 40) & (v >= 30) & inside).astype(
        np.uint8
    ) * 255
    brown_pixels = ((h < 18) & (s >= 40) & (v >= 20) & (v <= 140) & inside) | (
        (v < 55) & inside
    )
    yellow_pixels = (
        (h >= 10) & (h <= 35) & (s >= 40) & (v >= 80) & inside & ~brown_pixels
    )
    yellow = yellow_pixels.astype(np.uint8) * 255
    brown = brown_pixels.astype(np.uint8) * 255

    green_pct = _percentage(green, fruit_area)
    yellow_pct = _percentage(yellow, fruit_area)
    brown_pct = _percentage(brown, fruit_area)

    # Connected-component filtering removes isolated dark pixels caused by noise.
    dark = (
        (((v < 105) & (s >= 25)) | ((h < 18) & (s >= 55) & (v < 170))) & inside
    ).astype(np.uint8) * 255
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

    # Map the measured surface to the published BBCH Musaceae maturity scale:
    # stages 1-3 are green-dominant, stages 4-7 are yellow-dominant, stage 8
    # has 20-50% brown surface, and stage 9 has more than 50% brown surface.
    ripeness = classify_ripeness(green_pct, yellow_pct, brown_pct)

    grade = grade_quality(defect_pct)
    shelf_life = _SHELF_LIFE[ripeness]
    if ripeness == "Spoiled" or grade == "C":
        recommendation = (
            "Use for food processing or composting; discard if mouldy or unsafe."
        )
    elif grade == "B" or ripeness == "Overripe":
        recommendation = "Consume early or offer for quick, discounted sale."
    elif ripeness == "Unripe":
        recommendation = (
            "Store at room temperature until ripe; keep away from direct heat."
        )
    else:
        recommendation = "Ready for normal sale or consumption."

    # Confidence is an explainable colour-separation heuristic, not a probability.
    scores = sorted((green_pct, yellow_pct, brown_pct), reverse=True)
    coverage = min(100.0, sum(scores))
    confidence = round(
        min(99.0, 55 + 0.3 * (scores[0] - scores[1]) + 0.15 * coverage), 1
    )

    # Create a readable evidence overlay and a compact label map for the
    # interface's per-pixel hover explanation. Defects take visual priority.
    overlay = original.copy()
    region_map = np.zeros(banana_mask.shape, np.uint8)

    def tint(mask: np.ndarray, colour: tuple[int, int, int]) -> None:
        selected = mask > 0
        overlay[selected] = np.clip(
            original[selected] * 0.4 + np.asarray(colour) * 0.6,
            0,
            255,
        ).astype(np.uint8)

    tint(green, (80, 190, 40))
    tint(yellow, (0, 215, 255))
    tint(defect_mask, (30, 30, 230))
    region_map[green > 0] = 1
    region_map[yellow > 0] = 2
    region_map[defect_mask > 0] = 3
    contours, _ = cv2.findContours(
        banana_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)

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
    return Analysis(
        record, original, enhanced, banana_mask, defect_mask, overlay, region_map
    )


def analyse_bytes(
    data: bytes, file_name: str = "image.jpg", input_type: str = "image"
) -> Analysis:
    """Decode uploaded bytes and run the shared image-assessment pipeline."""
    return assess_image(decode_image(data), file_name, input_type)


# SQLite persistence


_FIELDS = (
    "record_id file_name input_type date_time image_width image_height fruit_area green_pixels "
    "yellow_pixels brown_pixels green_percentage yellow_percentage brown_percentage ripeness_stage "
    "defect_count defect_area defect_percentage quality_grade shelf_life_days recommendation confidence "
    "processing_time status"
).split()


def _connect(db_path: str | Path) -> sqlite3.Connection:
    """Open the history database and create its schema on first use."""
    connection = sqlite3.connect(db_path)
    columns = ", ".join(f"{name} TEXT" for name in _FIELDS)
    connection.execute(
        f"CREATE TABLE IF NOT EXISTS analyses ({columns}, PRIMARY KEY (record_id))"
    )
    return connection


def save_record(record: dict[str, Any], db_path: str | Path = "smartbanana.db") -> None:
    """Insert or replace one successful analysis record."""
    with _connect(db_path) as connection:
        names = ", ".join(_FIELDS)
        placeholders = ", ".join("?" for _ in _FIELDS)
        connection.execute(
            f"INSERT OR REPLACE INTO analyses ({names}) VALUES ({placeholders})",
            [record.get(name, "") for name in _FIELDS],
        )


def load_history(db_path: str | Path = "smartbanana.db") -> list[dict[str, Any]]:
    """Return saved records in reverse chronological order."""
    with _connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        return [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM analyses ORDER BY date_time DESC"
            )
        ]


def clear_history(db_path: str | Path = "smartbanana.db") -> None:
    """Delete all saved analysis records while retaining the database schema."""
    with _connect(db_path) as connection:
        connection.execute("DELETE FROM analyses")


# PDF report generation


def _png(image: np.ndarray) -> BytesIO:
    """Encode an OpenCV image as an in-memory PNG for ReportLab."""
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise AnalysisError("A report image could not be encoded.")
    return BytesIO(encoded.tobytes())


def create_pdf_report(
    analyses: Sequence[Analysis],
    report_title: str = "SmartBanana Analysis Report",
) -> bytes:
    """Build a PDF with an overall page followed by one page per analysis."""
    if not analyses:
        raise AnalysisError(
            "At least one completed analysis is required for a PDF report."
        )

    # Import ReportLab lazily because most analyses are viewed without a PDF export.
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Image as ReportImage
    from reportlab.platypus import (
        KeepTogether,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    output = BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=landscape(A4),
        rightMargin=12 * mm,
        leftMargin=12 * mm,
        topMargin=11 * mm,
        bottomMargin=12 * mm,
        title=report_title,
        author="SmartBanana",
    )
    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "SmartTitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#365314"),
        alignment=TA_CENTER,
        fontSize=18,
        leading=20,
        spaceAfter=2 * mm,
    )
    subtitle = ParagraphStyle(
        "SmartSubtitle",
        parent=styles["BodyText"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#525252"),
        fontSize=8,
        leading=10,
    )
    heading = ParagraphStyle(
        "SmartHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#4d7c0f"),
        fontSize=12,
        leading=14,
        spaceAfter=2 * mm,
    )
    cell = ParagraphStyle(
        "Cell", parent=styles["BodyText"], fontSize=8, leading=9, wordWrap="CJK"
    )
    label = ParagraphStyle("Label", parent=cell, fontName="Helvetica-Bold")
    caption = ParagraphStyle(
        "Caption", parent=cell, alignment=TA_CENTER, fontName="Helvetica-Bold"
    )
    note = ParagraphStyle(
        "Note", parent=cell, fontSize=7, leading=8, textColor=colors.HexColor("#525252")
    )

    def paragraph(value: Any, style: ParagraphStyle = cell) -> Paragraph:
        return Paragraph(escape(str(value)), style)

    def report_image(image: np.ndarray) -> ReportImage:
        height, width = image.shape[:2]
        scale = min(82 * mm / width, 51 * mm / height)
        return ReportImage(_png(image), width=width * scale, height=height * scale)

    def footer(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#737373"))
        canvas.drawString(
            12 * mm, 6 * mm, "SmartBanana - external visual assessment only"
        )
        canvas.drawRightString(landscape(A4)[0] - 12 * mm, 6 * mm, f"Page {doc.page}")
        canvas.restoreState()

    story = []
    if len(analyses) > 1:
        records = [analysis.record for analysis in analyses]
        total = len(records)
        ripeness_order = ("Unripe", "Ripe", "Overripe", "Spoiled")
        grade_order = ("A", "B", "C")
        ripeness_counts = Counter(record["ripeness_stage"] for record in records)
        grade_counts = Counter(record["quality_grade"] for record in records)
        dominant_ripeness = max(
            ripeness_order, key=lambda value: ripeness_counts[value]
        )
        dominant_grade = max(grade_order, key=lambda value: grade_counts[value])

        def average(field: str) -> float:
            return round(fmean(float(record[field]) for record in records), 2)

        def distribution_table(
            category: str, values: Sequence[str], counts: Counter
        ) -> Table:
            rows = [
                [
                    paragraph(category, label),
                    paragraph("Count", label),
                    paragraph("Share", label),
                ]
            ]
            rows.extend(
                [
                    paragraph(value),
                    paragraph(counts[value]),
                    paragraph(f"{100 * counts[value] / total:.1f}%"),
                ]
                for value in values
            )
            table = Table(rows, colWidths=[48 * mm, 18 * mm, 22 * mm])
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#365314")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#a3a3a3")),
                        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                        ("PADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
            return table

        overview = Table(
            [
                [
                    paragraph("Successful analyses", label),
                    paragraph(total),
                    paragraph("Dominant ripeness", label),
                    paragraph(dominant_ripeness),
                    paragraph("Dominant quality", label),
                    paragraph(f"Grade {dominant_grade}"),
                    paragraph("Average defect", label),
                    paragraph(f"{average('defect_percentage')}%"),
                ],
                [
                    paragraph("Average confidence", label),
                    paragraph(f"{average('confidence')}%"),
                    paragraph("Average processing", label),
                    paragraph(f"{average('processing_time')} s"),
                    paragraph("Input type", label),
                    paragraph(
                        ", ".join(
                            sorted({str(record["input_type"]) for record in records})
                        )
                    ),
                    paragraph("Report pages", label),
                    paragraph(total + 1),
                ],
            ],
            colWidths=[36 * mm, 32 * mm] * 4,
        )
        overview.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7fee7")),
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#ecfccb")),
                    ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#ecfccb")),
                    ("BACKGROUND", (4, 0), (4, -1), colors.HexColor("#ecfccb")),
                    ("BACKGROUND", (6, 0), (6, -1), colors.HexColor("#ecfccb")),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#a3a3a3")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        averages = Table(
            [
                [
                    paragraph("Average measurement", label),
                    paragraph("Percentage", label),
                ],
                [paragraph("Green peel"), paragraph(f"{average('green_percentage')}%")],
                [
                    paragraph("Yellow peel"),
                    paragraph(f"{average('yellow_percentage')}%"),
                ],
                [
                    paragraph("Brown / dark"),
                    paragraph(f"{average('brown_percentage')}%"),
                ],
                [
                    paragraph("Detected defect"),
                    paragraph(f"{average('defect_percentage')}%"),
                ],
            ],
            colWidths=[58 * mm, 30 * mm],
        )
        averages.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#365314")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#a3a3a3")),
                    ("ALIGN", (1, 1), (1, -1), "RIGHT"),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        aggregate_tables = Table(
            [
                [
                    distribution_table("Ripeness", ripeness_order, ripeness_counts),
                    distribution_table("Quality grade", grade_order, grade_counts),
                    averages,
                ]
            ],
            colWidths=[91 * mm] * 3,
        )
        aggregate_tables.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 1.5 * mm),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 1.5 * mm),
                ]
            )
        )

        story.extend(
            [
                KeepTogether(
                    [
                        paragraph(report_title, title),
                        paragraph(
                            f"Overall analysis of {total} successfully processed items",
                            subtitle,
                        ),
                        Spacer(1, 5 * mm),
                        overview,
                        Spacer(1, 9 * mm),
                        paragraph("Overall distributions and measurements", heading),
                        aggregate_tables,
                        Spacer(1, 9 * mm),
                        paragraph(
                            "The statistics on this page aggregate successful analyses only. "
                            "Each following page contains the evidence and result for one image or sampled video frame.",
                            cell,
                        ),
                        Spacer(1, 4 * mm),
                        paragraph(
                            "Disclaimer: SmartBanana assesses visible external condition only. "
                            "It is not a food-safety or internal-quality test.",
                            note,
                        ),
                    ]
                ),
                PageBreak(),
            ]
        )

    for index, analysis in enumerate(analyses):
        if index:
            story.append(PageBreak())
        r = analysis.record
        summary = Table(
            [
                [
                    paragraph("Ripeness", label),
                    paragraph(r["ripeness_stage"]),
                    paragraph("Quality", label),
                    paragraph(f"Grade {r['quality_grade']}"),
                    paragraph("Defect area", label),
                    paragraph(f"{r['defect_percentage']}%"),
                    paragraph("Confidence", label),
                    paragraph(f"{r['confidence']}%"),
                ],
                [
                    paragraph("Shelf life", label),
                    paragraph(r["shelf_life_days"]),
                    paragraph("Defects", label),
                    paragraph(r["defect_count"]),
                    paragraph("Processing", label),
                    paragraph(f"{r['processing_time']} s"),
                    paragraph("Input", label),
                    paragraph(r["input_type"]),
                ],
            ],
            colWidths=[
                22 * mm,
                44 * mm,
                19 * mm,
                34 * mm,
                24 * mm,
                30 * mm,
                23 * mm,
                76 * mm,
            ],
        )
        summary.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f7fee7")),
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#ecfccb")),
                    ("BACKGROUND", (2, 0), (2, -1), colors.HexColor("#ecfccb")),
                    ("BACKGROUND", (4, 0), (4, -1), colors.HexColor("#ecfccb")),
                    ("BACKGROUND", (6, 0), (6, -1), colors.HexColor("#ecfccb")),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#a3a3a3")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )

        evidence = Table(
            [
                [
                    [
                        report_image(analysis.original),
                        paragraph("Original image", caption),
                    ],
                    [
                        report_image(analysis.banana_mask),
                        paragraph("Banana segmentation mask", caption),
                    ],
                    [
                        report_image(analysis.overlay),
                        paragraph("Colour and defect analysis", caption),
                    ],
                ]
            ],
            colWidths=[91 * mm] * 3,
        )
        evidence.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#d4d4d4")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e5e5e5")),
                    ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                    ("PADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )

        colour_table = Table(
            [
                [
                    paragraph(f"Green peel: {r['green_percentage']}%", label),
                    paragraph(f"Yellow peel: {r['yellow_percentage']}%", label),
                    paragraph(f"Brown / dark: {r['brown_percentage']}%", label),
                    paragraph(f"Detected defect: {r['defect_percentage']}%", label),
                ]
            ],
            colWidths=[68.25 * mm] * 4,
        )
        colour_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#dcfce7")),
                    ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#fef9c3")),
                    ("BACKGROUND", (2, 0), (2, 0), colors.HexColor("#f5e7d6")),
                    ("BACKGROUND", (3, 0), (3, 0), colors.HexColor("#fee2e2")),
                    ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#a3a3a3")),
                    ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#a3a3a3")),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        recommendation = Table(
            [[paragraph("Recommendation", label), paragraph(r["recommendation"])]],
            colWidths=[31 * mm, 242 * mm],
        )
        recommendation.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#365314")),
                    ("TEXTCOLOR", (0, 0), (0, 0), colors.white),
                    ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#ecfccb")),
                    ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#a3a3a3")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )

        page = [
            paragraph(report_title, title),
            paragraph(f"{r['file_name']} | Analysed {r['date_time']}", subtitle),
            Spacer(1, 3 * mm),
            summary,
            Spacer(1, 4 * mm),
            evidence,
            Spacer(1, 4 * mm),
            colour_table,
            Spacer(1, 4 * mm),
            recommendation,
            Spacer(1, 3 * mm),
            paragraph(
                "Method: Gaussian filtering, CLAHE, HSV segmentation, morphology, contour isolation and connected-component defect measurement. "
                "Disclaimer: this result describes visible external condition only and is not a food-safety or internal-quality test.",
                note,
            ),
        ]
        story.append(KeepTogether(page))

    document.build(story, onFirstPage=footer, onLaterPages=footer)
    return output.getvalue()


def create_pdf(analysis: Analysis) -> bytes:
    """Build a one-page PDF report for a single completed analysis."""
    return create_pdf_report([analysis], "SmartBanana Inspection Report")
