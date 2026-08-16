"""Render the as-implemented SmartBanana system flowchart as a high-resolution PNG.

The chart deliberately uses conventional flowchart symbols so it can be inserted
directly into the assignment report without relying on a diagramming application.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output" / "flowchart"
OUTPUT_PATH = OUTPUT_DIR / "SmartBanana_Current_System_Flowchart.png"

WIDTH, HEIGHT = 3000, 7600
BACKGROUND = "#F7F9FC"
INK = "#183153"
MUTED = "#5E7184"
LINE = "#35536E"
PANEL = "#FFFFFF"
ACCESS = "#FFF4CF"
INPUT = "#DDF0FF"
PROCESS = "#DFF5E5"
DECISION = "#FFF0BE"
DATA = "#E8E0FF"
ERROR = "#FFE0E0"
OUTPUT = "#E0F4F3"
MANUAL = "#FFE8C7"


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Load a readable Windows font, with a Pillow fallback."""
    candidates = [
        Path("C:/Windows/Fonts/seguisb.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


FONT_TITLE = load_font(72, bold=True)
FONT_SUBTITLE = load_font(34)
FONT_SECTION = load_font(42, bold=True)
FONT_NODE = load_font(31, bold=True)
FONT_SMALL = load_font(27)
FONT_EDGE = load_font(26, bold=True)
FONT_LEGEND = load_font(25)


image = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
draw = ImageDraw.Draw(image)


def wrapped_lines(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    """Wrap node labels by measured pixel width."""
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = words[0]
        for word in words[1:]:
            trial = f"{current} {word}"
            if draw.textlength(trial, font=font) <= max_width:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def centered_text(
    box: tuple[int, int, int, int],
    text: str,
    *,
    font: ImageFont.ImageFont = FONT_NODE,
    fill: str = INK,
    padding: int = 34,
) -> None:
    """Draw wrapped text centered inside a bounding box."""
    x1, y1, x2, y2 = box
    lines = wrapped_lines(text, font, x2 - x1 - 2 * padding)
    spacing = 10
    heights = [
        draw.textbbox((0, 0), line or " ", font=font)[3]
        - draw.textbbox((0, 0), line or " ", font=font)[1]
        for line in lines
    ]
    total_height = sum(heights) + spacing * (len(lines) - 1)
    y = y1 + (y2 - y1 - total_height) / 2
    for line, height in zip(lines, heights):
        width = draw.textlength(line, font=font)
        draw.text((x1 + (x2 - x1 - width) / 2, y), line, font=font, fill=fill)
        y += height + spacing


def arrow_head(start: tuple[int, int], end: tuple[int, int], color: str = LINE) -> None:
    """Draw a triangular arrowhead at the end of a line."""
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    length, spread = 25, 0.55
    points = [
        end,
        (
            end[0] - length * math.cos(angle - spread),
            end[1] - length * math.sin(angle - spread),
        ),
        (
            end[0] - length * math.cos(angle + spread),
            end[1] - length * math.sin(angle + spread),
        ),
    ]
    draw.polygon(points, fill=color)


def connector(
    points: Iterable[tuple[int, int]],
    *,
    label: str | None = None,
    label_at: tuple[int, int] | None = None,
    color: str = LINE,
    width: int = 7,
    arrow: bool = True,
    dashed: bool = False,
) -> None:
    """Draw an orthogonal connector with an optional edge label."""
    pts = list(points)
    if dashed:
        for first, second in zip(pts, pts[1:]):
            distance = math.dist(first, second)
            steps = max(1, int(distance // 28))
            for index in range(0, steps, 2):
                t1 = index / steps
                t2 = min((index + 1) / steps, 1)
                p1 = (
                    first[0] + (second[0] - first[0]) * t1,
                    first[1] + (second[1] - first[1]) * t1,
                )
                p2 = (
                    first[0] + (second[0] - first[0]) * t2,
                    first[1] + (second[1] - first[1]) * t2,
                )
                draw.line([p1, p2], fill=color, width=width)
    else:
        draw.line(pts, fill=color, width=width, joint="curve")
    if arrow and len(pts) >= 2:
        arrow_head(pts[-2], pts[-1], color)
    if label and label_at:
        bbox = draw.textbbox((0, 0), label, font=FONT_EDGE)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x, y = label_at
        draw.rounded_rectangle(
            (x - 12, y - 8, x + text_width + 12, y + text_height + 12),
            radius=10,
            fill=BACKGROUND,
        )
        draw.text((x, y), label, font=FONT_EDGE, fill=INK)


def process(box: tuple[int, int, int, int], text: str, fill: str = PROCESS) -> None:
    draw.rounded_rectangle(box, radius=20, fill=fill, outline=INK, width=5)
    centered_text(box, text)


def terminator(box: tuple[int, int, int, int], text: str, fill: str = ACCESS) -> None:
    draw.rounded_rectangle(
        box,
        radius=(box[3] - box[1]) // 2,
        fill=fill,
        outline=INK,
        width=5,
    )
    centered_text(box, text)


def input_output(box: tuple[int, int, int, int], text: str, fill: str = INPUT) -> None:
    x1, y1, x2, y2 = box
    skew = 45
    points = [(x1 + skew, y1), (x2, y1), (x2 - skew, y2), (x1, y2)]
    draw.polygon(points, fill=fill, outline=INK)
    draw.line(points + [points[0]], fill=INK, width=5, joint="curve")
    centered_text(box, text, padding=65)


def decision(
    center: tuple[int, int],
    size: tuple[int, int],
    text: str,
    fill: str = DECISION,
) -> tuple[int, int, int, int]:
    cx, cy = center
    width, height = size
    box = (cx - width // 2, cy - height // 2, cx + width // 2, cy + height // 2)
    points = [(cx, box[1]), (box[2], cy), (cx, box[3]), (box[0], cy)]
    draw.polygon(points, fill=fill, outline=INK)
    draw.line(points + [points[0]], fill=INK, width=5, joint="curve")
    centered_text(box, text, padding=95)
    return box


def database(box: tuple[int, int, int, int], text: str) -> None:
    x1, y1, x2, y2 = box
    cap = 34
    draw.rectangle((x1, y1 + cap, x2, y2 - cap), fill=DATA)
    draw.ellipse((x1, y1, x2, y1 + 2 * cap), fill=DATA, outline=INK, width=5)
    draw.ellipse((x1, y2 - 2 * cap, x2, y2), fill=DATA)
    # Cover the upper half of the lower ellipse so only the conventional
    # bottom cylinder arc remains and never crosses the database label.
    draw.rectangle((x1, y2 - 2 * cap, x2, y2 - cap), fill=DATA)
    draw.arc((x1, y2 - 2 * cap, x2, y2), 0, 180, fill=INK, width=5)
    draw.line((x1, y1 + cap, x1, y2 - cap), fill=INK, width=5)
    draw.line((x2, y1 + cap, x2, y2 - cap), fill=INK, width=5)
    centered_text(
        (x1, y1 + cap, x2, y2 - cap),
        text,
        font=FONT_SMALL,
        padding=55,
    )


def manual_operation(box: tuple[int, int, int, int], text: str) -> None:
    x1, y1, x2, y2 = box
    inset = 45
    points = [(x1, y1), (x2, y1), (x2 - inset, y2), (x1 + inset, y2)]
    draw.polygon(points, fill=MANUAL, outline=INK)
    draw.line(points + [points[0]], fill=INK, width=5)
    centered_text(box, text, padding=60)


def document(box: tuple[int, int, int, int], text: str) -> None:
    x1, y1, x2, y2 = box
    wave = 22
    points = [
        (x1, y1),
        (x2, y1),
        (x2, y2 - wave),
        (x2 - (x2 - x1) * 0.25, y2),
        (x2 - (x2 - x1) * 0.50, y2 - 2 * wave),
        (x2 - (x2 - x1) * 0.75, y2),
        (x1, y2 - wave),
    ]
    draw.polygon(points, fill=OUTPUT, outline=INK)
    draw.line(points + [points[0]], fill=INK, width=5, joint="curve")
    centered_text((x1, y1, x2, y2 - 10), text, padding=55)


def offpage(center: tuple[int, int], label: str, caption: str = "") -> None:
    cx, cy = center
    radius = 44
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        fill=INK,
        outline=INK,
    )
    bbox = draw.textbbox((0, 0), label, font=FONT_NODE)
    draw.text(
        (cx - (bbox[2] - bbox[0]) / 2, cy - (bbox[3] - bbox[1]) / 2 - 3),
        label,
        font=FONT_NODE,
        fill="white",
    )
    if caption:
        draw.text((cx + 60, cy - 18), caption, font=FONT_SMALL, fill=MUTED)


def panel(box: tuple[int, int, int, int], title: str, subtitle: str) -> None:
    draw.rounded_rectangle(box, radius=32, fill=PANEL, outline="#C9D6E2", width=4)
    x1, y1, _, _ = box
    draw.text((x1 + 55, y1 + 35), title, font=FONT_SECTION, fill=INK)
    draw.text((x1 + 55, y1 + 94), subtitle, font=FONT_SUBTITLE, fill=MUTED)


# Header
draw.text((WIDTH / 2, 70), "SmartBanana Current System Flowchart", font=FONT_TITLE, fill=INK, anchor="ma")
draw.text(
    (WIDTH / 2, 160),
    "As implemented in the Streamlit application and image-processing pipeline",
    font=FONT_SUBTITLE,
    fill=MUTED,
    anchor="ma",
)

# Panels
panel(
    (90, 245, 2910, 2310),
    "A. Access, workspace selection and input handling",
    "The three analysis modes share one image-analysis sub-process; History and About remain separate.",
)
panel(
    (90, 2370, 2910, 7050),
    "B. Shared banana-analysis sub-process",
    "Each uploaded image or sampled video frame follows this pipeline.",
)

# Connectors in panel A (drawn before nodes)
connector([(1500, 390), (1500, 480)])
connector([(1500, 610), (1500, 705)])
connector([(1780, 805), (2440, 805)], label="No", label_at=(2030, 760))
connector([(2440, 875), (2440, 960)])
connector([(1500, 905), (1500, 1000)], label="Yes", label_at=(1530, 930))
connector([(1500, 1135), (1500, 1235)])

# Workspace branches
branch_x = [350, 900, 1450, 2000, 2550]
for x in branch_x:
    connector([(1500, 1435), (1500, 1480), (x, 1480), (x, 1530)], arrow=True)

# Analyse/Batch/Video to their preparation nodes
for x in branch_x[:3]:
    connector([(x, 1680), (x, 1770)])

# Merge inputs into A
for x in branch_x[:3]:
    connector([(x, 1910), (x, 2055), (1450, 2055)], arrow=False)
connector([(1450, 2055), (1450, 2140)])

# History branch
connector([(2000, 1680), (2000, 1760)])
connector([(2000, 1920), (2000, 2000)])
connector([(2000, 2160), (2000, 2210)], arrow=False)

# About branch
connector([(2550, 1680), (2550, 1800)])
connector([(2550, 1960), (2550, 2210)], arrow=False)

# Panel A nodes
terminator((1320, 300, 1680, 390), "Start")
process((1130, 480, 1870, 610), "Display safety disclaimer", ACCESS)
decision((1500, 805), (560, 200), "User agrees?")
process((2140, 740, 2740, 875), "Access is ended", ERROR)
terminator((2260, 960, 2620, 1050), "End", ERROR)
input_output((1120, 1000, 1880, 1135), "Select workspace")
decision((1500, 1335), (650, 200), "Selected workspace?")

input_output((90, 1530, 610, 1680), "Analyse: upload image or use camera")
input_output((640, 1530, 1160, 1680), "Batch: upload multiple images")
input_output((1190, 1530, 1710, 1680), "Video: upload MP4 and sampling settings")
input_output((1740, 1530, 2260, 1680), "History")
input_output((2290, 1530, 2810, 1680), "About")

process((90, 1770, 610, 1910), "Validate JPG, JPEG or PNG and 15 MB limit", INPUT)
process((640, 1770, 1160, 1910), "Iterate uploaded image files", INPUT)
process((1190, 1770, 1710, 1910), "Extract and sample video frames", INPUT)
database((1740, 1760, 2260, 1920), "Read SQLite history")
input_output((1740, 2000, 2260, 2160), "Display records; export CSV or delete history", OUTPUT)
input_output((2290, 1800, 2810, 1960), "Display method, capture guidance and limitations", OUTPUT)
offpage((1450, 2190), "A", "Send one image or frame to analysis")
offpage((2000, 2235), "W", "Return to workspace")
offpage((2550, 2235), "W")

# Connectors in panel B
connector([(1500, 2570), (1500, 2580)])
connector([(1500, 2720), (1500, 2790)])
connector([(1500, 2930), (1500, 3000)])
connector([(1500, 3140), (1500, 3210)])
connector([(1500, 3350), (1500, 3420)])
connector([(1500, 3560), (1500, 3635)])

# Segmentation decision branches
connector([(1500, 3865), (1500, 3970)], label="Yes", label_at=(1530, 3880))
connector([(1810, 3750), (2360, 3750)], label="No", label_at=(2050, 3705))
connector([(2360, 3820), (2360, 3960)])
connector([(2360, 4110), (2360, 5880), (1800, 5880)], arrow=True)

# Successful pipeline
connector([(1500, 4110), (1500, 4180)])
connector([(1500, 4330), (1500, 4400)])
connector([(1500, 4550), (1500, 4620)])
connector([(1500, 4770), (1500, 4840)])
connector([(1500, 4990), (1500, 5060)])
connector([(1500, 5210), (1500, 5290)])

# Manual correction branch
connector([(1180, 5620), (570, 5620)], label="Yes", label_at=(835, 5575))
connector([(570, 5700), (570, 5780)])
connector(
    [(570, 5940), (250, 5940), (250, 4255), (1140, 4255)],
    label="Repeat measurements",
    label_at=(300, 4200),
)

# Normal path to item loop
connector([(1500, 5720), (1500, 5800)], label="No", label_at=(1530, 5735))
connector([(1500, 5960), (1500, 6030)], label="No", label_at=(1530, 5965))

# More-items loop and completed-analysis path
connector(
    [(1200, 5880), (900, 5880), (900, 6200), (150, 6200), (150, 2570), (1450, 2570)],
    label="Yes: next batch item or video frame",
    label_at=(390, 6155),
)

# Panel B nodes
offpage((1500, 2525), "A", "Receive one image or sampled frame")
process((1110, 2580, 1890, 2720), "Decode image and resize to a maximum of 1024 px")
process((1110, 2790, 1890, 2930), "Apply Gaussian filtering and CLAHE contrast enhancement")
process((1110, 3000, 1890, 3140), "Convert BGR image to HSV colour space")
process((1110, 3210, 1890, 3350), "Threshold green, yellow, orange and brown banana colours")
process((1110, 3420, 1890, 3560), "Morphological opening/closing and largest contour")
decision((1500, 3750), (620, 230), "Plausible banana mask?")
input_output((2050, 3680, 2670, 3820), "Display clearer-image error or batch/video warning", ERROR)
process((2050, 3960, 2670, 4110), "Do not save result; skip failed item", ERROR)

process((1110, 3970, 1890, 4110), "Calculate banana peel surface area")
process((1110, 4180, 1890, 4330), "Calculate green, yellow and brown percentages")
process((1110, 4400, 1890, 4550), "Detect dark defects using morphology and connected components")
process((1110, 4620, 1890, 4770), "Assign ripeness, Grade A/B/C, shelf life, recommendation and confidence")
process((1110, 4840, 1890, 4990), "Create colour-coded evidence overlay and hover percentages")
database((1110, 5060, 1890, 5210), "Save successful analysis to smartbanana.db")
input_output((1110, 5290, 1890, 5470), "Display result, measurements and visual evidence", OUTPUT)
decision((1500, 5620), (640, 200), "Single image: use manual boundary correction?")
manual_operation((300, 5540, 840, 5700), "Draw banana polygon with mouse")
process((300, 5780, 840, 5940), "Use polygon as mask and reassess")
decision((1500, 5880), (600, 160), "More items or frames?")
decision((1500, 6140), (600, 220), "Generate report?")

# Report and completion section
connector([(1800, 6140), (2360, 6140), (2360, 6380)], label="Yes", label_at=(2050, 6095))
connector([(1500, 6250), (1500, 6800)], label="No", label_at=(1530, 6320))
document(
    (2020, 6380, 2700, 6570),
    "PDF: overall summary first; individual illustrated pages follow",
)
connector([(2360, 6570), (2360, 6858), (1880, 6858)])
terminator((1120, 6800, 1880, 6915), "Return to workspace or end session")
offpage((1020, 6858), "W")

# Small note clarifying output rules
draw.rounded_rectangle((130, 6380, 980, 6720), radius=22, fill="#F2F6FA", outline="#B7C8D8", width=3)
draw.text((170, 6420), "Output rules", font=FONT_NODE, fill=INK)
note_lines = [
    "• Analyse: optional one-result PDF",
    "• Batch/video: summary first page,",
    "  then one illustrated result per page",
    "• History: CSV export",
    "• Failed segmentation is not recorded",
]
for index, line in enumerate(note_lines):
    draw.text((170, 6480 + index * 40), line, font=FONT_LEGEND, fill=MUTED)

# Legend
legend_y = 7460
draw.text((120, legend_y), "Standard symbols:", font=FONT_NODE, fill=INK)
items = [
    ("Terminator", ACCESS),
    ("Input / output", INPUT),
    ("Process", PROCESS),
    ("Decision", DECISION),
    ("Database", DATA),
    ("Manual operation", MANUAL),
    ("Document", OUTPUT),
]
x = 500
for label, colour in items:
    draw.rounded_rectangle((x, legend_y - 4, x + 48, legend_y + 34), radius=8, fill=colour, outline=INK, width=2)
    draw.text((x + 62, legend_y), label, font=FONT_LEGEND, fill=MUTED)
    x += 62 + int(draw.textlength(label, font=FONT_LEGEND)) + 55

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
image.save(OUTPUT_PATH, "PNG", optimize=True, dpi=(300, 300))
print(f"Saved {OUTPUT_PATH}")
