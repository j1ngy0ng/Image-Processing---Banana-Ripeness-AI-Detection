"""Generate the editable diagrams.net flowchart for the implemented system."""

from __future__ import annotations

from pathlib import Path
from xml.dom import minidom
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "flowchart" / "SmartBanana_Current_System_Flowchart.drawio"

COLORS = {
    "ink": "#183153",
    "line": "#35536E",
    "panel": "#FFFFFF",
    "process": "#DFF5E5",
    "input": "#DDF0FF",
    "decision": "#FFF0BE",
    "data": "#E8E0FF",
    "error": "#FFE0E0",
    "output": "#E0F4F3",
    "manual": "#FFE8C7",
    "access": "#FFF4CF",
}


def build() -> ET.Element:
    mxfile = ET.Element(
        "mxfile",
        {
            "host": "app.diagrams.net",
            "agent": "SmartBanana flowchart generator",
            "version": "24.7.17",
            "compressed": "false",
        },
    )
    diagram = ET.SubElement(mxfile, "diagram", {"id": "smartbanana-current", "name": "Current System Flowchart"})
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        {
            "dx": "2200",
            "dy": "5000",
            "grid": "1",
            "gridSize": "10",
            "guides": "1",
            "tooltips": "1",
            "connect": "1",
            "arrows": "1",
            "fold": "1",
            "page": "1",
            "pageScale": "1",
            "pageWidth": "2200",
            "pageHeight": "5000",
            "math": "0",
            "shadow": "0",
        },
    )
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})

    cells: dict[str, ET.Element] = {}

    def vertex(
        cell_id: str,
        value: str,
        x: int,
        y: int,
        width: int,
        height: int,
        style: str,
        *,
        parent: str = "1",
    ) -> None:
        cell = ET.SubElement(
            root,
            "mxCell",
            {
                "id": cell_id,
                "value": value,
                "style": style,
                "vertex": "1",
                "parent": parent,
            },
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {"x": str(x), "y": str(y), "width": str(width), "height": str(height), "as": "geometry"},
        )
        cells[cell_id] = cell

    def edge(
        cell_id: str,
        source: str,
        target: str,
        label: str = "",
        *,
        dashed: bool = False,
    ) -> None:
        style = (
            "edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;"
            f"html=1;endArrow=block;endFill=1;strokeWidth=2;strokeColor={COLORS['line']};"
            "fontFamily=Segoe UI;fontSize=18;fontStyle=1;labelBackgroundColor=#F7F9FC;"
        )
        if dashed:
            style += "dashed=1;dashPattern=6 4;"
        cell = ET.SubElement(
            root,
            "mxCell",
            {
                "id": cell_id,
                "value": label,
                "style": style,
                "edge": "1",
                "parent": "1",
                "source": source,
                "target": target,
            },
        )
        ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})

    base = (
        "whiteSpace=wrap;html=1;align=center;verticalAlign=middle;"
        f"fontFamily=Segoe UI;fontSize=20;fontColor={COLORS['ink']};"
        f"strokeColor={COLORS['ink']};strokeWidth=2;spacing=8;"
    )
    process = base + f"rounded=0;fillColor={COLORS['process']};"
    input_output = base + f"shape=parallelogram;perimeter=parallelogramPerimeter;fillColor={COLORS['input']};"
    decision = base + f"shape=rhombus;perimeter=rhombusPerimeter;fillColor={COLORS['decision']};"
    terminator = base + f"rounded=1;arcSize=50;fillColor={COLORS['access']};"
    database = base + f"shape=cylinder3;boundedLbl=1;backgroundOutline=1;fillColor={COLORS['data']};"
    document = base + f"shape=document;boundedLbl=1;fillColor={COLORS['output']};"
    manual = base + f"shape=trapezoid;perimeter=trapezoidPerimeter;fixedSize=1;fillColor={COLORS['manual']};"
    error_output = base + f"shape=parallelogram;perimeter=parallelogramPerimeter;fillColor={COLORS['error']};"
    connector = base + f"shape=ellipse;aspect=fixed;fillColor={COLORS['ink']};fontColor=#FFFFFF;fontStyle=1;"
    text = (
        "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;"
        f"fontFamily=Segoe UI;fontColor={COLORS['ink']};whiteSpace=wrap;"
    )
    panel = (
        "rounded=1;arcSize=12;whiteSpace=wrap;html=1;align=left;verticalAlign=top;"
        "spacingTop=16;spacingLeft=18;fontFamily=Segoe UI;fontSize=26;fontStyle=1;"
        f"fontColor={COLORS['ink']};fillColor={COLORS['panel']};strokeColor=#C9D6E2;strokeWidth=2;"
    )

    # Title and section containers.
    vertex("title", "<b>SmartBanana Current System Flowchart</b>", 40, 20, 2120, 62, text + "fontSize=34;align=center;fontStyle=1;")
    vertex("subtitle", "As implemented in the Streamlit application and classical image-processing pipeline", 40, 78, 2120, 40, text + "fontSize=18;align=center;fontColor=#5E7184;")
    vertex("panel_access", "A. Access, workspace selection and input handling", 25, 130, 2150, 1100, panel)
    vertex("panel_analysis", "B. Shared banana-analysis sub-process", 25, 1250, 2150, 2680, panel)
    vertex("panel_output", "C. Completion, aggregate output and reports", 25, 3950, 2150, 900, panel)

    # Access and workspace selection.
    vertex("start", "Start", 960, 190, 280, 70, terminator)
    vertex("disclaimer", "Display first-visit safety disclaimer", 790, 300, 620, 95, process)
    vertex("agree", "User agrees?", 950, 440, 300, 160, decision)
    vertex("access_end", "Display access-ended message", 1510, 455, 430, 90, error_output)
    vertex("end_declined", "End", 1585, 585, 280, 70, terminator + f"fillColor={COLORS['error']};")
    vertex("workspace", "Select workspace", 870, 650, 460, 160, decision)

    vertex("analyse", "<b>Analyse</b><br>Upload image or use camera", 40, 865, 360, 105, input_output)
    vertex("batch", "<b>Batch</b><br>Upload multiple images", 470, 865, 360, 105, input_output)
    vertex("video", "<b>Video</b><br>Upload MP4 and sampling settings", 900, 865, 360, 105, input_output)
    vertex("history", "<b>History</b>", 1330, 865, 360, 105, input_output)
    vertex("about", "<b>About</b>", 1760, 865, 360, 105, input_output)

    vertex("analyse_prep", "Validate size/type and decode image", 40, 1010, 360, 105, process)
    vertex("batch_prep", "Iterate uploaded image files", 470, 1010, 360, 105, process)
    vertex("video_prep", "Extract and sample video frames", 900, 1010, 360, 105, process)
    vertex("history_db", "Load smartbanana.db", 1330, 995, 360, 105, database)
    vertex("history_out", "Display records; export CSV or delete history", 1330, 1125, 360, 85, input_output)
    vertex("about_out", "Display method, capture guidance, thresholds and limitations", 1760, 1010, 360, 110, input_output)

    vertex("a_analyse", "A", 190, 1155, 48, 48, connector)
    vertex("a_batch", "A", 620, 1155, 48, 48, connector)
    vertex("a_video", "A", 1050, 1155, 48, 48, connector)
    vertex("w_history", "W", 1486, 1162, 48, 48, connector)
    vertex("w_about", "W", 1916, 1162, 48, 48, connector)

    edge("e_start_disclaimer", "start", "disclaimer")
    edge("e_disclaimer_agree", "disclaimer", "agree")
    edge("e_agree_yes", "agree", "workspace", "Yes")
    edge("e_agree_no", "agree", "access_end", "No")
    edge("e_access_end", "access_end", "end_declined")
    for mode in ("analyse", "batch", "video", "history", "about"):
        edge(f"e_workspace_{mode}", "workspace", mode)
    edge("e_analyse_prep", "analyse", "analyse_prep")
    edge("e_batch_prep", "batch", "batch_prep")
    edge("e_video_prep", "video", "video_prep")
    edge("e_history_db", "history", "history_db")
    edge("e_history_out", "history_db", "history_out")
    edge("e_history_w", "history_out", "w_history")
    edge("e_about_out", "about", "about_out")
    edge("e_about_w", "about_out", "w_about")
    edge("e_analyse_a", "analyse_prep", "a_analyse")
    edge("e_batch_a", "batch_prep", "a_batch")
    edge("e_video_a", "video_prep", "a_video")

    # Shared image-analysis pipeline.
    vertex("a_entry", "A", 1076, 1305, 48, 48, connector)
    vertex("decode_resize", "Decode image and resize to a maximum of 1024 px", 760, 1385, 680, 90, process)
    vertex("preprocess", "Apply 5×5 Gaussian filtering and CLAHE local-contrast enhancement", 760, 1505, 680, 90, process)
    vertex("hsv", "Convert the enhanced BGR image to HSV", 760, 1625, 680, 90, process)
    vertex("threshold", "Threshold green, yellow, orange and brown banana-colour ranges", 760, 1745, 680, 90, process)
    vertex("morphology", "Apply morphological opening and closing", 760, 1865, 680, 90, process)
    vertex("largest", "Fill and retain the largest external contour", 760, 1985, 680, 90, process)
    vertex("plausible", "Plausible banana mask?", 940, 2120, 320, 170, decision)

    vertex("clearer_error", "Display clearer-image error or batch/video warning", 1550, 2160, 500, 90, error_output)
    vertex("skip_failed", "Do not save failed result; skip the item", 1550, 2280, 500, 85, process + f"fillColor={COLORS['error']};")
    vertex("b_failed", "B", 1776, 2400, 48, 48, connector)

    vertex("surface", "Calculate detected banana surface area", 760, 2350, 680, 85, process)
    vertex("colours", "Calculate green, yellow and brown peel percentages", 760, 2465, 680, 85, process)
    vertex("defects", "Detect dark defects using morphology and connected components", 760, 2580, 680, 85, process)
    vertex("classify", "Assign ripeness, Grade A/B/C, shelf-life guidance, recommendation and confidence", 760, 2695, 680, 100, process)
    vertex("overlay", "Create colour-coded overlay with hover percentages", 760, 2825, 680, 85, process)
    vertex("save", "Save successful analysis to smartbanana.db", 760, 2940, 680, 100, database)
    vertex("display", "Display result, measurements and visual evidence", 760, 3070, 680, 90, input_output)
    vertex("single", "Single-image workspace?", 940, 3200, 320, 170, decision)

    vertex("manual_choice", "Apply manual boundary correction?", 280, 3200, 330, 170, decision)
    vertex("manual_draw", "Draw and close banana polygon with mouse", 245, 3420, 400, 95, manual)
    vertex("manual_mask", "Validate polygon and use it as the banana mask", 245, 3545, 400, 90, process)
    vertex("c_manual", "C", 420, 3670, 48, 48, connector)
    vertex("c_measure", "C", 680, 2368, 48, 48, connector)

    vertex("b_success", "B", 1076, 3410, 48, 48, connector)
    vertex("b_merge", "B", 1076, 3490, 48, 48, connector)
    vertex("more_items", "More batch images or video frames?", 915, 3580, 370, 190, decision)
    vertex("a_loop", "A", 700, 3648, 48, 48, connector)
    vertex("successes", "Any successful analyses?", 1450, 3585, 340, 180, decision)
    vertex("w_no_results", "W", 1900, 3650, 48, 48, connector)

    edge("e_a_decode", "a_entry", "decode_resize")
    edge("e_decode_preprocess", "decode_resize", "preprocess")
    edge("e_preprocess_hsv", "preprocess", "hsv")
    edge("e_hsv_threshold", "hsv", "threshold")
    edge("e_threshold_morph", "threshold", "morphology")
    edge("e_morph_largest", "morphology", "largest")
    edge("e_largest_plausible", "largest", "plausible")
    edge("e_plausible_yes", "plausible", "surface", "Yes")
    edge("e_plausible_no", "plausible", "clearer_error", "No")
    edge("e_error_skip", "clearer_error", "skip_failed")
    edge("e_skip_b", "skip_failed", "b_failed")

    edge("e_surface_colours", "surface", "colours")
    edge("e_colours_defects", "colours", "defects")
    edge("e_defects_classify", "defects", "classify")
    edge("e_classify_overlay", "classify", "overlay")
    edge("e_overlay_save", "overlay", "save")
    edge("e_save_display", "save", "display")
    edge("e_display_single", "display", "single")
    edge("e_single_yes", "single", "manual_choice", "Yes")
    edge("e_manual_yes", "manual_choice", "manual_draw", "Yes")
    edge("e_manual_draw_mask", "manual_draw", "manual_mask")
    edge("e_manual_mask_c", "manual_mask", "c_manual")
    edge("e_c_measure", "c_measure", "surface", "Reassess")
    edge("e_single_no", "single", "b_success", "No")
    edge("e_manual_no", "manual_choice", "successes", "No")
    edge("e_b_success_merge", "b_success", "b_merge", dashed=True)
    edge("e_b_failed_merge", "b_failed", "b_merge", dashed=True)
    edge("e_b_more", "b_merge", "more_items")
    edge("e_more_yes", "more_items", "a_loop", "Yes")
    edge("e_more_no", "more_items", "successes", "No")
    edge("e_success_no", "successes", "w_no_results", "No")

    # Completion, aggregate output, and report generation.
    vertex("success_entry", "Successful result(s)", 835, 4025, 530, 85, process)
    vertex("multi", "Batch or video?", 950, 4150, 300, 160, decision)
    vertex("aggregate", "Display overall table/chart and individual or last-frame results", 1450, 4185, 520, 95, input_output)
    vertex("pdf_choice", "Download PDF report?", 950, 4380, 300, 160, decision)
    vertex("pdf", "PDF report<br><b>Single:</b> illustrated result page<br><b>Batch/video:</b> overall summary first, then illustrated pages", 1425, 4400, 560, 145, document)
    vertex("return", "Return to workspace or end session", 780, 4655, 640, 80, terminator)
    vertex("w_return", "W", 1076, 4770, 48, 48, connector)

    vertex("output_note", "<b>Output rules</b><br>• Failed segmentation is not recorded.<br>• History alone exports CSV.<br>• Analysis, batch and video offer illustrated PDF reports.<br>• Results describe visible external condition—not food safety.", 90, 4120, 650, 270, text + "fontSize=19;fillColor=#F2F6FA;strokeColor=#B7C8D8;rounded=1;spacing=16;")

    edge("e_success_yes", "successes", "success_entry", "Yes")
    edge("e_success_entry_multi", "success_entry", "multi")
    edge("e_multi_yes", "multi", "aggregate", "Yes")
    edge("e_multi_no", "multi", "pdf_choice", "No")
    edge("e_aggregate_pdf_choice", "aggregate", "pdf_choice")
    edge("e_pdf_yes", "pdf_choice", "pdf", "Yes")
    edge("e_pdf_no", "pdf_choice", "return", "No")
    edge("e_pdf_return", "pdf", "return")
    edge("e_return_w", "return", "w_return")

    # A connector legend makes the repeated A/B/C/W routing explicit.
    vertex(
        "connector_note",
        "<b>Connector key:</b> A = next image/frame enters shared analysis · B = item attempt completed · C = manual mask re-enters measurement · W = return to workspace",
        90,
        4870,
        2020,
        70,
        text + "fontSize=17;align=center;fontColor=#5E7184;",
    )

    # Self-check: keep every connector attached to real cells.
    all_ids = {cell.get("id") for cell in root}
    for cell in root.findall("mxCell"):
        if cell.get("edge") == "1":
            assert cell.get("source") in all_ids and cell.get("target") in all_ids
    assert len(root.findall("mxCell")) >= 90
    return mxfile


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rough = ET.tostring(build(), encoding="utf-8")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ", encoding="UTF-8")
    OUTPUT.write_bytes(pretty)
    ET.parse(OUTPUT)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
