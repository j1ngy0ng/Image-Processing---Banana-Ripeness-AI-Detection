"""Streamlit interface for SmartBanana's image, batch, video, and history workflows."""

from __future__ import annotations

import csv
import tempfile
from io import StringIO
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from interactive_overlay import show_colour_overlay
from smartbanana import (
    Analysis,
    AnalysisError,
    analyse_bytes,
    assess_image,
    clear_history,
    create_pdf_report,
    load_history,
    mask_from_drawing,
    save_record,
)


st.set_page_config(page_title="SmartBanana", page_icon="🍌", layout="wide")


def require_safety_agreement() -> None:
    """Block analysis features until the user accepts the safety disclaimer."""
    st.session_state.setdefault("safety_agreement", None)
    decision = st.session_state.safety_agreement
    if decision is True:
        return

    st.title("SmartBanana safety notice")
    if decision is False:
        st.error(
            "You did not accept the safety disclaimer, so access to SmartBanana has ended for this session."
        )
        st.write(
            "Close this browser tab to quit the system. Open a new session if you want to review the agreement again."
        )
        st.stop()

    with st.container(border=True):
        st.warning(
            "SmartBanana is a visual decision-support tool, not a food-safety test.",
            icon=":material/warning:",
        )
        st.markdown(
            """
            - Results assess only the banana surface visible in the submitted image or video frame.
            - Lighting, background colour, blur, obstruction and camera quality can produce an incorrect result.
            - The system cannot detect internal spoilage, contamination, odour, texture or every unsafe condition.
            - Do not consume a banana that is mouldy, leaking, smells unusual or otherwise appears unsafe, regardless of the result.
            - By continuing, you acknowledge these limitations and accept responsibility for how the result is used.
            """
        )
        with st.container(horizontal=True):
            if st.button(
                "I agree and continue", type="primary", icon=":material/check_circle:"
            ):
                st.session_state.safety_agreement = True
                st.rerun()
            if st.button("I do not agree", icon=":material/cancel:"):
                st.session_state.safety_agreement = False
                st.rerun()
    st.stop()


require_safety_agreement()

st.markdown(
    """
    <style>
    .stApp { background: #fbfdf5; }
    [data-testid="stMetric"] { background: white; border: 1px solid #e5e7d8; border-radius: 14px; padding: 14px; }
    .hero { padding: 1.1rem 1.4rem; border-radius: 18px; color: white; background: linear-gradient(120deg,#365314,#65a30d); margin-bottom: 1rem; }
    .hero h1 { margin: 0; font-size: 2rem; }
    .hero p { margin: .35rem 0 0; color: #ecfccb; }
    </style>
    <div class="hero"><h1>🍌 SmartBanana</h1><p>Explainable ripeness classification and surface-quality assessment</p></div>
    """,
    unsafe_allow_html=True,
)

DB_PATH = Path("smartbanana.db")


# Shared presentation and export helpers


def rgb(image):
    """Convert OpenCV's BGR images to the RGB order expected by Streamlit."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image


def records_csv(records: list[dict]) -> bytes:
    """Serialize result records for Streamlit download buttons."""
    if not records:
        return b""
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)
    return output.getvalue().encode("utf-8")


@st.cache_data(show_spinner=False, max_entries=6)
def report_bytes(analyses: tuple[Analysis, ...], title: str) -> bytes:
    """Cache potentially expensive illustrated report generation by its inputs."""
    return create_pdf_report(analyses, title)


def show_result(result: Analysis, key: str = "result") -> None:
    """Render one analysis result, its evidence, measurements, and PDF export."""
    r = result.record
    cols = st.columns(5)
    cols[0].metric("Ripeness", r["ripeness_stage"])
    cols[1].metric("Quality", f"Grade {r['quality_grade']}")
    cols[2].metric("Defect area", f"{r['defect_percentage']}%")
    cols[3].metric("Shelf life", r["shelf_life_days"])
    cols[4].metric("Confidence", f"{r['confidence']}%")
    st.info(r["recommendation"])

    tabs = st.tabs(["Visual evidence", "Measurements"])
    with tabs[0]:
        st.caption(
            "Hover over a green, yellow or red highlighted area to see its percentage of the detected banana surface."
        )
        with st.container(horizontal_alignment="center"):
            show_colour_overlay(
                result.overlay,
                result.region_map,
                green_percentage=r["green_percentage"],
                yellow_percentage=r["yellow_percentage"],
                defect_percentage=r["defect_percentage"],
                key=f"hover_{key}",
            )
        views = st.columns(3)
        views[0].image(rgb(result.original), caption="Original", width="stretch")
        views[1].image(
            rgb(result.enhanced), caption="Gaussian + CLAHE", width="stretch"
        )
        views[2].image(result.banana_mask, caption="Banana mask", width="stretch")
    with tabs[1]:
        left, right = st.columns([1, 2])
        colour_data = pd.DataFrame(
            {
                "Percentage": [
                    r["green_percentage"],
                    r["yellow_percentage"],
                    r["brown_percentage"],
                ]
            },
            index=["Green", "Yellow", "Brown / dark"],
        )
        left.bar_chart(colour_data, color="#84cc16")
        right.dataframe(
            pd.DataFrame({"Value": [str(value) for value in r.values()]}, index=r),
            width="stretch",
        )
    st.download_button(
        "Download PDF report",
        report_bytes((result,), "SmartBanana Inspection Report"),
        f"SmartBanana_{Path(r['file_name']).stem}.pdf",
        "application/pdf",
        key=f"pdf_{key}",
        icon=":material/picture_as_pdf:",
    )


def manual_boundary_editor(result: Analysis) -> None:
    """Let the user replace the automatic mask with a mouse-drawn polygon."""
    enabled = st.checkbox(
        "Manually correct the banana boundary",
        key=f"manual_enabled_{result.record['record_id']}",
    )
    if not enabled:
        return

    st.caption(
        "Click points around the banana. Right-click the final point to close and fill the polygon; use the toolbar to undo or reset."
    )
    height, width = result.original.shape[:2]
    # Limit canvas width while retaining the original aspect ratio.
    scale = min(1.0, 700 / width)
    canvas = st_canvas(
        fill_color="rgba(132, 204, 22, 0.35)",
        stroke_width=3,
        stroke_color="#65a30d",
        background_image=Image.fromarray(rgb(result.original)),
        update_streamlit=True,
        height=round(height * scale),
        width=round(width * scale),
        drawing_mode="polygon",
        display_toolbar=True,
        key=f"manual_canvas_{result.record['record_id']}",
    )
    if st.button(
        "Apply manual boundary",
        type="primary",
        key=f"apply_manual_{result.record['record_id']}",
    ):
        try:
            mask = mask_from_drawing(canvas.image_data, result.original.shape[:2])
            corrected = assess_image(
                result.original,
                result.record["file_name"],
                f"{result.record['input_type']} + manual boundary",
                manual_mask=mask,
            )
            save_record(corrected.record, DB_PATH)
            st.session_state.single_result = corrected
            st.session_state.boundary_applied = True
            st.rerun()
        except AnalysisError as error:
            st.error(str(error))


# Workspace navigation
page = st.sidebar.radio("Workspace", ["Analyse", "Batch", "Video", "History", "About"])
st.sidebar.caption("Classical OpenCV pipeline • No AI model")

# Single-image workspace
if page == "Analyse":
    st.subheader("Single banana inspection")
    method = st.radio("Input method", ["Upload", "Camera"], horizontal=True)
    uploaded = (
        st.file_uploader("Upload one clear banana image", ["jpg", "jpeg", "png"])
        if method == "Upload"
        else st.camera_input("Capture one clear banana")
    )
    if uploaded and st.button("Analyse banana", type="primary"):
        # A failed attempt must not leave the previous successful result on screen.
        st.session_state.pop("single_result", None)
        st.session_state.pop("boundary_applied", None)
        try:
            if uploaded.size > 15 * 1024 * 1024:
                raise AnalysisError("The image is larger than 15 MB.")
            result = analyse_bytes(uploaded.getvalue(), uploaded.name, method.lower())
            save_record(result.record, DB_PATH)
            st.session_state.single_result = result
        except AnalysisError as error:
            st.error(str(error))
    if result := st.session_state.get("single_result"):
        if st.session_state.pop("boundary_applied", False):
            st.success(
                "Manual boundary applied. All measurements and the PDF now use the corrected mask."
            )
        show_result(result, "single")
        manual_boundary_editor(result)

# Batch workspace: invalid files are reported without stopping the remaining files.
elif page == "Batch":
    st.subheader("Batch inspection")
    files = st.file_uploader(
        "Upload multiple banana images",
        ["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )
    if files and st.button("Analyse batch", type="primary"):
        results, errors = [], []
        progress = st.progress(0)
        for index, file in enumerate(files):
            try:
                result = analyse_bytes(file.getvalue(), file.name, "batch image")
                save_record(result.record, DB_PATH)
                results.append(result)
            except AnalysisError as error:
                errors.append(f"{file.name}: {error}")
            progress.progress((index + 1) / len(files))
        st.session_state.batch_results = results
        if errors:
            st.warning("\n".join(errors))
    results = st.session_state.get("batch_results", [])
    if results:
        records = [result.record for result in results]
        st.success(f"Processed {len(results)} image(s).")
        st.dataframe(pd.DataFrame(records), width="stretch", hide_index=True)
        st.download_button(
            "Download batch PDF report",
            report_bytes(tuple(results), "SmartBanana Batch Analysis Report"),
            "SmartBanana_batch_report.pdf",
            "application/pdf",
            icon=":material/picture_as_pdf:",
        )
        summary = (
            pd.DataFrame(records)["ripeness_stage"]
            .value_counts()
            .rename_axis("Ripeness")
            .to_frame("Images")
        )
        st.bar_chart(summary, color="#65a30d")
        for index, result in enumerate(results):
            with st.expander(result.record["file_name"]):
                show_result(result, f"batch_{index}")

# Video workspace: sample frames instead of processing every frame unnecessarily.
elif page == "Video":
    st.subheader("Video frame analysis")
    video = st.file_uploader("Upload an MP4 video", ["mp4"])
    interval = st.slider("Sample every N frames", 5, 120, 30)
    maximum = st.slider("Maximum sampled frames", 5, 100, 30)
    if video and st.button("Analyse video", type="primary"):
        results, errors = [], 0
        suffix = Path(video.name).suffix
        # OpenCV's VideoCapture requires a filesystem path, so persist the upload briefly.
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
            temporary.write(video.getvalue())
            path = temporary.name
        capture = cv2.VideoCapture(path)
        frame_index = 0
        while capture.isOpened() and len(results) < maximum:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % interval == 0:
                try:
                    result = assess_image(
                        frame, f"{video.name} frame {frame_index}", "video frame"
                    )
                    save_record(result.record, DB_PATH)
                    results.append(result)
                except AnalysisError:
                    errors += 1
            frame_index += 1
        capture.release()
        Path(path).unlink(missing_ok=True)
        st.session_state.video_results = results
        if errors:
            st.warning(f"Skipped {errors} frame(s) where no clear banana was detected.")
    results = st.session_state.get("video_results", [])
    if results:
        records = [result.record for result in results]
        st.dataframe(pd.DataFrame(records), width="stretch", hide_index=True)
        st.download_button(
            "Download video PDF report",
            report_bytes(tuple(results), "SmartBanana Video Analysis Report"),
            "SmartBanana_video_report.pdf",
            "application/pdf",
            icon=":material/picture_as_pdf:",
        )
        st.subheader("Last successfully analysed frame")
        show_result(results[-1], "video_last")

# History workspace
elif page == "History":
    st.subheader("Inspection history")
    records = load_history(DB_PATH)
    if records:
        st.dataframe(pd.DataFrame(records), width="stretch", hide_index=True)
        st.download_button(
            "Download history CSV",
            records_csv(records),
            "SmartBanana_history.csv",
            "text/csv",
        )
        if st.checkbox("I want to delete all history") and st.button("Delete history"):
            clear_history(DB_PATH)
            st.rerun()
    else:
        st.info("No inspections have been saved yet.")

# About workspace
else:
    st.subheader("How SmartBanana works")
    st.write(
        "SmartBanana follows the assignment report exactly: image resizing → Gaussian filtering → CLAHE → HSV segmentation → morphology → largest-contour isolation → peel-colour and connected-defect analysis → rule-based recommendations."
    )
    st.markdown("""
    **Best capture conditions**

    - Show one complete banana against a plain, contrasting background.
    - Use even lighting and avoid strong shadows, blur, and yellow background objects.
    - Supported inputs are JPG/JPEG/PNG images and MP4 videos.

    **Interpretation**

    - Grade A: less than 20% detected surface defects.
    - Grade B: 20% to 50% detected surface defects.
    - Grade C: more than 50% detected surface defects.
    - The 20% and 50% boundaries follow the published BBCH scale for edible Musaceae.
    - FAO ambient-life ranges are shown for unripe and ripe bananas; later stages use handling guidance because no universal remaining-day value is stated.

    Results describe visible external condition only; they are not a food-safety or internal-quality test.
    """)
