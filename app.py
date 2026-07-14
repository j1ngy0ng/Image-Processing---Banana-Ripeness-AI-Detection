"""Streamlit interface for SmartBanana."""

from __future__ import annotations

import csv
import tempfile
from io import StringIO
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from smartbanana import Analysis, AnalysisError, analyse_bytes, assess_image, clear_history, create_pdf, load_history, save_record


st.set_page_config(page_title="SmartBanana", page_icon="🍌", layout="wide")
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


def rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image


def records_csv(records: list[dict]) -> bytes:
    if not records:
        return b""
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)
    return output.getvalue().encode("utf-8")


def show_result(result: Analysis, key: str = "result") -> None:
    r = result.record
    cols = st.columns(5)
    cols[0].metric("Ripeness", r["ripeness_stage"])
    cols[1].metric("Quality", f"Grade {r['quality_grade']}")
    cols[2].metric("Defect area", f"{r['defect_percentage']}%")
    cols[3].metric("Shelf life", f"{r['shelf_life_days']} day(s)")
    cols[4].metric("Confidence", f"{r['confidence']}%")
    st.info(r["recommendation"])

    tabs = st.tabs(["Visual evidence", "Measurements"])
    with tabs[0]:
        views = st.columns(4)
        views[0].image(rgb(result.original), caption="Original", use_container_width=True)
        views[1].image(rgb(result.enhanced), caption="Gaussian + CLAHE", use_container_width=True)
        views[2].image(result.banana_mask, caption="Banana mask", use_container_width=True)
        views[3].image(rgb(result.overlay), caption="Defects in red", use_container_width=True)
    with tabs[1]:
        left, right = st.columns([1, 2])
        colour_data = pd.DataFrame({"Percentage": [r["green_percentage"], r["yellow_percentage"], r["brown_percentage"]]}, index=["Green", "Yellow", "Brown / dark"])
        left.bar_chart(colour_data, color="#84cc16")
        right.dataframe(pd.DataFrame([r]).T.rename(columns={0: "Value"}), use_container_width=True)
    st.download_button("Download PDF report", create_pdf(result), f"SmartBanana_{Path(r['file_name']).stem}.pdf", "application/pdf", key=f"pdf_{key}")


page = st.sidebar.radio("Workspace", ["Analyse", "Batch", "Video", "History", "About"])
st.sidebar.caption("Classical OpenCV pipeline • No AI model")

if page == "Analyse":
    st.subheader("Single banana inspection")
    method = st.radio("Input method", ["Upload", "Camera"], horizontal=True)
    uploaded = st.file_uploader("Upload one clear banana image", ["jpg", "jpeg", "png"]) if method == "Upload" else st.camera_input("Capture one clear banana")
    if uploaded and st.button("Analyse banana", type="primary"):
        try:
            if uploaded.size > 15 * 1024 * 1024:
                raise AnalysisError("The image is larger than 15 MB.")
            result = analyse_bytes(uploaded.getvalue(), uploaded.name, method.lower())
            save_record(result.record, DB_PATH)
            st.session_state.single_result = result
        except AnalysisError as error:
            st.error(str(error))
    if result := st.session_state.get("single_result"):
        show_result(result, "single")

elif page == "Batch":
    st.subheader("Batch inspection")
    files = st.file_uploader("Upload multiple banana images", ["jpg", "jpeg", "png"], accept_multiple_files=True)
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
        st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)
        st.download_button("Download batch CSV", records_csv(records), "SmartBanana_batch.csv", "text/csv")
        summary = pd.DataFrame(records)["ripeness_stage"].value_counts().rename_axis("Ripeness").to_frame("Images")
        st.bar_chart(summary, color="#65a30d")
        for index, result in enumerate(results):
            with st.expander(result.record["file_name"]):
                show_result(result, f"batch_{index}")

elif page == "Video":
    st.subheader("Video frame analysis")
    video = st.file_uploader("Upload an MP4 video", ["mp4"])
    interval = st.slider("Sample every N frames", 5, 120, 30)
    maximum = st.slider("Maximum sampled frames", 5, 100, 30)
    if video and st.button("Analyse video", type="primary"):
        results, errors = [], 0
        suffix = Path(video.name).suffix
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
                    result = assess_image(frame, f"{video.name} frame {frame_index}", "video frame")
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
        st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)
        st.download_button("Download video results", records_csv(records), "SmartBanana_video.csv", "text/csv")
        st.image(rgb(results[-1].overlay), caption="Last sampled frame", use_container_width=True)

elif page == "History":
    st.subheader("Inspection history")
    records = load_history(DB_PATH)
    if records:
        st.dataframe(pd.DataFrame(records), use_container_width=True, hide_index=True)
        st.download_button("Download history CSV", records_csv(records), "SmartBanana_history.csv", "text/csv")
        if st.checkbox("I want to delete all history") and st.button("Delete history"):
            clear_history(DB_PATH)
            st.rerun()
    else:
        st.info("No inspections have been saved yet.")

else:
    st.subheader("How SmartBanana works")
    st.write("SmartBanana follows the assignment report exactly: image resizing → Gaussian filtering → CLAHE → HSV segmentation → morphology → largest-contour isolation → peel-colour and connected-defect analysis → rule-based recommendations.")
    st.markdown("""
    **Best capture conditions**

    - Show one complete banana against a plain, contrasting background.
    - Use even lighting and avoid strong shadows, blur, and yellow background objects.
    - Supported inputs are JPG/JPEG/PNG images and MP4 videos.

    **Interpretation**

    - Grade A: less than 5% detected defect area.
    - Grade B: 5% to less than 10% detected defect area.
    - Grade C: 10% or more detected defect area.

    Results describe visible external condition only; they are not a food-safety or internal-quality test.
    """)

