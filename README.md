# SmartBanana

SmartBanana is an explainable, classical image-processing system for banana ripeness classification and external quality assessment. It implements the submitted assignment report without a trained AI model.

## Features

- JPG/JPEG/PNG upload and camera capture
- Four outcomes: Unripe, Ripe, Overripe, and Spoiled
- Gaussian filtering, CLAHE, HSV segmentation, morphology, and contour isolation
- Peel-colour percentages and connected-component defect measurement
- Grade A/B/C, shelf-life estimate, recommendation, and confidence score
- Batch image and MP4 frame processing
- SQLite history, CSV export, dashboard charts, and illustrated PDF reports

## Run

```powershell
py -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\streamlit run app.py
```

Open the local URL shown by Streamlit (normally `http://localhost:8501`). For reliable results, use one clearly visible banana on a plain contrasting background with even lighting.

## Test

```powershell
.venv\Scripts\python -m pytest -q
```

## Method and limits

The pipeline resizes each image, applies a 5×5 Gaussian filter and CLAHE, segments banana-coloured pixels in HSV, cleans the mask with opening/closing, and retains the largest contour. HSV colour ranges determine ripeness, while connected dark regions determine defect percentage. Following the report flowchart, defect area below 5% is Grade A, below 10% is Grade B, and 10% or more is Grade C.

This is an external visual assessment, not a food-safety test. Complex backgrounds, yellow background objects, overlapping fruit, extreme lighting, blur, and hidden fruit can reduce reliability. Thresholds are intentionally kept in `smartbanana.py` so they can be calibrated for the final camera environment.
