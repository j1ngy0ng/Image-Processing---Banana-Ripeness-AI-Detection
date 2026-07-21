# SmartBanana

SmartBanana is an explainable, classical image-processing system for banana ripeness classification and external quality assessment. It implements the submitted assignment report without a trained AI model.

## Features

- JPG/JPEG/PNG upload and camera capture
- Four outcomes: Unripe, Ripe, Overripe, and Spoiled
- Gaussian filtering, CLAHE, HSV segmentation, morphology, and contour isolation
- Peel-colour percentages and connected-component defect measurement
- BBCH-based surface-condition Grade A/B/C, referenced shelf-life guidance, recommendation, and confidence score
- Batch image and MP4 frame processing
- Optional mouse-drawn boundary correction for difficult backgrounds
- First-visit safety agreement before analysis features are available
- Interactive colour overlay with green, yellow, and defect percentages on hover
- SQLite history, dashboard charts, and illustrated single, batch, and video PDF reports

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

The pipeline resizes each image, applies a 5×5 Gaussian filter and CLAHE, segments banana-coloured pixels in HSV, cleans the mask with opening/closing, and retains the largest contour. If automatic segmentation is inaccurate, enable the manual boundary editor, click around the banana, right-click to close the polygon, and apply it. HSV colour ranges determine ripeness, while connected dark regions determine defect percentage. The decision rules follow the published BBCH scale for edible Musaceae: green-dominant fruit is unripe, yellow-dominant fruit with less than 20% brown surface is ripe, 20–50% brown surface is overripe, and more than 50% brown surface is spoiled. Project quality grades reuse the same referenced surface boundaries: Grade A is below 20%, Grade B is 20–50%, and Grade C is above 50%.

Shelf-life output follows FAO ambient-condition guidance: 4–10 days for mature-green bananas and 2–4 days for ripe bananas. Overripe and spoiled results display handling guidance because the cited source does not define a universal remaining-day value for those conditions.

Single-image, batch-image, and sampled-video results can be downloaded as illustrated PDF reports. Batch and video reports begin with an overall summary of distributions and average measurements. Each analysed image or frame then receives one landscape page containing the original image, segmentation mask, annotated analysis, measurements, recommendation, method note, and safety disclaimer. CSV export remains available only for inspection-history data.

This is an external visual assessment, not a food-safety test. Complex backgrounds, yellow background objects, overlapping fruit, extreme lighting, blur, and hidden fruit can reduce reliability. The BBCH decision boundaries remain fixed; only acquisition-dependent HSV colour bands should be recalibrated for a different camera or lighting environment.
