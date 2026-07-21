"""Interactive colour-region evidence for the SmartBanana Streamlit interface."""

from __future__ import annotations

import base64

import cv2
import numpy as np
import streamlit as st


_HTML = """
<div class="analysis-card">
  <div class="image-wrap">
    <img class="analysis-image" alt="Colour-coded banana analysis" />
    <div class="tooltip" role="status" aria-live="polite"></div>
  </div>
  <div class="legend" aria-label="Analysis colour legend">
    <span><i class="green"></i>Green peel</span>
    <span><i class="yellow"></i>Yellow peel</span>
    <span><i class="red"></i>Detected defect</span>
  </div>
</div>
"""

_CSS = """
.analysis-card {
  max-width: 240px;
  margin: 0 auto;
  color: var(--st-text-color);
  font-family: var(--st-font), sans-serif;
}
.image-wrap {
  position: relative;
  width: 100%;
  overflow: hidden;
  border: 1px solid var(--st-border-color);
  border-radius: var(--st-base-radius);
  background: var(--st-secondary-background-color);
}
.analysis-image {
  display: block;
  width: 100%;
  height: auto;
  cursor: crosshair;
}
.tooltip {
  display: none;
  position: absolute;
  z-index: 2;
  pointer-events: none;
  padding: 6px 9px;
  border-radius: 6px;
  color: white;
  background: rgba(23, 23, 23, 0.92);
  font-size: 13px;
  font-weight: 600;
  white-space: nowrap;
  transform: translate(12px, 12px);
}
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 14px;
  margin-top: 8px;
  font-size: 13px;
}
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { width: 12px; height: 12px; border-radius: 3px; }
.green { background: #22c55e; }
.yellow { background: #facc15; }
.red { background: #ef4444; }
"""

_JS = """
export default function(component) {
  const { data, parentElement } = component
  const image = parentElement.querySelector('.analysis-image')
  const wrapper = parentElement.querySelector('.image-wrap')
  const tooltip = parentElement.querySelector('.tooltip')
  if (!image || !wrapper || !tooltip) return

  const maskImage = new Image()
  const maskCanvas = document.createElement('canvas')
  const maskContext = maskCanvas.getContext('2d', { willReadFrequently: true })
  const regions = {
    1: ['Green peel', data.greenPercentage],
    2: ['Yellow peel', data.yellowPercentage],
    3: ['Detected defect', data.defectPercentage],
  }

  image.src = data.imageUrl
  maskImage.onload = () => {
    maskCanvas.width = maskImage.naturalWidth
    maskCanvas.height = maskImage.naturalHeight
    maskContext.drawImage(maskImage, 0, 0)
  }
  maskImage.src = data.maskUrl

  image.onmousemove = (event) => {
    if (!maskCanvas.width || !image.clientWidth || !image.clientHeight) return
    const bounds = image.getBoundingClientRect()
    const x = Math.min(maskCanvas.width - 1, Math.max(0, Math.floor((event.clientX - bounds.left) * maskCanvas.width / bounds.width)))
    const y = Math.min(maskCanvas.height - 1, Math.max(0, Math.floor((event.clientY - bounds.top) * maskCanvas.height / bounds.height)))
    const region = regions[maskContext.getImageData(x, y, 1, 1).data[0]]
    if (!region) {
      tooltip.style.display = 'none'
      return
    }
    tooltip.textContent = `${region[0]}: ${region[1]}% of banana surface`
    tooltip.style.left = `${event.clientX - bounds.left}px`
    tooltip.style.top = `${event.clientY - bounds.top}px`
    tooltip.style.display = 'block'
  }
  image.onmouseleave = () => { tooltip.style.display = 'none' }

  return () => {
    image.onmousemove = null
    image.onmouseleave = null
    maskImage.onload = null
  }
}
"""

_COLOUR_OVERLAY = st.components.v2.component(
    "smartbanana_colour_overlay_thumbnail",
    html=_HTML,
    css=_CSS,
    js=_JS,
)


def _png_data_url(image: np.ndarray) -> str:
    """Encode an OpenCV image for direct use by the browser component."""
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("The analysis image could not be displayed.")
    payload = base64.b64encode(encoded).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _thumbnail_images(
    overlay: np.ndarray,
    region_map: np.ndarray,
    maximum: int = 240,
) -> tuple[np.ndarray, np.ndarray]:
    """Downscale the visible image and hover map without changing label values."""
    height, width = overlay.shape[:2]
    scale = min(1.0, maximum / max(height, width))
    if scale == 1.0:
        return overlay, region_map
    size = (round(width * scale), round(height * scale))
    return (
        cv2.resize(overlay, size, interpolation=cv2.INTER_AREA),
        cv2.resize(region_map, size, interpolation=cv2.INTER_NEAREST),
    )


def show_colour_overlay(
    overlay: np.ndarray,
    region_map: np.ndarray,
    *,
    green_percentage: float,
    yellow_percentage: float,
    defect_percentage: float,
    key: str,
) -> None:
    """Display the annotated image with percentage tooltips under the cursor."""
    # Resize the source data as well as the CSS box. This keeps the component
    # genuinely compact and avoids sending a full-resolution base64 image.
    overlay, region_map = _thumbnail_images(overlay, region_map)

    _COLOUR_OVERLAY(
        key=key,
        width=240,
        data={
            "imageUrl": _png_data_url(overlay),
            "maskUrl": _png_data_url(region_map),
            "greenPercentage": green_percentage,
            "yellowPercentage": yellow_percentage,
            "defectPercentage": defect_percentage,
        },
    )
