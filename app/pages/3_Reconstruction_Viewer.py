from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from utils.reconstruction import get_reconstruction_sample
from utils.streamlit_helpers import render_app_header, set_app_style
from utils.visualization import show_image_pair

set_app_style()
render_app_header("Reconstruction Viewer", "Compare the original digit, reconstruction, and absolute error heatmap.")

index = st.number_input("Test image index", min_value=0, max_value=9999, value=10, step=1)
original, reconstructed, heatmap, label = get_reconstruction_sample(int(index))

st.caption(f"MNIST label: {label}")
# `get_reconstruction_sample` now returns numpy arrays shaped (H, W)
show_image_pair(original, reconstructed, heatmap)
