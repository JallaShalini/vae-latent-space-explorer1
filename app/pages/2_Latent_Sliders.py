from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import streamlit as st
from PIL import Image

from src.config.config import settings
from utils.streamlit_helpers import render_app_header, set_app_style
from utils.latent_space import decode_latent_vector

set_app_style()
render_app_header("Latent Sliders", "Move each latent dimension and watch the generated digit update in real time.")

with st.sidebar:
    st.subheader("Latent Controls")
    slider_values = []
    for index in range(settings.latent_dim):
        slider_values.append(st.slider(f"z{index + 1}", min_value=-3.0, max_value=3.0, value=0.0, step=0.1))

latent_vector = np.array(slider_values, dtype=np.float32)
generated_image = decode_latent_vector(latent_vector)

# Upscale the 28x28 image for a clearer display and convert to uint8
img_uint8 = (np.clip(generated_image, 0.0, 1.0) * 255.0).astype(np.uint8)
pil_img = Image.fromarray(img_uint8)
display_size = (560, 560)
pil_img = pil_img.resize(display_size, resample=Image.BILINEAR)
st.image(pil_img, caption="Generated image", clamp=True, use_column_width=True)
