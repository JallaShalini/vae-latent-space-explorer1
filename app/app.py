from __future__ import annotations

import streamlit as st

from utils.bootstrap import PROJECT_ROOT  # noqa: F401
from utils.streamlit_helpers import render_app_header, set_app_style


# Page config must be the first Streamlit command in the main script
st.set_page_config(page_title="VAE Latent Space Explorer", page_icon="🧠", layout="wide")

set_app_style()

render_app_header(
    "VAE Latent Space Explorer",
    "Inspect the learned latent geometry, generate samples, and diagnose posterior collapse.",
)

st.write(
    "Use the pages in the left sidebar to explore the latent map, manipulate latent dimensions, inspect reconstructions, and review KL diagnostics."
)

st.info("The trained checkpoint is loaded from models/vae.pt and the MNIST test set is used for all analysis views.")
