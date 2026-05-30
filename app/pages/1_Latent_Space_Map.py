from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from utils.bootstrap import PROJECT_ROOT as _PROJECT_ROOT  # noqa: F401
from utils.latent_space import get_latent_space_data
from utils.streamlit_helpers import render_app_header, set_app_style
from utils.visualization import render_latent_scatter

set_app_style()
render_app_header("Latent Space Map", "PCA and t-SNE projections of the encoded MNIST test set.")

latent_vectors, labels, pca_embedding, tsne_embedding = get_latent_space_data()

tab_pca, tab_tsne = st.tabs(["PCA", "t-SNE"])
with tab_pca:
    render_latent_scatter(pca_embedding, labels, "PCA Projection")

with tab_tsne:
    render_latent_scatter(tsne_embedding, labels, "t-SNE Projection")
