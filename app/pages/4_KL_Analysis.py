from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.config.config import settings
from src.data.dataloader import get_mnist_dataloaders
from src.evaluation.kl_analysis import collect_mu_logvar, compute_kl_per_dimension
from utils.model_loader import load_vae_model
from utils.streamlit_helpers import render_app_header, set_app_style, show_metric_grid

set_app_style()
render_app_header("KL Analysis", "Inspect the average KL divergence for each latent dimension and flag dead dimensions.")

model = load_vae_model()
_, test_loader = get_mnist_dataloaders(batch_size=settings.batch_size, download=True)
mu, logvar = collect_mu_logvar(model, test_loader)
kl_values = compute_kl_per_dimension(mu, logvar)

dead_threshold = st.slider("Dead dimension threshold", min_value=0.0, max_value=0.25, value=0.01, step=0.01)
dead_dimensions = [index + 1 for index, value in enumerate(kl_values) if value < dead_threshold]

show_metric_grid(
    {
        "Latent Dimensions": float(len(kl_values)),
        "Dead Dimensions": float(len(dead_dimensions)),
    }
)

fig, axis = plt.subplots(figsize=(10, 4))
x_positions = np.arange(len(kl_values))
bar_colors = ["#d62728" if value < dead_threshold else "#4c78a8" for value in kl_values]
axis.bar(x_positions, kl_values, color=bar_colors)
axis.set_xlabel("Latent Dimension")
axis.set_ylabel("Mean KL Divergence")
axis.set_title("KL Divergence Per Latent Dimension")
axis.set_xticks(x_positions)
axis.set_xticklabels([str(index + 1) for index in x_positions])
axis.grid(axis="y", linestyle="--", alpha=0.3)
fig.tight_layout()
st.pyplot(fig, clear_figure=True)

if dead_dimensions:
    st.warning(f"Dead dimensions below threshold {dead_threshold:.2f}: {dead_dimensions}")
else:
    st.success("No dead dimensions detected at the current threshold.")
