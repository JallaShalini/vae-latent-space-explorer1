"""Visualization helpers for Streamlit pages."""

from __future__ import annotations

import numpy as np
import plotly.express as px
import streamlit as st


def render_latent_scatter(embedding: np.ndarray, labels: np.ndarray, title: str):
    if embedding.size == 0:
        st.info("No latent vectors available.")
        return None

    plot_data = {
        "x": embedding[:, 0],
        "y": embedding[:, 1],
        "label": labels,
    }
    fig = px.scatter(plot_data, x="x", y="y", color="label", title=title, color_continuous_scale="Viridis")
    fig.update_traces(marker=dict(size=7, opacity=0.8))
    fig.update_layout(height=650, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)
    return fig


def show_image_pair(original, reconstructed, heatmap):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    # Use bilinear interpolation when scaling small images for smoother display
    axes[0].imshow(original, cmap="gray", interpolation="bilinear")
    axes[0].set_title("Original")
    axes[1].imshow(reconstructed, cmap="gray", interpolation="bilinear")
    axes[1].set_title("Reconstruction")
    heatmap_image = axes[2].imshow(heatmap, cmap="magma", interpolation="bilinear")
    axes[2].set_title("Error Heatmap")
    for axis in axes:
        axis.axis("off")
    figure.colorbar(heatmap_image, ax=axes[2], fraction=0.046, pad=0.04)
    st.pyplot(figure, clear_figure=True)
