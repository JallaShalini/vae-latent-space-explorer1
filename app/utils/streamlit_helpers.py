"""Common Streamlit UI helpers."""

from __future__ import annotations

import streamlit as st


def set_app_style() -> None:
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 2rem;
            }
            .app-title {
                font-size: 2.2rem;
                font-weight: 700;
                margin-bottom: 0.25rem;
            }
            .app-subtitle {
                color: rgba(255, 255, 255, 0.7);
                margin-bottom: 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_app_header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="app-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="app-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def show_metric_grid(metrics: dict[str, float]) -> None:
    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics.items(), strict=False):
        column.metric(label, f"{value:.4f}")
