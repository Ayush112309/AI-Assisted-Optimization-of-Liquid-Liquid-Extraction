"""
Heatmap visualizations for extraction stage results.

Provides annotated Seaborn heatmaps for:
    - Stage-wise compositions (A, C, B in raffinate and extract)
    - Flow rates (R and E per stage)
    - Percent removal of solute (per-stage and cumulative)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from mass_transfer.core.crosscurrent import CrosscurrentResult


def composition_heatmap(
    result: CrosscurrentResult,
    phase: str = "raffinate",
    title: Optional[str] = None,
    figsize: tuple = (12, 5),
) -> Figure:
    """
    Create an annotated heatmap of stage compositions.

    Parameters
    ----------
    result : CrosscurrentResult from solver
    phase : "raffinate" or "extract"
    title : optional custom title
    figsize : figure size

    Returns
    -------
    Matplotlib Figure.
    """
    stages = result.stages
    n = len(stages)
    stage_labels = [f"Stage {s.stage_number}" for s in stages]

    if phase == "raffinate":
        data = {
            "A (Carrier)": [s.A_raff for s in stages],
            "C (Solute)": [s.C_raff for s in stages],
            "B (Solvent)": [s.B_raff for s in stages],
        }
        default_title = "Raffinate Composition (wt%) by Stage"
    else:
        data = {
            "A (Carrier)": [s.A_ext for s in stages],
            "C (Solute)": [s.C_ext for s in stages],
            "B (Solvent)": [s.B_ext for s in stages],
        }
        default_title = "Extract Composition (wt%) by Stage"

    df = pd.DataFrame(data, index=stage_labels).T

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.heatmap(
        df, annot=True, fmt=".2f", cmap="YlOrRd",
        linewidths=0.5, ax=ax, cbar_kws={"label": "wt%"},
    )
    ax.set_title(title or default_title, fontsize=14)
    ax.set_ylabel("Component")
    ax.set_xlabel("Stage")

    fig.tight_layout()
    return fig


def flowrate_heatmap(
    result: CrosscurrentResult,
    title: str = "Flow Rates (kg) by Stage",
    figsize: tuple = (12, 4),
) -> Figure:
    """
    Create a heatmap of raffinate and extract flow rates per stage.
    """
    stages = result.stages
    stage_labels = [f"Stage {s.stage_number}" for s in stages]

    data = {
        "Raffinate (R)": [s.R_flow for s in stages],
        "Extract (E)": [s.E_flow for s in stages],
    }
    df = pd.DataFrame(data, index=stage_labels).T

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.heatmap(
        df, annot=True, fmt=".1f", cmap="Blues",
        linewidths=0.5, ax=ax, cbar_kws={"label": "kg"},
    )
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Stream")
    ax.set_xlabel("Stage")

    fig.tight_layout()
    return fig


def removal_heatmap(
    result: CrosscurrentResult,
    title: str = "Solute Removal (%) by Stage",
    figsize: tuple = (12, 4),
) -> Figure:
    """
    Create a heatmap showing per-stage and cumulative % removal of solute.
    """
    stages = result.stages
    stage_labels = [f"Stage {s.stage_number}" for s in stages]

    data = {
        "Per-Stage Removal (%)": [s.pct_removal_stage for s in stages],
        "Cumulative Removal (%)": [s.pct_removal_cumul for s in stages],
    }
    df = pd.DataFrame(data, index=stage_labels).T

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.heatmap(
        df, annot=True, fmt=".2f", cmap="Greens",
        linewidths=0.5, ax=ax, cbar_kws={"label": "%"},
    )
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Metric")
    ax.set_xlabel("Stage")

    fig.tight_layout()
    return fig


def combined_heatmap(
    result: CrosscurrentResult,
    title: str = "Crosscurrent Extraction Summary",
    figsize: tuple = (14, 10),
) -> Figure:
    """Create a combined figure with all three heatmaps."""
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    stages = result.stages
    stage_labels = [f"S{s.stage_number}" for s in stages]

    # Composition (raffinate)
    comp_data = pd.DataFrame({
        "A (wt%)": [s.A_raff for s in stages],
        "C (wt%)": [s.C_raff for s in stages],
        "B (wt%)": [s.B_raff for s in stages],
        "X (sf)": [s.X_raff for s in stages],
    }, index=stage_labels).T
    sns.heatmap(comp_data, annot=True, fmt=".2f", cmap="YlOrRd",
                linewidths=0.5, ax=axes[0], cbar_kws={"label": "Value"})
    axes[0].set_title("Raffinate Composition", fontsize=12)

    # Flow rates
    flow_data = pd.DataFrame({
        "R (kg)": [s.R_flow for s in stages],
        "E (kg)": [s.E_flow for s in stages],
    }, index=stage_labels).T
    sns.heatmap(flow_data, annot=True, fmt=".1f", cmap="Blues",
                linewidths=0.5, ax=axes[1], cbar_kws={"label": "kg"})
    axes[1].set_title("Flow Rates", fontsize=12)

    # Removal
    rem_data = pd.DataFrame({
        "Stage (%)": [s.pct_removal_stage for s in stages],
        "Cumul. (%)": [s.pct_removal_cumul for s in stages],
    }, index=stage_labels).T
    sns.heatmap(rem_data, annot=True, fmt=".2f", cmap="Greens",
                linewidths=0.5, ax=axes[2], cbar_kws={"label": "%"})
    axes[2].set_title("Solute Removal", fontsize=12)

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    return fig
