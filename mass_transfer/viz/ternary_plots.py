"""
Ternary diagram visualizations for liquid-liquid extraction.

Right-angle triangle convention (standard LLE):
    x-axis = wt% B (solvent / Propane)
    y-axis = wt% C (solute  / Oleic Acid)
    A (carrier) = 100 - B - C  (implied, not plotted)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from mass_transfer.core.equilibrium import EquilibriumModel


def plot_right_angle_triangle(
    eq_model: EquilibriumModel,
    stage_points: Optional[list] = None,
    title: str = "Right-Angle Ternary Diagram",
    figsize: tuple = (10, 8),
) -> Figure:
    """
    Plot the phase envelope on a right-angle triangle diagram.

    Parameters
    ----------
    eq_model : fitted EquilibriumModel
    stage_points : optional list of (A, C) tuples to overlay (e.g. from solver)
    title : plot title
    figsize : figure size

    Returns
    -------
    Matplotlib Figure.
    """
    data = eq_model.tie_line_data
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Phase envelope curves — plotted on B (solvent) vs C (solute) axes
    B_raff_dense = np.linspace(min(data.B_raff), max(data.B_raff), 200)
    C_raff_dense = np.array([eq_model.C_raff_from_B(b) for b in B_raff_dense])

    B_ext_dense = np.linspace(min(data.B_ext), max(data.B_ext), 200)
    C_ext_dense = np.array([eq_model.C_ext_from_B(b) for b in B_ext_dense])

    ax.plot(B_raff_dense, C_raff_dense, "b-", linewidth=2, label="Raffinate curve")
    ax.plot(B_ext_dense, C_ext_dense, "r-", linewidth=2, label="Extract curve")

    # Data points
    ax.scatter(data.B_raff, data.C_raff, c="blue", s=40, zorder=5)
    ax.scatter(data.B_ext, data.C_ext, c="red", s=40, zorder=5)

    # Tie lines  (connect raffinate to extract end of each tie line)
    for i in range(len(data.B_raff)):
        ax.plot([data.B_raff[i], data.B_ext[i]],
                [data.C_raff[i], data.C_ext[i]],
                "k--", linewidth=0.8, alpha=0.5)

    # Plait point (stored as (A, C, B), so index [2]=B, [1]=C)
    if eq_model.plait_point is not None:
        pp = eq_model.plait_point
        ax.plot(pp[2], pp[1], "g*", markersize=15, label="Plait point (est.)")

    # Stage overlay — expects (B, C) tuples when called with the corrected convention
    if stage_points is not None:
        for i, (B, C) in enumerate(stage_points):
            ax.plot(B, C, "ko", markersize=8)
            ax.annotate(f"  {i+1}", (B, C), fontsize=9)

    # Triangle boundary
    # Vertices: pure-A=(0,0), pure-B=(100,0), pure-C=(0,100)
    # Right angle at origin; hypotenuse is B+C=100 (A=0 line)
    ax.plot([0, 100], [0, 0], "k-", linewidth=1.5)   # bottom leg  (C=0)
    ax.plot([0, 0], [0, 100], "k-", linewidth=1.5)   # left leg    (B=0)
    ax.plot([100, 0], [0, 100], "k-", linewidth=1.5)  # hypotenuse  (A=0)

    ax.set_xlabel("wt% B (Solvent / Propane)", fontsize=12)
    ax.set_ylabel("wt% C (Solute / Oleic Acid)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlim(-2, 105)
    ax.set_ylim(-2, 105)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_N_vs_XY(
    eq_model: EquilibriumModel,
    title: str = "Solvent Ratio (N) vs Solvent-Free Composition",
    figsize: tuple = (10, 7),
) -> Figure:
    """
    Plot N_raff vs X and N_ext vs Y on the same axes.

    This is the Janecke diagram used for the difference-point method.
    """
    data = eq_model.tie_line_data
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Dense curves
    X_dense = np.linspace(0, max(data.X) * 1.05, 200)
    Y_dense = np.linspace(0, max(data.Y) * 1.05, 200)

    N_raff_dense = np.array([eq_model.N_raff_from_X(x) for x in X_dense])
    N_ext_dense = np.array([eq_model.N_ext_from_Y(y) for y in Y_dense])

    ax.plot(X_dense, N_raff_dense, "b-", linewidth=2, label="N_raff vs X (raffinate)")
    ax.plot(Y_dense, N_ext_dense, "r-", linewidth=2, label="N_ext vs Y (extract)")

    # Data points
    ax.scatter(data.X, data.N_raff, c="blue", s=40, zorder=5)
    ax.scatter(data.Y, data.N_ext, c="red", s=40, zorder=5)

    # Tie lines on N-X/Y diagram
    for i in range(len(data.X)):
        ax.plot([data.X[i], data.Y[i]],
                [data.N_raff[i], data.N_ext[i]],
                "k--", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("X or Y (solvent-free composition)", fontsize=12)
    ax.set_ylabel("N = B/(A+C) (solvent ratio)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_distribution(
    eq_model: EquilibriumModel,
    title: str = "Distribution Diagram (Y vs X)",
    figsize: tuple = (8, 8),
) -> Figure:
    """
    Plot the distribution curve Y vs X with a 45° reference line.
    """
    data = eq_model.tie_line_data
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Dense curve
    X_dense = np.linspace(0, max(data.X) * 1.05, 200)
    Y_dense = np.array([eq_model.Y_from_X(x) for x in X_dense])

    ax.plot(X_dense, Y_dense, "b-", linewidth=2, label="Y = f(X) distribution")
    ax.scatter(data.X, data.Y, c="blue", s=50, zorder=5, label="Data points")

    # 45° reference line
    max_val = max(max(data.X), max(data.Y)) * 1.1
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=1, alpha=0.5, label="Y = X (45° line)")

    ax.set_xlabel("X = C/(A+C) in Raffinate", fontsize=12)
    ax.set_ylabel("Y = C/(A+C) in Extract", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlim(-0.02, max_val)
    ax.set_ylim(-0.02, max_val)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_all_equilibrium(eq_model: EquilibriumModel) -> Figure:
    """Create a 2x2 subplot with all equilibrium plots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    data = eq_model.tie_line_data

    # 1. Right-angle triangle (B-C axes — standard LLE convention)
    ax = axes[0, 0]
    B_raff_dense = np.linspace(min(data.B_raff), max(data.B_raff), 200)
    C_raff_dense = [eq_model.C_raff_from_B(b) for b in B_raff_dense]
    B_ext_dense = np.linspace(min(data.B_ext), max(data.B_ext), 200)
    C_ext_dense = [eq_model.C_ext_from_B(b) for b in B_ext_dense]

    ax.plot(B_raff_dense, C_raff_dense, "b-", lw=2, label="Raffinate")
    ax.plot(B_ext_dense, C_ext_dense, "r-", lw=2, label="Extract")
    for i in range(len(data.B_raff)):
        ax.plot([data.B_raff[i], data.B_ext[i]],
                [data.C_raff[i], data.C_ext[i]], "k--", lw=0.7, alpha=0.5)
    ax.scatter(data.B_raff, data.C_raff, c="blue", s=30, zorder=5)
    ax.scatter(data.B_ext, data.C_ext, c="red", s=30, zorder=5)
    if eq_model.plait_point is not None:
        pp = eq_model.plait_point
        ax.plot(pp[2], pp[1], "g*", markersize=12, label="Plait pt.")
    ax.plot([0, 100], [0, 0], "k-", lw=1.5)
    ax.plot([0, 0], [0, 100], "k-", lw=1.5)
    ax.plot([100, 0], [0, 100], "k-", lw=1.5)
    ax.set_xlabel("wt% B (Solvent / Propane)")
    ax.set_ylabel("wt% C (Solute / Oleic Acid)")
    ax.set_title("Right-Angle Triangle"); ax.legend(fontsize=9)
    ax.set_xlim(-2, 105); ax.set_ylim(-2, 105); ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # 2. N vs X/Y
    ax = axes[0, 1]
    X_d = np.linspace(0, max(data.X)*1.05, 200)
    Y_d = np.linspace(0, max(data.Y)*1.05, 200)
    ax.plot(X_d, [eq_model.N_raff_from_X(x) for x in X_d], "b-", lw=2, label="N_raff(X)")
    ax.plot(Y_d, [eq_model.N_ext_from_Y(y) for y in Y_d], "r-", lw=2, label="N_ext(Y)")
    for i in range(len(data.X)):
        ax.plot([data.X[i], data.Y[i]], [data.N_raff[i], data.N_ext[i]], "k--", lw=0.7, alpha=0.5)
    ax.scatter(data.X, data.N_raff, c="blue", s=30, zorder=5)
    ax.scatter(data.Y, data.N_ext, c="red", s=30, zorder=5)
    ax.set_xlabel("X or Y"); ax.set_ylabel("N = B/(A+C)")
    ax.set_title("Janecke Diagram (N vs X/Y)"); ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 3. Distribution curve
    ax = axes[1, 0]
    Y_fit = [eq_model.Y_from_X(x) for x in X_d]
    ax.plot(X_d, Y_fit, "b-", lw=2, label="Y = f(X)")
    ax.scatter(data.X, data.Y, c="blue", s=50, zorder=5)
    max_v = max(max(data.X), max(data.Y))*1.1
    ax.plot([0, max_v], [0, max_v], "k--", lw=1, alpha=0.5, label="Y = X")
    ax.set_xlabel("X (raffinate)"); ax.set_ylabel("Y (extract)")
    ax.set_title("Distribution Diagram"); ax.legend(fontsize=9)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)

    # 4. R² summary
    ax = axes[1, 1]
    names = list(eq_model.r_squared.keys())
    vals = [eq_model.r_squared[n] for n in names]
    colors = ["green" if v > 0.95 else "orange" if v > 0.9 else "red" for v in vals]
    bars = ax.barh(names, vals, color=colors)
    ax.set_xlim(0.85, 1.01)
    ax.set_xlabel("R²"); ax.set_title("Curve Fit Quality")
    for bar, v in zip(bars, vals):
        ax.text(v + 0.002, bar.get_y() + bar.get_height()/2, f"{v:.4f}",
                va="center", fontsize=9)
    ax.axvline(0.95, color="gray", ls="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="x")

    fig.suptitle("Equilibrium Model Summary", fontsize=16, y=1.01)
    fig.tight_layout()
    return fig
