"""
3D response surface and contour plot visualizations.

Uses Plotly for interactive 3D surfaces and Matplotlib for static contours.
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def response_surface_3d(
    X_grid: np.ndarray,
    Y_grid: np.ndarray,
    Z_grid: np.ndarray,
    x_label: str = "X",
    y_label: str = "Y",
    z_label: str = "Z",
    title: str = "Response Surface",
    optimal_point: Optional[Tuple[float, float, float]] = None,
    colorscale: str = "Viridis",
) -> go.Figure:
    """
    Create an interactive 3D response surface plot using Plotly.

    Parameters
    ----------
    X_grid, Y_grid, Z_grid : 2D arrays from meshgrid
    x_label, y_label, z_label : axis labels
    title : plot title
    optimal_point : (x, y, z) tuple for the optimal operating point
    colorscale : Plotly colorscale name

    Returns
    -------
    Plotly Figure (can be displayed in browser or embedded in GUI).
    """
    if not HAS_PLOTLY:
        raise ImportError("Plotly is required for 3D surface plots")

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=X_grid, y=Y_grid, z=Z_grid,
        colorscale=colorscale,
        colorbar=dict(title=z_label),
        hovertemplate=(
            f"{x_label}: %{{x:.2f}}<br>"
            f"{y_label}: %{{y:.2f}}<br>"
            f"{z_label}: %{{z:.2f}}<extra></extra>"
        ),
    ))

    if optimal_point is not None:
        fig.add_trace(go.Scatter3d(
            x=[optimal_point[0]],
            y=[optimal_point[1]],
            z=[optimal_point[2]],
            mode="markers+text",
            marker=dict(size=8, color="red", symbol="diamond"),
            text=[f"Optimal: {optimal_point[2]:.2f}"],
            textposition="top center",
            name="Optimal Point",
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label,
        ),
        width=800,
        height=700,
    )

    return fig


def contour_plot(
    X_grid: np.ndarray,
    Y_grid: np.ndarray,
    Z_grid: np.ndarray,
    x_label: str = "X",
    y_label: str = "Y",
    z_label: str = "Z",
    title: str = "Contour Plot",
    optimal_point: Optional[Tuple[float, float, float]] = None,
    colorscale: str = "Viridis",
    n_contours: int = 20,
) -> go.Figure:
    """
    Create an interactive 2D contour plot using Plotly.

    Parameters
    ----------
    X_grid, Y_grid, Z_grid : 2D arrays
    x_label, y_label, z_label : axis labels
    title : plot title
    optimal_point : (x, y, z) for optimal point marker
    colorscale : Plotly colorscale
    n_contours : number of contour levels

    Returns
    -------
    Plotly Figure.
    """
    if not HAS_PLOTLY:
        raise ImportError("Plotly is required for contour plots")

    fig = go.Figure()

    fig.add_trace(go.Contour(
        x=X_grid[0, :] if X_grid.ndim == 2 else X_grid,
        y=Y_grid[:, 0] if Y_grid.ndim == 2 else Y_grid,
        z=Z_grid,
        colorscale=colorscale,
        colorbar=dict(title=z_label),
        ncontours=n_contours,
        contours=dict(showlabels=True, labelfont=dict(size=10)),
        hovertemplate=(
            f"{x_label}: %{{x:.2f}}<br>"
            f"{y_label}: %{{y:.2f}}<br>"
            f"{z_label}: %{{z:.2f}}<extra></extra>"
        ),
    ))

    if optimal_point is not None:
        fig.add_trace(go.Scatter(
            x=[optimal_point[0]],
            y=[optimal_point[1]],
            mode="markers+text",
            marker=dict(size=12, color="red", symbol="star"),
            text=[f"Optimal ({optimal_point[2]:.1f}%)"],
            textposition="top center",
            name="Optimal Point",
        ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        width=800,
        height=600,
    )

    return fig


def response_surface_matplotlib(
    X_grid: np.ndarray,
    Y_grid: np.ndarray,
    Z_grid: np.ndarray,
    x_label: str = "X",
    y_label: str = "Y",
    z_label: str = "Z",
    title: str = "Response Surface",
    optimal_point: Optional[Tuple[float, float, float]] = None,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """
    Create a static 3D surface plot using Matplotlib (for embedding in Qt).

    Returns
    -------
    Matplotlib Figure.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X_grid, Y_grid, Z_grid,
        cmap="viridis", alpha=0.8, edgecolor="none",
    )
    fig.colorbar(surf, ax=ax, label=z_label, shrink=0.6)

    if optimal_point is not None:
        ax.scatter(
            [optimal_point[0]], [optimal_point[1]], [optimal_point[2]],
            c="red", s=100, marker="*", zorder=10, label="Optimal",
        )
        ax.legend()

    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_zlabel(z_label, fontsize=11)
    ax.set_title(title, fontsize=14)

    fig.tight_layout()
    return fig


def contour_matplotlib(
    X_grid: np.ndarray,
    Y_grid: np.ndarray,
    Z_grid: np.ndarray,
    x_label: str = "X",
    y_label: str = "Y",
    z_label: str = "Z",
    title: str = "Contour Plot",
    optimal_point: Optional[Tuple[float, float, float]] = None,
    figsize: tuple = (10, 8),
    n_levels: int = 20,
) -> plt.Figure:
    """
    Create a static 2D contour plot using Matplotlib.

    Returns
    -------
    Matplotlib Figure.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    contour = ax.contourf(X_grid, Y_grid, Z_grid, levels=n_levels, cmap="viridis")
    fig.colorbar(contour, ax=ax, label=z_label)

    cs = ax.contour(X_grid, Y_grid, Z_grid, levels=n_levels, colors="k",
                    linewidths=0.5, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=8)

    if optimal_point is not None:
        ax.plot(optimal_point[0], optimal_point[1], "r*", markersize=15,
                label=f"Optimal: {z_label}={optimal_point[2]:.1f}")
        ax.legend(fontsize=10)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
