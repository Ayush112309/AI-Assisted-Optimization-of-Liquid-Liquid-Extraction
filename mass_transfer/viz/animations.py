"""
Animated visualizations of extraction processes.

Creates Matplotlib FuncAnimation objects that can be:
  - played live inside a Qt FigureCanvas (via canvas.draw + timer)
  - saved as GIF using PillowWriter (no ffmpeg required)

Four animation types:
  1. Stage-by-stage X-Y stepping (crosscurrent or countercurrent)
  2. Ternary diagram build-up (stages appearing on right-angle triangle)
  3. Composition profile evolution (bar chart growing per stage)
  4. Parameter sensitivity sweep (solve + render for varying parameter)
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from mass_transfer.core.equilibrium import EquilibriumModel


# ============================================================================
# 1. Stage-by-Stage X-Y Stepping
# ============================================================================

def animate_xy_stepping(
    eq_model: "EquilibriumModel",
    stages: list,
    result,
    *,
    fps: int = 3,
    figsize: tuple = (10, 8),
) -> FuncAnimation:
    """
    Animate the extraction stepping on the X-Y distribution diagram.

    Each frame adds one stage to the diagram (staircase for crosscurrent,
    dot+line for countercurrent).

    Parameters
    ----------
    eq_model : fitted EquilibriumModel
    stages : list of stage objects from a solver result
    result : the full CrosscurrentResult or CountercurrentResult
    fps : frames per second
    figsize : figure size

    Returns
    -------
    FuncAnimation that can be saved via .save() or played in Qt.
    """
    from mass_transfer.core.crosscurrent import CrosscurrentResult

    data = eq_model.tie_line_data
    is_cc = isinstance(result, CrosscurrentResult)

    fig, ax = plt.subplots(figsize=figsize)

    # Static background: equilibrium curve + 45° line
    X_dense = np.linspace(0, max(data.X) * 1.05, 300)
    Y_dense = np.array([eq_model.Y_from_X(float(x)) for x in X_dense])
    mv = max(max(data.X), max(data.Y)) * 1.1

    ax.plot(X_dense, Y_dense, "b-", lw=2, label="Equilibrium curve")
    ax.plot([0, mv], [0, mv], "k--", lw=1, alpha=0.5, label="Y = X")
    ax.set_xlabel("X = C/(A+C) in Raffinate", fontsize=12)
    ax.set_ylabel("Y = C/(A+C) in Extract", fontsize=12)
    ax.set_xlim(-0.01, mv)
    ax.set_ylim(-0.01, mv)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")

    # Annotation text box
    info_text = ax.text(
        0.98, 0.02, "", transform=ax.transAxes, fontsize=10,
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.85),
    )

    # Elements that get added per frame
    lines = []
    dots = []

    if is_cc:
        feed_X = result.feed_C / (result.feed_A + result.feed_C) \
            if (result.feed_A + result.feed_C) > 0 else 0.0
        ax.plot(feed_X, 0, "gs", ms=10, zorder=5, label="Feed")
        ax.set_title("Crosscurrent Extraction — Stage-by-Stage Animation", fontsize=13)
    else:
        ax.set_title("Countercurrent Extraction — Stage-by-Stage Animation", fontsize=13)

    n_frames = len(stages) + 2  # +1 blank + 1 hold at end

    def _update(frame):
        if frame == 0:
            info_text.set_text("Starting extraction…")
            return

        idx = frame - 1
        if idx >= len(stages):
            info_text.set_text(
                f"Complete — {len(stages)} stages\n"
                + (f"Total removal: {result.total_pct_removal:.2f}%"
                   if is_cc else f"Final X: {stages[-1].X_raff:.4f}")
            )
            return

        s = stages[idx]

        if is_cc:
            # Vertical line: X_raff → Y_ext at same X
            ln_v, = ax.plot([s.X_raff, s.X_raff], [s.X_raff if idx == 0 else stages[idx-1].Y_ext if False else s.X_raff, s.Y_ext],
                            color="#d62728", lw=2, zorder=4)
            # Actually for crosscurrent, each stage is independent (fresh solvent).
            # Draw vertical from X-axis (or from previous Y on op-line) up to eq curve.
            # Simpler: just show the point
            dot, = ax.plot(s.X_raff, s.Y_ext, "o", color="#d62728", ms=8, zorder=5)
            ax.annotate(f" S{s.stage_number}", (s.X_raff, s.Y_ext), fontsize=9,
                        color="#d62728", fontweight="bold")
            lines.append(ln_v)
            dots.append(dot)

            info_text.set_text(
                f"Stage {s.stage_number}/{len(stages)}\n"
                f"X_raff = {s.X_raff:.4f}\n"
                f"Y_ext  = {s.Y_ext:.4f}\n"
                f"Removal = {s.pct_removal_cumul:.2f}%"
            )
        else:
            dot, = ax.plot(s.X_raff, s.Y_ext, "o", color="#d62728", ms=8, zorder=5)
            ax.annotate(f" S{s.stage_number}", (s.X_raff, s.Y_ext), fontsize=9,
                        color="#d62728", fontweight="bold")
            if idx > 0:
                prev = stages[idx - 1]
                ln, = ax.plot([prev.X_raff, s.X_raff], [prev.Y_ext, s.Y_ext],
                              "-", color="#d62728", lw=1.5, zorder=4)
                lines.append(ln)
            dots.append(dot)

            info_text.set_text(
                f"Stage {s.stage_number}/{len(stages)}\n"
                f"X_raff = {s.X_raff:.4f}\n"
                f"Y_ext  = {s.Y_ext:.4f}\n"
                f"Section: {getattr(s, 'section', 'N/A')}"
            )

    anim = FuncAnimation(fig, _update, frames=n_frames,
                         interval=1000 // fps, repeat=False)
    anim._fig = fig  # keep reference so caller can access figure
    return anim


# ============================================================================
# 2. Ternary Diagram Build-up
# ============================================================================

def animate_ternary_buildup(
    eq_model: "EquilibriumModel",
    stages: list,
    result,
    *,
    fps: int = 3,
    figsize: tuple = (10, 9),
) -> FuncAnimation:
    """
    Animate stages appearing on the right-angle ternary diagram (B vs C axes).
    """
    from mass_transfer.core.crosscurrent import CrosscurrentResult

    data = eq_model.tie_line_data
    is_cc = isinstance(result, CrosscurrentResult)
    fig, ax = plt.subplots(figsize=figsize)

    # Phase envelope curves
    B_raff_d = np.linspace(min(data.B_raff), max(data.B_raff), 200)
    C_raff_d = np.array([eq_model.C_raff_from_B(b) for b in B_raff_d])
    B_ext_d = np.linspace(min(data.B_ext), max(data.B_ext), 200)
    C_ext_d = np.array([eq_model.C_ext_from_B(b) for b in B_ext_d])

    ax.plot(B_raff_d, C_raff_d, "b-", lw=2, label="Raffinate curve")
    ax.plot(B_ext_d, C_ext_d, "r-", lw=2, label="Extract curve")

    # Tie lines
    for i in range(len(data.B_raff)):
        ax.plot([data.B_raff[i], data.B_ext[i]],
                [data.C_raff[i], data.C_ext[i]], "k--", lw=0.6, alpha=0.35)

    # Triangle boundary
    ax.plot([0, 100], [0, 0], "k-", lw=1.5)
    ax.plot([0, 0], [0, 100], "k-", lw=1.5)
    ax.plot([100, 0], [0, 100], "k-", lw=1.5)

    ax.set_xlabel("wt% B (Solvent / Propane)", fontsize=12)
    ax.set_ylabel("wt% C (Solute / Oleic Acid)", fontsize=12)
    ax.set_xlim(-2, 105); ax.set_ylim(-2, 105)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("Ternary Diagram — Stage Build-up Animation", fontsize=13)

    info_text = ax.text(
        0.98, 0.6, "", transform=ax.transAxes, fontsize=10,
        ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9),
    )

    n_frames = len(stages) + 2

    def _update(frame):
        if frame == 0:
            info_text.set_text("Loading stages…")
            return
        idx = frame - 1
        if idx >= len(stages):
            info_text.set_text(f"All {len(stages)} stages shown")
            return

        s = stages[idx]
        # Raffinate point
        ax.plot(s.B_raff, s.C_raff, "o", color="#1f77b4", ms=10, zorder=5)
        ax.annotate(f" R{s.stage_number}" if is_cc else f" S{s.stage_number}",
                    (s.B_raff, s.C_raff), fontsize=8, color="#1f77b4", fontweight="bold")

        # Extract point
        ax.plot(s.B_ext, s.C_ext, "s", color="#d62728", ms=8, zorder=5)

        # Tie-line for this stage
        ax.plot([s.B_raff, s.B_ext], [s.C_raff, s.C_ext],
                "g-", lw=1.5, alpha=0.7, zorder=4)

        info_text.set_text(
            f"Stage {s.stage_number}/{len(stages)}\n"
            f"Raff: A={s.A_raff:.1f} C={s.C_raff:.1f} B={s.B_raff:.1f}\n"
            f"Ext:  A={s.A_ext:.1f} C={s.C_ext:.1f} B={s.B_ext:.1f}"
        )

    anim = FuncAnimation(fig, _update, frames=n_frames,
                         interval=1000 // fps, repeat=False)
    anim._fig = fig
    return anim


# ============================================================================
# 3. Composition Profile Evolution
# ============================================================================

def animate_composition_profile(
    stages: list,
    result,
    *,
    fps: int = 3,
    figsize: tuple = (12, 7),
) -> FuncAnimation:
    """
    Animate a bar chart of raffinate compositions growing one stage at a time.
    """
    from mass_transfer.core.crosscurrent import CrosscurrentResult
    is_cc = isinstance(result, CrosscurrentResult)

    fig, axes = plt.subplots(1, 2, figsize=figsize, width_ratios=[3, 1])
    ax_bar, ax_removal = axes

    n_stages = len(stages)
    x_pos = np.arange(1, n_stages + 1)
    bar_width = 0.25
    colors_A = "#4e79a7"
    colors_C = "#e15759"
    colors_B = "#76b7b2"

    ax_bar.set_xlim(0.3, n_stages + 0.7)
    if is_cc:
        max_comp = max(max(s.A_raff for s in stages), max(s.B_raff for s in stages)) * 1.1
    else:
        max_comp = 100
    ax_bar.set_ylim(0, max_comp + 5)
    ax_bar.set_xlabel("Stage Number", fontsize=12)
    ax_bar.set_ylabel("Raffinate Composition (wt%)", fontsize=12)
    ax_bar.set_title("Raffinate Composition — Stage-by-Stage Build-up", fontsize=13)
    ax_bar.set_xticks(x_pos)
    ax_bar.grid(True, alpha=0.3, axis="y")

    # Right panel: cumulative removal line
    ax_removal.set_xlim(0.5, n_stages + 0.5)
    ax_removal.set_ylim(0, 105)
    ax_removal.set_xlabel("Stage", fontsize=11)
    ax_removal.set_ylabel("Cumul. Removal (%)", fontsize=11)
    ax_removal.set_title("% Removal", fontsize=12)
    ax_removal.grid(True, alpha=0.3)

    info_text = fig.text(
        0.5, 0.01, "", ha="center", fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85),
    )

    n_frames = n_stages + 2  # blank + stages + hold

    # Store bar containers for clearing
    drawn_bars = []
    removal_xs, removal_ys = [], []

    def _update(frame):
        if frame == 0:
            info_text.set_text("Starting composition profile…")
            return
        idx = frame - 1
        if idx >= n_stages:
            info_text.set_text(f"All {n_stages} stages complete")
            return

        s = stages[idx]
        pos = idx + 1

        ax_bar.bar(pos - bar_width, s.A_raff, bar_width, color=colors_A,
                   label="A (Carrier)" if idx == 0 else "", edgecolor="white")
        ax_bar.bar(pos, s.C_raff, bar_width, color=colors_C,
                   label="C (Solute)" if idx == 0 else "", edgecolor="white")
        ax_bar.bar(pos + bar_width, s.B_raff, bar_width, color=colors_B,
                   label="B (Solvent)" if idx == 0 else "", edgecolor="white")

        if idx == 0:
            ax_bar.legend(fontsize=9, loc="upper right")

        if is_cc:
            removal_xs.append(pos)
            removal_ys.append(s.pct_removal_cumul)
            ax_removal.plot(removal_xs, removal_ys, "o-", color="#e15759", lw=2, ms=6)

            info_text.set_text(
                f"Stage {s.stage_number}: A={s.A_raff:.1f}% C={s.C_raff:.1f}% "
                f"B={s.B_raff:.1f}%  |  Cumul. removal = {s.pct_removal_cumul:.1f}%"
            )
        else:
            info_text.set_text(
                f"Stage {s.stage_number}: A={s.A_raff:.1f}% C={s.C_raff:.1f}% B={s.B_raff:.1f}%"
            )

    anim = FuncAnimation(fig, _update, frames=n_frames,
                         interval=1000 // fps, repeat=False)
    anim._fig = fig
    return anim


# ============================================================================
# 4. Parameter Sensitivity Sweep
# ============================================================================

def animate_parameter_sweep(
    eq_model: "EquilibriumModel",
    sweep_var: str = "solvent_per_stage",
    sweep_range: Tuple[float, float] = (100, 5000),
    fixed_vals: Optional[dict] = None,
    n_frames: int = 30,
    *,
    fps: int = 3,
    figsize: tuple = (13, 8),
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> FuncAnimation:
    """
    Animate the effect of varying one parameter on extraction results.

    For each frame, solves a crosscurrent extraction and renders the
    X-Y diagram + a rolling % removal curve.

    Parameters
    ----------
    eq_model : fitted EquilibriumModel
    sweep_var : parameter to vary ("n_stages", "solvent_per_stage", "feed_acid_pct")
    sweep_range : (min, max) for the swept parameter
    fixed_vals : dict with fixed values for the other two parameters
    n_frames : number of animation frames
    fps : frames per second
    progress_callback : optional (current, total) callback
    """
    from mass_transfer.core.crosscurrent import solve_crosscurrent

    if fixed_vals is None:
        fixed_vals = {
            "n_stages": 5,
            "solvent_per_stage": 1000.0,
            "feed_acid_pct": 25.0,
        }

    var_labels = {
        "n_stages": "Number of Stages",
        "solvent_per_stage": "Solvent per Stage (kg)",
        "feed_acid_pct": "Feed Acid (%)",
    }

    sweep_vals = np.linspace(sweep_range[0], sweep_range[1], n_frames)

    # Pre-solve all frames
    results_list = []
    for i, sv in enumerate(sweep_vals):
        if progress_callback:
            progress_callback(i + 1, n_frames)
        params = dict(fixed_vals)
        params[sweep_var] = float(sv)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                n_stg = max(1, int(round(params["n_stages"])))
                res = solve_crosscurrent(
                    feed_A=100.0 - params["feed_acid_pct"],
                    feed_C=params["feed_acid_pct"],
                    feed_flow=100.0,
                    solvent_per_stage=params["solvent_per_stage"],
                    n_stages=n_stg,
                    eq_model=eq_model,
                )
                results_list.append((sv, res))
        except Exception:
            results_list.append((sv, None))

    # Set up figure
    data = eq_model.tie_line_data
    X_dense = np.linspace(0, max(data.X) * 1.05, 300)
    Y_dense = np.array([eq_model.Y_from_X(float(x)) for x in X_dense])
    mv = max(max(data.X), max(data.Y)) * 1.1

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.3, wspace=0.3)
    ax_xy = fig.add_subplot(gs[0, 0])
    ax_ternary = fig.add_subplot(gs[0, 1])
    ax_metric = fig.add_subplot(gs[1, :])

    # Static parts
    ax_xy.set_xlim(-0.01, mv); ax_xy.set_ylim(-0.01, mv)
    ax_xy.set_xlabel("X (raffinate)", fontsize=11)
    ax_xy.set_ylabel("Y (extract)", fontsize=11)
    ax_xy.grid(True, alpha=0.3)

    # Ternary
    B_raff_d = np.linspace(min(data.B_raff), max(data.B_raff), 200)
    C_raff_d = [eq_model.C_raff_from_B(b) for b in B_raff_d]
    B_ext_d = np.linspace(min(data.B_ext), max(data.B_ext), 200)
    C_ext_d = [eq_model.C_ext_from_B(b) for b in B_ext_d]

    ax_ternary.set_xlim(-2, 105); ax_ternary.set_ylim(-2, 105)
    ax_ternary.set_xlabel("wt% B (Solvent)", fontsize=10)
    ax_ternary.set_ylabel("wt% C (Solute)", fontsize=10)
    ax_ternary.set_aspect("equal")

    # Metric axes
    ax_metric.set_xlim(sweep_range[0], sweep_range[1])
    ax_metric.set_ylim(0, 105)
    ax_metric.set_xlabel(var_labels.get(sweep_var, sweep_var), fontsize=11)
    ax_metric.set_ylabel("% Removal", fontsize=11)
    ax_metric.grid(True, alpha=0.3)

    metric_xs, metric_ys = [], []

    def _update(frame):
        ax_xy.clear()
        ax_ternary.clear()

        # Redraw static backgrounds
        ax_xy.plot(X_dense, Y_dense, "b-", lw=2, label="Equilibrium")
        ax_xy.plot([0, mv], [0, mv], "k--", lw=1, alpha=0.5, label="Y=X")
        ax_xy.set_xlim(-0.01, mv); ax_xy.set_ylim(-0.01, mv)
        ax_xy.set_xlabel("X (raffinate)", fontsize=11)
        ax_xy.set_ylabel("Y (extract)", fontsize=11)
        ax_xy.grid(True, alpha=0.3)

        ax_ternary.plot(B_raff_d, C_raff_d, "b-", lw=2)
        ax_ternary.plot(B_ext_d, C_ext_d, "r-", lw=2)
        ax_ternary.plot([0, 100], [0, 0], "k-", lw=1.5)
        ax_ternary.plot([0, 0], [0, 100], "k-", lw=1.5)
        ax_ternary.plot([100, 0], [0, 100], "k-", lw=1.5)
        ax_ternary.set_xlim(-2, 105); ax_ternary.set_ylim(-2, 105)
        ax_ternary.set_xlabel("wt% B", fontsize=10)
        ax_ternary.set_ylabel("wt% C", fontsize=10)
        ax_ternary.set_aspect("equal")

        sv, res = results_list[frame]

        if res is not None:
            stages = res.stages
            for s in stages:
                ax_xy.plot(s.X_raff, s.Y_ext, "ro", ms=6, zorder=5)
                ax_ternary.plot(s.B_raff, s.C_raff, "bo", ms=6, zorder=5)
                ax_ternary.plot(s.B_ext, s.C_ext, "rs", ms=5, zorder=5)

            xs = [s.X_raff for s in stages]
            ys = [s.Y_ext for s in stages]
            if len(xs) > 1:
                # Just connect the dots
                pass
            removal = res.total_pct_removal

            ax_xy.set_title(
                f"{var_labels.get(sweep_var, sweep_var)} = {sv:.1f}\n"
                f"{len(stages)} stages, Removal = {removal:.1f}%",
                fontsize=11,
            )
            ax_ternary.set_title(
                f"Ternary Diagram\n{len(stages)} stages", fontsize=11
            )

            metric_xs.append(sv)
            metric_ys.append(removal)
        else:
            ax_xy.set_title(f"{var_labels.get(sweep_var, sweep_var)} = {sv:.1f}\n(solver failed)", fontsize=11)

        ax_xy.legend(fontsize=8, loc="upper left")

        # Update metric plot
        ax_metric.clear()
        ax_metric.set_xlim(sweep_range[0], sweep_range[1])
        ax_metric.set_ylim(0, 105)
        ax_metric.set_xlabel(var_labels.get(sweep_var, sweep_var), fontsize=11)
        ax_metric.set_ylabel("% Removal", fontsize=11)
        ax_metric.set_title("Process Performance vs Parameter", fontsize=11)
        ax_metric.grid(True, alpha=0.3)
        if metric_xs:
            ax_metric.plot(metric_xs, metric_ys, "o-", color="#1f77b4", lw=2, ms=5)
            # Highlight current point
            ax_metric.plot(metric_xs[-1], metric_ys[-1], "o", color="red", ms=10, zorder=5)

    anim = FuncAnimation(fig, _update, frames=len(results_list),
                         interval=1000 // fps, repeat=False)
    anim._fig = fig
    return anim


# ============================================================================
# Utility: save to GIF
# ============================================================================

def save_animation_gif(anim: FuncAnimation, path: str, fps: int = 3, dpi: int = 100):
    """Save a FuncAnimation as a GIF using Pillow."""
    writer = PillowWriter(fps=fps)
    anim.save(path, writer=writer, dpi=dpi)
