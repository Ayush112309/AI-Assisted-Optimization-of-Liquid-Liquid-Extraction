"""
Comparison tab: run any two extraction modes with identical parameters
and display side-by-side stage diagrams, heatmaps, and a summary table.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from ..core.equilibrium import EquilibriumModel

from .animation_tab import AnimationTab
from .ui_helpers import animate_widget_in, countercurrent_removal_percentages, draw_empty_figure


# ---------------------------------------------------------------------------
# Worker thread (same pattern as simulation_tab.py)
# ---------------------------------------------------------------------------

class _ComparisonWorker(QThread):
    """Runs a single solver and emits the result."""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, solver_func, kwargs: dict):
        super().__init__()
        self.solver_func = solver_func
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.solver_func(**self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Mode helpers
# ---------------------------------------------------------------------------

MODE_NAMES = [
    "Crosscurrent",
    "Countercurrent",
]

def _build_solver(mode_idx: int, eq_model, *, feed_A, feed_C, feed_flow,
                  solvent, n_stages):
    """Return (solver_func, kwargs) for the given mode index."""
    if mode_idx == 0:
        from ..core.crosscurrent import solve_crosscurrent
        return solve_crosscurrent, dict(
            feed_A=feed_A, feed_C=feed_C, feed_flow=feed_flow,
            solvent_per_stage=solvent, n_stages=n_stages, eq_model=eq_model,
        )
    elif mode_idx == 1:
        from ..core.countercurrent import solve_countercurrent
        return solve_countercurrent, dict(
            feed_A=feed_A, feed_C=feed_C, feed_flow=feed_flow,
            solvent_flow=solvent, n_stages=n_stages, eq_model=eq_model,
        )


# ---------------------------------------------------------------------------
# Main Comparison Tab
# ---------------------------------------------------------------------------

class ComparisonTab(QWidget):
    """Side-by-side comparison of any two extraction modes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.eq_model: Optional[EquilibriumModel] = None
        self._result_A = None
        self._result_B = None
        self._mode_A_idx = 0
        self._mode_B_idx = 1
        self._workers_done = 0
        self._worker_A: Optional[_ComparisonWorker] = None
        self._worker_B: Optional[_ComparisonWorker] = None
        self.animation_tab: Optional[AnimationTab] = None
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        root = QHBoxLayout(self)
        root.setSpacing(16)

        # ---- LEFT: shared inputs ----
        left = QVBoxLayout()
        left.setSpacing(6)

        intro = QLabel(
            "Run two extraction modes with the same feed and operating inputs to compare their stage behavior, heatmaps, and summary metrics side by side."
        )
        intro.setWordWrap(True)
        intro.setProperty("class", "sectionIntro")
        left.addWidget(intro)

        # Mode selectors
        mode_group = QGroupBox("Modes to Compare")
        mg_layout = QHBoxLayout(mode_group)
        mg_layout.setSpacing(12)

        mode_a_col = QVBoxLayout()
        mode_a_label = QLabel("Mode A")
        mode_a_label.setProperty("class", "helperText")
        self.mode_A_combo = QComboBox()
        self.mode_A_combo.addItems(MODE_NAMES)
        self.mode_A_combo.setCurrentIndex(0)
        mode_a_col.addWidget(mode_a_label)
        mode_a_col.addWidget(self.mode_A_combo)

        mode_b_col = QVBoxLayout()
        mode_b_label = QLabel("Mode B")
        mode_b_label.setProperty("class", "helperText")
        self.mode_B_combo = QComboBox()
        self.mode_B_combo.addItems(MODE_NAMES)
        self.mode_B_combo.setCurrentIndex(1)
        mode_b_col.addWidget(mode_b_label)
        mode_b_col.addWidget(self.mode_B_combo)

        mg_layout.addLayout(mode_a_col)
        mg_layout.addLayout(mode_b_col)

        left.addWidget(mode_group)

        # Feed conditions
        feed_group = QGroupBox("Feed Conditions (shared)")
        feed_layout = QFormLayout(feed_group)

        self.feed_A_spin = QDoubleSpinBox()
        self.feed_A_spin.setRange(0, 100); self.feed_A_spin.setValue(75.0)
        self.feed_A_spin.setSuffix(" wt%")
        feed_layout.addRow("Carrier (A):", self.feed_A_spin)

        self.feed_C_spin = QDoubleSpinBox()
        self.feed_C_spin.setRange(0, 100); self.feed_C_spin.setValue(25.0)
        self.feed_C_spin.setSuffix(" wt%")
        feed_layout.addRow("Solute (C):", self.feed_C_spin)

        self.feed_flow_spin = QDoubleSpinBox()
        self.feed_flow_spin.setRange(1, 1_000_000); self.feed_flow_spin.setValue(100.0)
        self.feed_flow_spin.setSuffix(" kg")
        feed_layout.addRow("Feed flow:", self.feed_flow_spin)

        left.addWidget(feed_group)

        # Operating parameters
        op_group = QGroupBox("Operating Parameters (shared)")
        self._op_layout = QFormLayout(op_group)

        self.n_stages_spin = QSpinBox()
        self.n_stages_spin.setRange(1, 50); self.n_stages_spin.setValue(3)
        self._op_layout.addRow("Stages:", self.n_stages_spin)

        self.solvent_label = QLabel("Solvent input:")
        self.solvent_spin = QDoubleSpinBox()
        self.solvent_spin.setRange(1, 1_000_000); self.solvent_spin.setValue(1000.0)
        self.solvent_spin.setSuffix(" kg")
        self.solvent_spin.setToolTip(
            "Crosscurrent: solvent per stage. Countercurrent: total solvent flow rate."
        )
        self._op_layout.addRow(self.solvent_label, self.solvent_spin)

        left.addWidget(op_group)

        # Run button
        self.run_btn = QPushButton("Run Comparison")
        self.run_btn.setProperty("class", "primary")
        self.run_btn.setMinimumHeight(44)
        self.run_btn.clicked.connect(self._run_comparison)
        left.addWidget(self.run_btn)

        self.status_label = QLabel("Choose two modes and run the comparison.")
        self.status_label.setProperty("class", "statusCard")
        self.status_label.setWordWrap(True)
        left.addWidget(self.status_label)
        left.addStretch()

        root.addLayout(left, stretch=1)

        # ---- RIGHT: results notebook ----
        self.results_tabs = QTabWidget()

        # Tab 1: Stage Diagram
        stage_widget = QWidget()
        sv = QVBoxLayout(stage_widget)
        self.stage_canvas = FigureCanvas(Figure(figsize=(14, 6)))
        sv.addWidget(self.stage_canvas)
        export_stage_btn = QPushButton("Export PNG")
        export_stage_btn.clicked.connect(lambda: self._export(self.stage_canvas))
        sv.addWidget(export_stage_btn)
        self.results_tabs.addTab(stage_widget, "Stage Diagram")

        # Tab 2: Heatmap
        heatmap_widget = QWidget()
        hv = QVBoxLayout(heatmap_widget)

        hm_ctrl = QHBoxLayout()
        self.hm_comp_btn   = QPushButton("Composition");  self.hm_comp_btn.setCheckable(True)
        self.hm_flow_btn   = QPushButton("Flow Rates");   self.hm_flow_btn.setCheckable(True)
        self.hm_rem_btn    = QPushButton("% Removal");    self.hm_rem_btn.setCheckable(True)
        self.hm_comb_btn   = QPushButton("Combined");     self.hm_comb_btn.setCheckable(True); self.hm_comb_btn.setChecked(True)
        self._hm_btns = [self.hm_comp_btn, self.hm_flow_btn, self.hm_rem_btn, self.hm_comb_btn]
        self.hm_comp_btn.clicked.connect(lambda: self._show_heatmaps("composition"))
        self.hm_flow_btn.clicked.connect(lambda: self._show_heatmaps("flowrate"))
        self.hm_rem_btn.clicked.connect(lambda: self._show_heatmaps("removal"))
        self.hm_comb_btn.clicked.connect(lambda: self._show_heatmaps("combined"))
        for b in self._hm_btns:
            hm_ctrl.addWidget(b)
        hm_ctrl.addStretch()
        export_hm_btn = QPushButton("Export PNG")
        export_hm_btn.clicked.connect(lambda: self._export(self.heatmap_canvas))
        hm_ctrl.addWidget(export_hm_btn)
        hv.addLayout(hm_ctrl)

        self.heatmap_canvas = FigureCanvas(Figure(figsize=(14, 7)))
        hv.addWidget(self.heatmap_canvas)
        self.results_tabs.addTab(heatmap_widget, "Heatmap")

        # Tab 3: Animation
        animation_widget = QWidget()
        av = QVBoxLayout(animation_widget)
        anim_ctrl = QHBoxLayout()
        anim_ctrl.addWidget(QLabel("Animate:"))
        self.anim_mode_combo = QComboBox()
        self.anim_mode_combo.addItems(["Mode A", "Mode B"])
        self.anim_mode_combo.currentIndexChanged.connect(self._on_animation_target_changed)
        anim_ctrl.addWidget(self.anim_mode_combo)
        anim_ctrl.addStretch()
        av.addLayout(anim_ctrl)
        self.animation_tab = AnimationTab(self, show_source_controls=False)
        self.animation_tab.set_solver_factory(self._build_animation_solver)
        av.addWidget(self.animation_tab)
        self.results_tabs.addTab(animation_widget, "Animation")

        # Tab 4: Summary Table
        summary_widget = QWidget()
        suv = QVBoxLayout(summary_widget)
        self.summary_table = QTableWidget()
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.summary_table.setAlternatingRowColors(True)
        suv.addWidget(self.summary_table)
        self.results_tabs.addTab(summary_widget, "Summary Table")

        root.addWidget(self.results_tabs, stretch=3)

        # Placeholder text until first run
        self._show_placeholder()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_model(self, eq_model: "EquilibriumModel"):
        self.eq_model = eq_model
        if self.animation_tab is not None:
            self.animation_tab.set_model(eq_model)
        self.status_label.setText("Equilibrium model ready. Pick two modes and compare them.")

    def _on_animation_target_changed(self, index: int):
        if self.animation_tab is None:
            return
        if index == 0 and self._result_A is not None:
            self.animation_tab.set_result(self._result_A)
        elif index == 1 and self._result_B is not None:
            self.animation_tab.set_result(self._result_B)

    def _build_animation_solver(self):
        if self.eq_model is None:
            raise ValueError("Load equilibrium data first.")

        target_idx = self.anim_mode_combo.currentIndex()
        mode_idx = self.mode_A_combo.currentIndex() if target_idx == 0 else self.mode_B_combo.currentIndex()
        solver_func, kwargs = _build_solver(
            mode_idx,
            self.eq_model,
            feed_A=self.feed_A_spin.value(),
            feed_C=self.feed_C_spin.value(),
            feed_flow=self.feed_flow_spin.value(),
            solvent=self.solvent_spin.value(),
            n_stages=self.n_stages_spin.value(),
        )
        return solver_func, kwargs

    # ------------------------------------------------------------------
    # Running solvers
    # ------------------------------------------------------------------

    def _run_comparison(self):
        if self.eq_model is None:
            QMessageBox.warning(self, "No Model", "Load equilibrium data first.")
            return

        self.run_btn.setEnabled(False)
        self.status_label.setText("Running both comparison cases…")
        self._result_A = None
        self._result_B = None
        self._workers_done = 0

        # Common params
        feed_A = self.feed_A_spin.value()
        feed_C = self.feed_C_spin.value()
        feed_flow = self.feed_flow_spin.value()
        solvent = self.solvent_spin.value()
        n_stages = self.n_stages_spin.value()

        idx_a = self.mode_A_combo.currentIndex()
        idx_b = self.mode_B_combo.currentIndex()

        func_a, kw_a = _build_solver(
            idx_a, self.eq_model,
            feed_A=feed_A, feed_C=feed_C, feed_flow=feed_flow,
            solvent=solvent, n_stages=n_stages,
        )
        func_b, kw_b = _build_solver(
            idx_b, self.eq_model,
            feed_A=feed_A, feed_C=feed_C, feed_flow=feed_flow,
            solvent=solvent, n_stages=n_stages,
        )

        self._worker_A = _ComparisonWorker(func_a, kw_a)
        self._worker_A.finished.connect(lambda r: self._on_done(r, "A"))
        self._worker_A.error.connect(lambda e: self._on_error(e, "A"))

        self._worker_B = _ComparisonWorker(func_b, kw_b)
        self._worker_B.finished.connect(lambda r: self._on_done(r, "B"))
        self._worker_B.error.connect(lambda e: self._on_error(e, "B"))

        self._worker_A.start()
        self._worker_B.start()

    def _on_done(self, result, which: str):
        if which == "A":
            self._result_A = result
        else:
            self._result_B = result
        self._workers_done += 1
        if self._workers_done == 2:
            self._render_all()

    def _on_error(self, msg: str, which: str):
        self._workers_done += 1
        self.status_label.setText(f"Error in Mode {which}: {msg}")
        QMessageBox.critical(self, f"Solver Error (Mode {which})", msg)
        if self._workers_done >= 2:
            self.run_btn.setEnabled(True)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_all(self):
        self.run_btn.setEnabled(True)
        name_a = MODE_NAMES[self.mode_A_combo.currentIndex()]
        name_b = MODE_NAMES[self.mode_B_combo.currentIndex()]
        self.status_label.setText(f"Done! Showing: {name_a} vs {name_b}")

        self._plot_stage_diagram()
        self._show_heatmaps("combined")
        self._build_summary_table()
        self._on_animation_target_changed(self.anim_mode_combo.currentIndex())
        self.results_tabs.setCurrentIndex(0)

    def _plot_stage_diagram(self):
        """Side-by-side X-Y distribution diagrams with stage stepping."""
        name_a = MODE_NAMES[self.mode_A_combo.currentIndex()]
        name_b = MODE_NAMES[self.mode_B_combo.currentIndex()]

        fig = self.stage_canvas.figure
        fig.clear()
        axes = fig.subplots(1, 2)

        for ax, result, name, color in [
            (axes[0], self._result_A, name_a, "#1f77b4"),
            (axes[1], self._result_B, name_b, "#d62728"),
        ]:
            if result is None:
                ax.text(0.5, 0.5, "Error / No Result", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(name)
                continue

            data = self.eq_model.tie_line_data
            X_d = np.linspace(0, max(data.X) * 1.05, 300)
            Y_d = [self.eq_model.Y_from_X(float(x)) for x in X_d]
            mv = max(max(data.X), max(data.Y)) * 1.1

            ax.plot(X_d, Y_d, "b-", lw=2, label="Equilibrium curve")
            ax.plot([0, mv], [0, mv], "k--", lw=1, alpha=0.5, label="Y = X (45°)")

            stages = result.stages
            if stages:
                xs = [s.X_raff for s in stages]
                ys = [s.Y_ext  for s in stages]

                from ..core.crosscurrent import CrosscurrentResult
                if isinstance(result, CrosscurrentResult):
                    # Crosscurrent: staircase stepping
                    feed_X = result.feed_C / (result.feed_A + result.feed_C) if (result.feed_A + result.feed_C) > 0 else 0
                    ax.plot(feed_X, 0, "gs", markersize=9, label="Feed", zorder=5)
                    for s in stages:
                        ax.plot([s.X_raff, s.X_raff], [s.X_raff, s.Y_ext], color=color, lw=1.5)
                        ax.plot(s.X_raff, s.Y_ext, "o", color=color, ms=5)
                        ax.annotate(f" {s.stage_number}", (s.X_raff, s.Y_ext), fontsize=7, color=color)
                else:
                    # Countercurrent: connect stage dots
                    ax.plot(xs, ys, "o-", color=color, lw=1.5, ms=5, label="Stages")
                    for s in stages:
                        ax.annotate(f" {s.stage_number}", (s.X_raff, s.Y_ext), fontsize=7, color=color)

            ax.set_xlabel("X = C/(A+C) in Raffinate", fontsize=10)
            ax.set_ylabel("Y = C/(A+C) in Extract", fontsize=10)
            ax.set_title(f"{name}\n({len(stages)} stages)", fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.01, mv)
            ax.set_ylim(-0.01, mv)

        fig.suptitle("Stage-by-Stage Comparison (X-Y Diagram)", fontsize=13, y=1.01)
        fig.tight_layout()
        self.stage_canvas.draw()
        animate_widget_in(self.stage_canvas)

    def _show_heatmaps(self, hm_type: str):
        """Render side-by-side heatmaps (or N/A panels if result is None)."""
        # Update toggle buttons
        mapping = {
            "composition": self.hm_comp_btn,
            "flowrate":    self.hm_flow_btn,
            "removal":     self.hm_rem_btn,
            "combined":    self.hm_comb_btn,
        }
        for b in self._hm_btns:
            b.setChecked(False)
        if hm_type in mapping:
            mapping[hm_type].setChecked(True)

        if self._result_A is None and self._result_B is None:
            return  # nothing to show yet

        name_a = MODE_NAMES[self.mode_A_combo.currentIndex()]
        name_b = MODE_NAMES[self.mode_B_combo.currentIndex()]

        fig = self.heatmap_canvas.figure
        fig.clear()

        if hm_type == "combined":
            # 2 columns × 3 rows
            axes = fig.subplots(3, 2)
            for col, (result, name) in enumerate([
                (self._result_A, name_a),
                (self._result_B, name_b),
            ]):
                self._draw_heatmap_axes(result, hm_type, axes[:, col], name)
        else:
            # 1 row × 2 columns
            axes = fig.subplots(1, 2)
            for col, (result, name) in enumerate([
                (self._result_A, name_a),
                (self._result_B, name_b),
            ]):
                self._draw_heatmap_axes(result, hm_type, [axes[col]], name)

        label_map = {
            "composition": "Composition (wt%)",
            "flowrate": "Flow Rates (kg)",
            "removal": "% Removal",
            "combined": "Composition / Flow Rates / % Removal",
        }
        fig.suptitle(f"Heatmap Comparison — {label_map.get(hm_type, hm_type)}", fontsize=13)
        fig.tight_layout()
        self.heatmap_canvas.draw()
        animate_widget_in(self.heatmap_canvas)

    def _draw_heatmap_axes(self, result, hm_type: str, axes_list, title_prefix: str):
        """Draw heatmap(s) for one result into a list of axes."""
        import pandas as pd
        import seaborn as sns

        if result is None:
            for ax in axes_list:
                ax.text(0.5, 0.5, "No result", ha="center", va="center",
                        transform=ax.transAxes, fontsize=11)
                ax.set_title(title_prefix)
            return

        from ..core.crosscurrent import CrosscurrentResult
        stages = result.stages
        labels = [f"S{s.stage_number}" for s in stages]
        is_cc = isinstance(result, CrosscurrentResult)

        def _make_comp_df():
            if is_cc:
                return pd.DataFrame({
                    "A (Carrier)": [s.A_raff for s in stages],
                    "C (Solute)":  [s.C_raff  for s in stages],
                    "B (Solvent)": [s.B_raff  for s in stages],
                }, index=labels).T
            else:
                return pd.DataFrame({
                    "X_raff": [s.X_raff for s in stages],
                    "Y_ext":  [s.Y_ext  for s in stages],
                    "N_raff": [s.N_raff for s in stages],
                    "N_ext":  [s.N_ext  for s in stages],
                }, index=labels).T

        def _make_flow_df():
            if is_cc:
                return pd.DataFrame({
                    "Raffinate R (kg)": [s.R_flow for s in stages],
                    "Extract E (kg)":   [s.E_flow for s in stages],
                }, index=labels).T
            return pd.DataFrame({
                "Raffinate R (kg)": [s.R_flow for s in stages],
                "Extract E (kg)":   [s.E_flow for s in stages],
            }, index=labels).T

        def _make_removal_df():
            if is_cc:
                return pd.DataFrame({
                    "Stage (%)":     [s.pct_removal_stage for s in stages],
                    "Cumulative (%)": [s.pct_removal_cumul for s in stages],
                }, index=labels).T
            stage_removals, cumulative_removals = countercurrent_removal_percentages(
                stages, getattr(result, "X_feed", 0.0)
            )
            return pd.DataFrame({
                "Stage (%)": [float(v) for v in stage_removals],
                "Cumulative (%)": [float(v) for v in cumulative_removals],
            }, index=labels).T

        kwargs_base = dict(linewidths=0.5, linecolor="white")

        if hm_type == "combined":
            ax_comp, ax_flow, ax_rem = axes_list[0], axes_list[1], axes_list[2]
            df_c = _make_comp_df()
            sns.heatmap(df_c, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax_comp,
                        cbar_kws={"label": "wt%" if is_cc else "sf val"}, **kwargs_base)
            ax_comp.set_title(f"{title_prefix}\nComposition", fontsize=9)

            df_f = _make_flow_df()
            sns.heatmap(df_f, annot=True, fmt=".1f", cmap="Blues", ax=ax_flow,
                        cbar_kws={"label": "kg"}, **kwargs_base)
            ax_flow.set_title("Flow Rates", fontsize=9)

            df_r = _make_removal_df()
            sns.heatmap(df_r, annot=True, fmt=".2f", cmap="Greens", ax=ax_rem,
                        cbar_kws={"label": "%"}, **kwargs_base)
            ax_rem.set_title("% Removal", fontsize=9)

        elif hm_type == "composition":
            ax = axes_list[0]
            sns.heatmap(_make_comp_df(), annot=True, fmt=".2f", cmap="YlOrRd", ax=ax,
                        cbar_kws={"label": "wt%" if is_cc else "sf val"}, **kwargs_base)
            ax.set_title(f"{title_prefix}\nComposition", fontsize=10)

        elif hm_type == "flowrate":
            ax = axes_list[0]
            df_f = _make_flow_df()
            sns.heatmap(df_f, annot=True, fmt=".1f", cmap="Blues", ax=ax,
                        cbar_kws={"label": "kg"}, **kwargs_base)
            ax.set_title(f"{title_prefix}\nFlow Rates", fontsize=10)

        elif hm_type == "removal":
            ax = axes_list[0]
            df_r = _make_removal_df()
            sns.heatmap(df_r, annot=True, fmt=".2f", cmap="Greens", ax=ax,
                        cbar_kws={"label": "%"}, **kwargs_base)
            ax.set_title(f"{title_prefix}\n% Removal", fontsize=10)

    def _build_summary_table(self):
        """Populate the two-column summary comparison table."""
        name_a = MODE_NAMES[self.mode_A_combo.currentIndex()]
        name_b = MODE_NAMES[self.mode_B_combo.currentIndex()]

        from ..core.crosscurrent import CrosscurrentResult

        def _extract_metrics(result, mode_idx: int) -> list[tuple[str, str]]:
            if result is None:
                return [("(error)", "—")]
            rows = []
            stages = result.stages
            rows.append(("Number of stages", str(len(stages))))

            if isinstance(result, CrosscurrentResult):
                rows.append(("Total % removal", f"{result.total_pct_removal:.2f}%"))
                rows.append(("Final raffinate X (sf)", f"{result.final_raff_X:.4f}"))
                rows.append(("Mixed extract Y (sf)", f"{result.final_ext_Y_mixed:.4f}"))
                rows.append(("Mixed extract flow (kg)", f"{result.mixed_extract_flow:.2f}"))
                rows.append(("Mixed extract A wt%", f"{result.mixed_extract_A:.2f}%"))
                rows.append(("Mixed extract C wt%", f"{result.mixed_extract_C:.2f}%"))
                rows.append(("Mixed extract B wt%", f"{result.mixed_extract_B:.2f}%"))
                # Per-stage % removal
                for s in stages:
                    rows.append((f"  Stage {s.stage_number} removal", f"{s.pct_removal_stage:.2f}%"))
                    rows.append((f"  Stage {s.stage_number} cumul.", f"{s.pct_removal_cumul:.2f}%"))
                    rows.append((f"  Stage {s.stage_number} X_raff", f"{s.X_raff:.4f}"))
                    rows.append((f"  Stage {s.stage_number} Y_ext", f"{s.Y_ext:.4f}"))
            else:
                # Countercurrent
                first_s = stages[0]
                last_s = stages[-1]

                # Use the lead raffinate and terminal extract as the countercurrent
                # analogs of the crosscurrent final/mixed outlet summaries.
                X_feed = result.X_feed
                X_raff = first_s.X_raff
                pct = None
                if X_feed > 0:
                    pct = (1 - X_raff / X_feed) * 100
                    rows.append(("Total % removal", f"{pct:.2f}%"))

                rows.append(("Final raffinate X (sf)", f"{first_s.X_raff:.4f}"))
                rows.append(("Mixed extract Y (sf)", f"{last_s.Y_ext:.4f}"))
                rows.append(("Mixed extract flow (kg)", f"{last_s.E_flow:.2f}"))
                rows.append(("Mixed extract A wt%", f"{last_s.A_ext:.2f}%"))
                rows.append(("Mixed extract C wt%", f"{last_s.C_ext:.2f}%"))
                rows.append(("Mixed extract B wt%", f"{last_s.B_ext:.2f}%"))
                rows.append(("SF feed flow (kg/h)", f"{result.feed_flow_sf:.2f}"))

                stage_removals, cumulative_removals = countercurrent_removal_percentages(
                    stages, result.X_feed
                )
                for s, stage_pct, cumul_pct in zip(stages, stage_removals, cumulative_removals):
                    rows.append((f"  Stage {s.stage_number} removal", f"{stage_pct:.2f}%"))
                    rows.append((f"  Stage {s.stage_number} cumul.", f"{cumul_pct:.2f}%"))
                    rows.append((f"  Stage {s.stage_number} X_raff", f"{s.X_raff:.4f}"))
                    rows.append((f"  Stage {s.stage_number} Y_ext", f"{s.Y_ext:.4f}"))
            return rows

        rows_a = _extract_metrics(self._result_A, self.mode_A_combo.currentIndex())
        rows_b = _extract_metrics(self._result_B, self.mode_B_combo.currentIndex())

        # Merge: union of all row labels preserving A order, then B-only
        keys_a = [r[0] for r in rows_a]
        keys_b = [r[0] for r in rows_b]
        dict_a = dict(rows_a)
        dict_b = dict(rows_b)

        all_keys = list(dict.fromkeys(keys_a + [k for k in keys_b if k not in dict_a]))

        self.summary_table.setColumnCount(3)
        self.summary_table.setHorizontalHeaderLabels(["Metric", name_a, name_b])
        self.summary_table.setRowCount(len(all_keys))

        for row, key in enumerate(all_keys):
            val_a = dict_a.get(key, "—")
            val_b = dict_b.get(key, "—")
            for col, text in enumerate([key, val_a, val_b]):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if col == 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                # Bold section headers
                if key.startswith("  ") is False and col == 0:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self.summary_table.setItem(row, col, item)

        self.summary_table.resizeColumnsToContents()
        animate_widget_in(self.summary_table)

    # ------------------------------------------------------------------
    # Placeholder & helpers
    # ------------------------------------------------------------------

    def _show_placeholder(self):
        """Show placeholder content before the first comparison run."""
        draw_empty_figure(
            self.stage_canvas.figure,
            "Comparison Stage View",
            "Run a comparison to populate both stage diagrams.",
        )
        self.stage_canvas.draw()
        draw_empty_figure(
            self.heatmap_canvas.figure,
            "Comparison Heatmaps",
            "Run a comparison to populate the side-by-side heatmaps and summary views.",
        )
        self.heatmap_canvas.draw()

        self.summary_table.setColumnCount(3)
        self.summary_table.setHorizontalHeaderLabels(["Metric", "Mode A", "Mode B"])
        self.summary_table.setRowCount(1)
        self.summary_table.setItem(0, 0, QTableWidgetItem("Status"))
        self.summary_table.setItem(0, 1, QTableWidgetItem("Waiting"))
        self.summary_table.setItem(0, 2, QTableWidgetItem("Waiting"))

    def _export(self, canvas: FigureCanvas):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Figure", "comparison.png",
            "PNG Files (*.png);;SVG Files (*.svg);;All Files (*)"
        )
        if filepath:
            canvas.figure.savefig(filepath, dpi=200, bbox_inches="tight")
            QMessageBox.information(self, "Exported", f"Saved to {filepath}")
