"""
Simulation tab: run crosscurrent and countercurrent solvers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from ..core.equilibrium import EquilibriumModel

from .heatmap_tab import HeatmapTab
from .animation_tab import AnimationTab
from .ui_helpers import animate_widget_in, draw_empty_figure


class SolverWorker(QThread):
    """Background thread for running solvers."""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, solver_func, kwargs):
        super().__init__()
        self.solver_func = solver_func
        self.kwargs = kwargs

    def run(self):
        try:
            result = self.solver_func(**self.kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class SimulationTab(QWidget):
    """Tab for running extraction simulations."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.eq_model: Optional[EquilibriumModel] = None
        self.worker: Optional[SolverWorker] = None
        self.last_result = None
        self.heatmap_tab: Optional[HeatmapTab] = None
        self.animation_tab: Optional[AnimationTab] = None
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(16)

        # Left: inputs
        left = QVBoxLayout()
        left.setSpacing(12)

        intro = QLabel(
            "Choose a process mode, set the feed and solvent conditions, and run the solver to generate results, stage plots, heatmaps, and animations."
        )
        intro.setWordWrap(True)
        intro.setProperty("class", "sectionIntro")
        left.addWidget(intro)

        # Mode selector
        mode_group = QGroupBox("Extraction Mode")
        mode_layout = QVBoxLayout(mode_group)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Crosscurrent",
            "Countercurrent",
        ])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        left.addWidget(mode_group)

        # Feed parameters
        feed_group = QGroupBox("Feed Conditions")
        feed_layout = QFormLayout(feed_group)

        self.feed_A_spin = QDoubleSpinBox()
        self.feed_A_spin.setRange(0, 100); self.feed_A_spin.setValue(75.0)
        self.feed_A_spin.setSuffix(" wt%")
        self.feed_A_spin.setToolTip("Carrier fraction in the incoming feed stream.")
        feed_layout.addRow("Carrier (A):", self.feed_A_spin)

        self.feed_C_spin = QDoubleSpinBox()
        self.feed_C_spin.setRange(0, 100); self.feed_C_spin.setValue(25.0)
        self.feed_C_spin.setSuffix(" wt%")
        self.feed_C_spin.setToolTip("Solute fraction in the incoming feed stream.")
        feed_layout.addRow("Solute (C):", self.feed_C_spin)

        self.feed_flow_spin = QDoubleSpinBox()
        self.feed_flow_spin.setRange(1, 100000); self.feed_flow_spin.setValue(100.0)
        self.feed_flow_spin.setSuffix(" kg/h")
        feed_layout.addRow("Feed flow:", self.feed_flow_spin)

        left.addWidget(feed_group)

        # Operation parameters
        op_group = QGroupBox("Operating Parameters")
        self.op_layout = QFormLayout(op_group)

        self.n_stages_spin = QSpinBox()
        self.n_stages_spin.setRange(1, 50); self.n_stages_spin.setValue(2)
        self.n_stages_spin.setToolTip("Number of ideal extraction stages to calculate.")
        self.op_layout.addRow("Stages:", self.n_stages_spin)

        self.solvent_label = QLabel("Solvent per stage:")
        self.solvent_spin = QDoubleSpinBox()
        self.solvent_spin.setRange(1, 100000); self.solvent_spin.setValue(1000.0)
        self.solvent_spin.setSuffix(" kg")
        self.solvent_spin.setToolTip("Crosscurrent: fresh solvent per stage. Countercurrent: total solvent flow.")
        self.op_layout.addRow(self.solvent_label, self.solvent_spin)

        left.addWidget(op_group)

        # Run button
        self.run_btn = QPushButton("Run Simulation")
        self.run_btn.setProperty("class", "primary")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.clicked.connect(self._run_solver)
        left.addWidget(self.run_btn)

        self.status_label = QLabel("Set the operating conditions and run a case.")
        self.status_label.setProperty("class", "statusCard")
        self.status_label.setWordWrap(True)
        left.addWidget(self.status_label)
        left.addStretch()

        main_layout.addLayout(left, stretch=1)

        # Right: results
        right = QVBoxLayout()
        self.results_tabs = QTabWidget()
        self.results_tabs.setDocumentMode(True)

        # Results subtab
        results_page = QWidget()
        results_layout = QVBoxLayout(results_page)
        table_group = QGroupBox("Results")
        table_layout = QVBoxLayout(table_group)
        self.results_table = QTableWidget()
        self.results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        table_layout.addWidget(self.results_table)
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.summary_label.setProperty("class", "metricCard")
        table_layout.addWidget(self.summary_label)
        results_layout.addWidget(table_group)
        self.results_tabs.addTab(results_page, "Results")

        # Stage diagram subtab
        stage_page = QWidget()
        stage_layout = QVBoxLayout(stage_page)
        plot_group = QGroupBox("Stage Diagram")
        plot_layout = QVBoxLayout(plot_group)
        self.canvas = FigureCanvas(Figure(figsize=(8, 6)))
        plot_layout.addWidget(self.canvas)
        stage_layout.addWidget(plot_group)
        self.results_tabs.addTab(stage_page, "Stage Diagram")

        # Heatmaps subtab
        self.heatmap_scroll = QScrollArea()
        self.heatmap_scroll.setWidgetResizable(True)
        self.heatmap_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.heatmap_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.heatmap_tab = HeatmapTab(self)
        self.heatmap_scroll.setWidget(self.heatmap_tab)
        self.results_tabs.addTab(self.heatmap_scroll, "Heatmaps")

        # Animation subtab
        self.animation_tab = AnimationTab(self, show_source_controls=False)
        self.animation_tab.set_solver_factory(self._build_animation_solver)
        self.results_tabs.addTab(self.animation_tab, "Animation")

        right.addWidget(self.results_tabs)

        main_layout.addLayout(right, stretch=2)
        self._on_mode_changed(self.mode_combo.currentIndex())
        self._draw_empty_state()

    def set_model(self, eq_model: EquilibriumModel):
        self.eq_model = eq_model
        if self.heatmap_tab is not None:
            self.heatmap_tab.set_model(eq_model)
        if self.animation_tab is not None:
            self.animation_tab.set_model(eq_model)
        self.status_label.setText("Equilibrium model ready. Configure a case and run the solver.")

    def _build_animation_solver(self):
        if self.eq_model is None:
            raise ValueError("Load equilibrium data first.")

        mode = self.mode_combo.currentIndex()
        if mode == 0:
            from ..core.crosscurrent import solve_crosscurrent
            return solve_crosscurrent, dict(
                feed_A=self.feed_A_spin.value(),
                feed_C=self.feed_C_spin.value(),
                feed_flow=self.feed_flow_spin.value(),
                solvent_per_stage=self.solvent_spin.value(),
                n_stages=self.n_stages_spin.value(),
                eq_model=self.eq_model,
            )
        if mode == 1:
            from ..core.countercurrent import solve_countercurrent
            return solve_countercurrent, dict(
                feed_A=self.feed_A_spin.value(),
                feed_C=self.feed_C_spin.value(),
                feed_flow=self.feed_flow_spin.value(),
                solvent_flow=self.solvent_spin.value(),
                n_stages=self.n_stages_spin.value(),
                eq_model=self.eq_model,
            )

    def _on_mode_changed(self, index):
        if index == 1:
            self.solvent_label.setText("Solvent flow:")
            self.solvent_spin.setSuffix(" kg/h")
            self.solvent_spin.setToolTip("Countercurrent: total solvent flow rate.")
        else:
            self.solvent_label.setText("Solvent per stage:")
            self.solvent_spin.setSuffix(" kg")
            self.solvent_spin.setToolTip("Crosscurrent: fresh solvent per stage.")

    def _run_solver(self):
        if self.eq_model is None:
            QMessageBox.warning(self, "No Model", "Load equilibrium data first.")
            return

        mode = self.mode_combo.currentIndex()
        self.run_btn.setEnabled(False)
        self.status_label.setText("Running the solver for the current operating point.")

        if mode == 0:  # Crosscurrent
            from ..core.crosscurrent import solve_crosscurrent
            kwargs = dict(
                feed_A=self.feed_A_spin.value(),
                feed_C=self.feed_C_spin.value(),
                feed_flow=self.feed_flow_spin.value(),
                solvent_per_stage=self.solvent_spin.value(),
                n_stages=self.n_stages_spin.value(),
                eq_model=self.eq_model,
            )
            self.worker = SolverWorker(solve_crosscurrent, kwargs)

        elif mode == 1:  # Countercurrent
            from ..core.countercurrent import solve_countercurrent
            kwargs = dict(
                feed_A=self.feed_A_spin.value(),
                feed_C=self.feed_C_spin.value(),
                feed_flow=self.feed_flow_spin.value(),
                solvent_flow=self.solvent_spin.value(),
                n_stages=self.n_stages_spin.value(),
                eq_model=self.eq_model,
            )
            self.worker = SolverWorker(solve_countercurrent, kwargs)

        self.worker.finished.connect(self._on_solver_done)
        self.worker.error.connect(self._on_solver_error)
        self.worker.start()

    def _on_solver_done(self, result):
        self.run_btn.setEnabled(True)
        self.last_result = result
        self.status_label.setText("Simulation complete. Review the results, plots, and heatmaps.")

        mode = self.mode_combo.currentIndex()

        if mode == 0:
            self._display_crosscurrent_results(result)
        else:
            self._display_countercurrent_results(result)

        if self.heatmap_tab is not None:
            self.heatmap_tab.set_result(result)
        if self.animation_tab is not None:
            self.animation_tab.set_result(result)
        self.results_tabs.setCurrentWidget(self.heatmap_scroll)


    def _on_solver_error(self, msg):
        self.run_btn.setEnabled(True)
        self.status_label.setText("Solver error!")
        QMessageBox.critical(self, "Solver Error", msg)

    def _draw_empty_state(self):
        draw_empty_figure(
            self.canvas.figure,
            "Stage Diagram",
            "Run a simulation to populate the X-Y stage plot and the ternary process path.",
        )
        self.canvas.draw()

    def _display_crosscurrent_results(self, result):
        stages = result.stages
        cols = ["Stage", "A_raff%", "C_raff%", "B_raff%", "R(kg)",
                "A_ext%", "C_ext%", "B_ext%", "E(kg)", "X", "Y", "Removal%"]
        self.results_table.setColumnCount(len(cols))
        self.results_table.setHorizontalHeaderLabels(cols)
        self.results_table.setRowCount(len(stages))

        for i, s in enumerate(stages):
            vals = [
                str(s.stage_number), f"{s.A_raff:.2f}", f"{s.C_raff:.2f}", f"{s.B_raff:.2f}",
                f"{s.R_flow:.1f}", f"{s.A_ext:.2f}", f"{s.C_ext:.2f}", f"{s.B_ext:.2f}",
                f"{s.E_flow:.1f}", f"{s.X_raff:.4f}", f"{s.Y_ext:.4f}", f"{s.pct_removal_cumul:.2f}",
            ]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.results_table.setItem(i, j, item)

        self.summary_label.setText(
            f"<b>Crosscurrent Summary</b><br>"
            f"Stages: <b>{result.n_stages}</b>  |  "
            f"Removal: <b>{result.total_pct_removal:.2f}%</b>  |  "
            f"Final X_raff: <b>{result.final_raff_X:.4f}</b><br>"
            f"Mixed extract flow: {result.mixed_extract_flow:.1f} kg"
        )
        self._plot_crosscurrent(result)

    def _display_countercurrent_results(self, result):
        stages = result.stages
        cols = ["Stage", "X_raff", "Y_ext", "A_raff%", "C_raff%", "B_raff%",
                "A_ext%", "C_ext%", "B_ext%"]
        self.results_table.setColumnCount(len(cols))
        self.results_table.setHorizontalHeaderLabels(cols)
        self.results_table.setRowCount(len(stages))

        for i, s in enumerate(stages):
            vals = [
                str(s.stage_number), f"{s.X_raff:.4f}", f"{s.Y_ext:.4f}",
                f"{s.A_raff:.2f}", f"{s.C_raff:.2f}", f"{s.B_raff:.2f}",
                f"{s.A_ext:.2f}", f"{s.C_ext:.2f}", f"{s.B_ext:.2f}",
            ]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.results_table.setItem(i, j, item)

        self.summary_label.setText(
            f"<b>Countercurrent Summary</b><br>"
            f"Stages: <b>{result.n_stages}</b>  |  "
            f"Lead raffinate X: <b>{stages[0].X_raff:.4f}</b>  |  "
            f"Terminal extract Y: <b>{stages[-1].Y_ext:.4f}</b>"
        )
        self._plot_countercurrent(result)

    def _plot_crosscurrent(self, result):
        import numpy as np
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(121)
        ax_ternary = self.canvas.figure.add_subplot(122)

        data = self.eq_model.tie_line_data
        X_d = np.linspace(0, max(data.X)*1.05, 200)
        Y_d = [self.eq_model.Y_from_X(x) for x in X_d]

        ax.plot(X_d, Y_d, "b-", lw=2, label="Equilibrium")
        mv = max(max(data.X), max(data.Y))*1.1
        ax.plot([0, mv], [0, mv], "k--", lw=1, alpha=0.5, label="Y=X")

        # Stage stepping
        X_f = result.feed_C / (result.feed_A + result.feed_C)
        ax.plot(X_f, 0, "gs", markersize=10, label="Feed")

        for s in result.stages:
            # Vertical: X → Y (equilibrium)
            ax.plot([s.X_raff, s.X_raff], [s.X_raff, s.Y_ext], "r-", lw=1.5)
            # Horizontal: Y → next X
            ax.plot([s.X_raff], [s.Y_ext], "ro", markersize=5)

        ax.set_xlabel("X (raffinate)")
        ax.set_ylabel("Y (extract)")
        ax.set_title("Crosscurrent Stage Diagram")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        self._draw_ternary_background(ax_ternary)
        feed_B = getattr(result, "feed_B", max(0.0, 100.0 - result.feed_A - result.feed_C))
        ax_ternary.plot(feed_B, result.feed_C, "gs", markersize=9, label="Feed")

        raff_B = [s.B_raff for s in result.stages]
        raff_C = [s.C_raff for s in result.stages]
        ext_B = [s.B_ext for s in result.stages]
        ext_C = [s.C_ext for s in result.stages]

        for i, s in enumerate(result.stages, start=1):
            ax_ternary.plot([s.B_raff, s.B_ext], [s.C_raff, s.C_ext], "k--", lw=0.9, alpha=0.6)
            ax_ternary.annotate(f" {i}", (s.B_raff, s.C_raff), fontsize=8, color="#1f4e79")

        ax_ternary.plot(raff_B, raff_C, "o-", color="#1f77b4", lw=1.8, ms=6, label="Raffinate path")
        ax_ternary.plot(ext_B, ext_C, "s-", color="#d62728", lw=1.2, ms=5, alpha=0.85, label="Extract stages")
        ax_ternary.set_title("Ternary Process Path")
        ax_ternary.legend(fontsize=9, loc="best")

        self.canvas.figure.tight_layout()
        self.canvas.draw()
        animate_widget_in(self.canvas)

    def _plot_countercurrent(self, result):
        import numpy as np
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(121)
        ax_ternary = self.canvas.figure.add_subplot(122)

        data = self.eq_model.tie_line_data
        X_d = np.linspace(0, max(data.X)*1.05, 200)
        Y_d = [self.eq_model.Y_from_X(x) for x in X_d]

        ax.plot(X_d, Y_d, "b-", lw=2, label="Equilibrium")
        mv = max(max(data.X), max(data.Y))*1.1
        ax.plot([0, mv], [0, mv], "k--", lw=1, alpha=0.5, label="Y=X")

        # Plot stage points
        for s in result.stages:
            ax.plot(s.X_raff, s.Y_ext, "ro", markersize=6)
            ax.annotate(f" {s.stage_number}", (s.X_raff, s.Y_ext), fontsize=8)

        # Connect stages with lines
        X_vals = [s.X_raff for s in result.stages]
        Y_vals = [s.Y_ext for s in result.stages]
        ax.plot(X_vals, Y_vals, "r-", lw=1, alpha=0.7)

        ax.set_xlabel("X (raffinate)")
        ax.set_ylabel("Y (extract)")
        ax.set_title("Countercurrent Stage Diagram")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        self._draw_ternary_background(ax_ternary)

        feed_B = max(0.0, 100.0 - self.feed_A_spin.value() - self.feed_C_spin.value())
        ax_ternary.plot(feed_B, self.feed_C_spin.value(), "gs", markersize=9, label="Feed")

        raff_B = [s.B_raff for s in result.stages]
        raff_C = [s.C_raff for s in result.stages]
        ext_B = [s.B_ext for s in result.stages]
        ext_C = [s.C_ext for s in result.stages]

        for s in result.stages:
            color = "#ff7f0e" if getattr(s, "section", "") == "stripping" else "k"
            ax_ternary.plot([s.B_raff, s.B_ext], [s.C_raff, s.C_ext], "--", color=color, lw=0.9, alpha=0.65)
            ax_ternary.annotate(f" {s.stage_number}", (s.B_raff, s.C_raff), fontsize=8, color="#1f4e79")

        ax_ternary.plot(raff_B, raff_C, "o-", color="#1f77b4", lw=1.8, ms=6, label="Raffinate path")
        ax_ternary.plot(ext_B, ext_C, "s-", color="#d62728", lw=1.2, ms=5, alpha=0.85, label="Extract path")

        ax_ternary.set_title("Ternary Process Path")
        ax_ternary.legend(fontsize=9, loc="best")

        self.canvas.figure.tight_layout()
        self.canvas.draw()
        animate_widget_in(self.canvas)

    def _draw_ternary_background(self, ax):
        import numpy as np

        data = self.eq_model.tie_line_data
        B_raff_dense = np.linspace(min(data.B_raff), max(data.B_raff), 200)
        C_raff_dense = [self.eq_model.C_raff_from_B(b) for b in B_raff_dense]
        B_ext_dense = np.linspace(min(data.B_ext), max(data.B_ext), 200)
        C_ext_dense = [self.eq_model.C_ext_from_B(b) for b in B_ext_dense]

        ax.plot(B_raff_dense, C_raff_dense, "b-", lw=2, label="Raffinate envelope")
        ax.plot(B_ext_dense, C_ext_dense, "r-", lw=2, label="Extract envelope")
        ax.plot([0, 100], [0, 0], "k-", lw=1.5)
        ax.plot([0, 0], [0, 100], "k-", lw=1.5)
        ax.plot([100, 0], [0, 100], "k-", lw=1.5)
        ax.set_xlim(-2, 105)
        ax.set_ylim(-2, 105)
        ax.set_xlabel("wt% B (Solvent)")
        ax.set_ylabel("wt% C (Solute)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
