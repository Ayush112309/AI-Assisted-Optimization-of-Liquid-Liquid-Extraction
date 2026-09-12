"""
Data Input tab: table editor for tie-line data + equilibrium plot preview.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from ..core.equilibrium import EquilibriumModel, TieLineData

from .ui_helpers import animate_widget_in, draw_empty_figure


COLUMNS = ["A_raff", "C_raff", "B_raff", "A_ext", "C_ext", "B_ext"]


class DataInputTab(QWidget):
    """Tab for viewing/editing equilibrium tie-line data."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.eq_model: Optional[EquilibriumModel] = None
        self.data_source_path: Optional[str] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setSpacing(16)

        # Left: data table
        left_panel = QVBoxLayout()
        left_panel.setSpacing(12)

        intro = QLabel(
            "Load a tie-line dataset, review the phase compositions, and refit the equilibrium model after any table edits."
        )
        intro.setWordWrap(True)
        intro.setProperty("class", "sectionIntro")
        left_panel.addWidget(intro)

        self.source_label = QLabel("Active dataset: bundled default data")
        self.source_label.setWordWrap(True)
        self.source_label.setProperty("class", "helperText")
        left_panel.addWidget(self.source_label)

        table_group = QGroupBox("Tie-Line Data (wt%)")
        table_layout = QVBoxLayout(table_group)

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table_layout.addWidget(self.table)

        btn_layout = QHBoxLayout()
        self.upload_btn = QPushButton("Upload JSON")
        self.upload_btn.clicked.connect(self._upload_json)
        self.add_btn = QPushButton("Add Row")
        self.add_btn.clicked.connect(self._add_row)
        self.remove_btn = QPushButton("Remove Row")
        self.remove_btn.clicked.connect(self._remove_row)
        self.fit_btn = QPushButton("Fit Model")
        self.fit_btn.setProperty("class", "primary")
        self.fit_btn.clicked.connect(self._fit_model)
        btn_layout.addWidget(self.upload_btn)
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.remove_btn)
        btn_layout.addWidget(self.fit_btn)
        table_layout.addLayout(btn_layout)

        left_panel.addWidget(table_group)

        # R² display
        self.r2_label = QLabel("Fit diagnostics appear here after the model is updated.")
        self.r2_label.setWordWrap(True)
        self.r2_label.setProperty("class", "metricCard")
        left_panel.addWidget(self.r2_label)

        layout.addLayout(left_panel, stretch=1)

        # Right: plots
        right_panel = QVBoxLayout()
        plot_group = QGroupBox("Equilibrium Plots")
        plot_layout = QVBoxLayout(plot_group)

        self.canvas = FigureCanvas(Figure(figsize=(8, 10)))
        plot_layout.addWidget(self.canvas)
        right_panel.addWidget(plot_group)

        layout.addLayout(right_panel, stretch=2)
        self._draw_empty_state()

    def set_data(
        self,
        tie_data: TieLineData,
        eq_model: EquilibriumModel,
        data_source_path: Optional[str] = None,
    ):
        """Populate the table and plots from loaded data."""
        self.eq_model = eq_model
        self.data_source_path = data_source_path
        n = len(tie_data.A_raff)
        self.table.setRowCount(n)

        for i in range(n):
            vals = [
                tie_data.A_raff[i], tie_data.C_raff[i], tie_data.B_raff[i],
                tie_data.A_ext[i], tie_data.C_ext[i], tie_data.B_ext[i],
            ]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(f"{v:.2f}")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(i, j, item)

        self._update_source_label()
        self._update_r2_display()
        self._update_plots()

    def _update_source_label(self) -> None:
        if self.data_source_path:
            self.source_label.setText(
                f"Active data source: {Path(self.data_source_path).name}"
            )
        else:
            self.source_label.setText("Active dataset: bundled default data")

    def _upload_json(self) -> None:
        main = self.window()
        if hasattr(main, "_on_load"):
            main._on_load()

    def _add_row(self):
        self.table.insertRow(self.table.rowCount())

    def _remove_row(self):
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)

    def _fit_model(self):
        """Re-fit equilibrium model from current table data."""
        try:
            import numpy as np
            from ..core.equilibrium import TieLineData, fit_equilibrium_model

            n = self.table.rowCount()
            if n < 3:
                QMessageBox.warning(self, "Error", "Need at least 3 tie-lines.")
                return

            data_arrays = {col: [] for col in COLUMNS}
            for i in range(n):
                for j, col in enumerate(COLUMNS):
                    item = self.table.item(i, j)
                    if item is None or item.text().strip() == "":
                        QMessageBox.warning(self, "Error", f"Missing value at row {i+1}, col {col}")
                        return
                    data_arrays[col].append(float(item.text()))

            A_r = np.array(data_arrays["A_raff"])
            C_r = np.array(data_arrays["C_raff"])
            B_r = np.array(data_arrays["B_raff"])
            A_e = np.array(data_arrays["A_ext"])
            C_e = np.array(data_arrays["C_ext"])
            B_e = np.array(data_arrays["B_ext"])

            AC_r = A_r + C_r
            AC_e = A_e + C_e
            X = np.where(AC_r > 0, C_r / AC_r, 0.0)
            Y = np.where(AC_e > 0, C_e / AC_e, 0.0)
            N_r = np.where(AC_r > 0, B_r / AC_r, 0.0)
            N_e = np.where(AC_e > 0, B_e / AC_e, 0.0)

            tie_data = TieLineData(
                A_raff=A_r, C_raff=C_r, B_raff=B_r,
                A_ext=A_e, C_ext=C_e, B_ext=B_e,
                X=X, Y=Y, N_raff=N_r, N_ext=N_e,
            )
            self.eq_model = fit_equilibrium_model(tie_data)

            # Update parent
            parent = self.parent()
            if hasattr(parent, "eq_model"):
                parent.eq_model = self.eq_model
            main = self.window()
            if hasattr(main, "eq_model"):
                main.eq_model = self.eq_model
                if hasattr(main, "data_tab"):
                    main.data_tab.eq_model = self.eq_model
                if hasattr(main, "sim_tab"):
                    main.sim_tab.set_model(self.eq_model)
                if hasattr(main, "surrogate_tab"):
                    data_path = getattr(main, "current_data_path", None)
                    if data_path is None:
                        data_path = self.data_source_path
                    if data_path is None:
                        data_path = getattr(main.surrogate_tab, "data_path", None)
                    if data_path is not None:
                        main.surrogate_tab.set_model(self.eq_model, data_path)
                    else:
                        main.surrogate_tab.set_model(self.eq_model)
                if hasattr(main, "comparison_tab"):
                    main.comparison_tab.set_model(self.eq_model)

            self._update_r2_display()
            self._update_plots()
            QMessageBox.information(self, "Success", "Equilibrium model fitted successfully.")

        except Exception as e:
            QMessageBox.critical(self, "Fit Error", f"Failed to fit model:\n{e}")

    def _update_r2_display(self):
        if self.eq_model is None:
            return
        lines = ["<b>R² Values:</b><br>"]
        for name, val in self.eq_model.r_squared.items():
            color = "green" if val > 0.95 else "orange" if val > 0.9 else "red"
            lines.append(f'<span style="color:{color}">{name}: {val:.4f}</span><br>')
        self.r2_label.setText("".join(lines))

    def _draw_empty_state(self):
        draw_empty_figure(
            self.canvas.figure,
            "Equilibrium Preview",
            "Load a dataset or refit the table to preview the phase envelope and distribution curve.",
        )
        self.canvas.draw()

    def _update_plots(self):
        if self.eq_model is None:
            self._draw_empty_state()
            return

        import numpy as np
        self.canvas.figure.clear()
        data = self.eq_model.tie_line_data

        # 2x1: triangle + distribution
        ax1 = self.canvas.figure.add_subplot(211)
        ax2 = self.canvas.figure.add_subplot(212)

        # Triangle diagram (B-C axes — standard LLE convention)
        B_raff_dense = np.linspace(min(data.B_raff), max(data.B_raff), 200)
        C_raff_dense = [self.eq_model.C_raff_from_B(b) for b in B_raff_dense]
        B_ext_dense = np.linspace(min(data.B_ext), max(data.B_ext), 200)
        C_ext_dense = [self.eq_model.C_ext_from_B(b) for b in B_ext_dense]

        ax1.plot(B_raff_dense, C_raff_dense, "b-", lw=2, label="Raffinate")
        ax1.plot(B_ext_dense, C_ext_dense, "r-", lw=2, label="Extract")
        for i in range(len(data.B_raff)):
            ax1.plot([data.B_raff[i], data.B_ext[i]],
                     [data.C_raff[i], data.C_ext[i]], "k--", lw=0.7, alpha=0.5)
        ax1.scatter(data.B_raff, data.C_raff, c="blue", s=25, zorder=5)
        ax1.scatter(data.B_ext, data.C_ext, c="red", s=25, zorder=5)
        if self.eq_model.plait_point is not None:
            pp = self.eq_model.plait_point
            ax1.plot(pp[2], pp[1], "g*", markersize=12, label="Plait pt.")
        ax1.plot([0, 100], [0, 0], "k-", lw=1.5)
        ax1.plot([0, 0], [0, 100], "k-", lw=1.5)
        ax1.plot([100, 0], [0, 100], "k-", lw=1.5)
        ax1.set_xlabel("wt% B (Solvent / Propane)"); ax1.set_ylabel("wt% C (Solute / Oleic Acid)")
        ax1.set_title("Phase Envelope"); ax1.legend(fontsize=9)
        ax1.set_xlim(-2, 105); ax1.set_ylim(-2, 105); ax1.set_aspect("equal")
        ax1.grid(True, alpha=0.3)

        # Distribution
        X_d = np.linspace(0, max(data.X)*1.05, 200)
        Y_d = [self.eq_model.Y_from_X(x) for x in X_d]
        ax2.plot(X_d, Y_d, "b-", lw=2, label="Y = f(X)")
        ax2.scatter(data.X, data.Y, c="blue", s=40, zorder=5)
        mv = max(max(data.X), max(data.Y))*1.1
        ax2.plot([0, mv], [0, mv], "k--", lw=1, alpha=0.5, label="Y = X")
        ax2.set_xlabel("X (raffinate)"); ax2.set_ylabel("Y (extract)")
        ax2.set_title("Distribution Diagram"); ax2.legend(fontsize=9)
        ax2.set_aspect("equal"); ax2.grid(True, alpha=0.3)

        self.canvas.figure.tight_layout()
        self.canvas.draw()
        animate_widget_in(self.canvas)
