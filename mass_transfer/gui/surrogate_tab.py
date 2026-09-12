"""
Surrogate Model tab: data generation, ANN training, prediction, and response surfaces.
"""

from __future__ import annotations

from importlib import resources
from typing import TYPE_CHECKING, Optional

import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from ..core.equilibrium import EquilibriumModel

from .ui_helpers import animate_widget_in, draw_empty_figure


DEFAULT_DATA_PATH = str(
    resources.files("mass_transfer").joinpath("resources/data/default_tie_lines.json")
)


# ---------------------------------------------------------------------------
# Worker threads
# ---------------------------------------------------------------------------

class DataGenWorker(QThread):
    """Background thread for dataset generation."""
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, eq_model, config):
        super().__init__()
        self.eq_model = eq_model
        self.config = config

    def run(self):
        try:
            from ..ml.data_generator import generate_crosscurrent_dataset_serial
            df = generate_crosscurrent_dataset_serial(
                self.eq_model, self.config,
                progress_callback=lambda cur, tot: self.progress.emit(cur, tot),
            )
            self.finished.emit(df)
        except Exception as e:
            self.error.emit(str(e))


class TrainWorker(QThread):
    """Background thread for model training."""
    progress = pyqtSignal(int, int, float, float)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, df, config):
        super().__init__()
        self.df = df
        self.config = config

    def run(self):
        try:
            from ..ml.neural_net import train_model
            result = train_model(
                self.df, self.config,
                progress_callback=lambda ep, tot, tl, vl: self.progress.emit(ep, tot, tl, vl),
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class NNComparisonWorker(QThread):
    """
    Runs the actual crosscurrent solver over a test grid so the predictions
    can be compared against the trained NN surrogate.
    """
    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object)   # emits a dict with arrays
    error = pyqtSignal(str)

    def __init__(self, eq_model, mode: str, sweep_var: str,
                 sweep_range: tuple, fixed_vals: dict, n_pts: int):
        super().__init__()
        self.eq_model = eq_model
        self.mode = mode            # 'scatter' or 'sweep'
        self.sweep_var = sweep_var
        self.sweep_range = sweep_range
        self.fixed_vals = fixed_vals
        self.n_pts = n_pts

    def run(self):
        import warnings
        import numpy as np
        from ..core.crosscurrent import solve_crosscurrent

        try:
            eq = self.eq_model

            if self.mode == 'sweep':
                xs = np.linspace(self.sweep_range[0], self.sweep_range[1], self.n_pts)
                y_solver = []
                for i, x in enumerate(xs):
                    self.progress.emit(i + 1, self.n_pts)
                    params = dict(self.fixed_vals)
                    params[self.sweep_var] = float(x)
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            n_stg = max(1, int(round(params.get('n_stages', 5))))
                            sol = solve_crosscurrent(
                                feed_A=100.0 - params['feed_acid_pct'],
                                feed_C=params['feed_acid_pct'],
                                feed_flow=100.0,
                                solvent_per_stage=params['solvent_per_stage'],
                                n_stages=n_stg,
                                eq_model=eq,
                            )
                            y_solver.append(sol.total_pct_removal)
                    except Exception:
                        y_solver.append(float('nan'))
                self.finished.emit({'mode': 'sweep', 'xs': xs, 'y_solver': np.array(y_solver),
                                    'sweep_var': self.sweep_var})

            else:  # scatter – generate a fresh test grid
                import pandas as pd
                from scipy.stats.qmc import LatinHypercube
                sampler = LatinHypercube(d=3, seed=99)
                samples = sampler.random(n=self.n_pts)
                n_stages_arr = np.round(samples[:, 0] * 14 + 1).astype(int)
                solvent_arr  = samples[:, 1] * 4900 + 100
                acid_arr     = samples[:, 2] * 40 + 5

                X_vals, y_solver = [], []
                for i in range(self.n_pts):
                    self.progress.emit(i + 1, self.n_pts)
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            sol = solve_crosscurrent(
                                feed_A=100.0 - acid_arr[i],
                                feed_C=acid_arr[i],
                                feed_flow=100.0,
                                solvent_per_stage=float(solvent_arr[i]),
                                n_stages=int(n_stages_arr[i]),
                                eq_model=eq,
                            )
                            X_vals.append([n_stages_arr[i], solvent_arr[i], acid_arr[i]])
                            y_solver.append(sol.total_pct_removal)
                    except Exception:
                        continue
                self.finished.emit({'mode': 'scatter',
                                    'X': np.array(X_vals),
                                    'y_solver': np.array(y_solver)})
        except Exception as e:
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Surrogate Tab
# ---------------------------------------------------------------------------

class SurrogateTab(QWidget):
    """Tab for surrogate model operations."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.eq_model: Optional[EquilibriumModel] = None
        self.data_path: str = DEFAULT_DATA_PATH
        self.dataset = None
        self.training_result = None
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(16)

        # Left panel: controls
        left_panel = QWidget()
        left_panel.setMinimumWidth(420)
        left = QVBoxLayout(left_panel)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(12)

        intro = QLabel(
            "Generate operating data, train the surrogate, and then use the trained model for prediction, optimization, and solver comparison."
        )
        intro.setWordWrap(True)
        intro.setProperty("class", "sectionIntro")
        left.addWidget(intro)

        self.workflow_tabs = QTabWidget()
        self.workflow_tabs.setDocumentMode(True)
        left.addWidget(self.workflow_tabs)

        # Default workflow tab: Data generation + training
        setup_page = QWidget()
        setup_layout = QVBoxLayout(setup_page)
        setup_layout.setContentsMargins(0, 0, 0, 0)
        setup_layout.setSpacing(12)

        setup_hint = QLabel(
            "Start by generating a dataset, then train the surrogate. Use the other subtabs for prediction, optimization, and solver comparison."
        )
        setup_hint.setWordWrap(True)
        setup_hint.setProperty("class", "helperText")
        setup_layout.addWidget(setup_hint)

        # Data generation
        gen_group = QGroupBox("1. Generate Training Data")
        gen_layout = QFormLayout(gen_group)

        self.n_samples_spin = QSpinBox()
        self.n_samples_spin.setRange(100, 50000)
        self.n_samples_spin.setValue(2000)
        self.n_samples_spin.setSingleStep(500)
        gen_layout.addRow("Samples:", self.n_samples_spin)

        self.gen_btn = QPushButton("Generate Data")
        self.gen_btn.setProperty("class", "primary")
        self.gen_btn.clicked.connect(self._generate_data)
        gen_layout.addRow(self.gen_btn)

        self.gen_progress = QProgressBar()
        gen_layout.addRow(self.gen_progress)

        self.gen_info = QLabel("Generate a dataset to begin.")
        self.gen_info.setProperty("class", "statusCard")
        self.gen_info.setWordWrap(True)
        gen_layout.addRow(self.gen_info)

        setup_layout.addWidget(gen_group)

        # Training
        train_group = QGroupBox("2. Train Model")
        train_layout = QFormLayout(train_group)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 1000); self.epochs_spin.setValue(200)
        self.epochs_spin.setToolTip("Maximum epochs used during ANN training.")
        train_layout.addRow("Epochs:", self.epochs_spin)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 0.1); self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(4); self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setToolTip("Learning rate for the optimizer.")
        train_layout.addRow("Learning rate:", self.lr_spin)

        self.test_size_spin = QDoubleSpinBox()
        self.test_size_spin.setRange(0.05, 0.50); self.test_size_spin.setValue(0.15)
        self.test_size_spin.setDecimals(2); self.test_size_spin.setSingleStep(0.05)
        self.test_size_spin.setToolTip("Fraction of data held out for testing (not used in training or validation)")
        train_layout.addRow("Test split:", self.test_size_spin)

        self.val_size_spin = QDoubleSpinBox()
        self.val_size_spin.setRange(0.05, 0.40); self.val_size_spin.setValue(0.15)
        self.val_size_spin.setDecimals(2); self.val_size_spin.setSingleStep(0.05)
        self.val_size_spin.setToolTip("Fraction of data used for validation (early stopping)")
        train_layout.addRow("Val split:", self.val_size_spin)

        self.train_btn = QPushButton("Train Model")
        self.train_btn.setProperty("class", "primary")
        self.train_btn.clicked.connect(self._train_model)
        train_layout.addRow(self.train_btn)

        self.split_viz_btn = QPushButton("Show Data Split Visuals")
        self.split_viz_btn.setEnabled(False)
        self.split_viz_btn.clicked.connect(self._plot_data_split)
        train_layout.addRow(self.split_viz_btn)

        self.train_progress = QProgressBar()
        train_layout.addRow(self.train_progress)

        self.train_info = QLabel("No training run yet.")
        self.train_info.setProperty("class", "statusCard")
        self.train_info.setWordWrap(True)
        train_layout.addRow(self.train_info)

        setup_layout.addWidget(train_group)
        setup_layout.addStretch()
        self.workflow_tabs.addTab(setup_page, "Workflow")

        # Prediction tab
        predict_page = QWidget()
        predict_layout_wrap = QVBoxLayout(predict_page)
        predict_layout_wrap.setContentsMargins(0, 0, 0, 0)
        pred_group = QGroupBox("3. Predict")
        pred_layout = QFormLayout(pred_group)

        self.pred_stages = QSpinBox()
        self.pred_stages.setRange(1, 20); self.pred_stages.setValue(3)
        pred_layout.addRow("Stages:", self.pred_stages)

        self.pred_solvent = QDoubleSpinBox()
        self.pred_solvent.setRange(1, 50000); self.pred_solvent.setValue(1000)
        pred_layout.addRow("Solvent (kg):", self.pred_solvent)

        self.pred_acid = QDoubleSpinBox()
        self.pred_acid.setRange(1, 50); self.pred_acid.setValue(25)
        pred_layout.addRow("Feed acid %:", self.pred_acid)

        self.pred_btn = QPushButton("Predict")
        self.pred_btn.clicked.connect(self._predict)
        pred_layout.addRow(self.pred_btn)

        self.pred_result = QLabel("Prediction results will appear here.")
        self.pred_result.setProperty("class", "metricCard")
        self.pred_result.setWordWrap(True)
        pred_layout.addRow(self.pred_result)

        predict_layout_wrap.addWidget(pred_group)
        predict_layout_wrap.addStretch()
        self.workflow_tabs.addTab(predict_page, "Predict")

        # Optimization tab
        optimize_page = QWidget()
        optimize_layout_wrap = QVBoxLayout(optimize_page)
        optimize_layout_wrap.setContentsMargins(0, 0, 0, 0)
        opt_group = QGroupBox("4. Optimize")
        opt_layout = QFormLayout(opt_group)

        self.target_removal_spin = QDoubleSpinBox()
        self.target_removal_spin.setRange(10, 99); self.target_removal_spin.setValue(95)
        self.target_removal_spin.setSuffix(" %")
        opt_layout.addRow("Target removal:", self.target_removal_spin)

        self.opt_btn = QPushButton("Find Optimal")
        self.opt_btn.clicked.connect(self._optimize)
        opt_layout.addRow(self.opt_btn)

        self.opt_result = QLabel("Optimization suggestions will appear here.")
        self.opt_result.setProperty("class", "metricCard")
        self.opt_result.setWordWrap(True)
        opt_layout.addRow(self.opt_result)

        optimize_layout_wrap.addWidget(opt_group)
        optimize_layout_wrap.addStretch()
        self.workflow_tabs.addTab(optimize_page, "Optimize")

        # Compare tab
        compare_page = QWidget()
        compare_layout_wrap = QVBoxLayout(compare_page)
        compare_layout_wrap.setContentsMargins(0, 0, 0, 0)
        cmp_group = QGroupBox("5. Compare NN vs Solver")
        cmp_layout = QFormLayout(cmp_group)

        self.cmp_type_combo = QComboBox()
        self.cmp_type_combo.addItems(["Scatter Plot (random grid)", "Parameter Sweep"])
        self.cmp_type_combo.currentIndexChanged.connect(self._on_cmp_type_changed)
        cmp_layout.addRow("Mode:", self.cmp_type_combo)

        self.cmp_n_pts_spin = QSpinBox()
        self.cmp_n_pts_spin.setRange(10, 500); self.cmp_n_pts_spin.setValue(80)
        cmp_layout.addRow("Test points:", self.cmp_n_pts_spin)

        # Sweep controls (shown only in sweep mode)
        self.cmp_sweep_var_combo = QComboBox()
        self.cmp_sweep_var_combo.addItems(["n_stages", "solvent_per_stage", "feed_acid_pct"])
        self.cmp_sweep_var_label = QLabel("Sweep variable:")
        cmp_layout.addRow(self.cmp_sweep_var_label, self.cmp_sweep_var_combo)

        self.cmp_fixed_stages_spin = QSpinBox()
        self.cmp_fixed_stages_spin.setRange(1, 20); self.cmp_fixed_stages_spin.setValue(5)
        self.cmp_fixed_stages_label = QLabel("Fixed stages:")
        cmp_layout.addRow(self.cmp_fixed_stages_label, self.cmp_fixed_stages_spin)

        self.cmp_fixed_solvent_spin = QDoubleSpinBox()
        self.cmp_fixed_solvent_spin.setRange(1, 50000); self.cmp_fixed_solvent_spin.setValue(1000)
        self.cmp_fixed_solvent_label = QLabel("Fixed solvent (kg):")
        cmp_layout.addRow(self.cmp_fixed_solvent_label, self.cmp_fixed_solvent_spin)

        self.cmp_fixed_acid_spin = QDoubleSpinBox()
        self.cmp_fixed_acid_spin.setRange(1, 50); self.cmp_fixed_acid_spin.setValue(25)
        self.cmp_fixed_acid_label = QLabel("Fixed feed acid %:")
        cmp_layout.addRow(self.cmp_fixed_acid_label, self.cmp_fixed_acid_spin)

        self.cmp_btn = QPushButton("Run Comparison")
        self.cmp_btn.clicked.connect(self._run_nn_comparison)
        cmp_layout.addRow(self.cmp_btn)

        self.cmp_progress = QProgressBar()
        cmp_layout.addRow(self.cmp_progress)

        self.cmp_info = QLabel("Train the model first, then compare its predictions against new solver evaluations.")
        self.cmp_info.setProperty("class", "statusCard")
        self.cmp_info.setWordWrap(True)
        cmp_layout.addRow(self.cmp_info)

        compare_layout_wrap.addWidget(cmp_group)
        compare_layout_wrap.addStretch()
        self.workflow_tabs.addTab(compare_page, "Compare NN")

        left.addStretch()

        # Hide sweep-only widgets initially
        self._sweep_widgets = [
            (self.cmp_sweep_var_label, self.cmp_sweep_var_combo),
            (self.cmp_fixed_stages_label, self.cmp_fixed_stages_spin),
            (self.cmp_fixed_solvent_label, self.cmp_fixed_solvent_spin),
            (self.cmp_fixed_acid_label, self.cmp_fixed_acid_spin),
        ]
        for lbl, wid in self._sweep_widgets:
            lbl.hide(); wid.hide()

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setWidget(left_panel)

        main_layout.addWidget(left_scroll, stretch=1)

        # Right panel: plots
        right = QVBoxLayout()

        # Response surface controls
        surf_ctrl = QHBoxLayout()
        surf_ctrl.addWidget(QLabel("Fixed var:"))
        self.fixed_var_combo = QComboBox()
        self.fixed_var_combo.addItems(["feed_acid_pct", "n_stages", "solvent_per_stage"])
        surf_ctrl.addWidget(self.fixed_var_combo)

        surf_ctrl.addWidget(QLabel("Value:"))
        self.fixed_var_slider = QDoubleSpinBox()
        self.fixed_var_slider.setRange(1, 5000)
        self.fixed_var_slider.setValue(25.0)
        surf_ctrl.addWidget(self.fixed_var_slider)

        self.surface_btn = QPushButton("Plot Surface")
        self.surface_btn.clicked.connect(self._plot_surface)
        surf_ctrl.addWidget(self.surface_btn)
        right.addLayout(surf_ctrl)

        # Canvas for loss curve and response surface
        self.canvas = FigureCanvas(Figure(figsize=(10, 8)))
        right.addWidget(self.canvas)

        main_layout.addLayout(right, stretch=2)
        self._draw_empty_state()

    def set_model(self, eq_model: EquilibriumModel, data_path: str = DEFAULT_DATA_PATH):
        self.eq_model = eq_model
        self.data_path = data_path
        self.gen_info.setText("Equilibrium model ready. Generate data to start.")

    def _draw_empty_state(self):
        draw_empty_figure(
            self.canvas.figure,
            "Surrogate Workspace",
            "Generate data and train a model to populate this workspace with losses, response surfaces, and comparison plots.",
        )
        self.canvas.draw()

    def _generate_data(self):
        if self.eq_model is None:
            QMessageBox.warning(self, "No Model", "Load equilibrium data first.")
            return

        from ..ml.data_generator import DataGenConfig
        config = DataGenConfig(n_samples=self.n_samples_spin.value())

        self.gen_btn.setEnabled(False)
        self.gen_info.setText("Generating...")

        self.gen_worker = DataGenWorker(self.eq_model, config)
        self.gen_worker.progress.connect(self._on_gen_progress)
        self.gen_worker.finished.connect(self._on_gen_done)
        self.gen_worker.error.connect(self._on_gen_error)
        self.gen_worker.start()

    def _on_gen_progress(self, current, total):
        self.gen_progress.setMaximum(total)
        self.gen_progress.setValue(current)

    def _on_gen_done(self, df):
        self.gen_btn.setEnabled(True)
        self.dataset = df
        self.gen_info.setText(f"Dataset ready: <b>{len(df)}</b> samples generated. Train the model when you are ready.")

    def _on_gen_error(self, msg):
        self.gen_btn.setEnabled(True)
        self.gen_info.setText("Error!")
        QMessageBox.critical(self, "Generation Error", msg)

    def _train_model(self):
        if self.dataset is None:
            QMessageBox.warning(self, "No Data", "Generate training data first.")
            return

        from ..ml.neural_net import TrainingConfig
        config = TrainingConfig(
            epochs=self.epochs_spin.value(),
            learning_rate=self.lr_spin.value(),
            test_size=self.test_size_spin.value(),
            val_size=self.val_size_spin.value(),
        )

        self.train_btn.setEnabled(False)
        self.train_info.setText("Training...")

        self.train_worker = TrainWorker(self.dataset, config)
        self.train_worker.progress.connect(self._on_train_progress)
        self.train_worker.finished.connect(self._on_train_done)
        self.train_worker.error.connect(self._on_train_error)
        self.train_worker.start()

    def _on_train_progress(self, epoch, total, train_loss, val_loss):
        self.train_progress.setMaximum(total)
        self.train_progress.setValue(epoch)
        self.train_info.setText(f"Epoch {epoch}/{total} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        # Update live loss curve
        if epoch % 5 == 0 or epoch == total:
            self._update_loss_plot()

    def _on_train_done(self, result):
        self.train_btn.setEnabled(True)
        self.split_viz_btn.setEnabled(True)
        self.training_result = result
        self.train_info.setText(
            f"<b>Training complete.</b><br>"
            f"R²: <b>{result.test_r_squared:.4f}</b>  |  "
            f"MAE: <b>{result.test_mae:.2f}%</b>  |  "
            f"RMSE: <b>{result.test_rmse:.2f}%</b><br>"
            f"Split: {result.n_train} train / {result.n_val} val / {result.n_test} test "
            f"(of {result.n_total} total)"
        )
        self._update_loss_plot()

    def _on_train_error(self, msg):
        self.train_btn.setEnabled(True)
        self.train_info.setText("Error!")
        QMessageBox.critical(self, "Training Error", msg)

    def _update_loss_plot(self):
        if self.training_result is None:
            return
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)
        ax.plot(self.training_result.train_losses, label="Train Loss")
        ax.plot(self.training_result.val_losses, label="Val Loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
        ax.set_title(f"Training Curve (R²={self.training_result.test_r_squared:.4f})")
        ax.legend(); ax.grid(True, alpha=0.3)
        if len(self.training_result.train_losses) > 10:
            ax.set_yscale("log")
        self.canvas.figure.tight_layout()
        self.canvas.draw()
        animate_widget_in(self.canvas)

    def _plot_data_split(self):
        """Visualize the train/val/test split and test-set predictions."""
        tr = self.training_result
        if tr is None or tr.X_test is None:
            QMessageBox.warning(self, "No Data", "Train a model first.")
            return

        import numpy as np

        self.canvas.figure.clear()
        fig = self.canvas.figure
        axes = fig.subplots(2, 2)

        # ---- 1. Pie chart of split sizes ----
        ax = axes[0, 0]
        sizes = [tr.n_train, tr.n_val, tr.n_test]
        labels = [f"Train\n({tr.n_train})", f"Val\n({tr.n_val})", f"Test\n({tr.n_test})"]
        colors = ["#4e79a7", "#f28e2b", "#e15759"]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors, autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 9},
        )
        ax.set_title(f"Data Split ({tr.n_total} total)", fontsize=11)

        # ---- 2. Predicted vs Actual on TEST set ----
        ax2 = axes[0, 1]
        y_test = tr.y_test.ravel()
        y_pred = tr.y_pred_test.ravel()
        errors = np.abs(y_test - y_pred)

        sc = ax2.scatter(y_test, y_pred, c=errors, cmap="RdYlGn_r",
                         s=30, alpha=0.7, edgecolors="none")
        fig.colorbar(sc, ax=ax2, label="|Error| (%)", shrink=0.8)
        lims = [min(y_test.min(), y_pred.min()) - 2,
                max(y_test.max(), y_pred.max()) + 2]
        ax2.plot(lims, lims, "k--", lw=1.5, label="Perfect fit")
        ax2.set_xlim(lims); ax2.set_ylim(lims)
        ax2.set_xlabel("Actual (% removal)", fontsize=10)
        ax2.set_ylabel("Predicted (% removal)", fontsize=10)
        ax2.set_title(
            f"Test Set: Predicted vs Actual\n"
            f"R²={tr.test_r_squared:.4f}  MAE={tr.test_mae:.2f}%",
            fontsize=10,
        )
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

        # ---- 3. Error histogram ----
        ax3 = axes[1, 0]
        errs = y_pred - y_test
        ax3.hist(errs, bins=25, color="steelblue", edgecolor="white", alpha=0.85)
        ax3.axvline(0, color="red", lw=1.5, label="Zero")
        mae = tr.test_mae
        ax3.axvline(mae, color="orange", ls="--", lw=1.5, label=f"+MAE ({mae:.2f}%)")
        ax3.axvline(-mae, color="orange", ls="--", lw=1.5, label=f"-MAE")
        ax3.set_xlabel("Prediction Error (%)", fontsize=10)
        ax3.set_ylabel("Count", fontsize=10)
        ax3.set_title("Test Set Error Distribution", fontsize=10)
        ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

        # ---- 4. Feature scatter (train vs val vs test) ----
        ax4 = axes[1, 1]
        feat_names = ["n_stages", "solvent/stage", "feed_acid%"]
        # Plot solvent vs feed_acid (features 1 & 2) coloured by split
        ax4.scatter(tr.X_train[:, 1], tr.X_train[:, 2], c="#4e79a7",
                    s=8, alpha=0.4, label=f"Train ({tr.n_train})")
        ax4.scatter(tr.X_val[:, 1], tr.X_val[:, 2], c="#f28e2b",
                    s=15, alpha=0.6, label=f"Val ({tr.n_val})")
        ax4.scatter(tr.X_test[:, 1], tr.X_test[:, 2], c="#e15759",
                    s=20, alpha=0.7, label=f"Test ({tr.n_test})", marker="x")
        ax4.set_xlabel("Solvent per Stage (kg)", fontsize=10)
        ax4.set_ylabel("Feed Acid (%)", fontsize=10)
        ax4.set_title("Feature Space: Train / Val / Test", fontsize=10)
        ax4.legend(fontsize=8, markerscale=2); ax4.grid(True, alpha=0.3)

        fig.suptitle("Train-Test Data Split Visualization", fontsize=13, y=1.01)
        fig.tight_layout()
        self.canvas.draw()
        animate_widget_in(self.canvas)

    def _predict(self):
        if self.training_result is None:
            QMessageBox.warning(self, "No Model", "Train a model first.")
            return

        from ..ml.neural_net import predict
        result = predict(
            self.training_result.model,
            self.training_result.scaler_X,
            self.training_result.scaler_y,
            n_stages=self.pred_stages.value(),
            solvent=self.pred_solvent.value(),
            feed_comp=self.pred_acid.value(),
        )
        self.pred_result.setText(
            f"<b>Predicted removal</b><br>"
            f"{result:.2f}% for {self.pred_stages.value()} stages, "
            f"{self.pred_solvent.value():.0f} kg solvent/stage, "
            f"and {self.pred_acid.value():.2f}% feed acid."
        )

    def _optimize(self):
        if self.training_result is None:
            QMessageBox.warning(self, "No Model", "Train a model first.")
            return

        from ..ml.optimization import find_optimal_conditions
        result = find_optimal_conditions(
            self.training_result,
            target_removal=self.target_removal_spin.value(),
        )
        self.opt_result.setText(
            f"<b>Recommended operating point</b><br>"
            f"Stages: <b>{result['n_stages']}</b><br>"
            f"Solvent/stage: <b>{result['solvent_per_stage']:.0f} kg</b><br>"
            f"Predicted removal: <b>{result['predicted_removal']:.2f}%</b><br>"
            f"Total solvent: {result['total_solvent']:.0f} kg"
        )

    # ------------------------------------------------------------------
    # NN vs Solver comparison
    # ------------------------------------------------------------------

    def _on_cmp_type_changed(self, idx):
        is_sweep = (idx == 1)
        for lbl, wid in self._sweep_widgets:
            if is_sweep:
                lbl.show(); wid.show()
            else:
                lbl.hide(); wid.hide()

    def _run_nn_comparison(self):
        if self.training_result is None:
            QMessageBox.warning(self, "No Model", "Train a surrogate model first.")
            return
        if self.eq_model is None:
            QMessageBox.warning(self, "No Data", "Load equilibrium data first.")
            return

        mode = 'sweep' if self.cmp_type_combo.currentIndex() == 1 else 'scatter'
        sweep_var = self.cmp_sweep_var_combo.currentText()
        n_pts = self.cmp_n_pts_spin.value()

        sweep_ranges = {
            'n_stages': (1, 15),
            'solvent_per_stage': (100, 5000),
            'feed_acid_pct': (5, 45),
        }
        fixed_vals = {
            'n_stages': float(self.cmp_fixed_stages_spin.value()),
            'solvent_per_stage': self.cmp_fixed_solvent_spin.value(),
            'feed_acid_pct': self.cmp_fixed_acid_spin.value(),
        }

        self.cmp_btn.setEnabled(False)
        self.cmp_info.setText("Running solver…")
        self.cmp_progress.setValue(0)

        self._cmp_worker = NNComparisonWorker(
            eq_model=self.eq_model,
            mode=mode,
            sweep_var=sweep_var,
            sweep_range=sweep_ranges[sweep_var],
            fixed_vals=fixed_vals,
            n_pts=n_pts,
        )
        self._cmp_worker.progress.connect(self._on_comparison_progress)
        self._cmp_worker.finished.connect(self._on_comparison_done)
        self._cmp_worker.error.connect(self._on_comparison_error)
        self._cmp_worker.start()

    def _on_comparison_progress(self, current, total):
        self.cmp_progress.setMaximum(total)
        self.cmp_progress.setValue(current)

    def _on_comparison_done(self, data: dict):
        self.cmp_btn.setEnabled(True)
        if data['mode'] == 'scatter':
            self._plot_scatter_comparison(data)
        else:
            self._plot_sweep_comparison(data)

    def _on_comparison_error(self, msg: str):
        self.cmp_btn.setEnabled(True)
        self.cmp_info.setText("Error!")
        QMessageBox.critical(self, "Comparison Error", msg)

    def _plot_scatter_comparison(self, data: dict):
        """Scatter plot: NN predicted vs actual solver for random test points."""
        import numpy as np
        from ..ml.neural_net import predict

        X = data['X']                    # (N, 3)  [n_stages, solvent, acid]
        y_solver = data['y_solver']      # (N,)

        tr = self.training_result
        y_nn = np.array([
            predict(tr.model, tr.scaler_X, tr.scaler_y,
                    n_stages=X[i, 0], solvent=X[i, 1], feed_comp=X[i, 2])
            for i in range(len(X))
        ])

        # Metrics
        valid = ~np.isnan(y_solver) & ~np.isnan(y_nn)
        y_s, y_n = y_solver[valid], y_nn[valid]
        ss_res = np.sum((y_s - y_n) ** 2)
        ss_tot = np.sum((y_s - np.mean(y_s)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        mae  = float(np.mean(np.abs(y_s - y_n)))
        rmse = float(np.sqrt(np.mean((y_s - y_n) ** 2)))
        max_err = float(np.max(np.abs(y_s - y_n)))

        self.cmp_info.setText(
            f"<b>Scatter vs Solver</b> — {valid.sum()} points<br>"
            f"R² = {r2:.4f} &nbsp; MAE = {mae:.2f}% &nbsp; "
            f"RMSE = {rmse:.2f}% &nbsp; Max err = {max_err:.2f}%"
        )

        self.canvas.figure.clear()
        axes = self.canvas.figure.subplots(1, 2)

        # Left: scatter predicted vs actual
        ax = axes[0]
        sc = ax.scatter(y_s, y_n, c=np.abs(y_s - y_n), cmap='RdYlGn_r',
                        s=40, alpha=0.75, edgecolors='none')
        self.canvas.figure.colorbar(sc, ax=ax, label='|Error| (%)', shrink=0.8)
        lims = [min(y_s.min(), y_n.min()) - 2, max(y_s.max(), y_n.max()) + 2]
        ax.plot(lims, lims, 'k--', lw=1.5, label='Perfect fit (y = x)')
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('Actual Solver (% removal)', fontsize=11)
        ax.set_ylabel('NN Predicted (% removal)', fontsize=11)
        ax.set_title(f'NN vs Solver — Scatter Plot\nR²={r2:.4f}  MAE={mae:.2f}%  RMSE={rmse:.2f}%', fontsize=11)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        # Right: histogram of errors
        ax2 = axes[1]
        errors = y_n - y_s
        ax2.hist(errors, bins=20, color='steelblue', edgecolor='white', alpha=0.85)
        ax2.axvline(0, color='red', lw=1.5, label='Zero error')
        ax2.axvline(mae,  color='orange', lw=1.5, ls='--', label=f'MAE={mae:.2f}%')
        ax2.axvline(-mae, color='orange', lw=1.5, ls='--')
        ax2.set_xlabel('NN Prediction Error (%)', fontsize=11)
        ax2.set_ylabel('Count', fontsize=11)
        ax2.set_title('Distribution of Prediction Errors', fontsize=11)
        ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def _plot_sweep_comparison(self, data: dict):
        """Line chart: NN vs solver as one parameter varies."""
        import numpy as np
        from ..ml.neural_net import predict

        xs = data['xs']
        y_solver = data['y_solver']
        sweep_var = data['sweep_var']

        fixed_vals = {
            'n_stages': float(self.cmp_fixed_stages_spin.value()),
            'solvent_per_stage': self.cmp_fixed_solvent_spin.value(),
            'feed_acid_pct': self.cmp_fixed_acid_spin.value(),
        }

        tr = self.training_result
        y_nn = np.array([
            predict(
                tr.model, tr.scaler_X, tr.scaler_y,
                n_stages=xs[i] if sweep_var == 'n_stages' else fixed_vals['n_stages'],
                solvent=xs[i] if sweep_var == 'solvent_per_stage' else fixed_vals['solvent_per_stage'],
                feed_comp=xs[i] if sweep_var == 'feed_acid_pct' else fixed_vals['feed_acid_pct'],
            )
            for i in range(len(xs))
        ])

        valid = ~np.isnan(y_solver)
        errors = np.abs(y_nn[valid] - y_solver[valid])
        mae = float(np.mean(errors)) if valid.any() else float('nan')
        max_err = float(np.max(errors)) if valid.any() else float('nan')

        var_labels = {
            'n_stages': 'Number of Stages',
            'solvent_per_stage': 'Solvent per Stage (kg)',
            'feed_acid_pct': 'Feed Acid (%)',
        }
        fixed_desc = ", ".join(
            f"{var_labels[k]}={v:.1f}" for k, v in fixed_vals.items() if k != sweep_var
        )

        self.cmp_info.setText(
            f"<b>Sweep: {var_labels[sweep_var]}</b>  |  {fixed_desc}<br>"
            f"MAE = {mae:.2f}% &nbsp; Max error = {max_err:.2f}%"
        )

        self.canvas.figure.clear()
        axes = self.canvas.figure.subplots(2, 1, sharex=True,
                                            gridspec_kw={'height_ratios': [3, 1]})

        # Top: NN vs solver lines
        ax = axes[0]
        ax.plot(xs[valid], y_solver[valid], 'b-o', ms=4, lw=2, label='Actual Solver')
        ax.plot(xs, y_nn, 'r--s', ms=4, lw=2, label='NN Surrogate')
        ax.set_ylabel('% Removal', fontsize=11)
        ax.set_title(
            f'NN Surrogate vs Actual Solver\nSweep: {var_labels[sweep_var]}  |  {fixed_desc}',
            fontsize=11
        )
        ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)

        # Bottom: absolute error
        ax2 = axes[1]
        ax2.fill_between(xs[valid], errors, alpha=0.4, color='orange', label='|Error|')
        ax2.plot(xs[valid], errors, 'o-', color='darkorange', ms=3, lw=1.5)
        ax2.axhline(mae, color='red', lw=1.5, ls='--', label=f'MAE={mae:.2f}%')
        ax2.set_xlabel(var_labels[sweep_var], fontsize=11)
        ax2.set_ylabel('|Error| (%)', fontsize=11)
        ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def _plot_surface(self):
        if self.training_result is None:
            QMessageBox.warning(self, "No Model", "Train a model first.")
            return

        from ..ml.optimization import generate_response_surface

        fixed_var = self.fixed_var_combo.currentText()
        fixed_val = self.fixed_var_slider.value()

        # Determine the other two variables
        all_vars = ["n_stages", "solvent_per_stage", "feed_acid_pct"]
        var_ranges = {
            "n_stages": (1, 15),
            "solvent_per_stage": (100, 5000),
            "feed_acid_pct": (5, 45),
        }
        var_labels = {
            "n_stages": "Number of Stages",
            "solvent_per_stage": "Solvent per Stage (kg)",
            "feed_acid_pct": "Feed Acid (%)",
        }

        others = [v for v in all_vars if v != fixed_var]

        X, Y, Z = generate_response_surface(
            self.training_result,
            var1_name=others[0],
            var2_name=others[1],
            fixed_var_name=fixed_var,
            fixed_var_value=fixed_val,
            var1_range=var_ranges[others[0]],
            var2_range=var_ranges[others[1]],
            grid_size=30,
        )

        self.canvas.figure.clear()

        # 3D surface
        ax = self.canvas.figure.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8, edgecolor="none")
        self.canvas.figure.colorbar(surf, ax=ax, label="% Removal", shrink=0.6)
        ax.set_xlabel(var_labels[others[0]])
        ax.set_ylabel(var_labels[others[1]])
        ax.set_zlabel("% Removal")
        ax.set_title(f"Response Surface ({var_labels[fixed_var]} = {fixed_val:.1f})")

        self.canvas.figure.tight_layout()
        self.canvas.draw()
