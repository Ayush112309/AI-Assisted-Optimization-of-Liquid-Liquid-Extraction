"""
Animation tab: generate and preview animated GIF visualizations of
extraction processes — stage-by-stage, ternary, composition, and
parameter sweeps.
"""

from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING

from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from ..core.equilibrium import EquilibriumModel

from .ui_helpers import animate_widget_in, draw_empty_figure


class AnimationWorker(QThread):
    """Generate an animation object (and optional GIF bytes) off-thread."""

    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, anim_type: str, eq_model, result, params: dict):
        super().__init__()
        self.anim_type = anim_type
        self.eq_model = eq_model
        self.result = result
        self.params = params

    def run(self):
        try:
            from ..viz.animations import (
                animate_composition_profile,
                animate_parameter_sweep,
                animate_ternary_buildup,
                animate_xy_stepping,
                save_animation_gif,
            )
            import tempfile

            fps = self.params.get("fps", 3)

            if self.anim_type == "xy_stepping":
                anim = animate_xy_stepping(
                    self.eq_model, self.result.stages, self.result, fps=fps,
                )
            elif self.anim_type == "ternary":
                anim = animate_ternary_buildup(
                    self.eq_model, self.result.stages, self.result, fps=fps,
                )
            elif self.anim_type == "composition":
                anim = animate_composition_profile(
                    self.result.stages, self.result, fps=fps,
                )
            elif self.anim_type == "param_sweep":
                anim = animate_parameter_sweep(
                    self.eq_model,
                    sweep_var=self.params["sweep_var"],
                    sweep_range=self.params["sweep_range"],
                    fixed_vals=self.params["fixed_vals"],
                    n_frames=self.params.get("n_frames", 30),
                    fps=fps,
                    progress_callback=lambda c, t: self.progress.emit(c, t),
                )
            else:
                raise ValueError(f"Unknown animation type: {self.anim_type}")

            tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
            tmp_path = tmp.name
            tmp.close()
            save_animation_gif(anim, tmp_path, fps=fps, dpi=90)

            self.finished.emit(
                {
                    "anim": anim,
                    "gif_path": tmp_path,
                    "fps": fps,
                    "n_frames": anim._fig is not None,
                }
            )
        except Exception as e:
            import traceback

            self.error.emit(f"{e}\n{traceback.format_exc()}")


class _SolverWorker(QThread):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, solver_func, kwargs):
        super().__init__()
        self.solver_func = solver_func
        self.kwargs = kwargs

    def run(self):
        try:
            self.finished.emit(self.solver_func(**self.kwargs))
        except Exception as e:
            self.error.emit(str(e))


ANIM_TYPES = [
    ("Stage-by-Stage X-Y Stepping", "xy_stepping"),
    ("Ternary Diagram Build-up", "ternary"),
    ("Composition Profile Evolution", "composition"),
    ("Parameter Sensitivity Sweep", "param_sweep"),
]


class AnimationTab(QWidget):
    """Embedded animation widget tied to the active solver context."""

    def __init__(
        self,
        parent=None,
        *,
        show_source_controls: bool = True,
        solver_factory: Optional[Callable[[], tuple]] = None,
    ):
        super().__init__(parent)
        self.eq_model: Optional[EquilibriumModel] = None
        self._show_source_controls = show_source_controls
        self._solver_factory = solver_factory
        self._last_result = None
        self._gif_path: Optional[str] = None
        self._anim = None
        self._frame_idx = 0
        self._frame_images = []
        self._timer: Optional[QTimer] = None
        self._worker: Optional[AnimationWorker] = None
        self._solver_worker: Optional[_SolverWorker] = None
        self._setup_ui()

    def _setup_ui(self):
        root = QHBoxLayout(self)
        root.setSpacing(16)

        left = QVBoxLayout()
        left.setSpacing(12)

        intro = QLabel(
            "Generate an animation for the current solver context and switch between stage, ternary, profile, and sweep views as needed."
        )
        intro.setWordWrap(True)
        intro.setProperty("class", "sectionIntro")
        left.addWidget(intro)

        type_group = QGroupBox("Animation Type")
        type_layout = QFormLayout(type_group)
        self.type_combo = QComboBox()
        for label, _ in ANIM_TYPES:
            self.type_combo.addItem(label)
        self.type_combo.currentIndexChanged.connect(self._on_type_changed)
        type_layout.addRow("Type:", self.type_combo)
        left.addWidget(type_group)

        source_group = QGroupBox("Animation Source")
        source_layout = QVBoxLayout(source_group)
        source_label = QLabel("This panel always uses the active solver context.")
        source_label.setWordWrap(True)
        source_label.setProperty("class", "helperText")
        source_layout.addWidget(source_label)
        left.addWidget(source_group)
        source_group.setVisible(self._show_source_controls)

        speed_group = QGroupBox("Playback")
        speed_layout = QFormLayout(speed_group)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 10)
        self.fps_spin.setValue(3)
        self.fps_spin.setSuffix(" FPS")
        self.fps_spin.setToolTip("Preview and export playback speed.")
        speed_layout.addRow("Speed:", self.fps_spin)
        left.addWidget(speed_group)

        self.sweep_group = QGroupBox("Parameter Sweep")
        sweep_layout = QFormLayout(self.sweep_group)
        self.sweep_var_combo = QComboBox()
        self.sweep_var_combo.addItems(["solvent_per_stage", "n_stages", "feed_acid_pct"])
        sweep_layout.addRow("Sweep variable:", self.sweep_var_combo)

        self.sweep_min_spin = QDoubleSpinBox()
        self.sweep_min_spin.setRange(1, 50000)
        self.sweep_min_spin.setValue(100)
        sweep_layout.addRow("Range min:", self.sweep_min_spin)

        self.sweep_max_spin = QDoubleSpinBox()
        self.sweep_max_spin.setRange(1, 50000)
        self.sweep_max_spin.setValue(5000)
        sweep_layout.addRow("Range max:", self.sweep_max_spin)

        self.sweep_frames_spin = QSpinBox()
        self.sweep_frames_spin.setRange(5, 100)
        self.sweep_frames_spin.setValue(30)
        sweep_layout.addRow("Frames:", self.sweep_frames_spin)

        self.sweep_fixed_stages = QSpinBox()
        self.sweep_fixed_stages.setRange(1, 20)
        self.sweep_fixed_stages.setValue(5)
        sweep_layout.addRow("Fixed stages:", self.sweep_fixed_stages)

        self.sweep_fixed_solvent = QDoubleSpinBox()
        self.sweep_fixed_solvent.setRange(1, 50000)
        self.sweep_fixed_solvent.setValue(1000)
        sweep_layout.addRow("Fixed solvent:", self.sweep_fixed_solvent)

        self.sweep_fixed_acid = QDoubleSpinBox()
        self.sweep_fixed_acid.setRange(1, 50)
        self.sweep_fixed_acid.setValue(25)
        sweep_layout.addRow("Fixed acid %:", self.sweep_fixed_acid)

        left.addWidget(self.sweep_group)
        self.sweep_group.hide()

        button_row = QHBoxLayout()
        self.gen_btn = QPushButton("Generate Animation")
        self.gen_btn.setProperty("class", "primary")
        self.gen_btn.setMinimumHeight(44)
        self.gen_btn.clicked.connect(self._generate)
        button_row.addWidget(self.gen_btn)

        self.save_btn = QPushButton("Save GIF")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._save_gif)
        button_row.addWidget(self.save_btn)
        left.addLayout(button_row)

        self.progress = QProgressBar()
        left.addWidget(self.progress)

        self.status_label = QLabel("Generate an animation for the active case.")
        self.status_label.setProperty("class", "statusCard")
        self.status_label.setWordWrap(True)
        left.addWidget(self.status_label)
        left.addStretch()

        root.addLayout(left, stretch=1)

        right = QVBoxLayout()
        preview_group = QGroupBox("Animation Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.canvas = FigureCanvas(Figure(figsize=(11, 8)))
        preview_layout.addWidget(self.canvas)

        footer = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self._toggle_playback)
        footer.addWidget(self.play_btn)
        self.frame_label = QLabel("Frame: —")
        footer.addWidget(self.frame_label)
        footer.addStretch()
        preview_layout.addLayout(footer)
        right.addWidget(preview_group)
        root.addLayout(right, stretch=3)

        self._draw_empty_state()

    def set_model(self, eq_model: "EquilibriumModel"):
        self.eq_model = eq_model

    def set_result(self, result):
        self._last_result = result
        self.status_label.setText("Results ready. Generate an animation for this case.")

    def set_solver_factory(self, solver_factory: Callable[[], tuple]) -> None:
        self._solver_factory = solver_factory

    def _draw_empty_state(self):
        draw_empty_figure(
            self.canvas.figure,
            "Animation Preview",
            "Generate an animation to preview stage motion, composition trends, or a parameter sweep.",
        )
        self.canvas.draw()

    def _on_type_changed(self, idx):
        is_sweep = ANIM_TYPES[idx][1] == "param_sweep"
        self.sweep_group.setVisible(is_sweep)

    def _generate(self):
        if self.eq_model is None:
            QMessageBox.warning(self, "No Model", "Load equilibrium data first.")
            return

        self.gen_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.play_btn.setEnabled(False)
        self.progress.setValue(0)

        if self._solver_factory is not None:
            self.status_label.setText("Running the current case before generating the animation…")
            try:
                solver_func, kwargs = self._solver_factory()
            except Exception as e:
                self.gen_btn.setEnabled(True)
                QMessageBox.critical(self, "Animation Source Error", str(e))
                return

            self._solver_worker = _SolverWorker(solver_func, kwargs)
            self._solver_worker.finished.connect(self._on_solver_done)
            self._solver_worker.error.connect(self._on_solver_error)
            self._solver_worker.start()
            return

        if self._last_result is None:
            self.gen_btn.setEnabled(True)
            QMessageBox.information(
                self,
                "No Result",
                "Run the current case first so an animation can be generated.",
            )
            return

        self._start_animation_worker(self._last_result)

    def _on_solver_done(self, result):
        self._last_result = result
        self._start_animation_worker(result)

    def _on_solver_error(self, msg):
        self.gen_btn.setEnabled(True)
        self.status_label.setText("Solver error!")
        QMessageBox.critical(self, "Solver Error", msg)

    def _start_animation_worker(self, result):
        self.status_label.setText("Generating animation frames…")
        anim_key = ANIM_TYPES[self.type_combo.currentIndex()][1]
        params = {"fps": self.fps_spin.value()}
        if anim_key == "param_sweep":
            params["sweep_var"] = self.sweep_var_combo.currentText()
            params["sweep_range"] = (self.sweep_min_spin.value(), self.sweep_max_spin.value())
            params["n_frames"] = self.sweep_frames_spin.value()
            params["fixed_vals"] = {
                "n_stages": float(self.sweep_fixed_stages.value()),
                "solvent_per_stage": self.sweep_fixed_solvent.value(),
                "feed_acid_pct": self.sweep_fixed_acid.value(),
            }

        self._worker = AnimationWorker(anim_key, self.eq_model, result, params)
        self._worker.progress.connect(self._on_gen_progress)
        self._worker.finished.connect(self._on_gen_done)
        self._worker.error.connect(self._on_gen_error)
        self._worker.start()

    def _on_gen_progress(self, current, total):
        self.progress.setMaximum(total)
        self.progress.setValue(current)

    def _on_gen_done(self, data: dict):
        self.gen_btn.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.play_btn.setEnabled(True)
        self._gif_path = data["gif_path"]

        try:
            from PIL import Image

            img = Image.open(self._gif_path)
            self._frame_images = []
            try:
                while True:
                    self._frame_images.append(img.copy().convert("RGBA"))
                    img.seek(img.tell() + 1)
            except EOFError:
                pass

            n = len(self._frame_images)
            self.status_label.setText(
                f"<b>Done.</b> {n} frames, {n / data['fps']:.1f}s duration."
            )
            self._frame_idx = 0
            self._show_frame(0)
            self.frame_label.setText(f"Frame: 1 / {n}")
        except Exception as e:
            self.status_label.setText(f"Animation generated but preview failed: {e}")

    def _on_gen_error(self, msg):
        self.gen_btn.setEnabled(True)
        self.status_label.setText("Error generating animation!")
        QMessageBox.critical(self, "Animation Error", msg)

    def _show_frame(self, idx):
        if not self._frame_images or idx >= len(self._frame_images):
            return

        import numpy as np

        arr = np.array(self._frame_images[idx])
        fig = self.canvas.figure
        fig.clear()
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(arr)
        ax.set_axis_off()
        self.canvas.draw()
        animate_widget_in(self.canvas, duration=160)

    def _toggle_playback(self):
        if self._timer is not None and self._timer.isActive():
            self._timer.stop()
            self.play_btn.setText("▶ Play")
            return

        if not self._frame_images:
            return

        self._frame_idx = 0
        interval_ms = max(50, 1000 // self.fps_spin.value())
        if self._timer is None:
            self._timer = QTimer(self)
            self._timer.timeout.connect(self._advance_frame)
        self._timer.start(interval_ms)
        self.play_btn.setText("⏸ Pause")

    def _advance_frame(self):
        if self._frame_idx >= len(self._frame_images):
            self._timer.stop()
            self.play_btn.setText("▶ Play")
            return

        self._show_frame(self._frame_idx)
        self.frame_label.setText(f"Frame: {self._frame_idx + 1} / {len(self._frame_images)}")
        self._frame_idx += 1

    def _save_gif(self):
        if self._gif_path is None:
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Animation GIF",
            "extraction_animation.gif",
            "GIF Files (*.gif);;All Files (*)",
        )
        if filepath:
            import shutil

            shutil.copy2(self._gif_path, filepath)
            QMessageBox.information(self, "Saved", f"Animation saved to {filepath}")
