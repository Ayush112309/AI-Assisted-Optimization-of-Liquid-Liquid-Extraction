"""
Main window for the Multistage Extraction Digital Twin GUI.

Entry point: python -m mass_transfer.gui.main_window
"""

from __future__ import annotations

import sys
from importlib import resources
from pathlib import Path

from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
)

from ..core.equilibrium import load_tie_line_data, fit_equilibrium_model, EquilibriumModel
from .data_input_tab import DataInputTab
from .simulation_tab import SimulationTab
from .surrogate_tab import SurrogateTab
from .comparison_tab import ComparisonTab
from .theme import apply_app_theme


DEFAULT_DATA_PATH = resources.files("mass_transfer").joinpath(
    "resources/data/default_tie_lines.json"
)


class MainWindow(QMainWindow):
    """Main application window for the desktop workflow."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multistage Extraction Digital Twin")
        self.setMinimumSize(1320, 860)

        self.eq_model: EquilibriumModel | None = None
        self.current_data_path: str | None = None

        self._setup_menu()
        self._setup_tabs()
        self._setup_statusbar()

        # Auto-load default data and open directly into the working tabs.
        if DEFAULT_DATA_PATH.exists():
            self._load_data(str(DEFAULT_DATA_PATH))

    def _setup_menu(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        load_action = QAction("&Load Data...", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self._on_load)
        file_menu.addAction(load_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _setup_tabs(self):
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setMovable(False)
        self.setCentralWidget(self.tabs)

        self.data_tab = DataInputTab(self)
        self.sim_tab = SimulationTab(self)
        self.surrogate_tab = SurrogateTab(self)
        self.comparison_tab = ComparisonTab(self)

        self.tabs.addTab(self.data_tab, "Data Input")
        self.tabs.addTab(self.sim_tab, "Simulation")
        self.tabs.addTab(self.comparison_tab, "⚖ Comparison")
        self.tabs.addTab(self.surrogate_tab, "Surrogate Model")

    def _setup_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("Ready. The default equilibrium dataset loads automatically.")

    def _on_load(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Equilibrium Data", "", "JSON Files (*.json);;All Files (*)"
        )
        if filepath:
            self._load_data(filepath, switch_to_data_tab=True)

    def _load_data(self, filepath: str, switch_to_data_tab: bool = True):
        try:
            tie_data = load_tie_line_data(filepath)
            self.eq_model = fit_equilibrium_model(tie_data)
            self.current_data_path = filepath
            self.data_tab.set_data(tie_data, self.eq_model, filepath)
            self.sim_tab.set_model(self.eq_model)
            self.surrogate_tab.set_model(self.eq_model, filepath)
            self.comparison_tab.set_model(self.eq_model)
            self.statusbar.showMessage(
                f"Loaded {len(tie_data.A_raff)} tie-lines from {Path(filepath).name}"
            )
            if switch_to_data_tab:
                self.tabs.setCurrentWidget(self.data_tab)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load data:\n{e}")

    def _on_about(self):
        QMessageBox.about(
            self,
            "About",
            "<h3>Multistage Extraction Digital Twin</h3>"
            "<p>Mass Transfer 2 Course Project</p>"
            "<p>Simulates crosscurrent and countercurrent liquid-liquid extraction "
            "with PyTorch ANN surrogate modeling.</p>"
            "<p>Default system: Cottonseed Oil / Oleic Acid / Propane</p>"
        )


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    apply_app_theme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
