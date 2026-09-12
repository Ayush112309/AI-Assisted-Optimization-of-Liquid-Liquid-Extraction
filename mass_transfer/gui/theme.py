"""
Shared visual theme for the PyQt GUI.
"""

from __future__ import annotations

from PyQt6.QtGui import QFont


APP_STYLESHEET = """
QMainWindow, QWidget {
    background: qlineargradient(
        x1: 0, y1: 0, x2: 1, y2: 1,
        stop: 0 #181818,
        stop: 0.55 #111417,
        stop: 1 #0d1012
    );
    color: #f1eee6;
    font-family: "Avenir Next", "Trebuchet MS", sans-serif;
    font-size: 13px;
}

QMenuBar {
    background: #14181a;
    color: #f1eee6;
    border-bottom: 1px solid #2a3136;
}

QMenuBar::item:selected, QMenu::item:selected {
    background: #27343a;
}

QMenu {
    background: #171d20;
    color: #f1eee6;
    border: 1px solid #344148;
}

QStatusBar {
    background: #121719;
    color: #d8d3c5;
    border-top: 1px solid #2a3136;
}

QTabWidget::pane {
    border: 1px solid #2b353b;
    border-radius: 14px;
    top: -1px;
    background: rgba(10, 13, 15, 0.35);
}

QTabBar::tab {
    background: #1a2125;
    color: #c6c1b5;
    padding: 10px 18px;
    margin-right: 6px;
    border-top-left-radius: 12px;
    border-top-right-radius: 12px;
    border: 1px solid #2b353b;
}

QTabBar::tab:selected {
    background: #223038;
    color: #f6f2ea;
    border-bottom: 2px solid #c98b2e;
}

QGroupBox {
    background: rgba(24, 29, 33, 0.92);
    border: 1px solid #313c42;
    border-radius: 14px;
    margin-top: 14px;
    padding: 16px 14px 14px 14px;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 14px;
    padding: 0 8px;
    color: #f6c36d;
}

QLabel[class="sectionIntro"] {
    color: #d4cec0;
    font-size: 13px;
    padding: 2px 4px 10px 4px;
}

QLabel[class="helperText"] {
    color: #9fb0b5;
    font-size: 12px;
}

QLabel[class="metricCard"] {
    background: rgba(201, 139, 46, 0.12);
    border: 1px solid #6a552c;
    border-radius: 12px;
    padding: 12px 14px;
    color: #f3ede0;
}

QLabel[class="statusCard"] {
    background: rgba(39, 52, 58, 0.85);
    border: 1px solid #39515c;
    border-radius: 12px;
    padding: 10px 12px;
}

QPushButton {
    background: #243038;
    color: #f3ede0;
    border: 1px solid #3d5059;
    border-radius: 10px;
    min-height: 34px;
    padding: 6px 12px;
}

QPushButton:hover {
    background: #2b3a43;
}

QPushButton:pressed {
    background: #1d272d;
}

QPushButton:disabled {
    color: #78858a;
    background: #1a2023;
    border-color: #293035;
}

QPushButton[class="primary"] {
    background: #c98b2e;
    color: #111417;
    border: 1px solid #d8a657;
    font-weight: 700;
}

QPushButton[class="primary"]:hover {
    background: #d89b42;
}

QPushButton[class="subtle"] {
    background: rgba(31, 37, 41, 0.92);
}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTableWidget, QPlainTextEdit {
    background: #0f1417;
    color: #f1eee6;
    border: 1px solid #334047;
    border-radius: 9px;
    padding: 5px 8px;
    selection-background-color: #c98b2e;
    selection-color: #111417;
}

QComboBox::drop-down, QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    background: #182026;
    border: none;
}

QHeaderView::section {
    background: #1d252a;
    color: #f1eee6;
    border: 1px solid #334047;
    padding: 6px;
}

QProgressBar {
    background: #101518;
    border: 1px solid #324048;
    border-radius: 8px;
    text-align: center;
    min-height: 20px;
}

QProgressBar::chunk {
    background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
        stop: 0 #c98b2e, stop: 1 #d9b15a);
    border-radius: 7px;
}

QScrollArea {
    border: none;
    background: transparent;
}
"""


def apply_app_theme(app) -> None:
    app.setStyleSheet(APP_STYLESHEET)
    font = QFont("Avenir Next", 11)
    app.setFont(font)
