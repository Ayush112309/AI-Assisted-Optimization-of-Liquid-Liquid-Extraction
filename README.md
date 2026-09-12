# Mass Transfer Digital Twin

A packaged Python desktop application for multistage liquid-liquid extraction analysis. The project combines equilibrium fitting, crosscurrent and countercurrent extraction solvers, stage-wise visualization, and a PyTorch surrogate model inside a PyQt6 GUI.

The default bundled system is cottonseed oil / oleic acid / propane at 98.5 C and 625 lb/in² abs, but the application is built to load user-supplied tie-line JSON data as well.

## What It Does

- Fits ternary equilibrium data and constructs solvent-free relationships used by the solvers
- Solves crosscurrent extraction for any configured number of stages
- Solves countercurrent extraction in simple and reflux configurations
- Visualizes stage behavior through equilibrium plots, heatmaps, profiles, comparison charts, and animations
- Generates synthetic training data and trains a neural-network surrogate for fast prediction and optimization
- Provides a desktop GUI that ties simulation, analysis, and surrogate modeling into one workflow

## GUI Workflow

The GUI is organized around the main tasks a user actually performs:

1. `Home`
   Use the landing screen for quick navigation and a short workflow overview.
2. `Data Input`
   Load the default dataset or a custom JSON file, inspect tie-line data, and refit the equilibrium model.
3. `Simulation`
   Run crosscurrent or countercurrent calculations.
   Results are grouped into subtabs:
   `Results`
   `Stage Diagram`
   `Heatmaps`
4. `Surrogate Model`
   Generate data and train the ANN in the default workflow subtab.
   Additional subtabs are available for:
   `Predict`
   `Optimize`
   `Compare NN`
5. `Comparison`
   Compare two extraction modes side by side with common inputs.
6. `Animation`
   Generate animated visualizations and export GIFs.

## Main Features

### Equilibrium Modeling

- Loads tie-line data from JSON
- Converts ternary data to solvent-free coordinates
- Fits distribution and auxiliary correlations used by the extraction solvers
- Displays phase-envelope and distribution plots in the GUI

Default dataset:
- [`mass_transfer/resources/data/default_tie_lines.json`](/Users/17kaushalsingh/GitHub/Mass-Transfer2-Project/mass_transfer/resources/data/default_tie_lines.json)

### Extraction Solvers

- `Crosscurrent`
  Stage-wise extraction with fresh solvent added per stage
- `Countercurrent (Simple)`
  Countercurrent extraction without reflux
- `Countercurrent (Reflux)`
  Countercurrent extraction with reflux-specific inputs and design outputs

Core solver modules:
- [`mass_transfer/core/equilibrium.py`](/Users/17kaushalsingh/GitHub/Mass-Transfer2-Project/mass_transfer/core/equilibrium.py)
- [`mass_transfer/core/crosscurrent.py`](/Users/17kaushalsingh/GitHub/Mass-Transfer2-Project/mass_transfer/core/crosscurrent.py)
- [`mass_transfer/core/countercurrent.py`](/Users/17kaushalsingh/GitHub/Mass-Transfer2-Project/mass_transfer/core/countercurrent.py)

### Visualization

- Embedded equilibrium plots
- Simulation result tables
- Stage diagrams
- Heatmaps and profile plots tied directly to the active simulation result
- Side-by-side comparison views
- Animated GIF generation and preview

Visualization modules:
- [`mass_transfer/viz/ternary_plots.py`](/Users/17kaushalsingh/GitHub/Mass-Transfer2-Project/mass_transfer/viz/ternary_plots.py)
- [`mass_transfer/viz/heatmaps.py`](/Users/17kaushalsingh/GitHub/Mass-Transfer2-Project/mass_transfer/viz/heatmaps.py)
- [`mass_transfer/viz/surfaces.py`](/Users/17kaushalsingh/GitHub/Mass-Transfer2-Project/mass_transfer/viz/surfaces.py)
- [`mass_transfer/viz/animations.py`](/Users/17kaushalsingh/GitHub/Mass-Transfer2-Project/mass_transfer/viz/animations.py)

### Surrogate Model

- Synthetic dataset generation from solver sweeps
- Feedforward ANN training with train/validation/test split controls
- Prediction from operating conditions
- Surrogate-based operating-point optimization
- Neural-network vs solver comparison
- Response-surface plotting

ML modules:
- [`mass_transfer/ml/data_generator.py`](/Users/17kaushalsingh/GitHub/Mass-Transfer2-Project/mass_transfer/ml/data_generator.py)
- [`mass_transfer/ml/neural_net.py`](/Users/17kaushalsingh/GitHub/Mass-Transfer2-Project/mass_transfer/ml/neural_net.py)
- [`mass_transfer/ml/optimization.py`](/Users/17kaushalsingh/GitHub/Mass-Transfer2-Project/mass_transfer/ml/optimization.py)

## Project Structure

```text
Mass-Transfer2-Project/
├── pyproject.toml
├── requirements.txt
├── README.md
├── docs/
│   └── Project_Description.pdf
├── mass_transfer/
│   ├── core/
│   ├── gui/
│   ├── ml/
│   ├── resources/
│   │   └── data/
│   │       └── default_tie_lines.json
│   └── viz/
└── tests/
```

Key GUI files:
- [`mass_transfer/gui/main_window.py`](/Users/17kaushalsingh/GitHub/Mass-Transfer2-Project/mass_transfer/gui/main_window.py)
- [`mass_transfer/gui/home_tab.py`](/Users/17kaushalsingh/GitHub/Mass-Transfer2-Project/mass_transfer/gui/home_tab.py)
- [`mass_transfer/gui/data_input_tab.py`](/Users/17kaushalsingh/GitHub/Mass-Transfer2-Project/mass_transfer/gui/data_input_tab.py)
- [`mass_transfer/gui/simulation_tab.py`](/Users/17kaushalsingh/GitHub/Mass-Transfer2-Project/mass_transfer/gui/simulation_tab.py)
- [`mass_transfer/gui/heatmap_tab.py`](/Users/17kaushalsingh/GitHub/Mass-Transfer2-Project/mass_transfer/gui/heatmap_tab.py)
- [`mass_transfer/gui/surrogate_tab.py`](/Users/17kaushalsingh/GitHub/Mass-Transfer2-Project/mass_transfer/gui/surrogate_tab.py)
- [`mass_transfer/gui/comparison_tab.py`](/Users/17kaushalsingh/GitHub/Mass-Transfer2-Project/mass_transfer/gui/comparison_tab.py)
- [`mass_transfer/gui/animation_tab.py`](/Users/17kaushalsingh/GitHub/Mass-Transfer2-Project/mass_transfer/gui/animation_tab.py)

## Installation

```bash
git clone https://github.com/17kaushalsingh/Mass-Transfer2-Project.git
cd Mass-Transfer2-Project

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

## Running the App

Launch the desktop GUI with either of these:

```bash
python -m mass_transfer.gui
```

or:

```bash
mass-transfer-gui
```

## Running Tests

```bash
python -m pytest tests -v
```

## Data Format

The bundled dataset and expected custom datasets use this basic structure:

- `phase_1`
  Raffinate-side tie-line endpoints
- `phase_2`
  Extract-side tie-line endpoints

Each item in `phase_1` corresponds by index to the item at the same index in `phase_2`.

Supported keys:
- specific component names such as `cottonseed_oil`, `oleic_acid`, `propane`
- generic keys such as `A`, `B`, `C`

## Tech Stack

- `Python`
- `NumPy`
- `SciPy`
- `Pandas`
- `Matplotlib`
- `Seaborn`
- `Plotly`
- `PyTorch`
- `PyQt6`

## Notes

- The GUI auto-loads the bundled default dataset on startup
- Heatmaps are attached to the active simulation result inside the `Simulation` tab
- The Surrogate tab is compacted into internal subtabs so prediction and optimization tools do not crowd the training workflow
- Animated exports are generated as GIFs without requiring `ffmpeg`
