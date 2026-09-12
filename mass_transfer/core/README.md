# Core Mass Transfer Models (`mass_transfer/core/`)

This directory contains the foundational physics-based mathematical solvers and thermodynamic equilibrium models for the liquid-liquid extraction digital twin.

## Purpose
The core module handles all the heavy lifting of calculating phase equilibria, material balances, and multistage extraction performance. It strictly utilizes numerical methods (via `scipy.optimize.fsolve`) rather than relying on graphical approximations, ensuring high precision and stability for systems with a large number of stages.

## Components

1. **Equilibrium Modeling** (`equilibrium.py`):
   - Handles the loading, transformation, and high-fidelity interpolation of ternary liquid-liquid equilibrium data.
   - Converts standard weight fractions into a solvent-free coordinate system ($X$, $Y$, $N$) for rigorous calculations.
   - For detailed methodology, see [Equilibrium Formulation](EQUILIBRIUM_README.md).

2. **Crosscurrent Extraction Solver** (`crosscurrent.py`):
   - Simulates a multi-stage crosscurrent cascade where fresh solvent is added at each discrete stage.
   - Solves mass balances and equilibrium constraints simultaneously per stage.
   - For detailed mathematical formulation, see [Crosscurrent Formulation](CROSSCURRENT_README.md).

3. **Countercurrent Extraction Solver** (`countercurrent.py`):
   - Simulates a continuous multi-stage countercurrent cascade where the feed and solvent streams flow in opposite directions.
   - Formulates the global material balances and equilibrium linkages between counter-flowing stages.
   - For detailed mathematical formulation, see [Countercurrent Formulation](COUNTERCURRENT_README.md).

## Usage
These modules are completely decoupled from the GUI and Visualization tools. They can be imported and run in standalone scripts, backend servers, or Jupyter notebooks for programmatic process simulation, data generation, and mathematical optimization.