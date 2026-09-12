# Visualization Module (`mass_transfer/viz/`)

This directory contains the rendering engine for the mass transfer digital twin. It generates all static, interactive, and animated plots to help users understand complex liquid-liquid extraction phenomena.

## Purpose
Visualizing ternary phase diagrams, stage-by-stage compositions, and multi-dimensional performance spaces is critical in chemical engineering. This module abstracts away the plotting code (using Matplotlib, Seaborn, and Plotly) into clean, reusable functions that ingest simulation results and output high-quality figures.

## Components

1. **`ternary_plots.py`**:
   - Generates classical chemical engineering phase diagrams.
   - Plots right-angle ternary phase envelopes with empirical tie-lines and conjugate curves.
   - Plots distribution diagrams ($X$ vs $Y$) and solvent-ratio Janecke diagrams ($N$ vs $X/Y$) on a solvent-free basis.

2. **`heatmaps.py`**:
   - Uses Seaborn and Matplotlib to create dense visual matrices of stage-by-stage data.
   - Generates component composition gradients across stages, solvent/raffinate flow rate variations, and cumulative percentage removal charts.

3. **`surfaces.py`**:
   - Leverages Plotly to create fully interactive 3D response surfaces.
   - Maps process input variables (e.g., number of stages vs. solvent flow) to extraction performance metrics (percentage removal), aiding in rapid visual optimization and surrogate model evaluation.

4. **`animations.py`**:
   - Generates animated GIFs that illustrate the numerical stage-stepping process.
   - Visually steps through crosscurrent and countercurrent cascades, showing exactly how operating lines and equilibrium tie-lines dictate stage compositions on the diagram.

## Integration
The visualization module is primarily consumed by the `gui` package, which embeds these plots directly into PyQt6 canvas widgets. However, all functions are designed to be stateless and operate independently, meaning they can easily be imported into Jupyter notebooks or backend scripts to generate and save plots directly to disk.