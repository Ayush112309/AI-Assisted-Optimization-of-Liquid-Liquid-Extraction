# Optimization and Analysis (`optimization.py`)

This module utilizes the trained Neural Network surrogate model to perform fast optimizations and generate response surfaces. By bypassing the computationally expensive physics-based solver, these tasks can be executed nearly instantaneously.

## Key Features

*   **Response Surface Generation:** The `generate_response_surface` function creates 2D grids of predicted extraction performance over varying parameter ranges. This is highly useful for generating heatmaps and 3D surface plots to visualize the design space.
*   **Optimal Condition Finding:** The `find_optimal_conditions` function searches for the best operating conditions to achieve a desired outcome. It can optimize for minimum solvent usage given a target extraction percentage, leveraging the speed of the neural network combined with `scipy.optimize` (e.g., `brentq` root-finding).

## Main Components

*   `generate_response_surface(...)`: Sweeps two variables across defined ranges while holding a third constant, returning `X`, `Y`, and `Z` grids of predicted values suitable for plotting.
*   `find_optimal_conditions(...)`: Iterates over discrete values of stages and uses continuous root-finding/optimization to find the exact solvent flow rate required to hit a specific removal target, returning the most optimal configuration.
