"""
Optimization and response surface generation using the trained surrogate model.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from .neural_net import TrainingResult, predict


def generate_response_surface(
    training_result: TrainingResult,
    var1_name: str = "n_stages",
    var2_name: str = "solvent_per_stage",
    fixed_var_name: str = "feed_acid_pct",
    fixed_var_value: float = 25.0,
    var1_range: Tuple[float, float] = (1, 15),
    var2_range: Tuple[float, float] = (100, 5000),
    grid_size: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a 2D response surface from the trained ANN.

    Parameters
    ----------
    training_result : trained model result
    var1_name, var2_name : names of the two varying parameters
    fixed_var_name : name of the held-constant parameter
    fixed_var_value : value of the fixed parameter
    var1_range, var2_range : ranges for the two variables
    grid_size : number of grid points per axis

    Returns
    -------
    (X_grid, Y_grid, Z_grid) arrays suitable for plotting.
    """
    var1_vals = np.linspace(var1_range[0], var1_range[1], grid_size)
    var2_vals = np.linspace(var2_range[0], var2_range[1], grid_size)
    X_grid, Y_grid = np.meshgrid(var1_vals, var2_vals)
    Z_grid = np.zeros_like(X_grid)

    # Map variable names to positions
    var_map = {
        "n_stages": 0,
        "solvent_per_stage": 1,
        "feed_acid_pct": 2,
    }

    for i in range(grid_size):
        for j in range(grid_size):
            params = [0.0, 0.0, 0.0]
            params[var_map[var1_name]] = float(X_grid[i, j])
            params[var_map[var2_name]] = float(Y_grid[i, j])
            params[var_map[fixed_var_name]] = fixed_var_value

            Z_grid[i, j] = predict(
                training_result.model,
                training_result.scaler_X,
                training_result.scaler_y,
                n_stages=params[0],
                solvent=params[1],
                feed_comp=params[2],
            )

    return X_grid, Y_grid, Z_grid


def find_optimal_conditions(
    training_result: TrainingResult,
    target_removal: float = 95.0,
    objective: str = "min_solvent",
    n_stages_range: Tuple[int, int] = (1, 15),
    solvent_range: Tuple[float, float] = (100.0, 5000.0),
    feed_acid_pct: float = 25.0,
) -> dict:
    """
    Find optimal operating conditions using the trained surrogate model.

    Parameters
    ----------
    training_result : trained model result
    target_removal : target % removal (constraint)
    objective : "min_solvent" or "max_removal"
    n_stages_range : bounds for number of stages
    solvent_range : bounds for solvent per stage
    feed_acid_pct : fixed feed composition

    Returns
    -------
    dict with optimal n_stages, solvent_per_stage, predicted_removal.
    """
    model = training_result.model
    scaler_X = training_result.scaler_X
    scaler_y = training_result.scaler_y

    best_result = None
    best_objective = float("inf") if objective == "min_solvent" else -float("inf")

    # Grid search over integer stages, optimize solvent continuously
    for n_stages in range(n_stages_range[0], n_stages_range[1] + 1):

        if objective == "min_solvent":
            # Minimize solvent subject to removal >= target
            def neg_removal(s):
                return -predict(model, scaler_X, scaler_y, float(n_stages), float(s), feed_acid_pct)

            # Find minimum solvent that achieves target
            def removal_minus_target(s):
                return predict(model, scaler_X, scaler_y, float(n_stages), float(s), feed_acid_pct) - target_removal

            from scipy.optimize import brentq
            try:
                # Check if target is achievable at max solvent
                rem_max = predict(model, scaler_X, scaler_y, float(n_stages),
                                  solvent_range[1], feed_acid_pct)
                rem_min = predict(model, scaler_X, scaler_y, float(n_stages),
                                  solvent_range[0], feed_acid_pct)

                if rem_max < target_removal:
                    continue  # can't achieve target with this many stages

                if rem_min >= target_removal:
                    opt_solvent = solvent_range[0]
                else:
                    opt_solvent = brentq(removal_minus_target,
                                         solvent_range[0], solvent_range[1],
                                         xtol=1.0)

                opt_removal = predict(model, scaler_X, scaler_y,
                                      float(n_stages), opt_solvent, feed_acid_pct)

                if opt_solvent < best_objective:
                    best_objective = opt_solvent
                    best_result = {
                        "n_stages": n_stages,
                        "solvent_per_stage": opt_solvent,
                        "feed_acid_pct": feed_acid_pct,
                        "predicted_removal": opt_removal,
                        "total_solvent": opt_solvent * n_stages,
                    }
            except (ValueError, RuntimeError):
                continue

        else:  # max_removal
            # Maximize removal at given solvent
            opt_removal = predict(model, scaler_X, scaler_y,
                                  float(n_stages), solvent_range[1], feed_acid_pct)
            if opt_removal > best_objective:
                best_objective = opt_removal
                best_result = {
                    "n_stages": n_stages,
                    "solvent_per_stage": solvent_range[1],
                    "feed_acid_pct": feed_acid_pct,
                    "predicted_removal": opt_removal,
                    "total_solvent": solvent_range[1] * n_stages,
                }

    if best_result is None:
        return {
            "n_stages": n_stages_range[1],
            "solvent_per_stage": solvent_range[1],
            "feed_acid_pct": feed_acid_pct,
            "predicted_removal": predict(
                model, scaler_X, scaler_y,
                float(n_stages_range[1]), solvent_range[1], feed_acid_pct
            ),
            "total_solvent": solvent_range[1] * n_stages_range[1],
        }

    return best_result
