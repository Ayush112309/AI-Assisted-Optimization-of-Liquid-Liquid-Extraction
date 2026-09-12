"""
Data generation for the surrogate model.

Sweeps the crosscurrent solver over a grid of operating conditions using
Latin Hypercube Sampling and parallel execution.
"""

from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from importlib import resources
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube

from ..core.equilibrium import EquilibriumModel, load_tie_line_data, fit_equilibrium_model
from ..core.crosscurrent import solve_crosscurrent


DEFAULT_DATA_PATH = str(
    resources.files("mass_transfer").joinpath("resources/data/default_tie_lines.json")
)


@dataclass
class DataGenConfig:
    """Configuration for dataset generation."""
    n_stages_range: Tuple[int, int] = (1, 15)
    solvent_range: Tuple[float, float] = (100.0, 5000.0)
    feed_acid_pct_range: Tuple[float, float] = (5.0, 45.0)
    n_samples: int = 5000
    feed_flow: float = 100.0
    feed_carrier_pct: float = 75.0
    n_workers: int = 4
    random_seed: int = 42


def _solve_single_point(args: tuple) -> Optional[dict]:
    """
    Solve a single operating point. Designed for use with ProcessPoolExecutor.

    Parameters
    ----------
    args : (n_stages, solvent, feed_acid, feed_flow, data_path)

    Returns
    -------
    dict with inputs and output, or None if solver fails.
    """
    n_stages, solvent, feed_acid, feed_flow, data_path = args

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = load_tie_line_data(data_path)
            eq = fit_equilibrium_model(data)

            feed_A = 100.0 - feed_acid
            result = solve_crosscurrent(
                feed_A=feed_A,
                feed_C=feed_acid,
                feed_flow=feed_flow,
                solvent_per_stage=solvent,
                n_stages=int(n_stages),
                eq_model=eq,
            )

            return {
                "n_stages": int(n_stages),
                "solvent_per_stage": solvent,
                "feed_acid_pct": feed_acid,
                "pct_removal": result.total_pct_removal,
            }
    except Exception:
        return None


def generate_crosscurrent_dataset(
    eq_model: EquilibriumModel,
    config: Optional[DataGenConfig] = None,
    data_path: str = DEFAULT_DATA_PATH,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """
    Generate a dataset by sweeping the crosscurrent solver.

    Uses Latin Hypercube Sampling for efficient coverage of the parameter space.

    Parameters
    ----------
    eq_model : fitted equilibrium model (used for validation only; each worker loads its own)
    config : generation configuration
    data_path : path to equilibrium data JSON
    progress_callback : optional callback(current, total) for progress updates

    Returns
    -------
    DataFrame with columns: n_stages, solvent_per_stage, feed_acid_pct, pct_removal
    """
    if config is None:
        config = DataGenConfig()

    # Latin Hypercube Sampling in [0,1]^3
    sampler = LatinHypercube(d=3, seed=config.random_seed)
    samples = sampler.random(n=config.n_samples)

    # Scale to actual ranges
    n_stages_arr = np.round(
        samples[:, 0] * (config.n_stages_range[1] - config.n_stages_range[0])
        + config.n_stages_range[0]
    ).astype(int)
    n_stages_arr = np.clip(n_stages_arr, config.n_stages_range[0], config.n_stages_range[1])

    solvent_arr = (
        samples[:, 1] * (config.solvent_range[1] - config.solvent_range[0])
        + config.solvent_range[0]
    )

    feed_acid_arr = (
        samples[:, 2] * (config.feed_acid_pct_range[1] - config.feed_acid_pct_range[0])
        + config.feed_acid_pct_range[0]
    )

    # Build argument list
    args_list = [
        (int(n_stages_arr[i]), float(solvent_arr[i]), float(feed_acid_arr[i]),
         config.feed_flow, data_path)
        for i in range(config.n_samples)
    ]

    # Execute in parallel
    results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=config.n_workers) as executor:
        futures = {executor.submit(_solve_single_point, args): i
                   for i, args in enumerate(args_list)}

        for future in as_completed(futures):
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, config.n_samples)

            try:
                res = future.result()
                if res is not None:
                    results.append(res)
            except Exception:
                continue

    df = pd.DataFrame(results)

    # Clean: remove invalid rows
    df = df.dropna()
    df = df[(df["pct_removal"] >= 0) & (df["pct_removal"] <= 100)]
    df = df.reset_index(drop=True)

    return df


def generate_crosscurrent_dataset_serial(
    eq_model: EquilibriumModel,
    config: Optional[DataGenConfig] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """
    Generate dataset serially (no multiprocessing). Useful when eq_model is already loaded.
    """
    if config is None:
        config = DataGenConfig()

    sampler = LatinHypercube(d=3, seed=config.random_seed)
    samples = sampler.random(n=config.n_samples)

    n_stages_arr = np.round(
        samples[:, 0] * (config.n_stages_range[1] - config.n_stages_range[0])
        + config.n_stages_range[0]
    ).astype(int)
    n_stages_arr = np.clip(n_stages_arr, config.n_stages_range[0], config.n_stages_range[1])

    solvent_arr = (
        samples[:, 1] * (config.solvent_range[1] - config.solvent_range[0])
        + config.solvent_range[0]
    )

    feed_acid_arr = (
        samples[:, 2] * (config.feed_acid_pct_range[1] - config.feed_acid_pct_range[0])
        + config.feed_acid_pct_range[0]
    )

    results = []
    for i in range(config.n_samples):
        if progress_callback is not None:
            progress_callback(i + 1, config.n_samples)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                feed_acid = float(feed_acid_arr[i])
                feed_A = 100.0 - feed_acid

                result = solve_crosscurrent(
                    feed_A=feed_A,
                    feed_C=feed_acid,
                    feed_flow=config.feed_flow,
                    solvent_per_stage=float(solvent_arr[i]),
                    n_stages=int(n_stages_arr[i]),
                    eq_model=eq_model,
                )

                results.append({
                    "n_stages": int(n_stages_arr[i]),
                    "solvent_per_stage": float(solvent_arr[i]),
                    "feed_acid_pct": feed_acid,
                    "pct_removal": result.total_pct_removal,
                })
        except Exception:
            continue

    df = pd.DataFrame(results)
    df = df.dropna()
    df = df[(df["pct_removal"] >= 0) & (df["pct_removal"] <= 100)]
    df = df.reset_index(drop=True)
    return df
