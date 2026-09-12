"""
Equilibrium model for ternary liquid-liquid extraction.

Components:
    A = Carrier
    B = Solvent
    C = Solute

Ternary coordinates: wt% basis, right-angle triangle (x=A, y=C, B=100-A-C).
Solvent-free basis:
    X = C/(A+C) in raffinate phase
    Y = C/(A+C) in extract phase
    N = B/(A+C) solvent ratio
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import fsolve


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TieLineData:
    """Raw and solvent-free tie-line data."""
    # Raffinate phase (wt%)
    A_raff: np.ndarray
    C_raff: np.ndarray
    B_raff: np.ndarray
    # Extract phase (wt%)
    A_ext: np.ndarray
    C_ext: np.ndarray
    B_ext: np.ndarray
    # Solvent-free coordinates
    X: np.ndarray          # C/(A+C) raffinate
    Y: np.ndarray          # C/(A+C) extract
    N_raff: np.ndarray     # B/(A+C) raffinate
    N_ext: np.ndarray      # B/(A+C) extract


@dataclass
class EquilibriumModel:
    """Fitted equilibrium model with robust interpolation."""
    tie_line_data: TieLineData

    # Property mappings as functions of X (solvent-free raffinate composition)
    A_raff_from_X: Callable[[float], float] = field(repr=False)
    C_raff_from_X: Callable[[float], float] = field(repr=False)
    B_raff_from_X: Callable[[float], float] = field(repr=False)
    
    A_ext_from_X: Callable[[float], float] = field(repr=False)
    C_ext_from_X: Callable[[float], float] = field(repr=False)
    B_ext_from_X: Callable[[float], float] = field(repr=False)

    # Legacy/Forward distribution curve
    Y_from_X: Callable[[float], float] = field(repr=False)
    N_raff_from_X: Callable[[float], float] = field(repr=False)
    N_ext_from_Y: Callable[[float], float] = field(repr=False)
    X_ext_from_X_raff: Callable[[float], float] = field(repr=False)

    # Ternary boundary curves (for plotting)
    C_raff_from_A: Callable[[float], float] = field(repr=False)
    C_ext_from_A: Callable[[float], float] = field(repr=False)
    C_raff_from_B: Callable[[float], float] = field(repr=False)
    C_ext_from_B: Callable[[float], float] = field(repr=False)

    # Plait point (where the two phases merge)
    plait_point: Optional[Tuple[float, float, float]] = None

    # GUI Compatibility fields
    r_squared: dict = field(default_factory=dict)
    poly_coeffs: dict = field(default_factory=dict)

    # Bounds for valid interpolation
    X_range: Tuple[float, float] = (0.0, 1.0)
    Y_range: Tuple[float, float] = (0.0, 1.0)

    def X_from_Y(self, Y_val: float) -> float:
        """Inverse of Y_from_X via numerical root-finding."""
        Y_val = float(np.clip(Y_val, self.Y_range[0], self.Y_range[1]))
        idx = np.argmin(np.abs(self.tie_line_data.Y - Y_val))
        x0 = float(self.tie_line_data.X[idx])
        try:
            sol = fsolve(lambda x: self.Y_from_X(float(x)) - Y_val, x0)
            return float(sol[0])
        except:
            return x0

    def get_raffinate_point(self, X: float) -> Tuple[float, float, float]:
        X = float(np.clip(X, self.X_range[0], self.X_range[1]))
        return (self.A_raff_from_X(X), self.C_raff_from_X(X), self.B_raff_from_X(X))

    def get_extract_point(self, X: float) -> Tuple[float, float, float]:
        X = float(np.clip(X, self.X_range[0], self.X_range[1]))
        return (self.A_ext_from_X(X), self.C_ext_from_X(X), self.B_ext_from_X(X))


# ---------------------------------------------------------------------------
# Loading and Fitting
# ---------------------------------------------------------------------------

def load_tie_line_data(filepath: str | Path) -> TieLineData:
    filepath = Path(filepath)
    with open(filepath, "r") as f:
        raw = json.load(f)

    phase_1 = raw["phase_1"]
    phase_2 = raw["phase_2"]

    sample = phase_1[0]
    if "water" in sample:
        key_A, key_C, key_B = "water", "acetic_acid", "isopropyl_ether"
    elif "cottonseed_oil" in sample:
        key_A, key_C, key_B = "cottonseed_oil", "oleic_acid", "propane"
    elif "A" in sample:
        key_A, key_C, key_B = "A", "C", "B"
    else:
        keys = list(sample.keys())
        key_A, key_C, key_B = keys[0], keys[1], keys[2]

    A_raff = np.array([p[key_A] for p in phase_1], dtype=float)
    C_raff = np.array([p[key_C] for p in phase_1], dtype=float)
    B_raff = np.array([p[key_B] for p in phase_1], dtype=float)

    A_ext = np.array([p[key_A] for p in phase_2], dtype=float)
    C_ext = np.array([p[key_C] for p in phase_2], dtype=float)
    B_ext = np.array([p[key_B] for p in phase_2], dtype=float)

    AC_raff = A_raff + C_raff
    AC_ext = A_ext + C_ext

    X = np.where(AC_raff > 0, C_raff / AC_raff, 0.0)
    Y = np.where(AC_ext > 0, C_ext / AC_ext, 0.0)
    N_raff = np.where(AC_raff > 0, B_raff / AC_raff, 0.0)
    N_ext = np.where(AC_ext > 0, B_ext / AC_ext, 0.0)

    return TieLineData(
        A_raff=A_raff, C_raff=C_raff, B_raff=B_raff,
        A_ext=A_ext, C_ext=C_ext, B_ext=B_ext,
        X=X, Y=Y, N_raff=N_raff, N_ext=N_ext,
    )

def _fit_interp(x: np.ndarray, y: np.ndarray) -> Callable[[float], float]:
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    if len(np.unique(xs)) < len(xs):
        xs = xs + np.arange(len(xs)) * 1e-10
    interp = PchipInterpolator(xs, ys, extrapolate=True)
    def wrapper(val):
        return float(interp(float(val)))
    return wrapper

def fit_equilibrium_model(data: TieLineData) -> EquilibriumModel:
    order = np.argsort(data.X)
    X_sorted = data.X[order]
    
    A_raff_X = _fit_interp(X_sorted, data.A_raff[order])
    C_raff_X = _fit_interp(X_sorted, data.C_raff[order])
    B_raff_X = _fit_interp(X_sorted, data.B_raff[order])
    
    A_ext_X = _fit_interp(X_sorted, data.A_ext[order])
    C_ext_X = _fit_interp(X_sorted, data.C_ext[order])
    B_ext_X = _fit_interp(X_sorted, data.B_ext[order])
    
    Y_X = _fit_interp(X_sorted, data.Y[order])
    N_R_X = _fit_interp(X_sorted, data.N_raff[order])
    N_E_Y = _fit_interp(data.Y, data.N_ext)

    C_R_A = _fit_interp(data.A_raff, data.C_raff)
    C_R_B = _fit_interp(data.B_raff, data.C_raff)
    C_E_A = _fit_interp(data.A_ext, data.C_ext)
    C_E_B = _fit_interp(data.B_ext, data.C_ext)

    X_range = (float(np.min(data.X)), float(np.max(data.X)))
    Y_range = (float(np.min(data.Y)), float(np.max(data.Y)))

    r2 = {"Y(X)": 1.0, "N_raff(X)": 1.0, "N_ext(Y)": 1.0}

    plait = None
    try:
        sep = (data.A_raff - data.A_ext)**2 + (data.C_raff - data.C_ext)**2
        idx = np.argmin(sep)
        if sep[idx] < 5.0:
            plait = (float(data.A_raff[idx]), float(data.C_raff[idx]), float(data.B_raff[idx]))
    except: pass

    return EquilibriumModel(
        tie_line_data=data,
        A_raff_from_X=A_raff_X, C_raff_from_X=C_raff_X, B_raff_from_X=B_raff_X,
        A_ext_from_X=A_ext_X, C_ext_from_X=C_ext_X, B_ext_from_X=B_ext_X,
        Y_from_X=Y_X, N_raff_from_X=N_R_X, N_ext_from_Y=N_E_Y,
        X_ext_from_X_raff=Y_X,
        C_raff_from_A=C_R_A, C_ext_from_A=C_E_A,
        C_raff_from_B=C_R_B, C_ext_from_B=C_E_B,
        X_range=X_range, Y_range=Y_range,
        r_squared=r2,
        plait_point=plait
    )

# ---------------------------------------------------------------------------
# Legacy helper functions
# ---------------------------------------------------------------------------

def get_raffinate_point_from_X(eq: EquilibriumModel, X: float) -> Tuple[float, float, float]:
    return eq.get_raffinate_point(X)

def get_extract_point_from_Y(eq: EquilibriumModel, Y: float) -> Tuple[float, float, float]:
    X = eq.X_from_Y(Y)
    return eq.get_extract_point(X)

def get_extract_point_from_X(eq: EquilibriumModel, X: float) -> Tuple[float, float, float]:
    return eq.get_extract_point(X)

def get_equilibrium_extract_from_raffinate(
    eq: EquilibriumModel, A_raff: float, C_raff: float
) -> Tuple[float, float, float]:
    AC = A_raff + C_raff
    if AC <= 0: return (0.0, 0.0, 100.0)
    X = C_raff / AC
    return eq.get_extract_point(X)

def get_N_from_ternary(A: float, C: float, B: float) -> float:
    AC = A + C
    return B / AC if AC > 0 else float("inf")

def get_X_from_ternary(A: float, C: float) -> float:
    AC = A + C
    return C / AC if AC > 0 else 0.0
