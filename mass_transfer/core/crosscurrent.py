"""
Crosscurrent (single-section) multistage liquid-liquid extraction solver.

Each stage receives fresh solvent. Stages are solved sequentially:
    Feed → Stage 1 → Stage 2 → ... → Stage N → Final Raffinate
              ↓           ↓                ↓
           Extract 1   Extract 2        Extract N
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.optimize import fsolve

from .equilibrium import (
    EquilibriumModel,
    get_X_from_ternary,
    get_N_from_ternary,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    """Results for a single crosscurrent stage."""
    stage_number: int

    # Raffinate leaving this stage (wt%)
    A_raff: float
    C_raff: float
    B_raff: float
    R_flow: float  # total raffinate flow (kg or kg/h)

    # Extract leaving this stage (wt%)
    A_ext: float
    C_ext: float
    B_ext: float
    E_flow: float  # total extract flow

    # Solvent-free values
    X_raff: float   # C/(A+C) in raffinate
    Y_ext: float    # C/(A+C) in extract
    N_raff: float   # B/(A+C) in raffinate
    N_ext: float    # B/(A+C) in extract

    # Performance
    acid_removed_kg: float    # kg of C removed in this stage
    pct_removal_stage: float  # % of feed acid removed in this stage
    pct_removal_cumul: float  # cumulative % removal


@dataclass
class CrosscurrentResult:
    """Complete crosscurrent extraction results."""
    # Feed conditions
    feed_A: float
    feed_C: float
    feed_B: float
    feed_flow: float
    solvent_per_stage: float
    n_stages: int

    stages: List[StageResult] = field(default_factory=list)

    # Mixed extract totals
    mixed_extract_A: float = 0.0
    mixed_extract_C: float = 0.0
    mixed_extract_B: float = 0.0
    mixed_extract_flow: float = 0.0

    # Final products (solvent-free basis)
    final_raff_X: float = 0.0
    final_ext_Y_mixed: float = 0.0
    total_pct_removal: float = 0.0


# ---------------------------------------------------------------------------
# Single stage solver
# ---------------------------------------------------------------------------

def _solve_single_stage(
    R_prev: float,
    xA_prev: float,
    xC_prev: float,
    xB_prev: float,
    S: float,
    eq: EquilibriumModel,
) -> tuple:
    """
    Solve one crosscurrent stage by finding X_raff that satisfies mass balance.
    """
    M_in = R_prev + S
    mA_in = R_prev * xA_prev
    mC_in = R_prev * xC_prev
    mB_in = R_prev * xB_prev + S

    def balance_resid(X_raff_guess):
        X_raff = float(X_raff_guess[0])
        X_raff = max(0.0, min(1.0, X_raff))
        
        # Get properties from robust X-mapping
        A_R, C_R, B_R = eq.get_raffinate_point(X_raff)
        A_E, C_E, B_E = eq.get_extract_point(X_raff)
        
        # Solver unknowns: X_raff and R_out
        # But we can solve for R_out analytically from the carrier balance:
        # mA_in = R_out * wA_R + E_out * wA_E
        # mA_in = R_out * wA_R + (M_in - R_out) * wA_E
        # mA_in = R_out * (wA_R - wA_E) + M_in * wA_E
        # R_out = (mA_in - M_in * wA_E) / (wA_R - wA_E)
        
        wA_R, wA_E = A_R/100.0, A_E/100.0
        wC_R, wC_E = C_R/100.0, C_E/100.0
        
        denom = (wA_R - wA_E)
        if abs(denom) < 1e-10:
            R_out = M_in / 2.0
        else:
            R_out = (mA_in - M_in * wA_E) / denom
            
        E_out = M_in - R_out
        
        # Return solute balance residual
        return [mC_in - (R_out * wC_R + E_out * wC_E)]

    # Initial guess: current X
    X_start = mC_in / (mA_in + mC_in) if (mA_in + mC_in) > 0 else 0.0
    
    sol = fsolve(balance_resid, [X_start * 0.8])
    X_sol = float(sol[0])
    
    A_R, C_R, B_R = eq.get_raffinate_point(X_sol)
    A_E, C_E, B_E = eq.get_extract_point(X_sol)
    
    wA_R, wA_E = A_R/100.0, A_E/100.0
    R_sol = (mA_in - M_in * wA_E) / (wA_R - wA_E) if abs(wA_R - wA_E) > 1e-10 else M_in/2.0
    E_sol = M_in - R_sol
    
    return (
        R_sol, A_R/100.0, C_R/100.0, B_R/100.0,
        E_sol, A_E/100.0, C_E/100.0, B_E/100.0
    )


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_crosscurrent(
    feed_A: float,
    feed_C: float,
    feed_flow: float,
    solvent_per_stage: float,
    n_stages: int,
    eq_model: EquilibriumModel,
    feed_B: float = 0.0,
) -> CrosscurrentResult:
    feed_B_actual = feed_B if feed_B > 0 else (100.0 - feed_A - feed_C)
    
    result = CrosscurrentResult(
        feed_A=feed_A, feed_C=feed_C, feed_B=feed_B_actual,
        feed_flow=feed_flow, solvent_per_stage=solvent_per_stage, n_stages=n_stages
    )

    R_prev = feed_flow
    xA_prev, xC_prev, xB_prev = feed_A/100.0, feed_C/100.0, feed_B_actual/100.0
    total_acid_in_feed = feed_flow * feed_C/100.0
    cumul_acid_removed = 0.0
    
    total_ext_A, total_ext_C, total_ext_B, total_ext_flow = 0.0, 0.0, 0.0, 0.0

    for stage_num in range(1, n_stages + 1):
        R_out, wA_R, wC_R, wB_R, E_out, wA_E, wC_E, wB_E = _solve_single_stage(
            R_prev, xA_prev, xC_prev, xB_prev, solvent_per_stage, eq_model
        )

        acid_removed = (R_prev * xC_prev) - (R_out * wC_R)
        cumul_acid_removed += acid_removed

        AC_R = wA_R + wC_R
        X_raff = wC_R / AC_R if AC_R > 0 else 0.0
        AC_E = wA_E + wC_E
        Y_ext = wC_E / AC_E if AC_E > 0 else 0.0

        stage = StageResult(
            stage_number=stage_num,
            A_raff=wA_R*100, C_raff=wC_R*100, B_raff=wB_R*100, R_flow=R_out,
            A_ext=wA_E*100, C_ext=wC_E*100, B_ext=wB_E*100, E_flow=E_out,
            X_raff=X_raff, Y_ext=Y_ext,
            N_raff=wB_R/AC_R if AC_R > 0 else 0,
            N_ext=wB_E/AC_E if AC_E > 0 else 0,
            acid_removed_kg=acid_removed,
            pct_removal_stage=(acid_removed/total_acid_in_feed*100 if total_acid_in_feed>0 else 0),
            pct_removal_cumul=(cumul_acid_removed/total_acid_in_feed*100 if total_acid_in_feed>0 else 0),
        )
        result.stages.append(stage)

        total_ext_A += E_out * wA_E
        total_ext_C += E_out * wC_E
        total_ext_B += E_out * wB_E
        total_ext_flow += E_out

        R_prev, xA_prev, xC_prev, xB_prev = R_out, wA_R, wC_R, wB_R

    if total_ext_flow > 0:
        result.mixed_extract_A = total_ext_A / total_ext_flow * 100
        result.mixed_extract_C = total_ext_C / total_ext_flow * 100
        result.mixed_extract_B = total_ext_B / total_ext_flow * 100
    result.mixed_extract_flow = total_ext_flow
    result.final_raff_X = result.stages[-1].X_raff
    result.total_pct_removal = result.stages[-1].pct_removal_cumul

    return result
