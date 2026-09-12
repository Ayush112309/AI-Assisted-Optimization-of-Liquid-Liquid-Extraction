"""
Countercurrent multistage liquid-liquid extraction solver.

Uses rigorous stage-by-stage mass balances and equilibrium.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import fsolve, brentq

from .equilibrium import (
    EquilibriumModel,
    get_X_from_ternary,
    get_N_from_ternary,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CountercurrentStage:
    """Results for a single countercurrent stage."""
    stage_number: int
    X_raff: float   # solvent-free raffinate composition
    Y_ext: float    # solvent-free extract composition
    N_raff: float   # solvent ratio in raffinate
    N_ext: float    # solvent ratio in extract
    A_raff: float   # wt% A in raffinate
    C_raff: float   # wt% C in raffinate
    B_raff: float   # wt% B in raffinate
    A_ext: float    # wt% A in extract
    C_ext: float    # wt% C in extract
    B_ext: float    # wt% B in extract
    R_flow: float = 0.0
    E_flow: float = 0.0
    section: str = "countercurrent"


@dataclass
class CountercurrentResult:
    """Complete countercurrent extraction results."""
    n_stages: int
    stages: List[CountercurrentStage] = field(default_factory=list)

    # Product specifications (solvent-free)
    X_feed: float = 0.0
    X_raff_spec: float = 0.0
    X_ext_spec: float = 0.0

    # Flows (solvent-free basis, kg/h)
    feed_flow_sf: float = 0.0
    extract_product_flow_sf: float = 0.0
    raffinate_product_flow_sf: float = 0.0


# ---------------------------------------------------------------------------
# Countercurrent
# ---------------------------------------------------------------------------

def solve_countercurrent(
    feed_A: float,
    feed_C: float,
    feed_flow: float,
    solvent_flow: float,
    n_stages: int,
    eq_model: EquilibriumModel,
) -> CountercurrentResult:
    """
    Solve countercurrent extraction using rigorous difference-point (Ponchon-Savarit) method.
    
    This solver iterates on the final raffinate composition X_1 until the
    corresponding feed stage matches the input feed.
    """
    X_F = feed_C / (feed_A + feed_C) if (feed_A + feed_C) > 0 else 0.0
    F_sf = feed_flow * (feed_A + feed_C) / 100.0
    feed_B = 100.0 - feed_A - feed_C
    N_F = feed_B / (feed_A + feed_C) if (feed_A + feed_C) > 0 else 0.0

    # Mixing point M coordinates
    X_M = X_F
    N_M = (feed_flow * feed_B / 100.0 + solvent_flow) / F_sf if F_sf > 0 else 0.0

    def residual_X1(X_1_guess):
        X_1 = float(np.clip(X_1_guess, 0.0001, X_F * 0.9999))
        A_R1, C_R1, B_R1 = eq_model.get_raffinate_point(X_1)
        N_R1 = B_R1 / (A_R1 + C_R1) if (A_R1 + C_R1) > 0 else 0.0

        # Point EN lies on the intersection of the extract curve and line R1-M
        if abs(X_M - X_1) < 1e-10: return 1e6
        slope_M = (N_M - N_R1) / (X_M - X_1)

        def en_eq(X_try):
            A_E, C_E, B_E = eq_model.get_extract_point(X_try)
            N_E = B_E / (A_E + C_E) if (A_E + C_E) > 0 else 0.0
            Y_E = C_E / (A_E + C_E) if (A_E + C_E) > 0 else 0.0
            return N_R1 + slope_M * (Y_E - X_1) - N_E

        try:
            X_N_sol = brentq(en_eq, 0.0, 1.0, xtol=1e-6)
        except:
            sol = fsolve(en_eq, [X_M])
            X_N_sol = float(sol[0])

        A_EN, C_EN, B_EN = eq_model.get_extract_point(X_N_sol)
        Y_N = C_EN / (A_EN + C_EN)
        N_EN = B_EN / (A_EN + C_EN)

        # Lever rule for R1_sf
        R1_sf = F_sf * (Y_N - X_F) / (Y_N - X_1) if abs(Y_N - X_1) > 1e-10 else F_sf
        
        # Difference point Delta = R1 - S (net flow)
        # S is pure solvent (Y=0, N=inf?? No, S_sf = 0)
        # Delta = R - E = constant. Delta_sf = R1_sf - 0 = R1_sf
        # Delta_sf * X_delta = R1_sf * X_1 - 0 => X_delta = X_1
        # Delta_sf * N_delta = R1_sf * N_R1 - S => N_delta = N_R1 - solvent_flow / R1_sf
        X_delta = X_1
        N_delta = N_R1 - solvent_flow / R1_sf if R1_sf > 0 else -1e6

        # Step through n_stages
        X_curr = X_1
        for stage in range(n_stages):
            if stage < n_stages - 1:
                # Equilibrium: Y_curr from X_curr (already done via robust mapping)
                A_Ei, C_Ei, B_Ei = eq_model.get_extract_point(X_curr)
                Y_Ei = C_Ei / (A_Ei + C_Ei)
                N_Ei = B_Ei / (A_Ei + C_Ei)
                
                # Operating line through Delta: connects (Y_Ei, N_Ei) to (X_{i+1}, N_{R,i+1})
                if abs(Y_Ei - X_delta) < 1e-10:
                    X_next = X_curr
                else:
                    slope_op = (N_Ei - N_delta) / (Y_Ei - X_delta)
                    def op_eq(X_try):
                        A_R, C_R, B_R = eq_model.get_raffinate_point(X_try)
                        N_R = B_R / (A_R + C_R) if (A_R + C_R) > 0 else 0.0
                        return N_delta + slope_op * (X_try - X_delta) - N_R
                    
                    try:
                        X_next = brentq(op_eq, X_curr, 1.0, xtol=1e-6)
                    except:
                        sol = fsolve(op_eq, [X_curr * 1.1])
                        X_next = float(sol[0])
                X_curr = X_next

        # X_curr after n_stages should be X_F
        return X_curr - X_N_sol

    try:
        X_1_sol = brentq(residual_X1, 0.0001, X_F, xtol=1e-6)
    except:
        sol = fsolve(lambda x: residual_X1(x[0]), [X_F * 0.5])
        X_1_sol = float(sol[0])

    # Reconstruct final result
    X_1 = float(np.clip(X_1_sol, 0.0001, 0.9999))
    A_R1, C_R1, B_R1 = eq_model.get_raffinate_point(X_1)
    N_R1 = B_R1 / (A_R1 + C_R1)
    
    # Solve for Y_N and Delta again
    X_F_val = X_F
    X_M_val = X_M
    N_M_val = N_M
    slope_M = (N_M_val - N_R1) / (X_M_val - X_1)
    
    def en_eq_final(X_try):
        A_E, C_E, B_E = eq_model.get_extract_point(X_try)
        N_E = B_E / (A_E + C_E)
        Y_E = C_E / (A_E + C_E)
        return N_R1 + slope_M * (Y_E - X_1) - N_E
    
    X_N_final = fsolve(en_eq_final, [X_M_val])[0]
    A_EN, C_EN, B_EN = eq_model.get_extract_point(X_N_final)
    Y_N = C_EN / (A_EN + C_EN)
    R1_sf = F_sf * (Y_N - X_F) / (Y_N - X_1)
    X_delta = X_1
    N_delta = N_R1 - solvent_flow / R1_sf

    stages = []
    X_curr = X_1
    for i in range(n_stages):
        A_R, C_R, B_R = eq_model.get_raffinate_point(X_curr)
        A_E, C_E, B_E = eq_model.get_extract_point(X_curr)
        Y_E = C_E / (A_E + C_E)
        
        # Approximate flows
        frac = i / max(1, n_stages - 1)
        R_sf = R1_sf * (1 - frac) + F_sf * frac
        E_sf = R_sf - R1_sf
        if i == n_stages - 1: E_sf = F_sf - R1_sf
            
        stages.append(CountercurrentStage(
            stage_number=i + 1,
            X_raff=X_curr, Y_ext=Y_E,
            N_raff=B_R/(A_R+C_R), N_ext=B_E/(A_E+C_E),
            A_raff=A_R, C_raff=C_R, B_raff=B_R,
            A_ext=A_E, C_ext=C_E, B_ext=B_E,
            R_flow=R_sf * (1 + B_R/(A_R+C_R)) / ((A_R+C_R)/100),
            E_flow=E_sf * (1 + B_E/(A_E+C_E)) / ((A_E+C_E)/100),
        ))

        if i < n_stages - 1:
            slope_op = (B_E/(A_E+C_E) - N_delta) / (Y_E - X_delta)
            X_curr = fsolve(lambda x: N_delta + slope_op * (x - X_delta) - eq_model.B_raff_from_X(x)/(eq_model.A_raff_from_X(x)+eq_model.C_raff_from_X(x)), [X_curr * 1.1])[0]

    return CountercurrentResult(
        n_stages=n_stages,
        stages=stages,
        X_feed=X_F,
        X_raff_spec=stages[0].X_raff,
        X_ext_spec=stages[-1].Y_ext,
        feed_flow_sf=F_sf,
    )

def find_max_extract_purity(eq: EquilibriumModel) -> float:
    X_vals = np.linspace(0.0, eq.X_range[1], 500)
    Y_vals = np.array([eq.Y_from_X(float(x)) for x in X_vals])
    return float(np.max(Y_vals))
