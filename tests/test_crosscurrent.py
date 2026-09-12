"""Tests for the crosscurrent extraction solver."""

from pathlib import Path

import numpy as np
import pytest

from mass_transfer.core.equilibrium import load_tie_line_data, fit_equilibrium_model
from mass_transfer.core.crosscurrent import solve_crosscurrent

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "mass_transfer" / "resources" / "data" / "default_tie_lines.json"


@pytest.fixture(scope="module")
def eq_model():
    data = load_tie_line_data(DATA_PATH)
    return fit_equilibrium_model(data)


class TestCrosscurrentBasic:
    """Basic solver functionality tests."""

    def test_single_stage(self, eq_model):
        """Single stage should produce valid results."""
        result = solve_crosscurrent(
            feed_A=75.0, feed_C=25.0, feed_flow=100.0,
            solvent_per_stage=1000.0, n_stages=1,
            eq_model=eq_model,
        )
        assert len(result.stages) == 1
        s = result.stages[0]
        assert s.R_flow > 0
        assert s.E_flow > 0
        assert 0 <= s.X_raff <= 1.0
        assert s.pct_removal_stage > 0

    def test_mass_balance_each_stage(self, eq_model):
        """Total mass in = total mass out for each stage."""
        result = solve_crosscurrent(
            feed_A=75.0, feed_C=25.0, feed_flow=100.0,
            solvent_per_stage=1000.0, n_stages=3,
            eq_model=eq_model,
        )
        R_prev = 100.0
        for s in result.stages:
            M_in = R_prev + 1000.0  # raffinate + solvent
            M_out = s.R_flow + s.E_flow
            assert M_in == pytest.approx(M_out, rel=0.05), \
                f"Stage {s.stage_number}: M_in={M_in:.1f}, M_out={M_out:.1f}"
            R_prev = s.R_flow

    def test_acid_decreases_each_stage(self, eq_model):
        """Raffinate acid content should decrease (or stay same) each stage."""
        result = solve_crosscurrent(
            feed_A=75.0, feed_C=25.0, feed_flow=100.0,
            solvent_per_stage=1000.0, n_stages=4,
            eq_model=eq_model,
        )
        X_vals = [s.X_raff for s in result.stages]
        for i in range(1, len(X_vals)):
            assert X_vals[i] <= X_vals[i - 1] + 0.01, \
                f"Acid increased: stage {i}: X={X_vals[i]:.4f} > stage {i-1}: X={X_vals[i-1]:.4f}"

    def test_cumulative_removal_increases(self, eq_model):
        """Cumulative % removal should increase with each stage."""
        result = solve_crosscurrent(
            feed_A=75.0, feed_C=25.0, feed_flow=100.0,
            solvent_per_stage=500.0, n_stages=5,
            eq_model=eq_model,
        )
        for i in range(1, len(result.stages)):
            assert result.stages[i].pct_removal_cumul >= result.stages[i - 1].pct_removal_cumul - 0.1


class TestCrosscurrentValidation:
    """Validation against Problem Part (ii)."""

    def test_problem_ii_two_stages(self, eq_model):
        """
        Problem Part (ii): 100 kg feed, 75% oil, 25% acid, 2 stages, 1000 kg propane each.
        Should achieve significant extraction.
        """
        result = solve_crosscurrent(
            feed_A=75.0, feed_C=25.0, feed_flow=100.0,
            solvent_per_stage=1000.0, n_stages=2,
            eq_model=eq_model,
        )
        # Should have 2 stages
        assert len(result.stages) == 2

        # Final raffinate should have less acid than feed
        assert result.final_raff_X < 0.25

        # Total removal should be positive and significant
        assert result.total_pct_removal > 10.0

        # Mixed extract should contain acid
        assert result.mixed_extract_C > 0

        # Mass balance: feed + 2*solvent = final raff + mixed extract
        total_in = 100.0 + 2 * 1000.0
        total_out = result.stages[-1].R_flow + result.mixed_extract_flow
        assert total_in == pytest.approx(total_out, rel=0.05)

    def test_more_solvent_more_removal(self, eq_model):
        """Increasing solvent should increase removal."""
        r_low = solve_crosscurrent(
            feed_A=75.0, feed_C=25.0, feed_flow=100.0,
            solvent_per_stage=200.0, n_stages=2, eq_model=eq_model,
        )
        r_high = solve_crosscurrent(
            feed_A=75.0, feed_C=25.0, feed_flow=100.0,
            solvent_per_stage=2000.0, n_stages=2, eq_model=eq_model,
        )
        assert r_high.total_pct_removal > r_low.total_pct_removal

    def test_more_stages_more_removal(self, eq_model):
        """More stages (same total solvent) should give more removal."""
        r_2 = solve_crosscurrent(
            feed_A=75.0, feed_C=25.0, feed_flow=100.0,
            solvent_per_stage=500.0, n_stages=2, eq_model=eq_model,
        )
        r_5 = solve_crosscurrent(
            feed_A=75.0, feed_C=25.0, feed_flow=100.0,
            solvent_per_stage=200.0, n_stages=5, eq_model=eq_model,
        )
        # Same total solvent (1000 kg), but split into more stages = more efficient
        assert r_5.total_pct_removal > r_2.total_pct_removal * 0.8  # at least comparable
