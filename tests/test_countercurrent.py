"""Tests for the countercurrent extraction solver."""

from pathlib import Path

import numpy as np
import pytest

from mass_transfer.core.equilibrium import load_tie_line_data, fit_equilibrium_model
from mass_transfer.core.countercurrent import (
    solve_countercurrent_simple,
    find_min_stages,
    find_min_reflux_ratio,
    solve_countercurrent_reflux,
    find_max_extract_purity,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "mass_transfer" / "resources" / "data" / "default_tie_lines.json"


@pytest.fixture(scope="module")
def eq_model():
    data = load_tie_line_data(DATA_PATH)
    return fit_equilibrium_model(data)


class TestMinStages:
    """Tests for minimum stages at total reflux."""

    def test_min_stages_positive(self, eq_model):
        """Min stages should be a positive integer."""
        n = find_min_stages(0.25, 0.02, 0.90, eq_model)
        assert n > 0
        assert isinstance(n, int)

    def test_min_stages_reasonable_range(self, eq_model):
        """For 2%→90% separation, min stages should be in reasonable range (3-15)."""
        n = find_min_stages(0.25, 0.02, 0.90, eq_model)
        assert 3 <= n <= 15, f"Min stages = {n}, expected 3-15"

    def test_tighter_spec_needs_more_stages(self, eq_model):
        """Tighter specifications should require more stages."""
        n_easy = find_min_stages(0.25, 0.05, 0.80, eq_model)
        n_hard = find_min_stages(0.25, 0.02, 0.90, eq_model)
        assert n_hard >= n_easy


class TestMinReflux:
    """Tests for minimum reflux ratio."""

    def test_min_reflux_positive(self, eq_model):
        """Min reflux ratio should be positive."""
        r_min = find_min_reflux_ratio(75.0, 25.0, 1000.0, 0.02, 0.90, eq_model)
        assert r_min > 0

    def test_min_reflux_reasonable(self, eq_model):
        """Min reflux should be in a reasonable range."""
        r_min = find_min_reflux_ratio(75.0, 25.0, 1000.0, 0.02, 0.90, eq_model)
        assert 0.1 <= r_min <= 20.0, f"r_min = {r_min}, expected 0.1-20"


class TestCountercurrentReflux:
    """Tests for countercurrent extraction with reflux (Problem Part iii)."""

    def test_reflux_produces_stages(self, eq_model):
        """Solver should produce a valid number of stages."""
        result = solve_countercurrent_reflux(
            feed_A=75.0, feed_C=25.0, feed_flow=1000.0,
            reflux_ratio=4.5, X_raff_spec=0.02, X_ext_spec=0.90,
            eq_model=eq_model,
        )
        assert result.n_stages > 0
        assert len(result.stages) > 0

    def test_reflux_product_flows(self, eq_model):
        """Product flows should sum to feed on sf basis."""
        result = solve_countercurrent_reflux(
            feed_A=75.0, feed_C=25.0, feed_flow=1000.0,
            reflux_ratio=4.5, X_raff_spec=0.02, X_ext_spec=0.90,
            eq_model=eq_model,
        )
        total = result.extract_product_flow_sf + result.raffinate_product_flow_sf
        assert total == pytest.approx(result.feed_flow_sf, rel=0.01)

    def test_reflux_stages_have_valid_compositions(self, eq_model):
        """All stage compositions should be physically valid."""
        result = solve_countercurrent_reflux(
            feed_A=75.0, feed_C=25.0, feed_flow=1000.0,
            reflux_ratio=4.5, X_raff_spec=0.02, X_ext_spec=0.90,
            eq_model=eq_model,
        )
        for s in result.stages:
            assert 0 <= s.X_raff <= 1.0, f"Stage {s.stage_number}: X={s.X_raff}"
            assert 0 <= s.Y_ext <= 1.0, f"Stage {s.stage_number}: Y={s.Y_ext}"
            assert s.N_raff >= 0, f"Stage {s.stage_number}: N_raff={s.N_raff}"
            assert s.N_ext >= 0, f"Stage {s.stage_number}: N_ext={s.N_ext}"

    def test_actual_reflux_above_min(self, eq_model):
        """The design reflux (4.5) should be above the minimum."""
        result = solve_countercurrent_reflux(
            feed_A=75.0, feed_C=25.0, feed_flow=1000.0,
            reflux_ratio=4.5, X_raff_spec=0.02, X_ext_spec=0.90,
            eq_model=eq_model,
        )
        assert result.min_reflux_ratio is not None
        assert 4.5 > result.min_reflux_ratio

    def test_more_stages_than_minimum(self, eq_model):
        """Design stages should be more than minimum stages."""
        result = solve_countercurrent_reflux(
            feed_A=75.0, feed_C=25.0, feed_flow=1000.0,
            reflux_ratio=4.5, X_raff_spec=0.02, X_ext_spec=0.90,
            eq_model=eq_model,
        )
        assert result.min_stages is not None
        assert result.n_stages >= result.min_stages

    def test_difference_points_exist(self, eq_model):
        """Difference points should be computed."""
        result = solve_countercurrent_reflux(
            feed_A=75.0, feed_C=25.0, feed_flow=1000.0,
            reflux_ratio=4.5, X_raff_spec=0.02, X_ext_spec=0.90,
            eq_model=eq_model,
        )
        assert result.delta_E is not None
        assert result.delta_S is not None


class TestSimpleCountercurrent:
    """Tests for simple countercurrent (no reflux)."""

    def test_simple_produces_stages(self, eq_model):
        result = solve_countercurrent_simple(
            feed_A=75.0, feed_C=25.0, feed_flow=1000.0,
            solvent_flow=5000.0, n_stages=5,
            eq_model=eq_model,
        )
        assert len(result.stages) == 5

    def test_simple_compositions_decrease(self, eq_model):
        """X should decrease from feed end to raffinate end."""
        result = solve_countercurrent_simple(
            feed_A=75.0, feed_C=25.0, feed_flow=1000.0,
            solvent_flow=5000.0, n_stages=5,
            eq_model=eq_model,
        )
        X_vals = [s.X_raff for s in result.stages]
        # Stage 1 is raffinate end (lowest X), stage N is feed end (highest X)
        assert X_vals[0] < X_vals[-1]


class TestMaxPurity:
    """Test max achievable purity."""

    def test_max_purity_reasonable(self, eq_model):
        """Max purity should be between 80% and 100%."""
        max_Y = find_max_extract_purity(eq_model)
        assert 0.80 <= max_Y <= 1.0
