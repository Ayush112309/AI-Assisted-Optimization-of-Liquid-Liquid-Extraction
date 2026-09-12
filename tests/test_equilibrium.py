"""Tests for the equilibrium model."""

from pathlib import Path

import numpy as np
import pytest

from mass_transfer.core.equilibrium import (
    load_tie_line_data,
    fit_equilibrium_model,
    get_raffinate_point_from_X,
    get_extract_point_from_Y,
    get_equilibrium_extract_from_raffinate,
    get_X_from_ternary,
    get_N_from_ternary,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "mass_transfer" / "resources" / "data" / "default_tie_lines.json"


@pytest.fixture
def tie_data():
    return load_tie_line_data(DATA_PATH)


@pytest.fixture
def eq_model(tie_data):
    return fit_equilibrium_model(tie_data)


# --- Data loading tests ---

class TestDataLoading:
    def test_loads_correct_number_of_tie_lines(self, tie_data):
        assert len(tie_data.A_raff) == 12
        assert len(tie_data.A_ext) == 12

    def test_first_tie_line_is_binary(self, tie_data):
        """First tie-line has zero oleic acid."""
        assert tie_data.C_raff[0] == 0.0
        assert tie_data.C_ext[0] == 0.0

    def test_wt_percent_closure(self, tie_data):
        """A + C + B should sum to ~100 for all points."""
        raff_sum = tie_data.A_raff + tie_data.C_raff + tie_data.B_raff
        ext_sum = tie_data.A_ext + tie_data.C_ext + tie_data.B_ext
        np.testing.assert_allclose(raff_sum, 100.0, atol=0.5)
        np.testing.assert_allclose(ext_sum, 100.0, atol=0.5)


# --- Solvent-free coordinate tests ---

class TestSolventFreeCoords:
    def test_X_range(self, tie_data):
        """X should be between 0 and 1."""
        assert np.all(tie_data.X >= 0.0)
        assert np.all(tie_data.X <= 1.0)

    def test_Y_range(self, tie_data):
        """Y should be between 0 and 1."""
        assert np.all(tie_data.Y >= 0.0)
        assert np.all(tie_data.Y <= 1.0)

    def test_X_zero_at_binary(self, tie_data):
        """First tie-line: X=0 (no acid in raffinate)."""
        assert tie_data.X[0] == pytest.approx(0.0, abs=1e-10)

    def test_Y_zero_at_binary(self, tie_data):
        """First tie-line: Y=0 (no acid in extract)."""
        assert tie_data.Y[0] == pytest.approx(0.0, abs=1e-10)

    def test_N_positive(self, tie_data):
        """Solvent ratios should be positive."""
        assert np.all(tie_data.N_raff >= 0.0)
        assert np.all(tie_data.N_ext >= 0.0)

    def test_N_ext_much_larger_than_N_raff(self, tie_data):
        """Extract phase has much more solvent than raffinate."""
        # Skip the first (binary) tie-line
        for i in range(1, len(tie_data.N_ext)):
            assert tie_data.N_ext[i] > tie_data.N_raff[i]


# --- Curve fit tests ---

class TestCurveFits:
    def test_r_squared_above_threshold(self, eq_model):
        """All R² values should be > 0.90."""
        for name, r2_val in eq_model.r_squared.items():
            assert r2_val > 0.90, f"R² for {name} = {r2_val:.4f} < 0.90"

    def test_distribution_curve_at_origin(self, eq_model):
        """Y(0) should be approximately 0."""
        assert eq_model.Y_from_X(0.0) == pytest.approx(0.0, abs=0.05)

    def test_distribution_curve_monotonic_low_X(self, eq_model):
        """Y should increase with X in the low-X region (X < 0.7)."""
        X_vals = np.linspace(0.0, 0.7, 20)
        Y_vals = [eq_model.Y_from_X(x) for x in X_vals]
        for i in range(1, len(Y_vals)):
            assert Y_vals[i] >= Y_vals[i - 1] - 0.01, \
                f"Y not monotonic at X={X_vals[i]:.3f}: Y={Y_vals[i]:.4f} < Y={Y_vals[i-1]:.4f}"

    def test_inverse_distribution(self, eq_model):
        """X_from_Y should invert Y_from_X."""
        for x_orig in [0.1, 0.3, 0.5]:
            y = eq_model.Y_from_X(x_orig)
            x_back = eq_model.X_from_Y(y)
            assert x_back == pytest.approx(x_orig, abs=0.02)


# --- Ternary reconstruction tests ---

class TestTernaryReconstruction:
    def test_raffinate_point_from_X(self, eq_model, tie_data):
        """Reconstructed raffinate point should match data approximately."""
        for i in [1, 3, 5, 7]:  # skip binary
            X = tie_data.X[i]
            A, C, B = get_raffinate_point_from_X(eq_model, X)
            assert A + C + B == pytest.approx(100.0, abs=1.0)
            X_recon = get_X_from_ternary(A, C)
            assert X_recon == pytest.approx(X, abs=0.03)

    def test_extract_point_closure(self, eq_model, tie_data):
        """Reconstructed extract point should sum to ~100."""
        for i in [1, 3, 5]:
            Y = tie_data.Y[i]
            A, C, B = get_extract_point_from_Y(eq_model, Y)
            assert A + C + B == pytest.approx(100.0, abs=2.0)

    def test_equilibrium_extract(self, eq_model, tie_data):
        """get_equilibrium_extract should return valid extract compositions."""
        A_raff = tie_data.A_raff[3]
        C_raff = tie_data.C_raff[3]
        A_ext, C_ext, B_ext = get_equilibrium_extract_from_raffinate(
            eq_model, A_raff, C_raff
        )
        assert A_ext >= 0
        assert C_ext >= 0
        assert B_ext >= 0
        assert A_ext + C_ext + B_ext == pytest.approx(100.0, abs=2.0)


# --- Utility tests ---

class TestUtilities:
    def test_get_X_from_ternary(self):
        assert get_X_from_ternary(75.0, 25.0) == pytest.approx(0.25, abs=1e-10)
        assert get_X_from_ternary(0.0, 0.0) == 0.0

    def test_get_N_from_ternary(self):
        assert get_N_from_ternary(50.0, 10.0, 40.0) == pytest.approx(40.0 / 60.0, abs=1e-10)
