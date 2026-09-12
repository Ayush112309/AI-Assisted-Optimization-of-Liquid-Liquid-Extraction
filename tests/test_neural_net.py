"""Tests for the ML surrogate model."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mass_transfer.core.equilibrium import load_tie_line_data, fit_equilibrium_model
from mass_transfer.ml.data_generator import generate_crosscurrent_dataset_serial, DataGenConfig
from mass_transfer.ml.neural_net import (
    ExtractionANN,
    TrainingConfig,
    train_model,
    predict,
    save_model,
    load_model,
)
from mass_transfer.ml.optimization import generate_response_surface, find_optimal_conditions

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "mass_transfer" / "resources" / "data" / "default_tie_lines.json"


@pytest.fixture(scope="module")
def eq_model():
    data = load_tie_line_data(DATA_PATH)
    return fit_equilibrium_model(data)


@pytest.fixture(scope="module")
def small_dataset(eq_model):
    """Generate a small dataset for fast testing."""
    config = DataGenConfig(
        n_stages_range=(1, 5),
        solvent_range=(200, 2000),
        feed_acid_pct_range=(10, 40),
        n_samples=200,
        random_seed=42,
    )
    return generate_crosscurrent_dataset_serial(eq_model, config)


@pytest.fixture(scope="module")
def training_result(small_dataset):
    """Train a model on the small dataset."""
    config = TrainingConfig(
        hidden_sizes=(32, 16),
        learning_rate=1e-3,
        batch_size=32,
        epochs=50,
        patience=15,
    )
    return train_model(small_dataset, config)


class TestDataGeneration:
    def test_generates_data(self, small_dataset):
        assert len(small_dataset) > 50
        assert "pct_removal" in small_dataset.columns
        assert "n_stages" in small_dataset.columns

    def test_removal_in_valid_range(self, small_dataset):
        assert (small_dataset["pct_removal"] >= 0).all()
        assert (small_dataset["pct_removal"] <= 100).all()

    def test_no_nans(self, small_dataset):
        assert not small_dataset.isna().any().any()


class TestModelArchitecture:
    def test_model_forward_pass(self):
        model = ExtractionANN(hidden_sizes=(64, 32))
        x = np.random.randn(10, 3).astype(np.float32)
        import torch
        out = model(torch.tensor(x))
        assert out.shape == (10, 1)

    def test_model_custom_sizes(self):
        model = ExtractionANN(hidden_sizes=(128, 64, 32))
        x = np.random.randn(5, 3).astype(np.float32)
        import torch
        out = model(torch.tensor(x))
        assert out.shape == (5, 1)


class TestTraining:
    def test_training_completes(self, training_result):
        assert training_result.model is not None
        assert len(training_result.train_losses) > 0

    def test_loss_decreases(self, training_result):
        """Training loss should generally decrease."""
        losses = training_result.train_losses
        # Compare first 5 avg to last 5 avg
        if len(losses) >= 10:
            first = np.mean(losses[:5])
            last = np.mean(losses[-5:])
            assert last < first

    def test_r_squared_positive(self, training_result):
        """Test R² should be positive (model better than mean)."""
        assert training_result.test_r_squared > 0.0

    def test_prediction_in_range(self, training_result):
        """Predictions should be in [0, 100]."""
        pred = predict(
            training_result.model,
            training_result.scaler_X,
            training_result.scaler_y,
            n_stages=3, solvent=1000, feed_comp=25.0,
        )
        assert 0 <= pred <= 100

    def test_more_solvent_more_removal(self, training_result):
        """Increasing solvent should generally increase predicted removal."""
        pred_low = predict(
            training_result.model,
            training_result.scaler_X,
            training_result.scaler_y,
            n_stages=3, solvent=200, feed_comp=25.0,
        )
        pred_high = predict(
            training_result.model,
            training_result.scaler_X,
            training_result.scaler_y,
            n_stages=3, solvent=2000, feed_comp=25.0,
        )
        assert pred_high > pred_low


class TestSaveLoad:
    def test_save_and_load(self, training_result, tmp_path):
        path = tmp_path / "test_model.pt"
        save_model(training_result, path)
        loaded = load_model(path, hidden_sizes=(32, 16))
        assert loaded.model is not None
        assert loaded.test_r_squared == pytest.approx(training_result.test_r_squared)

        # Predictions should match
        pred_orig = predict(
            training_result.model, training_result.scaler_X, training_result.scaler_y,
            3, 1000, 25.0,
        )
        pred_loaded = predict(
            loaded.model, loaded.scaler_X, loaded.scaler_y,
            3, 1000, 25.0,
        )
        assert pred_orig == pytest.approx(pred_loaded, abs=0.1)


class TestResponseSurface:
    def test_surface_shape(self, training_result):
        X, Y, Z = generate_response_surface(
            training_result,
            var1_range=(1, 5),
            var2_range=(200, 2000),
            grid_size=10,
        )
        assert X.shape == (10, 10)
        assert Z.shape == (10, 10)

    def test_surface_values_in_range(self, training_result):
        _, _, Z = generate_response_surface(
            training_result,
            var1_range=(1, 5),
            var2_range=(200, 2000),
            grid_size=10,
        )
        assert np.all(Z >= -5)  # allow small negative due to model
        assert np.all(Z <= 105)


class TestOptimization:
    def test_find_optimal(self, training_result):
        result = find_optimal_conditions(
            training_result,
            target_removal=50.0,
            objective="min_solvent",
            n_stages_range=(1, 5),
            solvent_range=(200, 2000),
        )
        assert "n_stages" in result
        assert "solvent_per_stage" in result
        assert result["predicted_removal"] >= 40.0  # close to target
