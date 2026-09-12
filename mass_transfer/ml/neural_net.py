"""
PyTorch ANN surrogate model for crosscurrent extraction.

Architecture: Input(3) → Linear(64)+ReLU+BatchNorm → Linear(32)+ReLU+BatchNorm → Linear(1)
Inputs: (n_stages, solvent_per_stage, feed_acid_pct)
Output: pct_removal (0-100)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

class ExtractionANN(nn.Module):
    """
    Feedforward neural network for predicting extraction performance.

    Architecture: Input(3) → [Linear → ReLU → BatchNorm] × 2 → Linear(1)
    """

    def __init__(self, hidden_sizes: Tuple[int, ...] = (64, 32)):
        super().__init__()
        layers: list = []
        in_size = 3  # n_stages, solvent, feed_acid

        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            in_size = h

        layers.append(nn.Linear(in_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ---------------------------------------------------------------------------
# Configuration and results
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    hidden_sizes: Tuple[int, ...] = (64, 32)
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 200
    patience: int = 20
    test_size: float = 0.15
    val_size: float = 0.15
    random_state: int = 42
    device: str = "cpu"


@dataclass
class TrainingResult:
    """Results from model training."""
    model: ExtractionANN
    scaler_X: StandardScaler
    scaler_y: StandardScaler
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    test_r_squared: float = 0.0
    test_mae: float = 0.0
    test_rmse: float = 0.0
    best_epoch: int = 0
    # Split data arrays (unscaled, original units) for visualization
    X_train: Optional[np.ndarray] = None
    X_val: Optional[np.ndarray] = None
    X_test: Optional[np.ndarray] = None
    y_train: Optional[np.ndarray] = None
    y_val: Optional[np.ndarray] = None
    y_test: Optional[np.ndarray] = None
    y_pred_test: Optional[np.ndarray] = None
    n_total: int = 0
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    df: pd.DataFrame,
    config: Optional[TrainingConfig] = None,
    progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
) -> TrainingResult:
    """
    Train the ANN surrogate model.

    Parameters
    ----------
    df : DataFrame with columns [n_stages, solvent_per_stage, feed_acid_pct, pct_removal]
    config : training configuration
    progress_callback : optional callback(epoch, total_epochs, train_loss, val_loss)

    Returns
    -------
    TrainingResult with trained model, scalers, and metrics.
    """
    if config is None:
        config = TrainingConfig()

    device = torch.device(config.device)

    # Prepare data
    feature_cols = ["n_stages", "solvent_per_stage", "feed_acid_pct"]
    target_col = "pct_removal"

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32).reshape(-1, 1)

    # Train/val/test split
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state
    )
    val_frac = config.val_size / (1 - config.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_frac, random_state=config.random_state
    )

    # Scale
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_s = scaler_X.fit_transform(X_train)
    X_val_s = scaler_X.transform(X_val)
    X_test_s = scaler_X.transform(X_test)

    y_train_s = scaler_y.fit_transform(y_train)
    y_val_s = scaler_y.transform(y_val)
    y_test_s = scaler_y.transform(y_test)

    # Tensors
    train_ds = TensorDataset(
        torch.tensor(X_train_s, dtype=torch.float32),
        torch.tensor(y_train_s, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)

    X_val_t = torch.tensor(X_val_s, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val_s, dtype=torch.float32).to(device)

    # Model
    model = ExtractionANN(hidden_sizes=config.hidden_sizes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    # Training loop with early stopping
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(config.epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / n_batches
        train_losses.append(train_loss)

        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = criterion(val_pred, y_val_t).item()
        val_losses.append(val_loss)

        if progress_callback is not None:
            progress_callback(epoch + 1, config.epochs, train_loss, val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # Test metrics
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32).to(device)
    with torch.no_grad():
        y_pred_s = model(X_test_t).cpu().numpy()

    y_pred = scaler_y.inverse_transform(y_pred_s)
    y_actual = y_test

    # R²
    ss_res = np.sum((y_actual - y_pred) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # MAE and RMSE
    mae = float(np.mean(np.abs(y_actual - y_pred)))
    rmse = float(np.sqrt(np.mean((y_actual - y_pred) ** 2)))

    return TrainingResult(
        model=model,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        train_losses=train_losses,
        val_losses=val_losses,
        test_r_squared=r_squared,
        test_mae=mae,
        test_rmse=rmse,
        best_epoch=best_epoch,
        # Store split data (unscaled) for visualization
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        y_pred_test=y_pred,
        n_total=len(X),
        n_train=len(X_train),
        n_val=len(X_val),
        n_test=len(X_test),
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(
    model: ExtractionANN,
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    n_stages: float,
    solvent: float,
    feed_comp: float,
) -> float:
    """
    Predict % removal for given operating conditions.

    Parameters
    ----------
    model : trained ExtractionANN
    scaler_X, scaler_y : fitted scalers
    n_stages : number of extraction stages
    solvent : solvent per stage (kg)
    feed_comp : feed acid composition (wt%)

    Returns
    -------
    Predicted % removal (0-100).
    """
    model.eval()
    X = np.array([[n_stages, solvent, feed_comp]], dtype=np.float32)
    X_scaled = scaler_X.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        y_scaled = model(X_tensor).numpy()

    y = scaler_y.inverse_transform(y_scaled)
    return float(np.clip(y[0, 0], 0.0, 100.0))


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_model(result: TrainingResult, path: str | Path) -> None:
    """Save trained model and scalers to a checkpoint file."""
    path = Path(path)
    checkpoint = {
        "model_state_dict": result.model.state_dict(),
        "scaler_X_mean": result.scaler_X.mean_,
        "scaler_X_scale": result.scaler_X.scale_,
        "scaler_y_mean": result.scaler_y.mean_,
        "scaler_y_scale": result.scaler_y.scale_,
        "test_r_squared": result.test_r_squared,
        "test_mae": result.test_mae,
        "test_rmse": result.test_rmse,
        "train_losses": result.train_losses,
        "val_losses": result.val_losses,
        "best_epoch": result.best_epoch,
    }
    torch.save(checkpoint, path)


def load_model(path: str | Path, hidden_sizes: Tuple[int, ...] = (64, 32)) -> TrainingResult:
    """Load a trained model from checkpoint."""
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    model = ExtractionANN(hidden_sizes=hidden_sizes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler_X = StandardScaler()
    scaler_X.mean_ = checkpoint["scaler_X_mean"]
    scaler_X.scale_ = checkpoint["scaler_X_scale"]
    scaler_X.n_features_in_ = 3

    scaler_y = StandardScaler()
    scaler_y.mean_ = checkpoint["scaler_y_mean"]
    scaler_y.scale_ = checkpoint["scaler_y_scale"]
    scaler_y.n_features_in_ = 1

    return TrainingResult(
        model=model,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        train_losses=checkpoint.get("train_losses", []),
        val_losses=checkpoint.get("val_losses", []),
        test_r_squared=checkpoint.get("test_r_squared", 0.0),
        test_mae=checkpoint.get("test_mae", 0.0),
        test_rmse=checkpoint.get("test_rmse", 0.0),
        best_epoch=checkpoint.get("best_epoch", 0),
    )
