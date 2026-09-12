# Neural Network Surrogate Model (`neural_net.py`)

This module defines, trains, and manages the PyTorch-based Artificial Neural Network (ANN) that acts as a fast surrogate for the physics-based crosscurrent extraction solver.

## Architecture

The surrogate model is a standard Feedforward Neural Network (`ExtractionANN`):
*   **Inputs (3):** Number of stages (`n_stages`), solvent per stage (`solvent_per_stage`), and feed acid composition (`feed_acid_pct`).
*   **Hidden Layers:** Two hidden layers by default. Each consists of a Linear layer, a ReLU activation function, and 1D Batch Normalization. (Default sizes: 64 and 32 neurons).
*   **Output (1):** Predicted total percentage removal of the solute (`pct_removal`).

## Key Features

*   **PyTorch Implementation:** Leverages PyTorch for model definition, training loops, and tensor operations.
*   **Robust Training Loop:** The `train_model` function handles dataset splitting (train/val/test), standard scaling (using `sklearn.preprocessing.StandardScaler`), batching via `DataLoader`, and early stopping based on validation loss to prevent overfitting.
*   **Comprehensive Metrics:** The `TrainingResult` dataclass captures not just the trained model and scalers, but also loss histories, $R^2$ score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) on the test set.
*   **Serialization:** Provides `save_model` and `load_model` functions to easily persist and reload the trained model's state dictionary and data scalers.

## Main Components

*   `ExtractionANN`: The PyTorch `nn.Module` defining the network architecture.
*   `TrainingConfig`: Configuration dataclass for hyperparameters like learning rate, batch size, epochs, and early stopping patience.
*   `train_model(...)`: The core training pipeline.
*   `predict(...)`: A utility function to make single predictions using a loaded model and its corresponding scalers.
*   `save_model(...)` & `load_model(...)`: Utilities for saving and loading checkpoints.
