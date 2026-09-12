# Machine Learning Surrogate Model Module (`mass_transfer/ml/`)

This directory contains the machine learning components of the mass transfer digital twin. It implements a data-driven surrogate model designed to rapidly approximate the results of the rigorous, physics-based crosscurrent extraction solver.

## Purpose

Rigorous mass transfer calculations, especially those involving complex equilibrium models and multiple stages, can be computationally intensive. This is particularly problematic for tasks requiring thousands of evaluations, such as:
*   Real-time interactive user interfaces
*   Large-scale multi-variable optimization
*   High-resolution surface plotting

The modules in this folder address this by generating a large dataset from the physics solver and training a PyTorch Neural Network to predict the outcome. Once trained, the surrogate model provides near-instantaneous predictions.

## Contents

*   [`data_generator.py`](DATA_GENERATOR_README.md): Automates the creation of training datasets using Latin Hypercube Sampling and parallel multiprocessing over the physics solver.
*   [`neural_net.py`](NEURAL_NET_README.md): Defines the PyTorch Artificial Neural Network (ANN) architecture, training loop, evaluation metrics, and serialization routines.
*   [`optimization.py`](OPTIMIZATION_README.md): Leverages the trained surrogate model to perform rapid root-finding (e.g., minimum solvent for a target removal) and generate 2D response surfaces for visualization.

## Workflow

1.  **Generate Data:** Use `data_generator.generate_crosscurrent_dataset` to sample the parameter space (`n_stages`, `solvent`, `feed_composition`) and compute the true `pct_removal`.
2.  **Train Model:** Pass the generated dataset to `neural_net.train_model` to train the neural network and obtain a `TrainingResult` containing the model and scalers.
3.  **Utilize Surrogate:** Use the trained model directly with `neural_net.predict`, or pass it to `optimization.py` functions to find optimal operating conditions or generate rapid visualizations.
