# Data Generator (`data_generator.py`)

This module is responsible for generating datasets to train the machine learning surrogate model. It automates the process of sweeping the physics-based crosscurrent solver over a grid of operating conditions.

## Key Features

*   **Design of Experiments (DoE):** Utilizes **Latin Hypercube Sampling (LHS)** via `scipy.stats.qmc.LatinHypercube` to efficiently sample the 3D parameter space (`n_stages`, `solvent_per_stage`, `feed_acid_pct`), ensuring comprehensive coverage for training the neural network.
*   **Parallel Execution:** Employs `concurrent.futures.ProcessPoolExecutor` to run multiple instances of the crosscurrent solver in parallel, significantly speeding up the dataset generation process.
*   **Configurability:** The `DataGenConfig` dataclass allows easy customization of sampling ranges, the number of samples, and the number of parallel workers.

## Main Components

*   `DataGenConfig`: A dataclass storing configuration parameters like ranges for the number of stages, solvent amount, feed acid percentage, and total sample count.
*   `generate_crosscurrent_dataset(...)`: The primary function. It generates LHS samples, maps them to the specified variable ranges, and dispatches the solving tasks to a multiprocessing pool. It returns a cleaned `pandas.DataFrame` of valid results.
*   `generate_crosscurrent_dataset_serial(...)`: A serial version of the dataset generator, useful for debugging or environments where multiprocessing is not viable.
*   `_solve_single_point(...)`: An internal worker function designed to be picked and executed by the process pool. It loads the equilibrium data, fits the model, runs the crosscurrent solver, and returns the results.
