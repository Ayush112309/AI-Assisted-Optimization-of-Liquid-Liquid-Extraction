# Comparison of Approaches for Liquid-Liquid Extraction (LLE)

This document provides an in-depth comparison between the two implementations found in this repository: the **`mass_transfer/`** package (Modular Digital Twin) and the **`MT_Project/`** directory (Exploratory AI Script).

---

## 1. Mathematical Formulation & Coordinate Systems

### `mass_transfer/` (Solvent-Free Basis)
- **Coordinate System**: Uses the **solvent-free basis** ($X, Y, N$).
  - $X = C / (A + C)$ for raffinate
  - $Y = C / (A + C)$ for extract
  - $N = B / (A + C)$ for solvent ratio
- **Rationale**: This basis separates the solvent from the carrier-solute subsystem, making tie-lines appear as a direct relation between $X$ and $Y$. It simplifies operating-line constructions and allows for a more rigorous representation of systems where solvent mutual solubility is significant.
- **Graphical Method**: Employs the **Janecke Diagram** ($N$ vs. $X/Y$) for difference-point constructions.

### `MT_Project/` (Weight Fraction Basis)
- **Coordinate System**: Uses standard **ternary weight fractions** ($x_C, y_C$).
  - $x_C$ = weight fraction of solute in raffinate
  - $y_C$ = weight fraction of solute in extract
- **Rationale**: Direct use of experimental weight fraction data. This is more "standard" in introductory mass transfer courses but can be mathematically more coupled when solvent solubility varies significantly.
- **Graphical Method**: Relies on 1D distribution curves ($y_C$ vs $x_C$) and 3D response surfaces for performance visualization.

---

## 2. Solver Methodologies

### `mass_transfer/` (Stagewise Stepping & Ponchon-Savarit)
- **Crosscurrent**: Sequential stage-by-stage solution using `fsolve` to satisfy carrier ($A$) and solute ($C$) balances simultaneously at each stage.
- **Countercurrent (Simple)**: Iterative root-finding on the terminal raffinate ($X_1$) using a numerical McCabe-Thiele-like stepping pass.
- **Countercurrent (Reflux)**: A rigorous **Ponchon-Savarit** implementation using enriching and stripping difference points ($\Delta_E, \Delta_S$). It steps through stages by alternating between equilibrium and operating-line intersections on the Janecke diagram.

### `MT_Project/` (Simultaneous Equations)
- **Crosscurrent**: Similar sequential stage-by-stage approach.
- **Countercurrent**: Solves the entire column **simultaneously**.
  - It sets up a system of $2N$ non-linear equations (1 solute balance + 1 equilibrium relation per stage).
  - The entire vector of stage compositions $[x_1, \dots, x_N, y_1, \dots, y_N]$ is solved in one pass using `fsolve`.
  - **Advantage**: Faster for fixed-stage problems as it avoids inner-loop iterations.

---

## 3. Equilibrium Modeling

### `mass_transfer/` (Fitted Surrogates)
- **Mapping**: Fits high-order polynomials and cubic splines to both the phase envelope boundaries ($C$ vs $A$, $C$ vs $B$) and the distribution curve ($Y$ vs $X$).
- **Robustness**: Includes logic to choose between splines and polynomials based on $R^2$. It provides a "Numerical Equilibrium Engine" that can reconstruct full ternary points from any solvent-free coordinate.

### `MT_Project/` (1D Interpolation)
- **Mapping**: Uses `scipy.interpolate.interp1d` (cubic) to create a simple mapping of $y_C = f(x_C)$ based on provided tie-line data.
- **Simplicity**: Focuses strictly on the distribution of the solute without modeling the full phase envelope shape explicitly in the solvers.

---

## 4. Machine Learning & AI Integration

### `mass_transfer/` (Surrogate Performance Modeling)
- **Framework**: **PyTorch**.
- **Philosophy**: Trains an Artificial Neural Network (ANN) to act as a **performance surrogate**.
- **Inputs/Outputs**: Input $[n\_stages, solvent, feed\_acid]$ $\rightarrow$ Output $[\% removal]$.
- **Features**: Includes optimization routines to find the best operating point using the surrogate, and generates 3D response surfaces.

### `MT_Project/` (Sequential/Temporal Modeling)
- **Framework**: **TensorFlow/Keras**.
- **Philosophy**: Treats the stages as a **sequence** (Time-Series analogy).
- **Models**: Explores **RNN, LSTM, and GRU** architectures.
  - Recognizes that Stage $N$ is physically dependent on Stage $N-1$.
  - Uses `TimeDistributed` layers to predict compositions at *every* stage rather than just the final outcome.
- **Benchmarking**: Explicitly compares traditional ANNs against sequential models (RNN/LSTM) to show that sequential models better capture the physics of a staged column.

---

## 5. Software Engineering & Visualization

| Feature | `mass_transfer/` | `MT_Project/` |
| :--- | :--- | :--- |
| **Project Structure** | Modular package (`core`, `gui`, `ml`, `viz`). | Script-based (`gui.py`, `mt_project.py`). |
| **GUI Framework** | PyQt6 (Multi-tab, Animated). | PyQt5 (Setup Wizard, Dashboard). |
| **Visualizations** | Animated GIFs of stepping, Seaborn heatmaps. | 3D Matplotlib surfaces, Parity plots. |
| **Data Handling** | JSON-based equilibrium configs. | Excel-based results logging. |
| **Automation** | Latin Hypercube Sampling for ML data. | Nested loops/Parametric sweeps for ML data. |

---

## 6. Development History and Evolution of Approaches

Based on the version control history, several key decisions, mistakes, and creative pivots have shaped the project's mass transfer modeling and mathematical solving foundations:

### 6.1 Coordinate System Correction (Mistake & Fix)
- **The Issue:** Initially, the right-angle ternary diagrams were plotted with the x-axis representing **wt% A (carrier)**.
- **The Fix:** This was recognized as non-standard for this specific solvent-extraction context and corrected so the x-axis represents **wt% B (solvent)**. This adjustment aligns the visual intuition with classical chemical engineering conventions where the carrier is computed by difference, providing a much clearer view of the two-phase solvent envelope.

### 6.2 Equilibrium Interpolation Rigor (Creativity)
- **Evolution:** The project started with simple polynomial fitting for phase envelopes. 
- **Refinement:** The equilibrium model was significantly refactored to introduce **high-fidelity equilibrium interpolation**. The system was upgraded to intelligently choose between high-order polynomials and cubic splines (based on $R^2$ scores) for modeling the phase boundaries and the distribution curve. This dynamic selection made the "Numerical Equilibrium Engine" much more robust and capable of handling non-ideal, highly skewed tie-line data without suffering from Runge's phenomenon (oscillation artifacts).

### 6.3 Simplification of Countercurrent Solvers (Approach Update)
- **Initial Ambition:** The original implementation was highly complex (over 1200 lines of code) and attempted to rigorously solve countercurrent extraction **with extract reflux** using numerical difference-point (Ponchon-Savarit) constructions on a solvent-free basis.
- **Strategic Pivot:** The complex reflux workflow was ultimately removed. The team recognized that a full simultaneous mathematical solve for reflux-assisted extraction was overly brittle and added unnecessary complexity to the core system. The approach was streamlined to focus purely on a highly stable, standard **multistage countercurrent extraction** model. (Reflux was correctly deferred to "Future Work").

### 6.4 Mathematical Solvers over Graphical Methods
- **Core Philosophy:** Throughout the project's history, a strict rule was enforced to avoid relying on graphical stage-stepping for actual calculations. 
- **Execution:** All solvers (crosscurrent and countercurrent) were rigorously implemented as systems of non-linear algebraic equations solved via `scipy.optimize.fsolve`. Graphical constructions (Janecke diagrams, ternary plots) were restricted strictly to the `viz/` module for validation, verification, and pedagogical display, rather than being the source of truth for the digital twin.

---

## Conclusion

The **`mass_transfer/`** approach is a **production-ready Digital Twin** framework. It prioritizes rigorous thermodynamics (solvent-free basis), modular software design, and high-fidelity visualizations suitable for an engineering simulator.

The **`MT_Project/`** approach is a **research-oriented AI exploration**. It prioritizes the investigation of sequential neural networks (RNNs/LSTMs) to model the physical "flow" of extraction stages, using simultaneous equation solving for rapid data generation.