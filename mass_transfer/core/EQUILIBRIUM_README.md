# Equilibrium Model Formulation

This note explains in detail what happens inside [`equilibrium.py`](/Users/17kaushalsingh/GitHub/Courses/Mass-Transfer2-Project/mass_transfer/core/equilibrium.py), why the data is transformed the way it is, what mathematical objects are fitted, and how the resulting equilibrium model is used by the extraction solvers.

The file is not just a data loader. It converts experimental tie-line data into a reusable equilibrium representation that can answer questions like:

- Given a raffinate composition, what is the corresponding extract composition at equilibrium?
- Given a solvent-free composition `X`, where is that point on the ternary diagram?
- What solvent ratio `N` corresponds to a given raffinate or extract state?
- What curves should be plotted as the phase envelope and distribution relation?

## 1. What Problem This File Solves

Liquid-liquid extraction data is usually available as tie-lines on a ternary diagram.

Each tie-line gives two equilibrium endpoints:

- a raffinate-phase composition
- an extract-phase composition

For simulation, plotting, and stage calculations, raw tie-line tables are inconvenient because the solver repeatedly needs smooth answers between measured points. The purpose of `equilibrium.py` is therefore:

1. Read the tabulated equilibrium data.
2. convert it into solvent-free variables that are natural for extraction calculations.
3. Fit smooth mathematical relations to the measured points.
4. Expose those fitted relations through a structured `EquilibriumModel`.

So the file acts as the bridge between experimental equilibrium data and process-calculation routines.

## 2. Component Convention

The file uses a generic ternary notation:

- `A`: carrier
- `B`: solvent
- `C`: solute

This abstraction allows the same code to work for different chemical systems, provided the data file identifies which physical component plays which role.

## 3. Raw Data Structure

The expected JSON contains two arrays:

- `phase_1`
- `phase_2`

Each item at index `i` in `phase_1` is paired with the item at index `i` in `phase_2`. That pair is one tie-line.

Within the code:

- `phase_1` is interpreted as the raffinate-side endpoint
- `phase_2` is interpreted as the extract-side endpoint

This means the file assumes the phase pairing is already known and ordered correctly.

## 4. Why Component-Key Detection Exists

The code supports multiple naming conventions for the JSON keys.

It explicitly recognizes:

- `cottonseed_oil`, `oleic_acid`, `propane`
- `water`, `acetic_acid`, `isopropyl_ether`
- generic `A`, `C`, `B`

This matters because the equations are role-based, not name-based. The extraction solvers do not care whether the carrier is water or cottonseed oil. They care only about which variable is:

- the carrier `A`
- the solute `C`
- the solvent `B`

Once those roles are mapped, every downstream equation can be written in a system-independent form.

## 5. Conversion From Ternary Data to Solvent-Free Variables

This is the most important conceptual step in the file.

### Raw Ternary Representation

Each measured endpoint is stored on a weight-percent basis:

- `A`
- `B`
- `C`

with:

`A + B + C = 100`

This is the standard ternary-plot representation.

### Solvent-Free Representation

The code then computes:

- `X = C / (A + C)` for raffinate endpoints
- `Y = C / (A + C)` for extract endpoints
- `N = B / (A + C)` for both phases

Specifically:

- `X` is computed from raffinate points
- `Y` is computed from extract points
- `N_raff` is computed from raffinate points
- `N_ext` is computed from extract points

### Why `A + C` Is Used

The denominator `A + C` removes the solvent from the composition basis. This is useful because the most important extraction question is often:

- how much solute is present relative to the non-solvent material?

That is exactly what `X` and `Y` measure.

Similarly, `N = B / (A + C)` measures how much solvent is present relative to the solvent-free material. This makes operating-line and stage calculations more natural than working directly in raw ternary percentages.

### Physical Meaning

- `X` is the solute concentration in the raffinate on a solvent-free basis.
- `Y` is the solute concentration in the extract on a solvent-free basis.
- `N` is the solvent loading relative to the non-solvent mass.

These variables are the natural coordinates for the stage models in `crosscurrent.py` and `countercurrent.py`.

## 6. What `TieLineData` Represents

The `TieLineData` dataclass stores both forms of the same equilibrium information:

- raw ternary endpoints
- derived solvent-free variables

This is important because different parts of the application need different views of the same data:

- the GUI and ternary plots need ternary coordinates
- the solvers need `X`, `Y`, and `N`

So `TieLineData` is the shared data package that keeps both representations together.

## 7. Why the File Fits Curves Instead of Interpolating Tie-Lines Directly

Experimental tie-line tables are discrete. Process solvers need a continuous model.

If the user asks for a composition between two measured tie-lines, the application still needs an answer. Rather than treating the data as isolated points, `equilibrium.py` fits smooth functions that approximate the measured equilibrium structure.

This turns the equilibrium data into callable relationships.

That is why the final `EquilibriumModel` stores functions, not just arrays.

## 8. The Main Fitted Relationships

The file fits several different relationships because the application needs equilibrium information in multiple coordinate systems.

### 8.1 Raffinate Phase Boundary on the Ternary Diagram

The file fits:

`C_raff = f(A_raff)`

This is the raffinate branch of the binodal curve expressed as solute content versus carrier content.

Why this fit is needed:

- if the solver knows a solvent-free raffinate composition `X`, it later needs to reconstruct the actual ternary point `(A, C, B)`
- that reconstruction needs a relationship between `A` and `C` on the raffinate boundary

So this curve is part of the map:

`X -> (A, C, B)`

for the raffinate phase.

### 8.2 Extract Phase Boundary on the Ternary Diagram

The file fits:

`C_ext = f(A_ext)`

This is the extract branch of the binodal curve.

Why this fit is needed:

- once the distribution curve gives an extract solvent-free composition `Y`, the code must reconstruct the actual ternary extract point

So this curve supports the map:

`Y -> (A, C, B)`

for the extract phase.

### 8.3 Distribution Curve

The file fits:

`Y = f(X)`

This is the central equilibrium relation for stage calculations.

It tells us:

- if the raffinate on a stage has solvent-free composition `X`
- then the equilibrium extract has solvent-free composition `Y`

This is the liquid-liquid extraction analog of the key equilibrium curve used in stagewise design methods.

### 8.4 Solvent-Ratio Curves

The file fits:

- `N_raff = f(X)`
- `N_ext = f(Y)`

These are needed because the stage solvers often work with solvent-free compositions and later need to know how much solvent is associated with the phase at that composition.

These curves are especially important in the countercurrent solver, where solvent-free operating geometry and stage stepping depend on phase solvent loading.

### 8.5 Conjugate-Line Approximation

The file also fits:

`X_ext_from_X_raff = f(X_raff)`

In practice this is fitted using the same `X` and `Y` data pairs. It is effectively another form of the distribution mapping.

This function represents the conjugate equilibrium relation between raffinate-side and extract-side solvent-free compositions.

## 9. Polynomial Fits Versus Cubic Splines

Not every equilibrium relation is fitted the same way.

### Polynomial Fits

The helper `_fit_poly(...)` uses `numpy.polyfit` and returns:

- the polynomial function
- the coefficient of determination `R^2`
- the polynomial coefficients

Polynomials are used where:

- the data behaves smoothly over the range
- a compact global representation is acceptable

This is used for:

- `C_raff_from_A`
- `Y_from_X`
- `N_raff_from_X`
- `N_ext_from_Y`
- conjugate-line fit
- `C_raff_from_B`

### Cubic Splines

The helper `_fit_cubic_spline(...)` is used where the data range is narrow or where a polynomial may distort the shape.

The extract phase boundary often occupies a very narrow range in `A`, so a cubic spline can preserve the local geometry better than a low-order polynomial.

That is why the code compares:

- a spline fit
- a polynomial fit

for the extract-side boundary and chooses the one with the better `R^2`.

The same logic is used for the extract-side `C(B)` representation.

### Why the Selection Is Adaptive

This is a pragmatic modeling choice. The file does not assume that one fit type is always best. It evaluates both on the available data and keeps the one that reproduces the measured points more accurately.

## 10. Why There Are Both `C(A)` and `C(B)` Fits

This is easy to miss but important.

The file fits the phase envelope in two coordinate forms:

- `C = f(A)`
- `C = f(B)`

The first form is mainly used internally for reconstruction calculations.

The second form is useful for plotting the ternary boundary correctly when the x-axis is expressed through the solvent coordinate or when a different triangle orientation is needed.

So the model stores both descriptions of the same phase boundary for graphical and numerical convenience.

## 11. What `fit_equilibrium_model(...)` Actually Builds

This is the main model-construction routine.

Starting from `TieLineData`, it:

1. Fits the raffinate boundary.
2. Fits the extract boundary.
3. Fits the distribution curve `Y(X)`.
4. Fits solvent-ratio curves.
5. Fits the conjugate relation.
6. Fits `C(B)` versions of the phase boundaries.
7. estimates the valid interpolation ranges.
8. estimates the plait point.
9. packages everything into an `EquilibriumModel`.

The result is not just a dataset. It is a compact numerical equilibrium engine.

## 12. What Is Stored in `EquilibriumModel`

`EquilibriumModel` contains:

- the original processed tie-line data
- callable phase-boundary relations
- callable distribution relation
- callable solvent-ratio relations
- diagnostics such as `R^2`
- fitted polynomial coefficients where relevant
- estimated plait point
- valid `X` and `Y` ranges

This means the rest of the project can use one object to answer all equilibrium-related questions.

## 13. Why the Fit Functions Are Wrapped

`numpy.poly1d` and spline objects can return NumPy scalar types or arrays depending on how they are called.

The helper `_wrap(...)` standardizes them into scalar-returning functions for single-value use. This matters because:

- the solvers repeatedly call these functions inside root finders
- plotting and stage calculations are easier when the API always behaves like `float -> float`

So the wrapping step is mostly about keeping the model numerically consistent and simple to use.

## 14. Interpolation Ranges

The file stores:

- `X_range = [min(X), max(X)]`
- `Y_range = [min(Y), max(Y)]`

These are not arbitrary metadata. They define the range over which the fitted relations are supported by data.

This matters because extrapolating equilibrium curves beyond the measured tie-line region can easily become nonphysical.

The inverse relation `X_from_Y(...)` clamps `Y` into the valid fitted range before solving for `X`.

## 15. Plait Point Estimation

The plait point is where the raffinate and extract phases become identical, so the tie-line length shrinks to zero.

The function `_estimate_plait_point(...)` approximates this by:

1. evaluating the fitted raffinate and extract phase-boundary curves over a range of `A`
2. computing the difference between them
3. looking for a sign change that indicates an intersection

If such an intersection is found, the code estimates:

- `A_plait`
- `C_plait`
- `B_plait = 100 - A_plait - C_plait`

This is an approximate graphical estimate, not a rigorous thermodynamic calculation. Its purpose is mainly diagnostic and visual.

## 16. Inverse Use of the Distribution Curve

The model stores `Y_from_X` directly, but stage calculations sometimes need the inverse question:

- given `Y`, what `X` is in equilibrium with it?

That is what `X_from_Y(...)` does.

Since the model does not store an exact analytical inverse, it computes `X` numerically by solving:

`Y_from_X(X) - Y_target = 0`

using `fsolve`.

This is why the quality and monotonicity of the fitted `Y(X)` relation matter so much. If the curve is strongly non-monotonic, inversion becomes less straightforward and may depend on the starting guess.

## 17. Reconstructing Full Ternary Points From `X` or `Y`

The equilibrium model is not useful unless it can move back from solvent-free variables to actual ternary compositions.

That is what the convenience functions do.

### 17.1 `get_raffinate_point_from_X(...)`

This function solves:

- the raffinate phase-boundary equation `C = C_raff(A)`
- the solvent-free definition `X = C / (A + C)`

Together, these define one unknown `A`. Once `A` is found:

- `C` comes from the fitted boundary
- `B = 100 - A - C`

So this function maps:

`X -> (A, C, B)` on the raffinate branch

### 17.2 `get_extract_point_from_Y(...)`

This does the same thing on the extract branch:

- solve `C = C_ext(A)`
- enforce `Y = C / (A + C)`

and then recover:

- `B = 100 - A - C`

So this is the map:

`Y -> (A, C, B)` on the extract branch

### 17.3 Why Root Finding Is Needed

These reconstructions are implicit, not explicit.

The file knows:

- `C` as a function of `A`
- `X` or `Y` as a ratio involving both `A` and `C`

So `A` must be found numerically from the nonlinear equation created by combining those two relations.

That is why both helper functions use `fsolve`.

## 18. `get_equilibrium_extract_from_raffinate(...)`

This helper is a compact equilibrium workflow:

1. Start from a raffinate ternary point `(A_raff, C_raff)`.
2. Convert it to solvent-free `X`.
3. Use the distribution curve to get `Y`.
4. Reconstruct the extract ternary point from `Y`.

This function expresses the whole purpose of the model in one chain:

`raffinate ternary point -> solvent-free coordinate -> equilibrium relation -> extract ternary point`

That is exactly the mapping the stage solvers need.

## 19. Utility Functions `get_X_from_ternary(...)` and `get_N_from_ternary(...)`

These are small functions, but they are conceptually central because they encode the solvent-free basis:

- `X = C / (A + C)`
- `N = B / (A + C)`

They are used throughout the project whenever a ternary point must be translated into the coordinates used by the operating models.

## 20. How the Solvers Depend on This File

### Crosscurrent Solver

The crosscurrent model uses the equilibrium package to:

- reconstruct the raffinate point for a trial `X`
- compute `Y = f(X)`
- reconstruct the extract ternary point associated with that `Y`

So each stage solve is built directly on the equilibrium maps created here.

### Countercurrent Solver

The countercurrent models use:

- `Y_from_X`
- `X_from_Y`
- `N_raff_from_X`
- `N_ext_from_Y`
- ternary reconstruction helpers

So the countercurrent routines depend not only on the basic equilibrium relation but also on the solvent-ratio curves and inverse mapping support.

## 21. What Is Approximate in This File

The equilibrium model is empirical and fitted. That means it is not a first-principles thermodynamic model.

Its approximations include:

- finite measured tie-line data represented by smooth regressions
- polynomial or spline fits replacing exact thermodynamic relations
- approximate plait-point detection from curve crossing
- numerical inversion of fitted curves

This is completely reasonable for a course project or engineering design tool, but it should be understood clearly:

- the quality of all downstream simulations depends on the quality of the tie-line data and these fitted relationships

## 22. Why This File Is the Core of the Project

Everything else in the project assumes the equilibrium package can do three jobs reliably:

1. represent the two-phase region on the ternary diagram
2. provide the equilibrium mapping between raffinate and extract compositions
3. translate between ternary and solvent-free coordinates

That is why `equilibrium.py` is foundational. It defines the mathematical language used by the rest of the extraction models.

## 23. Relation to the Implementation

The main pieces of the implementation correspond to the formulation as follows:

- `load_tie_line_data(...)`
  reads the JSON tie-lines and converts them into `X`, `Y`, and `N`
- `_fit_poly(...)` and `_fit_cubic_spline(...)`
  build smooth surrogate relations for equilibrium data
- `fit_equilibrium_model(...)`
  assembles all fitted relations into one reusable model object
- `_estimate_plait_point(...)`
  gives a graphical estimate of where the two branches meet
- `X_from_Y(...)`
  numerically inverts the distribution relation
- `get_raffinate_point_from_X(...)` and `get_extract_point_from_Y(...)`
  convert solvent-free variables back to actual ternary compositions
- `get_equilibrium_extract_from_raffinate(...)`
  performs a full equilibrium mapping from one phase to the other

The key engineering takeaway is simple:

`equilibrium.py` converts discrete tie-line measurements into a continuous equilibrium model that the rest of the application can query as if it were a thermodynamic property package.
