# Crosscurrent Extraction Formulation

This note explains how the crosscurrent system in [`crosscurrent.py`](/Users/17kaushalsingh/GitHub/Courses/Mass-Transfer2-Project/mass_transfer/core/crosscurrent.py) is formulated and solved. The emphasis here is the process model, not the GUI or code structure.

## System Definition

The process is an `N`-stage crosscurrent liquid-liquid extraction cascade. Each stage receives:

- The raffinate leaving the previous stage
- A fresh charge of pure solvent

Each stage produces:

- A new raffinate stream
- A stage extract stream

The pattern is:

`Feed -> Stage 1 -> Stage 2 -> ... -> Stage N -> Final Raffinate`

with fresh solvent added independently to every stage.

## Component Convention

The equilibrium model uses a ternary system:

- `A`: carrier
- `B`: solvent
- `C`: solute

The code works in two coordinate systems:

1. Ternary weight percent:
   `A + B + C = 100`
2. Solvent-free coordinates:
   `X = C / (A + C)` for the raffinate phase
   `Y = C / (A + C)` for the extract phase
   `N = B / (A + C)` as the solvent ratio

The equilibrium package provides:

- A raffinate-side phase relation
- An extract-side phase relation
- A distribution curve `Y = f(X)`
- Solvent-ratio correlations `N_raff(X)` and `N_ext(Y)`

For the crosscurrent stage solver, the key closure is:

- Choose a raffinate composition `X`
- Use equilibrium to get the corresponding raffinate point
- Use `Y = f(X)` to get the conjugate extract composition on the same tie-line

So one stage can be represented by a single unknown raffinate composition and the associated phase split.

## Single-Stage Model

For stage `i`, let the incoming raffinate be:

- total flow `R_(i-1)`
- composition `(x_A,in, x_B,in, x_C,in)` on a total-mass basis

Fresh solvent added to the stage is:

- `S`
- pure component `B`

The outgoing streams are:

- raffinate: `R_i`
- extract: `E_i`

The total stage inlet is:

`M_in = R_(i-1) + S`

The component masses entering are:

- `m_A,in = R_(i-1) x_A,in`
- `m_C,in = R_(i-1) x_C,in`
- `m_B,in = R_(i-1) x_B,in + S`

## Unknowns Chosen

The implementation solves each stage with two unknowns:

- `X_raff`
- `R_out`

This is a compact choice because once `X_raff` is known:

1. The raffinate endpoint on the binodal curve is fixed.
2. The corresponding extract composition is fixed by the equilibrium distribution relation.
3. The extract flow follows from the total mass balance:
   `E_out = M_in - R_out`

So the phase compositions come from equilibrium, while the phase split comes from the balances.

## Equilibrium Closure

Given a trial `X_raff`, the solver reconstructs the equilibrium raffinate composition:

- `(A_R, C_R, B_R)`

Then it computes the associated extract composition using:

- `Y_ext = f(X_raff)`
- `(A_E, C_E, B_E)` from the extract-side equilibrium relation

This enforces that both leaving phases lie on the same equilibrium tie-line.

## Mass-Balance Equations

The stage is solved with two independent component balances, written here for `A` and `C`:

- `m_A,in = R_out w_A,R + E_out w_A,E`
- `m_C,in = R_out w_C,R + E_out w_C,E`

where the `w` values are phase weight fractions reconstructed from equilibrium.

The `B` balance is not solved explicitly because:

- total balance gives `E_out = M_in - R_out`
- if `A` and `C` are satisfied, the `B` balance closes automatically up to numerical error

This reduces the stage model to a `2 x 2` nonlinear system.

## Numerical Solution Strategy

The stage equations are nonlinear because:

- the phase compositions depend nonlinearly on `X_raff`
- the extract composition depends on the fitted equilibrium curve `Y = f(X)`
- the mass balances couple composition and flow split

The solver therefore uses a nonlinear root finder:

- `fsolve`

An initial guess is constructed from the inlet composition:

- the trial `X` is set below the feed `X`, because extraction should reduce solute in the raffinate
- the trial raffinate flow is taken as a fraction of the entering raffinate flow

After convergence, the full stage result is reconstructed:

- raffinate and extract compositions
- total stream flows
- solvent-free coordinates `X`, `Y`, `N`

## Cascade Construction

The multistage crosscurrent cascade is solved sequentially.

For stage `1`:

- inlet raffinate is the process feed

For stage `i > 1`:

- inlet raffinate is the raffinate leaving stage `i - 1`

At each stage:

1. Solve the single-stage nonlinear model.
2. Compute how much solute was removed in that stage.
3. Pass the raffinate outlet forward as the inlet to the next stage.
4. Accumulate each extract stream into a mixed extract total.

This is a direct stage-by-stage propagation problem, not a simultaneous column solve.

## Why the Crosscurrent Problem Is Simpler

Crosscurrent extraction is easier to formulate than countercurrent extraction because:

- each stage receives known fresh solvent
- each stage only depends on the previous raffinate
- there is no backward coupling from downstream extract streams
- each stage can be solved independently once the previous stage is known

So the overall problem decomposes naturally into repeated equilibrium-stage calculations.

## Performance Quantities Reported

The solver reports, for each stage:

- outlet raffinate and extract compositions
- stage flows
- solvent-free compositions `X_raff`, `Y_ext`
- solvent ratios `N_raff`, `N_ext`
- solute removed in that stage
- stagewise and cumulative percent removal

It also reports for the whole cascade:

- final raffinate composition
- mixed extract composition
- overall percent solute removal

## Modeling Assumptions

The current formulation assumes:

- each stage reaches equilibrium
- each solvent addition is pure `B`
- the equilibrium model fitted from tie-line data is valid over the operating region
- no kinetics, holdup, entrainment, or hydraulic limitations are modeled
- stage temperature and pressure effects are already embedded in the tie-line data

So this is an equilibrium-stage design model, not a rate-based contactor model.

## Relation to the Implementation

The formulation is implemented in two layers:

- `_solve_single_stage(...)`
  Solves one equilibrium stage from balances plus equilibrium closure
- `solve_crosscurrent(...)`
  Repeats the stage calculation through the cascade and aggregates results

The important point is that the implementation follows the formulation above: each stage is a nonlinear flash-like split constrained by equilibrium and two component balances.
