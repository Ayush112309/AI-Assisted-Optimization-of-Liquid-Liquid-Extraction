# Countercurrent Extraction Formulation

This note explains how the countercurrent model in [`countercurrent.py`](/Users/17kaushalsingh/GitHub/Courses/Mass-Transfer2-Project/mass_transfer/core/countercurrent.py) is formulated and solved.

The implementation in this project now contains a single countercurrent model:

- multistage countercurrent extraction

The focus here is the process formulation rather than the UI or code structure.

Advanced reflux-assisted countercurrent extraction is intentionally not part of the current implementation. It is treated as future work for the project rather than part of the present solver set.

## Coordinate System

The solver works in solvent-free variables:

- `X = C / (A + C)` for the raffinate phase
- `Y = C / (A + C)` for the extract phase
- `N = B / (A + C)` for the solvent ratio

These variables are useful because they separate the solvent from the carrier-solute subsystem and make the equilibrium data easier to use in stage calculations.

The equilibrium package supplies:

- `Y = f(X)`
- mappings from `X` or `Y` back to full ternary phase compositions
- solvent-ratio information for each phase

So the countercurrent model alternates between:

- an equilibrium step
- an operating-line step

## Physical Arrangement

In countercurrent extraction:

- feed enters at one end of the cascade
- fresh solvent enters at the opposite end
- raffinate and extract move in opposite directions

This creates stronger coupling than in crosscurrent extraction because each stage is influenced by streams coming from both neighboring stages.

## Why the Problem Is Harder Than Crosscurrent

For a true countercurrent cascade, the unknown stage compositions are globally coupled:

- the raffinate leaving one stage becomes the raffinate entering the next
- the extract leaving one stage becomes the extract entering the previous stage

That means the entire stage network is linked in both directions. A direct rigorous solve would normally involve a larger simultaneous nonlinear system.

The implementation here avoids a full simultaneous solve and instead uses a practical stage-stepping construction.

## Solvent-Free Feed Description

The feed solvent-free composition is:

`X_F = C_F / (A_F + C_F)`

The corresponding solvent-free feed flow is:

`F' = F (A_F + C_F) / 100`

where `F'` is the non-solvent portion of the feed.

The code also forms a mixing-point representation for the incoming feed and solvent so the countercurrent operating geometry can be approximated in solvent-free coordinates.

## Formulation Used in the Solver

The implemented method treats the countercurrent problem as a repeated stage construction:

1. choose a trial raffinate composition at the solvent-entry end
2. use equilibrium to determine the associated extract state
3. use an operating relation to step to the next stage
4. continue through the cascade
5. adjust the starting point until the terminal stage matches the feed-end condition

So the main unknown is the leading raffinate composition, and the rest of the stage sequence is generated from it.

## Mixing Point and Difference-Point Idea

The solver computes a solvent-free mixing point from:

- the feed
- the incoming solvent

This point represents the overall mixture on the solvent-free diagram.

From a guessed final raffinate state, the code builds a line connecting that raffinate state to the mixing point. That line is used to locate the terminal extract state and to estimate the operating geometry for the countercurrent cascade.

This is the basis for the stepping logic used in the current implementation.

## Stage-Stepping Logic

For each stage, the code does two conceptual moves:

1. Equilibrium move:
   take the current raffinate state and determine the conjugate extract state
2. Operating move:
   use the approximate countercurrent operating relation to determine the next raffinate state

This is repeated stage by stage across the number of requested ideal stages.

In other words, the countercurrent column is represented as alternating:

- phase-equilibrium mapping
- interstage balance mapping

## Nonlinear Solve

The leading raffinate composition is not known beforehand, so the code defines a residual:

- after stepping through all stages, does the terminal state match the feed-end requirement?

That residual is then driven to zero numerically.

The implementation uses:

- `brentq` when a clean bracket is available
- `fsolve` as fallback

Once the root is found, the full stage sequence is reconstructed.

## What Each Stage Stores

Each countercurrent stage includes:

- raffinate solvent-free composition `X_raff`
- extract solvent-free composition `Y_ext`
- solvent ratios `N_raff` and `N_ext`
- full ternary raffinate composition `(A_raff, C_raff, B_raff)`
- full ternary extract composition `(A_ext, C_ext, B_ext)`
- approximate phase flow rates

This gives the GUI enough information to render:

- stage tables
- X-Y stepping diagrams
- ternary stage paths
- heatmaps

## Modeling Assumptions

The current countercurrent model is an equilibrium-stage model with a simplified operating construction.

Its key assumptions are:

- each stage reaches phase equilibrium
- the fitted equilibrium model is valid over the simulated region
- the operating relation can be represented through a practical stepping construction rather than a fully rigorous simultaneous solve
- no kinetics, hydraulics, holdup, or thermal effects are included

So this is a useful design and visualization model, not a rate-based or full MESH-equation extractor model.

## Outputs

The countercurrent solver returns:

- the total number of stages solved
- the solvent-free feed composition
- the terminal raffinate and extract solvent-free compositions
- the solvent-free feed flow
- stage-by-stage phase compositions and approximate flowrates

These are then used by the GUI for simulation results, comparison plots, and animations.

## Relation to the Implementation

The main routine is:

- `solve_countercurrent(...)`

It performs the full countercurrent calculation, including:

- trial leading raffinate state
- stage stepping
- nonlinear correction of the initial guess
- final reconstruction of stage results

The important engineering point is that the countercurrent solver is built as a solvent-free stage-construction model driven by equilibrium relations and approximate operating geometry.
