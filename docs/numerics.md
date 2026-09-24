# Numerical validation and limitations

Run the regression suite with `julia --project=. test/runtests.jl`.
The independently loaded, time-dependent solver example is checked with
`julia --project=. test/standalone_solver.jl`.

The tests compare against exact coherent-state solutions, a finite-dimensional
spectral QFI expression, and direct truncated-Fock-space master equations.
They also check multimode trace norms, fixed-record filter sensitivities,
pure-state limits, and an analytic replica sum through order 20.
The TUR checks differentiate the stationary cumulant generating function in
high precision to recover the counting current and noise, and compare the
analytic rate `f=2χ²/κ` with the late-time joint TSME-QFI slope for time-rescaled
dynamics at three pump strengths.

For the inefficient OPO settings in the paper, run:

```sh
julia --project=. validation/replica_convergence.jl
julia --project=. validation/tsme_convergence.jl
julia --project=. validation/unravelling_ensemble.jl
julia --project=. validation/replica_shorttime.jl
julia --project=. validation/replica_midrange.jl
julia --project=. validation/replica_order30.jl
```

These scripts print comparisons or write tables under `validation/output/`,
checking derivative formulas, step sizes, tolerances, and ensemble moments.
Numerical stability of a fixed `I_n`
is distinct from convergence to the QFI as `n` increases. Alternating binomial
coefficients amplify ODE and finite-difference errors at high order. Smaller
steps can make this worse; validate steps and tolerances together. `expm1`,
shared invariants, and compensated sums reduce roundoff but do not certify all
parameter regimes. Never infer convergence simply because a curve is smooth.

The original finite-difference validation at `κt=0.1` found roughly 7% variation
for an order-20 rate near `1e-7`; order-30 finite differences can even turn
negative. This motivated the OPO sensitivity backend. Its supported scope is
an undriven, zero-mean initial state with parameter-independent jump operators
and initial covariance. `I_n_hamiltonian_timecourse` takes the Hamiltonian `H`
and its derivative `dH` explicitly. For other parameter dependencies, use the
general `I_n_timecourse` and validate finite-difference steps carefully.

The single-mode trace norm uses a compensated determinant formula to resolve
nearly pure states. The general multimode spectral formula remains sensitive
to roundoff near pure modes; inspect convergence before using tiny parameter
steps in that regime.

Homodyne/heterodyne means use Euler–Maruyama; deterministic covariances use RK4.
Decrease `dt_traj` and increase `Ntraj` to assess timestep and sampling error.
The RNG seed reproduces this implementation, not the ordering of random draws
in the original notebook.

The ensemble validation integrates the covariance of the linear Gaussian
filter and its sensitivity directly. It supplies a deterministic reference
and predicted sampling errors for the trajectory averages.

Further checks cover the multimode sensitivity implementation and the
above-threshold example:

```sh
julia --project=. validation/replica_sensitivity_multimode.jl
julia --project=. validation/above_threshold_physics.jl
julia --project=examples validation/above_threshold_numerics.jl
```

The last command reads the above-threshold production dataset; generate it
first using the [reproduction guide](reproduction.md). It checks a finite time
window, not existence of a stationary state. Extending the replica order in
this regime requires separate convergence checks.
