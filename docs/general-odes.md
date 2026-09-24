# General Gaussian moment solver

The general solver accepts complex and time-dependent coefficients directly.
It is independent of the OPO model and replica construction.

## Loading

From `julia --project=.` at the repository root:

```julia
include("src/GaussianMoments.jl")
using .GaussianMoments
```

Alternatively, `include("GaussianGMEs.jl")` loads all routines. Use one entry point
per session. Run the complete example with
`julia --project=. examples/general_odes.jl`.

## Equations and coefficients

Writing the generalized operator as `μ = exp(-ξ) ν`, where `Tr(ν) = 1`, the
solver integrates

```text
dσ/dt = σ Q σ + A σ + σ Aᵀ + D
dd/dt = (A + σ Q) d - σ Ω a + i c
dξ/dt = -dᵀ Q d - tr(σ Q)/2 - tr(W Ω)/2 - 2 aᵀ Ω d.
```

Transposes are ordinary transposes, not adjoints. A complex covariance is
symmetric, not necessarily Hermitian. For `N` modes, `params(t)` must return:

| Named-tuple field | Size | Role |
| --- | --- | --- |
| `Q` | `2N × 2N` | Quadratic term in the Riccati equation |
| `A` | `2N × 2N` | Linear drift |
| `D` | `2N × 2N` | Additive covariance term |
| `a`, `c` | vectors of length `2N` | Linear terms |
| `W` | `2N × 2N` | Scalar normalization contribution |
| `Ω` | `2N × 2N`, optional | Symplectic matrix; supplied automatically by default |

`A` here is the drift matrix, not the manuscript's operator coefficient `𝔸`.
Build these coefficients from the chosen GME before calling the solver. Zero
terms must still be supplied as arrays of the appropriate dimensions. The
solver does not impose physical-state positivity on generalized operators.

The default ordering is `(x₁,p₁,…,xₙ,pₙ)`, with `[rᵢ,rⱼ] = iΩᵢⱼ` and
`Ω = diag([0 1; -1 0], …)`. Vacuum covariance is the identity.

## Inputs and output

```julia
sol = solve_gaussian_odes(;
    N, tspan=(0.0, T), σ0, d0, params,
    ξ0=0.0, reltol=1e-9, abstol=1e-9, saveat=0.1,
)
```

- `σ0` is a `2N × 2N` covariance and `d0` a vector of length `2N`.
- `ξ0` is the initial negative log trace; zero gives unit initial trace.
- `params(t)` is called at integration times, not just output times. For constant
  coefficients use `params = t -> coefficients`.
- `saveat` can be a vector of times or a sampling interval. If omitted, the
  integrator chooses saved steps. `reltol` and `abstol` control integration error.
- The default integrator is `Tsit5()`. Supply `solver=...` to use another
  compatible SciML solver after adding its dependency to your environment.

The result is a SciML ODE solution. `sol.t` contains saved times. Retrieve the
moments at saved index `i` with

```julia
σ, d, ξ = get_σ_d_ξ(sol, N, i)
trace_μ = exp(-ξ)
```

The layout in `sol.u[i]` is `[vec(σ); d; ξ]` in column-major order and
`ComplexF64` arithmetic. The returned `σ` and `d` are views into that state;
copy them before modifying them. They are moments of the normalized operator
`ν`; the trace factor belongs to `μ`. Integrator failure raises an error.

For nonzero `Q`, the moment equations are nonlinear. For generalized operators,
the normalized parametrization can become singular when the trace vanishes.
Check convergence for the particular model and time window.

## Reuse from another project

Include the source using an absolute path, or a path relative to your script
using `@__DIR__`. Run your script with
`julia --project=/path/to/gaussian-generalised-master-equations your_script.jl`
to use this repository's dependencies.

If your script has its own Julia environment, add `OrdinaryDiffEqTsit5` and
`SciMLBase` there, then include `src/GaussianMoments.jl` from the checkout.
No plotting or replica dependencies are needed. Load the file once per Julia
module/session.
