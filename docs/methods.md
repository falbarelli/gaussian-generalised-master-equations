# Conventions and specialized routines

Run the snippets below from the repository root in `julia --project=.`.
Load `ReplicaME.jl` once per Julia session. The name identifies a local module,
not a package to install. For the general solver alone, see [general-odes.md](general-odes.md).


- Quadratures are **interleaved**: `(x₁,p₁,x₂,p₂,…)`, with `[rᵢ,rⱼ]=iΩᵢⱼ`.
  Vacuum covariance is the identity. Replicas concatenate entire systems.
- The Hamiltonian is `rᵀ H r / 2 + hᵀ Ω r`. In particular the linear drive
  contributes **h** to the mean equation, not `Ω*h`.
- Jump coefficients occupy rows: `Jₖ = L[k,:] * r`. Corresponding rows at
  different parameter values must describe the same physical channels.
- Generalized covariances are complex and symmetric under plain transpose,
  not Hermitian conjugation. The propagated scalar obeys `Tr(μ)=exp(-ξ)`.
- TSME fidelities are **root fidelities**. Their environment/joint QFI
  interpretation requires a parameter-independent **pure** initial system
  state. With a mixed initial state a purifying reference is included.
- Replica Bargmann invariants allow mixed initial system states, but the input
  state is parameter independent. Output fields initially are vacuum.
- Bargmann invariants require a periodic replica ring. The lower-level
  coefficient builder supports open chains, whose traces are different objects.

```julia
include("ReplicaME.jl")
using LinearAlgebra

χ, κ, η = 0.45, 1.0, 0.5
model = (
    N=1, tspan=(0.0, 2.0),
    σ0_single=Matrix{Float64}(I, 2, 2), d0_single=zeros(2),
    Hfun=θ -> [θ -χ; -χ θ], hfun=θ -> zeros(2),
    jfun=θ -> sqrt(η*κ/2) .* [1 im],
    Lfun=θ -> sqrt((1-η)*κ/2) .* [1 im],
)

logB = log_bargmann_invariant([0.1, 0.12]; model...)
times, estimates = I_n_timecourse(0.1, [0, 2, 5];
    times=collect(0.0:0.1:2.0), h=0.002,
    reltol=1e-11, abstol=1e-12, model...)
# estimates[:, j] corresponds to the j-th requested order.
```

`I_n` returns the terminal value. `I_n_timecourse` accepts one order or a vector;
the latter shares the calculation of all required `f_m`. Both default to the
mixed derivative formula, with fourth-order Richardson finite differences.
Use `method=:eq20` for the independent derivative-insertion formula (Eq. 20 in
arXiv v1, Eq. 30 in v2). Use `fd_method=:central` to disable extrapolation.
An explicit `cache=Dict()` can reuse invariant solves across calls with nearby
Richardson step sizes; numerical and model inputs are part of the cache key.

For the zero-mean Hamiltonian-only case used in the OPO figure, avoid finite
parameter differences with:

```julia
times, estimates = I_n_hamiltonian_timecourse([5, 10, 20, 25, 30];
    H=model.Hfun(0.1), dH=Matrix{Float64}(I, 2, 2),
    j=model.jfun(0.1), L=model.Lfun(0.1), σ0_single=model.σ0_single,
    tspan=model.tspan, times=collect(0.0:0.1:2.0))
```

The general `solve_gaussian_odes` accepts time-dependent `params(t)`.
`replica_params` and `tsme_params` intentionally precompute constant coefficients;
their model functions depend on the estimated parameter, not on time.

### Deterministic unravelling information

For the single-mode OPO, `solve_unravelling` computes the measurement-record CFI,
the mean QFI of the conditional states, and their sum without sampling trajectories:

```julia
times, Fs, Qc, Qu = solve_unravelling(
    0.1, 0.45, 1.0, 0.5, 0.0, "hom", 0.0, 20.0, 0.005,
)
# Arguments: ω, χ, κ, η, ϕ, measurement, nth, final time, timestep.
# Use "het" for heterodyne detection. Fs is the CFI alone; Qu = Fs + Qc.
```

This uses the deterministic moment closure in
[Zhang et al., Supplement Sec. IV.A, Eqs. S11–S12](https://arxiv.org/html/2511.22248v1#A4.SS1).
In this code's conventions, let `S = ∂ωσ`, `M = B - σ*B`, and
`z = [r; ∂ωr]`, with the derivative taken at a fixed measurement record. Then

```text
F = [A  0; ∂ωA  A + M*B']       G = [M; -S*B] / √2
V = E[z*z']                     dV/dt = F*V + V*F' + G*G'
P = V[3:4, 3:4]                 dFs/dt = 2tr(B' * P * B)
Qc = Qcov(σ, S) + 2tr(σ⁻¹ * P)  Qu = Fs + Qc
```

The conditional covariance and its derivative are deterministic, so the same
second moments also give the mean conditional QFI. `Qc` is the average of the
conditional-state QFIs, not the QFI of the averaged state. This closure applies
to the linear Gaussian model with fixed homodyne/heterodyne settings, including
thermal initial states and inefficient detection. Nonlinear dynamics or
measurement settings chosen from the observed record generally do not admit
this finite closure.

The solver integrates all coupled equations using RK4 and returns the same
time grid and tuple as `simulate_unravelling`, without its `Ntraj` argument.
Decrease `dt` to check integration error. The stochastic API remains available
for trajectory checks; its Euler–Maruyama timestep bias and sampling error
must both be accounted for when comparing it with the deterministic result.
