# Gaussian generalised master equations

Julia code accompanying **Efficient evaluation of fundamental sensitivity limits
and full counting statistics for continuously monitored Gaussian quantum systems**,
by Francesco Albarelli and Marco G. Genoni.

Paper: [arXiv:2602.23304](https://arxiv.org/abs/2602.23304).

The code integrates general Gaussian moment equations and applies them to
two-sided master equations, replica master equations, and homodyne/heterodyne
monitoring of an optical parametric oscillator (OPO).

This is a collection of reusable source files and paper-reproduction scripts.
Clone the repository, install its dependencies, and load the functions with
`include`. No package registration or `Pkg.develop` is needed.

## Getting started

The reference environments were generated with **Julia 1.13.0**. From a terminal:

```sh
git clone https://github.com/falbarelli/gaussian-generalised-master-equations.git
cd gaussian-generalised-master-equations
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. examples/general_odes.jl
```

The last command solves a damped mode with a time-dependent frequency and prints
the final covariance, first moments, and trace. It needs neither Python nor LaTeX.
The first run includes dependency installation and Julia compilation.

## Use the general ODE solver

Start Julia with `julia --project=.` in the repository root:

```julia
include("src/GaussianMoments.jl")
using .GaussianMoments
using LinearAlgebra

identity2 = Matrix{Float64}(I, 2, 2)
params(t) = (
    Q=zeros(2, 2), A=-0.5identity2, D=identity2,
    a=zeros(2), c=zeros(2), W=zeros(2, 2),
)
sol = solve_gaussian_odes(;
    N=1, tspan=(0.0, 5.0), σ0=2identity2, d0=[1.0, 0.0],
    params, saveat=0.1,
)
σ, d, ξ = get_σ_d_ξ(sol, 1, length(sol.t))
trace = exp(-ξ)
```

`params(t)` can return time-dependent complex coefficients. The covariance uses
the convention vacuum = identity, with quadratures ordered `(x₁,p₁,x₂,p₂,…)`.
See [the general solver guide](docs/general-odes.md) for the equations, input
dimensions, output layout, and use from another project.

To load the general solver together with the specialized QFI routines, use
`include("GaussianGMEs.jl")` instead. `GaussianGMEs` is a local module used to
organize the source files. See [conventions and specialized routines](docs/methods.md).

An optional short check compares the general solver with the analytic solution
of a damped mode with time-dependent frequency:

```sh
julia --project=. test/standalone_solver.jl
```

## Reproduce the figures

Install the calculation and plotting dependencies:

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
```

For a quick calculation and plot:

```sh
julia --project=examples examples/parametric_oscillator.jl --smoke
julia --project=examples examples/parametric_oscillator_plot.jl --smoke
```

Outputs are written to `examples/output/smoke/`. For the complete calculations,
the figure-to-script map, saved reference data, and optional LaTeX typesetting,
see **[Reproducing the paper figures](docs/reproduction.md)**.

The pipelines cover all three manuscript figures: perfect-detection QFI, the
analytic TUR comparison, and the two-panel inefficient-detection replica
comparison. An above-threshold example is also included. Reference outputs
come from the documented code and can differ from earlier manuscript drafts.

## Repository layout

| Location | Contents |
| --- | --- |
| `src/GaussianMoments.jl` | Standalone general Gaussian ODE solver and trace norm |
| `GaussianGMEs.jl` | Include entry point for all scientific routines |
| `src/` | Two-sided and replica coefficients, fidelities, QFI, unravellings |
| `examples/general_odes.jl` | Minimal time-dependent solver example |
| `examples/` | OPO calculation and plotting scripts; separate dependency environment |
| `data/reference/` | Saved reference datasets, figures, and provenance |
| `test/standalone_solver.jl` | Optional analytic check of the general solver |
| `docs/` | Reproduction instructions and conventions for using the routines |

The two `Project.toml` files describe dependency environments. Their manifests
record resolved versions for reproduction. Generated files in `examples/output/`
are ignored by Git. Earlier exploratory notebooks and
obsolete local outputs are not part of the maintained code.

## Citation and support

Please cite the accompanying paper when using these routines; citation metadata
is provided in [CITATION.cff](CITATION.cff), including the
[preprint DOI](https://doi.org/10.48550/arXiv.2602.23304).

For questions or reproducible bug reports, open an issue in this repository.
Include the Julia version, command, parameters, and error or unexpected result.

## License

This code is distributed under the [MIT license](LICENSE).
