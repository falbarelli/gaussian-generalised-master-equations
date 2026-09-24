# Reproducing the paper figures

Run commands from the repository root. The names and labels below refer to the
accompanying manuscript, so the mapping survives changes to figure numbering.

## Environment

```sh
julia --project=examples -e 'using Pkg; Pkg.instantiate()'
```

The manifests record Julia 1.13.0 and resolved Julia dependencies. Plotting uses
Matplotlib through PythonCall; `examples/CondaPkg.toml` declares the Python
dependency. On first use, PythonCall/CondaPkg provisions a Python environment,
which requires network access. The default PDF/PNG backend requires no LaTeX.
Use `--tex` for PGF typesetting with a local `pdflatex` installation and the
fonts/packages used by `examples/plot_common.jl`.

## Figure-to-script map

| Manuscript result | Calculation scripts | Plot script |
| --- | --- | --- |
| Perfect detection, `fig1` | `parametric_oscillator_eta=1.jl` | `parametric_oscillator_eta=1_plot.jl` |
| Inefficient detection, two panels, `fig3` | `parametric_oscillator.jl` and `parametric_oscillator_low_pump.jl` | `parametric_oscillator_comparison_plot.jl` |
| TUR, `fig2` | Analytic expressions in `tur_common.jl` | `parametric_oscillator_TUR_plot.jl` |
| Additional above-threshold example | `parametric_oscillator_above_threshold.jl` | `parametric_oscillator_above_threshold_plot.jl` |

All listed scripts live in `examples/`. The TUR script evaluates the analytic
stationary formulas for the counting current, noise, and time-rescaling QFI
rate at zero detuning. It was extracted from the research plotting notebook;
no ODE calculation or precomputed dataset is needed for this figure.

## Complete OPO calculations

```sh
julia --project=examples 'examples/parametric_oscillator_eta=1.jl'
julia --project=examples 'examples/parametric_oscillator_eta=1_plot.jl'

julia --project=examples examples/parametric_oscillator_TUR_plot.jl

julia --project=examples examples/parametric_oscillator.jl
julia --project=examples examples/parametric_oscillator_low_pump.jl
julia --project=examples examples/parametric_oscillator_comparison_plot.jl
```

The perfect-detection figure is written as
`examples/output/parametric_oscillator_QFI_eta=1_paper.{pdf,png}`; the two-panel
figure is `examples/output/parametric_oscillator_rates_comparison.{pdf,png}`.
Add `--tex` to the plot commands for the manuscript's LaTeX typesetting.
The TUR script writes `examples/output/parametric_oscillator_TUR.{pdf,png,tsv}`.
Its grid excludes zero pump, where the plotted ratios diverge, and stays below
the stationary threshold `χ/κ = 1/2`.

For separate replica panels:

```sh
julia --project=examples examples/parametric_oscillator_plot.jl
julia --project=examples examples/parametric_oscillator_plot.jl --output-dir=examples/output/chi_0p1
```

For the additional above-threshold case:

```sh
julia --project=examples examples/parametric_oscillator_above_threshold.jl
julia --project=examples examples/parametric_oscillator_above_threshold_plot.jl
```

This writes to `examples/output/chi_0p7/`. The finite-time state remains physical,
but no stationary unconditional covariance exists for these parameters. The
plotted replica orders are lower bounds, not converged output QFI estimates.

## Quick end-to-end checks

```sh
julia --project=examples examples/parametric_oscillator.jl --smoke
julia --project=examples examples/parametric_oscillator_plot.jl --smoke
julia --project=examples 'examples/parametric_oscillator_eta=1.jl' --smoke
julia --project=examples 'examples/parametric_oscillator_eta=1_plot.jl' --smoke
```

These use a short time window, four trajectories, and reduced replica orders.
They write only to `examples/output/smoke/` and check the pipeline rather than
the paper's numerical accuracy. Compilation can dominate these runs; full
calculations, especially high-order replica checks, take longer.

## Redraw saved reference data

`data/reference/` contains selected regenerated datasets and figures, with
provenance in its README. Copy datasets to a disposable output directory before
plotting, so reference files remain unchanged:

```sh
mkdir -p examples/output/from-reference/chi_0p1
cp data/reference/parametric_oscillator_eta=1_data.jld2 examples/output/from-reference/
cp data/reference/parametric_oscillator_data.jld2 examples/output/from-reference/
cp data/reference/chi_0p1/parametric_oscillator_data.jld2 examples/output/from-reference/chi_0p1/
julia --project=examples 'examples/parametric_oscillator_eta=1_plot.jl' --output-dir=examples/output/from-reference
julia --project=examples examples/parametric_oscillator_comparison_plot.jl --output-dir=examples/output/from-reference
```

## Change parameters and inspect results

`run_oscillator` in `examples/opo_common.jl` exposes physical and numerical
parameters as keyword arguments. In `julia --project=examples`, for example:

```julia
include("examples/opo_common.jl")
run_oscillator(χ=0.2, η=0.8, tFinal=5.0, n_list=[1, 2, 3],
               output_dir="examples/output/custom")
```

Saved JLD2 files contain a `payload` with `metadata`, `params`, and `series`:

```julia
using JLD2
payload = load("examples/output/parametric_oscillator_data.jld2", "payload")
payload.params       # physical parameters, seed, steps, tolerances
payload.metadata     # producer, schema, Julia version, creation time
payload.series       # time grids, information rates, complex replica estimates
```

The inefficient-detection default uses replica orders 5, 10, 15, 20, 25, and 30.
The weak-pump panel uses 1, 2, 3, 5, 10, 15, and 20. The general finite-difference
implementation remains available; these OPO calculations use Hamiltonian
sensitivity equations to avoid parameter-difference cancellation at high order.

Homodyne trajectories use a recorded RNG seed. Small differences can remain
across environments; Monte Carlo curves also require sampling-error and timestep
checks. Bitwise agreement of generated image files is not expected. See the
[numerical notes](methods.md#numerical-notes) before changing model parameters.

The historical `--archive` plotting option refers to earlier local datasets
that are not distributed. Use the reference-data workflow above instead.
