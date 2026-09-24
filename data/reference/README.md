# Reference data and figures

These are selected outputs of the maintained OPO calculation pipeline, saved
with the research code and included to make plotting and comparison possible
without rerunning the full simulations. Earlier exploratory datasets and
notebook outputs are not included.

The JLD2 files were copied without modification from the existing regenerated
outputs. Their metadata gives the following provenance:

| Dataset | Created | Schema | Parameters |
| --- | --- | --- | --- |
| `parametric_oscillator_eta=1_data.jld2` | 2026-09-17 | 2 | `χ/κ=0.45`, `η=1`, `κT=100`, 1000 trajectories |
| `parametric_oscillator_data.jld2` | 2026-09-17 | 3 | `χ/κ=0.45`, `η=0.5`, `κT=20`, orders through 30 |
| `chi_0p1/parametric_oscillator_data.jld2` | 2026-09-17 | 3 | `χ/κ=0.1`, `η=0.5`, `κT=20`, orders through 20 |
| `chi_0p7/parametric_oscillator_data.jld2` | 2026-09-19 | 4 | `χ/κ=0.7`, `η=0.5`, `κT=20`, orders 1, 2, 3, 5 |

All four files record Julia 1.13.0 and RNG seed 0. The inefficient-detection
datasets use 500 trajectories and the Hamiltonian sensitivity implementation
for the replica derivatives. Full model parameters and numerical tolerances
are recorded in `payload.params`, not inferred from filenames.

Schemas 2 and 3 predate metadata for adaptive TSME steps. The current generator
writes schema 4 and records the step at each time; the plots accept all three
schemas. There is no claim that these files are byte-identical to a fresh run,
or to every manuscript draft. Compare numerical series within integration and
sampling accuracy.

The QFI and replica PDF/PNG files are the corresponding saved research plots.
The TUR PDF/PNG and TSV files are generated from the extracted analytic
expressions in `examples/tur_common.jl` during repository preparation. The TSV
contains the grid and both plotted curves; `κ=1` and `ω=0`.

Read a dataset with:

```julia
using JLD2
payload = load("data/reference/parametric_oscillator_data.jld2", "payload")
payload.metadata
payload.params
payload.series
```

Run this in the `examples` Julia environment. See
[reproduction instructions](../../docs/reproduction.md) for commands that redraw
these figures in a separate output directory. `SHA256SUMS` records checksums of
the reference datasets and figures; verify with
`shasum -a 256 -c SHA256SUMS` from this directory.
