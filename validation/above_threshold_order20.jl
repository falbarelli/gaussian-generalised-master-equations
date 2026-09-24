# Feasibility and tolerance check, separate from the main figure datasets.
# Run with --project=examples; add --tight for tenfold tighter tolerances.
# Saved params describe the source dataset; orders and tolerances specify this check.
include(joinpath(@__DIR__, "..", "ReplicaME.jl"))
using LinearAlgebra, JLD2, Printf

BLAS.set_num_threads(1)
source = joinpath(@__DIR__, "..", "examples", "output", "chi_0p7",
                  "parametric_oscillator_data.jld2")
payload = load(source, "payload")
p = payload.params
orders = [5, 10, 15, 20]
times = payload.series.barg_times
tight = "--tight" in ARGS
scale = tight ? 0.1 : 1.0
tolerances = (reltol=p.replica_reltol*scale, abstol=p.replica_abstol*scale,
              sensitivity_abstol=p.replica_sensitivity_abstol*scale)

println("Above-threshold replica orders ", orders, "; tolerances: ", tolerances)
flush(stdout)
elapsed = @elapsed _, estimates = I_n_hamiltonian_timecourse(orders;
    H=[p.ω -p.χ; -p.χ p.ω], dH=Matrix{Float64}(I, 2, 2),
    j=sqrt(p.η*p.κ/2) .* [1 im], L=sqrt((1-p.η)*p.κ/2) .* [1 im],
    σ0_single=Matrix{Float64}(I, 2, 2), tspan=(0.0, p.tFinal), times,
    tolerances...)
@assert all(isfinite, estimates)
@printf("Elapsed time including compilation: %.1f seconds\n", elapsed)
for (column, order) in enumerate(orders)
    rates = estimates[2:end, column] ./ times[2:end]
    @printf("n=%d: endpoint %.12g; maximum imaginary rate %.4g\n",
            order, real(last(rates)), maximum(abs, imag.(rates)))
end

outdir = joinpath(@__DIR__, "output")
mkpath(outdir)
filename = tight ? "above_threshold_order20_tight.jld2" : "above_threshold_order20.jld2"
jldsave(joinpath(outdir, filename); params=p, orders, times, estimates, tolerances, elapsed)
println("Saved: ", joinpath(outdir, filename))
