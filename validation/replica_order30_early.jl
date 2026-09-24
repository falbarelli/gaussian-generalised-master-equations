# Resolve the smallest OPO information rates with direct derivatives.
# Run: julia --project=. validation/replica_order30_early.jl
include(joinpath(@__DIR__, "..", "ReplicaME.jl"))
using LinearAlgebra
using Printf

BLAS.set_num_threads(1)
orders = [5, 20, 25, 30]
model = (
    H=[0.1 -0.45; -0.45 0.1], dH=Matrix{Float64}(I, 2, 2),
    j=0.5 .* [1.0 im], L=0.5 .* [1.0 im],
    σ0_single=Matrix{Float64}(I, 2, 2),
)
settings = (
    (name="default", times=collect(0.1:0.1:1.0),
     reltol=2e-13, abstol=2e-14, sensitivity_abstol=1e-22),
    (name="tight", times=[0.1, 0.2],
     reltol=2e-14, abstol=2e-15, sensitivity_abstol=1e-23),
)
outdir = joinpath(@__DIR__, "output")
mkpath(outdir)
open(joinpath(outdir, "replica_order30_early_sensitivity.tsv"), "w") do io
    println(io, "setting\tn\ttime\trate_real\trate_imag")
    for setting in settings
        elapsed = @elapsed times, values = I_n_hamiltonian_timecourse(orders;
            model..., tspan=(0.0, last(setting.times)), times=setting.times,
            reltol=setting.reltol, abstol=setting.abstol,
            sensitivity_abstol=setting.sensitivity_abstol,
        )
        println(setting.name, ": ", round(elapsed; digits=1), " seconds")
        for (column, n) in enumerate(orders), (row, t) in enumerate(times)
            rate = values[row, column] / t
            println(io, join((setting.name, n, t, real(rate), imag(rate)), '\t'))
        end
        @printf("  I30(0.1)/0.1 = %.15g\n", real(values[1, end]) / times[1])
        flush(io); flush(stdout)
    end
end
