# Run from the repository root: julia --project=. validation/replica_convergence.jl
# This checks numerical stability of I_n at the paper parameters. Agreement
# between stencils does not establish convergence of the n→∞ replica series.
include(joinpath(@__DIR__, "..", "ReplicaME.jl"))
using LinearAlgebra
using Printf

BLAS.set_num_threads(1)
ω, χ, κ, η = 0.1, 0.45, 1.0, 0.5
times = [0.0, 1.0, 5.0, 10.0, 20.0]
orders = [5, 10, 15, 20]
model = (
    N=1, tspan=(0.0, last(times)), times,
    σ0_single=Matrix{Float64}(I, 2, 2), d0_single=zeros(2),
    Hfun=θ -> [θ -χ; -χ θ], hfun=θ -> zeros(2),
    jfun=θ -> sqrt(η * κ / 2) .* [1.0 im],
    Lfun=θ -> sqrt((1 - η) * κ / 2) .* [1.0 im],
)
settings = [
    (method=:mixed, h=2e-3, reltol=1e-11, abstol=1e-12),
    (method=:mixed, h=1e-3, reltol=2e-13, abstol=2e-14),
    (method=:eq20, h=2e-3, reltol=1e-11, abstol=1e-12),
    (method=:eq20, h=1e-3, reltol=2e-13, abstol=2e-14),
]
outdir = joinpath(@__DIR__, "output")
mkpath(outdir)
open(joinpath(outdir, "replica_convergence.tsv"), "w") do io
    println(io, "method\th\treltol\tabstol\ttime\torder\tI_n_real\tI_n_imag")
    for setting in settings
        elapsed = @elapsed _, estimates = I_n_timecourse(ω, orders; model..., setting...)
        println("Settings: ", setting, " (", round(elapsed; digits=2), " seconds)")
        for (column, order) in enumerate(orders), (row, t) in enumerate(times)
            value = estimates[row, column]
            println(io, join((setting.method, setting.h, setting.reltol, setting.abstol,
                              t, order, real(value), imag(value)), '\t'))
        end
        flush(io)
        for (column, order) in enumerate(orders)
            @printf("  I_%d(20)/20 = %.10g, max |Im I_n| = %.3g\n",
                    order, real(estimates[end, column]) / times[end],
                    maximum(abs, imag.(estimates[:, column])))
        end
        flush(stdout)
    end
end
