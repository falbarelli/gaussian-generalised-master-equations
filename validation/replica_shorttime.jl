# Short-time finite-difference stability at the paper parameters.
# Run from the repository root: julia --project=. validation/replica_shorttime.jl
include(joinpath(@__DIR__, "..", "ReplicaME.jl"))
using LinearAlgebra
using Printf

BLAS.set_num_threads(1)
ω, χ, κ, η = 0.1, 0.45, 1.0, 0.5
times, orders = [0.1, 0.2, 0.5, 1.0], [5, 10, 15, 20]
model = (
    N = 1, tspan = (0.0, 1.0), times,
    σ0_single = Matrix{Float64}(I, 2, 2), d0_single = zeros(2),
    Hfun = θ -> [θ -χ; -χ θ], hfun = θ -> zeros(2),
    jfun = θ -> sqrt(η * κ / 2) .* [1.0 im],
    Lfun = θ -> sqrt((1 - η) * κ / 2) .* [1.0 im],
    reltol = 2e-13, abstol = 2e-14,
)
mkpath(joinpath(@__DIR__, "output"))
open(joinpath(@__DIR__, "output", "replica_shorttime.tsv"), "w") do io
    println(io, "method\th\treltol\tabstol\ttime\torder\tI_n_real\tI_n_imag")
    settings = vcat([(method, h) for method in (:mixed, :eq20) for h in (0.1, 0.05, 0.025)],
                    [(:mixed, 0.8), (:mixed, 0.4), (:mixed, 0.2)])
    for (method, h) in settings
        elapsed = @elapsed _, estimates = I_n_timecourse(ω, orders; method, h, model...)
        println("method=", method, " h=", h, " seconds=", elapsed)
        for (column, order) in enumerate(orders), (row, t) in enumerate(times)
            value = estimates[row, column]
            println(io, join((method, h, model.reltol, model.abstol, t, order,
                              real(value), imag(value)), '\t'))
        end
        flush(io)
        for (row, t) in enumerate(times)
            @printf("  t=%.2g: I20/t=%.12g; I15/t=%.12g\n", t,
                    real(estimates[row, end]) / t, real(estimates[row, end-1]) / t)
        end
        flush(stdout)
    end
end
