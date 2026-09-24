# Finite-difference window check for the OPO figure, in units κ=1.
include(joinpath(@__DIR__, "..", "ReplicaME.jl"))
using LinearAlgebra

BLAS.set_num_threads(1)
times = [1.0, 2.0, 5.0]
orders = [5, 10, 15, 20]
model = (
    N=1, tspan=(0.0, 5.0), times,
    σ0_single=Matrix{Float64}(I, 2, 2), d0_single=zeros(2),
    Hfun=θ -> [θ -0.45; -0.45 θ], hfun=θ -> zeros(2),
    jfun=θ -> 0.5 .* [1 im], Lfun=θ -> 0.5 .* [1 im],
    reltol=2e-13, abstol=2e-14,
)
mkpath(joinpath(@__DIR__, "output"))
open(joinpath(@__DIR__, "output", "replica_midrange.tsv"), "w") do io
    println(io, "step\ttime\torder\trate")
    for h in [0.04, 0.02, 0.01]
        t, values = I_n_timecourse(0.1, orders; h, model...)
        rates = real.(values) ./ t
        for (column, n) in enumerate(orders), (row, time) in enumerate(t)
            println(io, join((h, time, n, rates[row, column]), '\t'))
        end
        println("h=", h, ": order-20 rates = ", rates[:, end])
        flush(io)
        flush(stdout)
    end
end
