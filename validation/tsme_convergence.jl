# Compare derivative steps at both early and late times. Small absolute QFI
# can be below the fidelity's floating-point resolution even with tight ODEs.
include(joinpath(@__DIR__, "..", "ReplicaME.jl"))
using LinearAlgebra

model = (
    N=1, tspan=(0.0, 100.0), σ0=Matrix{Float64}(I, 2, 2), d0=zeros(2),
    Hfun=θ -> [θ -0.45; -0.45 θ], Lfun=θ -> sqrt(0.5) .* [1 im],
    saveat=[0.1, 0.5, 1.0, 10.0, 20.0, 100.0],
    reltol=2e-13, abstol=2e-14,
)
outdir = joinpath(@__DIR__, "output")
mkpath(outdir)
open(joinpath(outdir, "tsme_convergence.tsv"), "w") do io
    println(io, "step\ttime\tenvironment_rate\tjoint_rate")
    for step in [0.004, 0.002, 0.001, 0.0005, 0.00025]
        times, environment, joint = qfi_timecourse_from_tsme_fidelity(0.1; ϵ=step, model...)
        for i in eachindex(times)
            println(io, join((step, times[i], environment[i] / times[i], joint[i] / times[i]), '\t'))
        end
        println("ϵ=", step, ": environment/t at t=0.1: ", environment[1] / times[1],
                "; at t=100: ", environment[end] / times[end])
    end
end

# Independent analytical check: the steady-state joint-QFI rate at ω=0.
# Use a late-time slope to remove the finite initial transient.
χ, κ = 0.2, 1.0
times, _, joint = qfi_timecourse_from_tsme_fidelity(0.0;
    ϵ=5e-4, N=1, tspan=(0.0, 120.0), σ0=Matrix{Float64}(I, 2, 2),
    d0=zeros(2), Hfun=θ -> [θ -χ; -χ θ], Lfun=θ -> sqrt(κ/2) .* [1 im],
    saveat=[100.0, 120.0], reltol=2e-12, abstol=2e-13)
exact = 8κ * χ^2 * (5κ^2 - 4χ^2) / (κ^2 - 4χ^2)^3
numerical = (joint[2] - joint[1]) / (times[2] - times[1])
@assert isapprox(numerical, exact; rtol=2e-7)
println("Analytical steady-state rate: ", exact, "; computed: ", numerical)
