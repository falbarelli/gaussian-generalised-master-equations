# Run from the repository root: julia --project=. examples/general_odes.jl
# This file loads only the general moment solver, without replica/QFI routines.
include(joinpath(@__DIR__, "..", "src", "GaussianMoments.jl"))
using .GaussianMoments
using LinearAlgebra

"""
    general_odes_example(; κ=0.7, ω0=0.4, modulation=0.2, final_time=5.0)

A damped mode with time-dependent frequency `ω0 + modulation*sin(t)`.
The initial displaced thermal state has covariance `2I` and mean `[1, 0.2]`.
Return the ODE solution and its final covariance, mean, negative log trace,
and trace. All coefficient arrays can be replaced by those of another GME.
"""
function general_odes_example(; κ=0.7, ω0=0.4, modulation=0.2, final_time=5.0)
    identity2 = Matrix{Float64}(I, 2, 2)
    Ω = symplectic_Ω(1)
    zero_matrix = zeros(2, 2)
    zero_vector = zeros(2)
    params(t) = (
        Q=zero_matrix,
        A=(ω0 + modulation * sin(t)) * Ω - (κ / 2) * identity2,
        D=κ * identity2,
        a=zero_vector,
        c=zero_vector,
        W=zero_matrix,
    )

    sol = solve_gaussian_odes(;
        N=1, tspan=(0.0, final_time), σ0=2identity2, d0=[1.0, 0.2],
        params, saveat=collect(range(0.0, final_time; length=101)),
        reltol=1e-10, abstol=1e-12,
    )
    σ, d, ξ = get_σ_d_ξ(sol, 1, length(sol.t))
    return (; solution=sol, covariance=σ, mean=d, log_normalization=ξ,
            trace=exp(-ξ))
end

if abspath(PROGRAM_FILE) == @__FILE__
    result = general_odes_example()
    println("Final covariance: ", result.covariance)
    println("Final first moments: ", result.mean)
    println("Final trace: ", result.trace)
end
