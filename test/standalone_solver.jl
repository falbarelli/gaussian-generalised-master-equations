# An independent process exercises the documented include-only entry point.
using Test
using LinearAlgebra
include(joinpath(@__DIR__, "..", "examples", "general_odes.jl"))

@testset "Standalone time-dependent Gaussian solver" begin
    result = general_odes_example()
    for i in eachindex(result.solution.t)
        t = result.solution.t[i]
        σ, d, ξ = get_σ_d_ξ(result.solution, 1, i)
        angle = 0.4t + 0.2 * (1 - cos(t))
        rotation = [cos(angle) sin(angle); -sin(angle) cos(angle)]
        @test σ ≈ (1 + exp(-0.7t)) * Matrix{Float64}(I, 2, 2) atol=2e-9
        @test d ≈ exp(-0.7t / 2) * rotation * [1.0, 0.2] atol=2e-9
        @test exp(-ξ) ≈ 1 atol=1e-12
    end
end
