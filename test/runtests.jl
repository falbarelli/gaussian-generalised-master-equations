using Test
using LinearAlgebra
using Random
include(joinpath(@__DIR__, "..", "ReplicaME.jl"))

BLAS.set_num_threads(1)

@testset "ReplicaME" begin
    include("gaussian_tsme_tests.jl")
    include("replica_tests.jl")
    include("hamiltonian_replica_tests.jl")
    include("replica_fock_tests.jl")
    include("unravelling_tests.jl")
    include("tur_tests.jl")
end
