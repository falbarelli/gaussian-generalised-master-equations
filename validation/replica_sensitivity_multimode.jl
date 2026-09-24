# Independent derivative check with coupled modes and multiple output channels.
# Run: julia --project=. validation/replica_sensitivity_multimode.jl
include(joinpath(@__DIR__, "..", "ReplicaME.jl"))
using LinearAlgebra
BLAS.set_num_threads(1)
H0 = [0.3 0.07 0.02 -0.01; 0.07 0.4 0.03 0.02;
      0.02 0.03 0.5 -0.09; -0.01 0.02 -0.09 0.6]
dH = [1.0 0.2 0.1 0.0; 0.2 -0.3 0.0 0.15;
      0.1 0.0 0.6 -0.07; 0.0 0.15 -0.07 -0.2]
θ = 0.13
j = [0.4 0.4im 0 0; 0 0 0.3 0.3im]
L = [0.2 0.2im 0 0; 0 0 0.1 0.1im]
σ0 = diagm([exp(0.3), exp(-0.3), 1.2, 1.2])
times, orders = [0.2, 0.7], [0, 1, 2]
tspan = (0.0, last(times))
_, sensitivities = I_n_hamiltonian_timecourse(orders;
    H=H0 + θ * dH, dH, j, L, σ0_single=σ0, tspan, times)
model = (
    N=2, σ0_single=σ0, d0_single=zeros(4), tspan, times,
    Hfun=x -> H0 + x * dH, hfun=_ -> zeros(4),
    jfun=_ -> j, Lfun=_ -> L, reltol=2e-13, abstol=2e-14,
)
cache = Dict()
for h in (0.01, 0.005)
    _, displaced = I_n_timecourse(θ, orders; h, method=:eq20, cache, model...)
    difference = abs.(sensitivities - displaced)
    println("h=", h, ": max absolute difference = ", maximum(difference),
            ", max relative difference = ", maximum(difference ./ abs.(sensitivities)))
end
