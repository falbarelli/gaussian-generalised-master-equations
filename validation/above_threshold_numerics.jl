# Generate the χ/κ=0.7 dataset first, then run with --project=examples.
# Check replica integration and fidelity parameter steps independently.
include(joinpath(@__DIR__, "..", "ReplicaME.jl"))
using LinearAlgebra, JLD2, Printf

BLAS.set_num_threads(1)
datafile = joinpath(@__DIR__, "..", "examples", "output", "chi_0p7",
                    "parametric_oscillator_data.jld2")
payload = load(datafile, "payload")
p, s = payload.params, payload.series
times = s.barg_times
positive = times .> 0

_, refined = I_n_hamiltonian_timecourse(p.n_list;
    H=[p.ω -p.χ; -p.χ p.ω], dH=Matrix{Float64}(I, 2, 2),
    j=sqrt(p.η*p.κ/2) .* [1 im], L=sqrt((1-p.η)*p.κ/2) .* [1 im],
    σ0_single=Matrix{Float64}(I, 2, 2), tspan=(0.0, p.tFinal), times,
    reltol=p.replica_reltol/10, abstol=p.replica_abstol/10,
    sensitivity_abstol=p.replica_sensitivity_abstol/10)
replica_error = maximum(abs.((refined[positive, :] - s.barg_values[positive, :]) ./
                              s.barg_values[positive, :]))
@assert replica_error < 1e-7
@printf("Replica tolerance refinement: max relative change %.4g\n", replica_error)
for (column, order) in enumerate(p.n_list)
    @printf("  n=%d: endpoint information/time %.12g\n", order,
            real(refined[end, column])/p.tFinal)
end

model = (N=1, σ0=Matrix{Float64}(I, 2, 2), d0=zeros(2),
         Hfun=θ -> [θ -p.χ; -p.χ θ],
         Lfun=θ -> sqrt(p.κ/2) .* [1 im])

function refine_fidelities(p, times, model)
    environment, joint = (zeros(length(times)) for _ in 1:2)
    previous = -Inf
    for (endpoint, step) in p.tsme_steps
        indices = findall(t -> previous < t <= endpoint, times)
        previous = endpoint
        isempty(indices) && continue
        samples = times[indices]
        _, environment[indices], joint[indices] = qfi_timecourse_from_tsme_fidelity(
            p.ω; ϵ=step/2, model..., tspan=(0.0, last(samples)), saveat=samples,
            reltol=p.reltol/10, abstol=p.abstol/10)
    end
    return environment, joint
end

environment, joint = refine_fidelities(p, times, model)
for (name, refined_values, base_rate) in (
    ("environment", environment, s.qfi_env_rate),
    ("joint", joint, s.qfi_joint_rate),
)
    refined_rate = refined_values[positive] ./ times[positive]
    difference = abs.(refined_rate .- base_rate)
    # At vanishingly small early-time fidelity curvature, report absolute
    # error as well; relative error alone magnifies harmless roundoff.
    @printf("%s QFI: max relative rate change %.4g; max absolute %.4g\n", name,
            maximum(difference ./ abs.(refined_rate)), maximum(difference))
    @printf("  endpoint default %.12g; refined %.12g\n", last(base_rate), last(refined_rate))
    @assert all(difference .< 2e-5 .* abs.(refined_rate) .+ 1e-9)
end
