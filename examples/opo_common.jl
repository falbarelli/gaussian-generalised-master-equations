include(joinpath(@__DIR__, "..", "GaussianGMEs.jl"))
using LinearAlgebra
using Random
using Dates
using JLD2

# These ODEs use small dense matrices; BLAS threading costs more than it saves.
BLAS.set_num_threads(1)

"""Return X(t)/t at positive times, with matching axes checked explicitly."""
function rate_over_time(times, values)
    length(times) == length(values) || throw(DimensionMismatch("times and values differ"))
    keep = times .> 0
    return times[keep], real.(values[keep]) ./ times[keep]
end

"""
    run_oscillator(; kwargs...)

Generate the OPO information curves and save both parameters and raw series.
The TSME includes the full output channel; replica and homodyne curves use η.
Replica derivatives default to Hamiltonian sensitivity equations. Set
`replica_method=:finite_difference` to use the independent step-based method;
its steps must be checked for convergence when parameters change.
`tsme_steps` optionally selects fidelity parameter steps over successive time
windows. Each window evolves from the original initial state; smaller late-time
steps can resolve rapidly growing QFI without losing early-time curvature.
"""
function run_oscillator(;
    ω=0.1, χ=0.45, κ=1.0, η=0.5, ϕ=0.0, meas="hom", nth=0.0,
    tFinal=20.0, dt_traj=0.001, Ntraj=500, dt_eval=0.1,
    ϵ=5e-4 * κ, h=2e-3 * κ, n_list=[5, 10, 15, 20, 25, 30], seed=0,
    reltol=1e-11, abstol=1e-12,
    tsme_steps=((Inf, ϵ),),
    replica_method=:sensitivity,
    replica_steps=((1 / κ, 0.2κ), (5 / κ, 0.02κ), (Inf, h)),
    replica_reltol=2e-13, replica_abstol=2e-14,
    replica_sensitivity_abstol=1e-22,
    output_dir=joinpath(@__DIR__, "output"), show_progress=isinteractive(),
)
    nth == 0 || throw(ArgumentError("these TSME QFI curves require a pure initial state (nth=0)"))
    κ > 0 || throw(ArgumentError("the figure pipeline uses κ as its unit and requires κ>0"))
    tFinal > 0 && dt_eval > 0 || throw(ArgumentError("tFinal and dt_eval must be positive"))
    replica_method in (:sensitivity, :finite_difference) ||
        throw(ArgumentError("replica_method must be :sensitivity or :finite_difference"))
    Hfun(θ) = [θ -χ; -χ θ]
    hfun(θ) = zeros(2)
    jfun(θ) = sqrt(η * κ / 2) .* [1.0 im]
    Lfun(θ) = sqrt((1 - η) * κ / 2) .* [1.0 im]
    Lfullfun(θ) = sqrt(κ / 2) .* [1.0 im]
    σ0_single = Matrix{Float64}(I, 2, 2)
    d0_single = zeros(2)
    tspan = (0.0, tFinal)
    times_eval = collect(0.0:dt_eval:tFinal)
    last(times_eval) < tFinal && push!(times_eval, tFinal)

    t_unr, Fs, Qc, Qu = simulate_unravelling(
        ω, χ, κ, η, ϕ, meas, nth, tFinal, dt_traj, Ntraj;
        rng=MersenneTwister(seed), show_progress,
    )
    t_rate, Fs_rate = rate_over_time(t_unr, Fs)
    _, Qc_rate = rate_over_time(t_unr, Qc)
    _, Qu_rate = rate_over_time(t_unr, Qu)

    # A separate output grid avoids storing the deterministic ODE solution at
    # every Monte Carlo timestep (200,001 points for the original η=1 run).
    t_tsme = times_eval
    qfi_env, qfi_joint, tsme_ϵ = (zeros(length(times_eval)) for _ in 1:3)
    previous = -Inf
    for (endpoint, step) in tsme_steps
        endpoint > previous && isfinite(step) && step > 0 ||
            throw(ArgumentError("tsme_steps needs increasing endpoints and positive finite steps"))
        indices = findall(t -> previous < t <= endpoint, times_eval)
        previous = endpoint
        isempty(indices) && continue
        stage_times = times_eval[indices]
        _, environment, joint = qfi_timecourse_from_tsme_fidelity(
            ω; ϵ=step, N=1, tspan=(0.0, last(stage_times)),
            σ0=σ0_single, d0=d0_single, Hfun, hfun, Lfun=Lfullfun,
            reltol, abstol, saveat=stage_times,
        )
        qfi_env[indices] .= environment
        qfi_joint[indices] .= joint
        tsme_ϵ[indices] .= step
    end
    all(>(0), tsme_ϵ) || throw(ArgumentError("tsme_steps must cover the output time grid"))
    t_env_rate, qfi_env_rate = rate_over_time(t_tsme, qfi_env)
    t_joint_rate, qfi_joint_rate = rate_over_time(t_tsme, qfi_joint)

    barg_t = Vector{Float64}[]
    barg_rate = Vector{Float64}[]
    barg_h = Float64[]
    barg_values = zeros(ComplexF64, length(times_eval), 0)
    if !isempty(n_list) && replica_method == :sensitivity
        println("Replica Hamiltonian sensitivities through order ", maximum(n_list),
                " (", maximum(n_list) + 2, " replicas)")
        _, barg_values = I_n_hamiltonian_timecourse(n_list;
            H=Hfun(ω), dH=Matrix{Float64}(I, 2, 2), j=jfun(ω), L=Lfun(ω),
            σ0_single, tspan, times=times_eval,
            reltol=replica_reltol, abstol=replica_abstol,
            sensitivity_abstol=replica_sensitivity_abstol,
        )
    elseif !isempty(n_list)
        # Short-time information is much smaller: a tiny parameter step loses
        # its curvature to cancellation. These explicit windows were checked
        # at the paper parameters; each window still evolves from t=0.
        # For a constant step, pass replica_steps=((Inf, h),).
        estimates = zeros(ComplexF64, length(times_eval), length(n_list))
        barg_h = zeros(length(times_eval))
        previous = -Inf
        for (endpoint, step) in replica_steps
            endpoint > previous && isfinite(step) && step > 0 ||
                throw(ArgumentError("replica_steps needs increasing endpoints and positive finite steps"))
            indices = findall(t -> previous < t <= endpoint, times_eval)
            previous = endpoint
            isempty(indices) && continue
            stage_times = times_eval[indices]
            println("Replica samples t=", first(stage_times), "…", last(stage_times),
                    ", parameter step h=", step)
            _, values = I_n_timecourse(
                ω, n_list; h=step, N=1, tspan=(0.0, last(stage_times)),
                times=stage_times, σ0_single, d0_single, Hfun, hfun, jfun, Lfun,
                reltol=replica_reltol, abstol=replica_abstol,
            )
            estimates[indices, :] = values
            barg_h[indices] .= step
        end
        all(>(0), barg_h) || throw(ArgumentError("replica_steps must cover the output time grid"))
        barg_h = barg_h[times_eval .> 0] # Same positive-time grid as each barg_t.
        barg_values = estimates # Retain imaginary residuals for numerical checks.
    end
    for column in axes(barg_values, 2)
        times, rates = rate_over_time(times_eval, barg_values[:, column])
        push!(barg_t, times)
        push!(barg_rate, rates)
    end

    payload = (
        metadata=(created_at=string(Dates.now()), producer="examples/opo_common.jl",
                  format="jld2", schema_version=4, julia_version=string(VERSION)),
        params=(; ω, χ, κ, η, ϕ, meas, nth, tFinal, dt_traj, Ntraj,
                 dt_eval, ϵ, h, n_list, seed, reltol, abstol, tsme_steps,
                 replica_method, replica_steps, replica_reltol, replica_abstol,
                 replica_sensitivity_abstol),
        series=(; t_rate, Fs_rate, Qc_rate, Qu_rate, t_env_rate, qfi_env_rate,
                 t_joint_rate, qfi_joint_rate, tsme_ϵ, barg_n_list=n_list, barg_t, barg_rate, barg_h,
                 barg_times=times_eval, barg_values,
                 t_unr, Fs, Qc, Qu),
    )
    mkpath(output_dir)
    filename = isempty(n_list) && η == 1 ? "parametric_oscillator_eta=1_data.jld2" :
                                          "parametric_oscillator_data.jld2"
    outfile = joinpath(output_dir, filename)
    jldsave(outfile; payload)
    println("Saved data to: ", outfile)
    println("Final rates: joint QFI = ", last(qfi_joint_rate),
            ", environment QFI = ", last(qfi_env_rate), ", signal CFI = ", last(Fs_rate))
    for (n, rates) in zip(n_list, barg_rate)
        println("  I_", n, "/t = ", last(rates))
    end
    return payload
end
