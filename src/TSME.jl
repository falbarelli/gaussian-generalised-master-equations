module TSME

using LinearAlgebra
using OrdinaryDiffEqTsit5: Tsit5

import ..GaussianMoments: symplectic_Ω, solve_gaussian_odes, get_σ_d_ξ, log_trace_norm_gaussian_nu
import ..ReplicaCoeffs: sym, skew

export build_tsme_coeffs,
    tsme_params,
    solve_tsme_moments,
    overlap_from_solution,
    trace_norm_from_solution,
    fidelity_from_solution,
    fidelity_timecourse_from_solution,
    absoverlap_timecourse_from_solution,
    qfi_timecourse_from_tsme_fidelity,
    qfi_from_tsme_fidelity

"""Build TSME coefficients for Hamiltonian `rᵀH(θ)r/2 + h(θ)ᵀΩr`.

Quadratures are interleaved `(x₁,p₁,…)`. Rows of `Lfun(θ)` identify the same
environment channels at both parameter values; use zero rows for absent
couplings. `nothing` denotes zero coupling to every channel.
"""
function build_tsme_coeffs(
    θ1,
    θ2;
    N::Int,
    Hfun::Function,
    hfun::Function = (θ -> zeros(2N)),
    Lfun::Function,
    Ω::AbstractMatrix = symplectic_Ω(N),
)
    H1 = ComplexF64.(Hfun(θ1))
    H2 = ComplexF64.(Hfun(θ2))
    h1 = ComplexF64.(hfun(θ1))
    h2 = ComplexF64.(hfun(θ2))

    L1_raw = Lfun(θ1)
    L2_raw = Lfun(θ2)
    L1 = (L1_raw === nothing) ? zeros(ComplexF64, 0, 2N) : ComplexF64.(L1_raw)
    L2 = (L2_raw === nothing) ? zeros(ComplexF64, 0, 2N) : ComplexF64.(L2_raw)

    size(H1) == (2N, 2N) || throw(ArgumentError("H(θ1) must be (2N,2N)"))
    size(H2) == (2N, 2N) || throw(ArgumentError("H(θ2) must be (2N,2N)"))
    length(h1) == 2N || throw(ArgumentError("h(θ1) must have length 2N"))
    length(h2) == 2N || throw(ArgumentError("h(θ2) must have length 2N"))
    size(L1, 2) == 2N || throw(ArgumentError("L(θ1) must have 2N columns"))
    size(L2, 2) == 2N || throw(ArgumentError("L(θ2) must have 2N columns"))
    size(Ω) == (2N, 2N) || throw(DimensionMismatch("Ω must have size (2N, 2N)"))
    if isempty(L1)
        L1 = zeros(ComplexF64, size(L2, 1), 2N)
    elseif isempty(L2)
        L2 = zeros(ComplexF64, size(L1, 1), 2N)
    end
    size(L1, 1) == size(L2, 1) || throw(ArgumentError("L(θ1) and L(θ2) must describe the same channels; pad absent couplings with zero rows"))

    Hplus = 0.5 .* (H1 + H2)
    Hminus = 0.5 .* (H1 - H2)

    G1 = adjoint(L1) * L1
    G2 = adjoint(L2) * L2
    Gplus = 0.5 .* (G1 + G2)
    Gminus = 0.5 .* (G1 - G2)

    K = transpose(adjoint(L2) * L1)

    Ωc = ComplexF64.(Ω)

    # The antisymmetric quadratic part of a commutator is a scalar and drops
    # out. Keeping skew(Gminus) here introduces a spurious drift whenever
    # jump strengths depend on θ (even vacuum would no longer be stationary).
    A = Ωc * (Hplus - 1im .* sym(Gminus) + 1im .* skew(K))
    D = Ωc * sym(K + 1im .* Hminus + Gplus) * transpose(Ωc)
    Q = sym(K - 1im .* Hminus - Gplus)
    W = 1im .* skew(K + 1im .* Hminus + Gplus)

    a = (-0.5im) .* (h1 - h2)
    c = (-0.5im) .* (h1 + h2)

    return (; Q, A, D, a, c, W, Ω = Ωc)
end

"""Return a constant `params(t)` closure for TSME coefficients."""
function tsme_params(
    θ1,
    θ2;
    N::Int,
    Hfun::Function,
    hfun::Function = (θ -> zeros(2N)),
    Lfun::Function,
)
    coeffs = build_tsme_coeffs(θ1, θ2; N = N, Hfun = Hfun, hfun = hfun, Lfun = Lfun)
    return (t -> coeffs)
end

"""Solve TSME Gaussian moment equations for `(θ1, θ2)` and initial moments."""
function solve_tsme_moments(
    θ1,
    θ2;
    N::Int,
    tspan,
    σ0,
    d0,
    ξ0::Number = 0.0,
    Hfun::Function,
    hfun::Function = (θ -> zeros(2N)),
    Lfun::Function,
    solver = Tsit5(),
    reltol::Real = 1e-9,
    abstol::Real = 1e-9,
    saveat = nothing,
)
    params = tsme_params(θ1, θ2; N = N, Hfun = Hfun, hfun = hfun, Lfun = Lfun)
    return solve_gaussian_odes(
        N = N,
        tspan = tspan,
        σ0 = σ0,
        d0 = d0,
        ξ0 = ξ0,
        params = params,
        solver = solver,
        reltol = reltol,
        abstol = abstol,
        saveat = saveat,
    )
end

"""Compute overlap `Tr[μ(T)]` from a TSME solution, using `Tr[μ]=exp(-ξ)`."""
function overlap_from_solution(sol)
    ξT = sol.u[end][end]
    return exp(-ξT)
end

"""Compute `‖μ(T)‖₁ = |exp(-ξ(T))| * ‖ν(T)‖₁` from a TSME solution."""
function trace_norm_from_solution(sol, N::Int)
    σT, dT, ξT = get_σ_d_ξ(sol, N, length(sol.u))
    return exp(_log_trace_norm(σT, dT, ξT))
end

_log_trace_norm(σ, d, ξ) = -real(ξ) + log_trace_norm_gaussian_nu(σ, d)

"""Environment root fidelity `‖μ(T)‖₁`, assuming a parameter-independent pure initial system state.

For a mixed initial state the same expression refers to the environment
together with a purifying reference, and need not equal environment fidelity.
"""
fidelity_from_solution(sol, N::Int) = trace_norm_from_solution(sol, N)

"""Compute fidelity timecourse `F(t)=‖μ(t)‖₁` from a TSME ODE solution."""
function fidelity_timecourse_from_solution(sol, N::Int)
    times = collect(sol.t)
    F = Vector{Float64}(undef, length(sol.u))

    @inbounds for i in eachindex(sol.u)
        σt, dt, ξt = get_σ_d_ξ(sol, N, i)
        Ft = exp(_log_trace_norm(σt, dt, ξt))
        F[i] = real(Ft)
    end

    return times, F
end

"""Compute overlap-modulus timecourse `F(t)=|Tr[μ(t)]|` from a TSME ODE solution."""
function absoverlap_timecourse_from_solution(sol, N::Int)
    times = collect(sol.t)
    F = Vector{Float64}(undef, length(sol.u))

    @inbounds for i in eachindex(sol.u)
        F[i] = exp(-real(sol.u[i][end]))
    end

    return times, F
end

@inline function _theta_pair(θ, δ::Real, scheme::Symbol)
    isfinite(δ) && δ > 0 || throw(ArgumentError("δ must be finite and > 0"))
    if scheme == :centered
        return θ - δ / 2, θ + δ / 2
    elseif scheme == :forward
        return θ, θ + δ
    end
    throw(ArgumentError("Unknown scheme=$scheme. Use :centered or :forward"))
end

@inline function _qfi_from_fidelity(
    F::AbstractVector,
    δ::Real;
    use_log_formula::Bool,
    fidelity_floor::Real,
)
    0 < fidelity_floor <= 1 || throw(ArgumentError("fidelity_floor must be in (0, 1]"))
    all(f -> isfinite(f) && -1e-10 <= real(f) <= 1 + 1e-10, F) ||
        throw(DomainError(F, "fidelity outside [0, 1] beyond roundoff; check initial state and integration accuracy"))
    return _qfi_from_logfidelity(log.(max.(real.(F), fidelity_floor)), δ;
        use_log_formula, fidelity_floor)
end

function _qfi_from_logfidelity(logF::AbstractVector, δ::Real;
    use_log_formula::Bool, fidelity_floor::Real)
    isfinite(δ) && δ > 0 || throw(ArgumentError("δ must be finite and > 0"))
    0 < fidelity_floor <= 1 || throw(ArgumentError("fidelity_floor must be in (0, 1]"))
    all(f -> !isnan(f) && f <= 1e-10, logF) ||
        throw(DomainError(logF, "log fidelity above zero beyond roundoff; check initial state and integration accuracy"))
    # Preserve small signed errors: clipping logF to zero can hide an error
    # that becomes substantial after division by δ². Material F>1 is rejected
    # above; callers can assess roundoff by varying δ and solver tolerances.
    logFsafe = max.(logF, log(fidelity_floor))
    if use_log_formula
        return (-8.0) .* logFsafe ./ (δ^2)
    end
    return (-8.0) .* expm1.(logFsafe) ./ (δ^2)
end

function _logfidelity_pair_timecourses(
    θ1,
    θ2;
    N::Int,
    tspan,
    σ0,
    d0,
    Hfun::Function,
    hfun::Function,
    Lfun::Function,
    solver,
    reltol::Real,
    abstol::Real,
    saveat,
)
    sol = solve_tsme_moments(
        θ1,
        θ2;
        N = N,
        tspan = tspan,
        σ0 = σ0,
        d0 = d0,
        ξ0 = 0.0 + 0.0im,
        Hfun = Hfun,
        hfun = hfun,
        Lfun = Lfun,
        solver = solver,
        reltol = reltol,
        abstol = abstol,
        saveat = saveat,
    )
    logF_env = [_log_trace_norm(get_σ_d_ξ(sol, N, i)...) for i in eachindex(sol.u)]
    logF_joint = [-real(u[end]) for u in sol.u]
    return collect(sol.t), logF_env, logF_joint
end

"""Compute TSME-QFI timecourses with stabilized finite differences.

Returns `(times, qfi_env, qfi_joint)`. By default this uses:
- centered parameter pair `(θ-ϵ/2, θ+ϵ/2)`
- log-fidelity formula `-8 log(F) / ϵ^2`
- Richardson extrapolation using `ϵ` and `ϵ/2`

The environment/joint interpretation requires a parameter-independent pure
initial system state. A mixed input instead includes a purifying reference.
Centered differences have O(ϵ²) error; forward differences have O(ϵ) error,
so their Richardson factors differ. Reduce solver tolerances when reducing ϵ:
fidelity errors are amplified by 1/ϵ².
"""
function qfi_timecourse_from_tsme_fidelity(
    θ;
    ϵ::Real,
    N::Int,
    tspan,
    σ0,
    d0,
    Hfun::Function,
    hfun::Function = (θ -> zeros(2N)),
    Lfun::Function,
    solver = Tsit5(),
    reltol::Real = 1e-9,
    abstol::Real = 1e-9,
    saveat = nothing,
    scheme::Symbol = :centered,
    use_log_formula::Bool = true,
    richardson::Bool = true,
    fidelity_floor::Real = 1e-300,
)
    ϵ > 0 || throw(ArgumentError("ϵ must be > 0"))

    θ1, θ2 = _theta_pair(θ, ϵ, scheme)
    times, logF_env_ϵ, logF_joint_ϵ = _logfidelity_pair_timecourses(
        θ1,
        θ2;
        N = N,
        tspan = tspan,
        σ0 = σ0,
        d0 = d0,
        Hfun = Hfun,
        hfun = hfun,
        Lfun = Lfun,
        solver = solver,
        reltol = reltol,
        abstol = abstol,
        saveat = saveat,
    )

    qfi_env_ϵ = _qfi_from_logfidelity(
        logF_env_ϵ,
        ϵ;
        use_log_formula = use_log_formula,
        fidelity_floor = fidelity_floor,
    )
    qfi_joint_ϵ = _qfi_from_logfidelity(
        logF_joint_ϵ,
        ϵ;
        use_log_formula = use_log_formula,
        fidelity_floor = fidelity_floor,
    )

    if richardson
        ϵ_half = ϵ / 2
        θ1_half, θ2_half = _theta_pair(θ, ϵ_half, scheme)
        saveat_half = (saveat === nothing) ? times : saveat
        _, logF_env_half, logF_joint_half = _logfidelity_pair_timecourses(
            θ1_half,
            θ2_half;
            N = N,
            tspan = tspan,
            σ0 = σ0,
            d0 = d0,
            Hfun = Hfun,
            hfun = hfun,
            Lfun = Lfun,
            solver = solver,
            reltol = reltol,
            abstol = abstol,
            saveat = saveat_half,
        )
        qfi_env_half = _qfi_from_logfidelity(
            logF_env_half,
            ϵ_half;
            use_log_formula = use_log_formula,
            fidelity_floor = fidelity_floor,
        )
        qfi_joint_half = _qfi_from_logfidelity(
            logF_joint_half,
            ϵ_half;
            use_log_formula = use_log_formula,
            fidelity_floor = fidelity_floor,
        )
        factor = scheme == :centered ? 4 : 2
        qfi_env = (factor .* qfi_env_half .- qfi_env_ϵ) ./ (factor - 1)
        qfi_joint = (factor .* qfi_joint_half .- qfi_joint_ϵ) ./ (factor - 1)
    else
        qfi_env = qfi_env_ϵ
        qfi_joint = qfi_joint_ϵ
    end

    return times, qfi_env, qfi_joint
end

"""Estimate terminal environment QFI for a parameter-independent pure initial state.

For a mixed input this includes a purifying reference. See
[`qfi_timecourse_from_tsme_fidelity`](@ref) for finite-difference conventions.
"""
function qfi_from_tsme_fidelity(θ; tspan, kwargs...)
    _, environment, _ = qfi_timecourse_from_tsme_fidelity(θ;
        tspan, saveat=[last(tspan)], kwargs...)
    return only(environment)
end

end
