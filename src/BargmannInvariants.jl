module BargmannInvariants

using OrdinaryDiffEqTsit5: Tsit5

import ..GaussianMoments: solve_gaussian_odes
import ..ReplicaCoeffs: replica_params, replicate_initial_moments

export log_bargmann_invariant,
    log_bargmann_invariant_timecourse,
    log_Lambda,
    log_Lambda_timecourse,
    D_lm,
    theta_list_G,
    mixed_d2_central,
    normalized_mixed_d2_G_lm_timecourse,
    f_m_normalized_timecourse,
    I_n_timecourse,
    normalized_mixed_d2_G_lm,
    f_m_normalized,
    theta_list_eq20_term,
    f_m_normalized_eq20_fd,
    I_n

function _check_times(times, tspan)
    isempty(times) && throw(ArgumentError("times must be non-empty"))
    length(tspan) == 2 && all(isfinite, tspan) && tspan[1] <= tspan[2] ||
        throw(ArgumentError("tspan must contain two finite, increasing times"))
    all(isfinite, times) && all(t -> tspan[1] <= t <= tspan[2], times) ||
        throw(ArgumentError("times must be finite and inside tspan"))
    all(diff(times) .> 0) || throw(ArgumentError("times must be strictly increasing"))
end

function _check_step(θ, h, fd_method)
    isfinite(h) && h > 0 || throw(ArgumentError("h must be finite and positive"))
    θ isa Real && isfinite(θ) || throw(ArgumentError("θ must be finite and real"))
    fd_method in (:central, :richardson) || throw(ArgumentError("fd_method must be :central or :richardson"))
    smallest_step = fd_method == :richardson ? h / 2 : h
    θ - smallest_step < θ < θ + smallest_step ||
        throw(ArgumentError("h is too small to resolve distinct parameter samples"))
    isfinite(θ - h) && isfinite(θ + h) || throw(ArgumentError("finite-difference samples must be finite"))
end

"""
    log_bargmann_invariant_timecourse(Θ; N, tspan, times, σ0_single, d0_single,
                                     Hfun, hfun, jfun, Lfun = _ -> nothing, ...)

Return `(times, logB)` for the waveguide Bargmann invariant
`B(Θ,t) = tr[Ξ(Θ[1],t) ⋯ Ξ(Θ[end],t)]`. The replica operator initially is a
product of the supplied, parameter-independent single-system state. Its trace
is `exp(-ξ)`, so `logB=-ξ` follows the continuous logarithm without branch jumps.
Periodic boundaries are required by Yang et al., Eq. (6).
"""
function log_bargmann_invariant_timecourse(
    Θ;
    N::Int,
    tspan,
    times::AbstractVector,
    σ0_single,
    d0_single,
    Hfun,
    hfun,
    jfun,
    Lfun = _ -> nothing,
    periodic::Bool = true,
    ξ0::Number = 0.0 + 0.0im,
    solver = Tsit5(),
    reltol::Real = 1e-9,
    abstol::Real = 1e-9,
)
    isempty(Θ) && throw(ArgumentError("Θ must be non-empty"))
    N > 0 || throw(ArgumentError("N must be positive"))
    periodic || throw(ArgumentError("Bargmann invariants require periodic replica boundaries"))
    size(σ0_single) == (2N, 2N) && length(d0_single) == 2N ||
        throw(DimensionMismatch("single-system moments must match N=$N"))
    _check_times(times, tspan)
    params = replica_params(Θ, Hfun, hfun, jfun, Lfun)
    size(params(tspan[1]).Q, 1) == 2N * length(Θ) ||
        throw(DimensionMismatch("model coefficients must match N=$N"))
    σ0, d0 = replicate_initial_moments(σ0_single, d0_single, length(Θ))
    sol = solve_gaussian_odes(;
        N = N * length(Θ), tspan, σ0, d0, ξ0 = complex(ξ0), params,
        solver, reltol, abstol, saveat = times,
    )
    sol.t == times || error("replica solver did not return the requested time grid")
    return collect(sol.t), [-ComplexF64(u[end]) for u in sol.u]
end

"""Return `log B(Θ,T)` at the end of `tspan`; see `log_bargmann_invariant_timecourse`."""
function log_bargmann_invariant(Θ; tspan, kwargs...)
    _, values = log_bargmann_invariant_timecourse(Θ; tspan, times = [last(tspan)], kwargs...)
    return only(values)
end

"""Return `log Λ(θ,t) = log(2) + log B([θ,θ],t)/2` on `times`."""
function log_Lambda_timecourse(θ; kwargs...)
    times, logpurity = log_bargmann_invariant_timecourse([θ, θ]; kwargs...)
    return times, log(2) .+ logpurity ./ 2
end

"""Return `log Λ(θ,T)`, where `Λ = 2 sqrt(tr[Ξ(θ,T)^2])`."""
log_Lambda(θ; kwargs...) = log(2) + log_bargmann_invariant([θ, θ]; kwargs...) / 2

# BigInt arithmetic prevents overflow in the binomial coefficients themselves.
# The ODEs still use Float64; this does not remove ill-conditioning at large n.
_binomial(m, l) = 0 <= l <= m ? binomial(big(m), l) : big(0)

"""Coefficient `D_l^m = 2 binomial(m,l) - binomial(m,l+1) - binomial(m,l-1)` (Eq. 5b)."""
function D_lm(m::Int, l::Int)
    m >= 0 || throw(ArgumentError("m must be nonnegative"))
    return 2 * _binomial(m, l) - _binomial(m, l + 1) - _binomial(m, l - 1)
end

"""Parameter list for `G_{l,m}(μ,ν) = tr[Ξ(μ)^(l+1) Ξ(ν)^(m-l+1)]`."""
function theta_list_G(μ, ν, m::Int, l::Int)
    0 <= l <= m || throw(ArgumentError("require 0 <= l <= m"))
    μp, νp = promote(μ, ν)
    return vcat(fill(μp, l + 1), fill(νp, m - l + 1))
end

"""
Parameter list for `tr[Ξ(θA) Ξ(θ)^(m-l) Ξ(θB) Ξ(θ)^l]`.
The historical function name refers to Eq. (20) of arXiv:2504.12400v1;
the same derivative-insertion formula is Eq. (30) in v2.
"""
function theta_list_eq20_term(θA, θ, θB, m::Int, l::Int)
    0 <= l <= m || throw(ArgumentError("require 0 <= l <= m"))
    A, center, B = promote(θA, θ, θB)
    return vcat(A, fill(center, m - l), B, fill(center, l))
end

"""Central mixed derivative at `(θ,θ)`, optionally extrapolated to fourth order."""
function mixed_d2_central(G, θ; h::Real, richardson::Bool = false)
    _check_step(θ, h, richardson ? :richardson : :central)
    d2(s) = (G(θ + s, θ + s) - G(θ + s, θ - s) -
             G(θ - s, θ + s) + G(θ - s, θ - s)) / (4s^2)
    coarse = d2(h)
    return richardson ? (4 * d2(h / 2) - coarse) / 3 : coarse
end

function _mixed_stencil_from_logs(logpp, logpm, logmp, logmm, step::Real;
                                  log_rescale::Bool = true, log_normalization = 0)
    if !log_rescale
        return (exp(logpp - log_normalization) - exp(logpm - log_normalization) -
                exp(logmp - log_normalization) + exp(logmm - log_normalization)) / (4step^2)
    end
    # Factoring a common exponential controls scale. expm1 also avoids losing
    # the small differences by first rounding four exponentials near unity.
    reference = (logpp, logpm, logmp, logmm)[argmax(real.((logpp, logpm, logmp, logmm)))]
    numerator = expm1(logpp - reference) - expm1(logpm - reference) -
                expm1(logmp - reference) + expm1(logmm - reference)
    # Subtract normalization only after taking log differences: subtracting
    # (m+1)logΛ from each log first destroys small differences at large m.
    return exp(reference - log_normalization) * numerator / (4step^2)
end

# Each context contains immutable snapshots of the numerical inputs. A caller
# reusing a cache across model functions, grids or tolerances cannot get stale
# results. Captured mutable state inside a function must not change during use.
_cache_input(x::AbstractArray) = (size(x), Tuple(x))
_cache_input(x) = x

function _cyclic_key(Θ)
    parameters = Tuple(Θ)
    all(x -> x isa Real, parameters) || return parameters
    return minimum(ntuple(i -> parameters[mod1(i + offset, length(parameters))], length(parameters))
                   for offset in 0:(length(parameters) - 1))
end

function _log_trace_reader(times; cache, kwargs...)
    context = (Tuple(times), map(_cache_input, (; kwargs...)))
    return Θ -> get!(cache, (:bargmann, context, _cyclic_key(Θ))) do
        last(log_bargmann_invariant_timecourse(Θ; times, kwargs...))
    end
end

function _mixed_timecourse(logtrace, make_list, θ, m, logΛ, h, fd_method, log_rescale)
    function stencil(step)
        plus, minus = θ + step, θ - step
        pp, pm, mp, mm = (logtrace(make_list(a, b)) for (a, b) in
                         ((plus, plus), (plus, minus), (minus, plus), (minus, minus)))
        return [_mixed_stencil_from_logs(pp[i], pm[i], mp[i], mm[i], step;
                    log_rescale, log_normalization = (m + 1) * logΛ[i])
                for i in eachindex(logΛ)]
    end
    coarse = stencil(h)
    return fd_method == :richardson ? (4 .* stencil(h / 2) .- coarse) ./ 3 : coarse
end

# Neumaier summation retains low-order bits in alternating binomial sums.
function _add_compensated!(total, correction, values, weight)
    for i in eachindex(total)
        value = weight * values[i]
        updated = total[i] + value
        correction[i] += abs(total[i]) >= abs(value) ?
            (total[i] - updated) + value : (value - updated) + total[i]
        total[i] = updated
    end
end

function _f_timecourse(logtrace, θ, m, logΛ, h, method, fd_method, log_rescale)
    total, correction = zeros(ComplexF64, length(logΛ)), zeros(ComplexF64, length(logΛ))
    # Cyclicity makes the complete mixed stencil invariant under l ↔ m-l.
    for l in 0:(m ÷ 2)
        weight = method == :mixed ? D_lm(m, l) : 2 * _binomial(m, l)
        iszero(weight) && continue
        l != m - l && (weight *= 2)
        make_list = method == :mixed ?
            ((a, b) -> theta_list_G(a, b, m, l)) :
            ((a, b) -> theta_list_eq20_term(a, θ, b, m, l))
        values = _mixed_timecourse(logtrace, make_list, θ, m, logΛ, h, fd_method, log_rescale)
        _add_compensated!(total, correction, values, Float64(weight))
    end
    return total .+ correction
end

"""Normalized mixed derivative of `G_{l,m}` on the requested time grid."""
function normalized_mixed_d2_G_lm_timecourse(
    θ, m::Int, l::Int;
    h::Real, times::AbstractVector, logΛθ_vec::AbstractVector,
    cache::AbstractDict = Dict(), fd_method::Symbol = :richardson,
    log_rescale::Bool = true, kwargs...,
)
    0 <= l <= m || throw(ArgumentError("require 0 <= l <= m"))
    length(logΛθ_vec) == length(times) || throw(DimensionMismatch("logΛθ_vec must match times"))
    _check_step(θ, h, fd_method)
    logtrace = _log_trace_reader(times; cache, kwargs...)
    return _mixed_timecourse(logtrace, (a, b) -> theta_list_G(a, b, m, l),
                            θ, m, logΛθ_vec, h, fd_method, log_rescale)
end

"""Scalar counterpart of `normalized_mixed_d2_G_lm_timecourse`."""
function normalized_mixed_d2_G_lm(θ, m::Int, l::Int; tspan, logΛθ::Number, kwargs...)
    return only(normalized_mixed_d2_G_lm_timecourse(θ, m, l;
                tspan, times = [last(tspan)], logΛθ_vec = [logΛθ], kwargs...))
end

"""
Return `(times, f_m/Λ^(m+1))` using Eq. (5b) (`method=:mixed`) or the
independent derivative-insertion formula (`method=:eq20`, Eq. 30 in v2).
Both methods support `fd_method=:central` and `:richardson`.
"""
function f_m_normalized_timecourse(
    θ, m::Int;
    h::Real, times::AbstractVector, cache::AbstractDict = Dict(),
    method::Symbol = :mixed, fd_method::Symbol = :richardson,
    log_rescale::Bool = true, kwargs...,
)
    m >= 0 || throw(ArgumentError("m must be nonnegative"))
    method in (:mixed, :eq20) || throw(ArgumentError("method must be :mixed or :eq20"))
    _check_step(θ, h, fd_method)
    logtrace = _log_trace_reader(times; cache, kwargs...)
    logΛ = log(2) .+ logtrace([θ, θ]) ./ 2
    return times, _f_timecourse(logtrace, θ, m, logΛ, h, method, fd_method, log_rescale)
end

"""Return normalized `f_m(θ,T)/Λ(θ,T)^(m+1)` using Eq. (5b)."""
function f_m_normalized(θ, m::Int; tspan, kwargs...)
    return only(last(f_m_normalized_timecourse(θ, m; tspan, times = [last(tspan)], kwargs...)))
end

"""Evaluate `f_m/Λ^(m+1)` by the independent derivative-insertion formula."""
function f_m_normalized_eq20_fd(θ, m::Int; kwargs...)
    return f_m_normalized(θ, m; method = :eq20, kwargs...)
end

"""
    I_n_timecourse(θ, n; h, times, ...)
    I_n_timecourse(θ, orders; h, times, ...)

Evaluate Yang et al.'s QFI approximants, Eq. (5a). For an integer `n`, return
`(times, values)`. For a vector `orders`, return `(times, values)` with one
matrix column per requested order. The latter computes each `f_m` once and
shares the purity calculation and invariant cache across all orders.
An optional `cache=Dict()` can also share invariant solves between calls, for
example when halving a Richardson step during a convergence check.

Scalar and timecourse APIs both default to `method=:mixed` and fourth-order
Richardson differences. `method=:eq20` provides an independent check with
less cancellation inside `f_m`. The outer alternating binomial sum can still
amplify ODE and finite-difference errors severely at large order. Validate
results by varying `h`, tightening the ODE tolerances and comparing methods;
compensated summation alone cannot establish convergence of an `n=20` curve.
"""
function I_n_timecourse(
    θ, orders::AbstractVector{<:Integer};
    h::Real, times::AbstractVector, method::Symbol = :mixed,
    fd_method::Symbol = :richardson, log_rescale::Bool = true,
    cache::AbstractDict = Dict(), kwargs...,
)
    isempty(orders) && throw(ArgumentError("orders must be non-empty"))
    all(>=(0), orders) || throw(ArgumentError("orders must be nonnegative"))
    method in (:mixed, :eq20) || throw(ArgumentError("method must be :mixed or :eq20"))
    _check_step(θ, h, fd_method)
    logtrace = _log_trace_reader(times; cache, kwargs...)
    logΛ = log(2) .+ logtrace([θ, θ]) ./ 2
    total = zeros(ComplexF64, length(times), length(orders))
    correction = similar(total)
    fill!(correction, 0)
    for m in 0:maximum(orders)
        fm = _f_timecourse(logtrace, θ, m, logΛ, h, method, fd_method, log_rescale)
        for (column, n) in enumerate(orders)
            m <= n || continue
            weight = Float64((-1)^m * _binomial(n + 1, m + 1))
            _add_compensated!(view(total, :, column), view(correction, :, column), fm, weight)
        end
    end
    return times, total .+ correction
end

function I_n_timecourse(θ, n::Int; kwargs...)
    times, values = I_n_timecourse(θ, [n]; kwargs...)
    return times, vec(values)
end

"""Return the terminal approximant `I_n`; options match `I_n_timecourse`."""
function I_n(θ, n::Int; tspan, kwargs...)
    return only(last(I_n_timecourse(θ, n; tspan, times = [last(tspan)], kwargs...)))
end

end
