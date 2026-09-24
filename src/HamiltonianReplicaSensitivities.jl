module HamiltonianReplicaSensitivities

using LinearAlgebra
using OrdinaryDiffEqTsit5: Tsit5
using SciMLBase: ODEProblem, solve, successful_retcode

import ..GaussianMoments: symplectic_Ω
import ..ReplicaCoeffs: build_replica_coeffs, replicate_initial_moments
import ..BargmannInvariants: _check_times, _binomial, _add_compensated!

export I_n_hamiltonian_timecourse

_trace_product(X, Q) = sum(X[i, j] * Q[j, i] for j in axes(X, 2), i in axes(X, 1))

# E is nonzero only in one replica block. Add EX + XEᵀ without building E.
function _add_local_drift!(destination, X, block, p)
    mul!(p.local_product, p.parameter_drift, view(X, block, :))
    view(destination, block, :) .+= p.local_product
    view(destination, :, block) .+= transpose(p.local_product)
    return nothing
end

function _sensitivity_rhs!(du, u, p, t)
    M, M2 = p.M, p.M^2
    σ = reshape(view(u, 1:M2), M, M)
    Y = reshape(view(u, M2+1:2M2), M, M)
    Z = reshape(view(u, 2M2+1:3M2), M, M)
    V = reshape(view(u, 3M2+1:4M2), M, M)
    dσ = reshape(view(du, 1:M2), M, M)
    dY = reshape(view(du, M2+1:2M2), M, M)
    dZ = reshape(view(du, 2M2+1:3M2), M, M)
    dV = reshape(view(du, 3M2+1:4M2), M, M)

    # σ̇ = Aσ + σAᵀ + σQσ + D. All covariance tangents are symmetric
    # under plain transpose, so forming each symmetric sum also preserves
    # that property to roundoff throughout the integration.
    mul!(p.work, σ, p.Q)
    p.drift .= p.A .+ 0.5 .* p.work
    mul!(p.product, p.drift, σ)
    dσ .= p.product .+ transpose(p.product) .+ p.D
    p.drift .= p.A .+ p.work

    # Y=∂ασ, Z=∂βσ, B=A+σQ:
    # Ẏ=BY+YBᵀ+Eσ+σEᵀ, with the analogous equation for Z and F.
    mul!(p.product, p.drift, Y)
    dY .= p.product .+ transpose(p.product)
    _add_local_drift!(dY, σ, p.first_block, p)
    mul!(p.product, p.drift, Z)
    dZ .= p.product .+ transpose(p.product)
    _add_local_drift!(dZ, σ, p.second_block, p)

    # V=∂α∂βσ:
    # V̇=BV+VBᵀ+EZ+ZEᵀ+FY+YFᵀ+YQZ+ZQY.
    mul!(p.product, p.drift, V)
    dV .= p.product .+ transpose(p.product)
    mul!(p.work, Y, p.Q)
    mul!(p.product, p.work, Z)
    dV .+= p.product .+ transpose(p.product)
    _add_local_drift!(dV, Z, p.first_block, p)
    _add_local_drift!(dV, Y, p.second_block, p)

    du[4M2+1] = -0.5 * _trace_product(σ, p.Q) - p.trace_constant
    du[4M2+2] = -0.5 * _trace_product(Y, p.Q)
    du[4M2+3] = -0.5 * _trace_product(Z, p.Q)
    du[4M2+4] = -0.5 * _trace_product(V, p.Q)
    return nothing
end

function _insertion_sensitivities(coefficients, σ0, parameter_drift, R, l;
                                   tspan, times, solver, reltol, abstol,
                                   sensitivity_abstol)
    dimension = size(parameter_drift, 1)
    M = dimension * R
    M2 = M^2
    σrep, _ = replicate_initial_moments(σ0, zeros(dimension), R)
    u0 = zeros(ComplexF64, 4M2 + 4)
    u0[1:M2] .= vec(σrep)
    second_site = R - l
    block(site) = ((site - 1) * dimension + 1):(site * dimension)
    p = (;
        M, coefficients.Q, coefficients.A, coefficients.D, parameter_drift,
        first_block = block(1), second_block = block(second_site),
        trace_constant = 0.5 * _trace_product(coefficients.W, coefficients.Ω),
        work = zeros(ComplexF64, M, M), drift = zeros(ComplexF64, M, M),
        product = zeros(ComplexF64, M, M),
        local_product = zeros(ComplexF64, dimension, M),
    )
    # These derivatives can be orders of magnitude smaller than σ. Giving
    # them σ's absolute tolerance loses the short-time information before
    # the alternating replica sum is formed.
    tolerances = fill(Float64(abstol), length(u0))
    tolerances[(4M2+2):end] .= sensitivity_abstol
    indices = collect((4M2+1):(4M2+4))
    sol = solve(ODEProblem(_sensitivity_rhs!, u0, tspan, p), solver;
        reltol, abstol = tolerances, saveat = times, save_idxs = indices,
        save_start = first(tspan) in times, save_end = last(tspan) in times)
    successful_retcode(sol) || error("Hamiltonian replica sensitivity integration failed with $(sol.retcode)")
    sol.t == times || error("replica sensitivity solver did not return the requested time grid")
    return sol.u
end

function _check_symmetric_real_matrix(matrix, dimension, name)
    matrix isa AbstractMatrix && size(matrix) == (dimension, dimension) ||
        throw(DimensionMismatch("$name must be $dimension × $dimension"))
    all(isfinite, matrix) && isreal(matrix) && issymmetric(matrix) ||
        throw(ArgumentError("$name must be finite, real, and symmetric"))
end

"""
    I_n_hamiltonian_timecourse(orders; H, dH, j, L=nothing, σ0_single,
                              tspan, times, reltol=2e-13, abstol=2e-14, ...)

Return `(times, values)` for the replica QFI approximants of an undriven,
zero-mean Gaussian system whose estimated parameter enters only its quadratic
Hamiltonian. `H` is the Hamiltonian matrix at the evaluation point, and `dH`
is its first parameter derivative. The monitored jump coefficients `j`,
unmonitored coefficients `L`, and initial covariance must be independent of
that parameter. Quadratures are interleaved `(x₁,p₁,…,xₙ,pₙ)`.

This evaluates the derivative-insertion formula (Eq. 30 of Yang et al., v2)
by integrating first and mixed covariance sensitivities at equal parameters.
It introduces no parameter finite-difference step. Only first derivatives of
`H` are needed because the two insertions act on different replicas.

For a vector `orders`, columns follow the requested order; for an integer,
return a vector. The largest order `n` requires up to `n+2` physical replicas.
The outer binomial sum still amplifies numerical integration errors at large
orders, so convergence with respect to tolerances must be checked.

`sensitivity_abstol` defaults to `min(abstol, 1e-22)` for the three logarithmic
trace derivatives; these may be much smaller than the covariance entries.
"""
function I_n_hamiltonian_timecourse(
    orders::AbstractVector{<:Integer};
    H, dH, j, L = nothing, σ0_single, tspan, times::AbstractVector,
    solver = Tsit5(), reltol::Real = 2e-13, abstol::Real = 2e-14,
    sensitivity_abstol::Real = min(abstol, 1e-22),
)
    isempty(orders) && throw(ArgumentError("orders must be non-empty"))
    all(>=(0), orders) || throw(ArgumentError("orders must be nonnegative"))
    H isa AbstractMatrix || throw(ArgumentError("H must be a matrix"))
    dimension = size(H, 1)
    dimension > 0 && iseven(dimension) || throw(ArgumentError("H must have positive even dimension"))
    for (matrix, name) in ((H, "H"), (dH, "dH"), (σ0_single, "σ0_single"))
        _check_symmetric_real_matrix(matrix, dimension, name)
    end
    for (tolerance, name) in ((reltol, "reltol"), (abstol, "abstol"),
                              (sensitivity_abstol, "sensitivity_abstol"))
        isfinite(tolerance) && tolerance > 0 || throw(ArgumentError("$name must be finite and positive"))
    end
    _check_times(times, tspan)

    model = (; H, h = zeros(dimension), j, L)
    parameter_drift = ComplexF64.(symplectic_Ω(dimension ÷ 2) * dH)
    logΛ = zeros(ComplexF64, length(times))
    total = zeros(ComplexF64, length(times), length(orders))
    correction = zeros(ComplexF64, size(total))
    for m in 0:maximum(orders)
        R = m + 2
        coefficients = build_replica_coeffs(fill(model, R))
        fm = zeros(ComplexF64, length(times))
        fm_correction = zero(fm)
        for l in 0:(m ÷ 2)
            values = _insertion_sensitivities(coefficients, σ0_single,
                parameter_drift, R, l;
                tspan, times, solver, reltol, abstol, sensitivity_abstol)
            if m == 0
                logΛ .= [log(2) - value[1] / 2 for value in values]
            end
            # log B=-ξ, so ∂α∂β B=B(ξα ξβ-ξαβ).
            normalized = [exp(-value[1] - (m + 1) * logΛ[i]) *
                          (value[2] * value[3] - value[4])
                          for (i, value) in enumerate(values)]
            weight = 2 * _binomial(m, l)
            l != m - l && (weight *= 2)
            _add_compensated!(fm, fm_correction, normalized, Float64(weight))
        end
        fm .+= fm_correction
        for (column, n) in enumerate(orders)
            m <= n || continue
            weight = Float64((-1)^m * _binomial(n + 1, m + 1))
            _add_compensated!(view(total, :, column), view(correction, :, column), fm, weight)
        end
    end
    return collect(times), total .+ correction
end

function I_n_hamiltonian_timecourse(n::Integer; kwargs...)
    times, values = I_n_hamiltonian_timecourse([n]; kwargs...)
    return times, vec(values)
end

end
