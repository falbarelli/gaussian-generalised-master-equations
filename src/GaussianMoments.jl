module GaussianMoments

using LinearAlgebra
using OrdinaryDiffEqTsit5: Tsit5
using SciMLBase: ODEProblem, solve, successful_retcode

export PackSpec,
    unpack_state,
    pack_deriv!,
    symplectic_Ω,
    gaussian_rhs_general!,
    solve_gaussian_odes,
    get_σ_d_ξ,
    trace_norm_gaussian_nu

"""Packing layout for y = [vec(σ); d; ξ] with phase-space size M."""
struct PackSpec
    M::Int
    nσ::Int
    nd::Int
    nξ::Int
end

"""Build a packing specification for a given phase-space dimension M."""
function PackSpec(M::Int)
    M > 0 && iseven(M) || throw(ArgumentError("phase-space dimension must be positive and even"))
    nσ = M * M
    nd = M
    nξ = 1
    return PackSpec(M, nσ, nd, nξ)
end

"""Unpack state vector y into (σ, d, ξ) views/scalar using PackSpec."""
@inline function unpack_state(y::AbstractVector, ps::PackSpec)
    M = ps.M
    σvec = @view y[1:ps.nσ]
    d = @view y[(ps.nσ + 1):(ps.nσ + ps.nd)]
    ξ = y[ps.nσ + ps.nd + 1]
    σ = reshape(σvec, M, M)
    return σ, d, ξ
end

"""Pack derivatives (dσ, dd, dξ) into the flattened derivative vector dy."""
@inline function pack_deriv!(
    dy::AbstractVector,
    dσ::AbstractMatrix,
    dd::AbstractVector,
    dξ::Number,
    ps::PackSpec,
)
    dy[1:ps.nσ] .= vec(dσ)
    dy[(ps.nσ + 1):(ps.nσ + ps.nd)] .= dd
    dy[ps.nσ + ps.nd + 1] = dξ
    return nothing
end

"""Return Ω for the interleaved quadratures `(x₁,p₁,…,xₙ,pₙ)` used in the paper."""
function symplectic_Ω(N::Int)
    N > 0 || throw(ArgumentError("N must be positive"))
    return kron(Matrix{Float64}(I, N, N), [0.0 1.0; -1.0 0.0])
end

"""General Gaussian moment RHS with plain transpose convention."""
function gaussian_rhs_general!(dy, y, p, t)
    ps = p.ps
    M = ps.M
    σ = reshape(@view(y[1:ps.nσ]), M, M)
    d = @view(y[(ps.nσ + 1):(ps.nσ + ps.nd)])

    par = p.params(t)
    Q, A, D, a, c, W, Ω = par.Q, par.A, par.D, par.a, par.c, par.W, par.Ω

    σQ = σ * Q
    drift = A + σQ
    dσ = drift * σ + σ * transpose(A) + D
    dd = drift * d - σ * (Ω * a) + 1im * c

    dξ = -transpose(d) * (Q * d)
    dξ -= 0.5 * tr(σQ)
    dξ -= 0.5 * sum(W .* transpose(Ω))
    dξ -= 2.0 * transpose(a) * (Ω * d)

    pack_deriv!(dy, dσ, dd, dξ, ps)
    return nothing
end

"""
    solve_gaussian_odes(; N, tspan, σ0, d0, params, ξ0=0,
                        solver=Tsit5(), reltol=1e-9, abstol=1e-9, saveat=nothing)

Integrate the general Gaussian moment equations for `N` bosonic modes.
`params(t)` returns a named tuple containing matrices `Q`, `A`, `D`, `W`
of size `(2N, 2N)` and vectors `a`, `c` of length `2N`. Coefficients may be
complex and time dependent. An optional `Ω` overrides the standard symplectic
matrix for interleaved quadratures `(x₁,p₁,…,xₙ,pₙ)`.

`σ0` is the covariance (vacuum = identity), `d0` the first moments, and `ξ0`
the initial negative log trace. Covariances are symmetric under ordinary
transpose, not necessarily Hermitian. The trace is `exp(-ξ)`.

Return a SciML ODE solution. Read the saved times as `sol.t` and retrieve
`(σ, d, ξ)` at saved index `i` with `get_σ_d_ξ(sol, N, i)`. `saveat` can be a
time grid or a sampling interval; without it the solver chooses saved steps.
The state is stored in `ComplexF64`. Failed integrations raise an error.

See `examples/general_odes.jl` for a standalone, time-dependent example.
"""
function solve_gaussian_odes(
    ;
    N::Int,
    tspan,
    σ0,
    d0,
    ξ0::Number = 0.0,
    params::Function,
    solver = Tsit5(),
    reltol = 1e-9,
    abstol = 1e-9,
    saveat = nothing,
)
    M = 2N
    ps = PackSpec(M)
    size(σ0) == (M, M) || throw(DimensionMismatch("σ0 must have size ($M, $M)"))
    length(d0) == M || throw(DimensionMismatch("d0 must have length $M"))

    y0 = Vector{ComplexF64}(undef, ps.nσ + ps.nd + ps.nξ)
    y0[1:ps.nσ] .= ComplexF64.(vec(σ0))
    y0[(ps.nσ + 1):(ps.nσ + ps.nd)] .= ComplexF64.(d0)
    y0[end] = ξ0

    Ω = symplectic_Ω(N)
    params_wrapped = (t) -> begin
        nt = params(t)
        haskey(nt, :Ω) ? nt : merge(nt, (; Ω = Ω))
    end

    p = (; ps = ps, params = params_wrapped)
    prob = ODEProblem(gaussian_rhs_general!, y0, tspan, p)

    sol = if saveat === nothing
        solve(prob, solver; reltol = reltol, abstol = abstol)
    elseif saveat isa AbstractVector
        isempty(saveat) && throw(ArgumentError("saveat must be nonempty"))
        solve(prob, solver; reltol, abstol, saveat,
            save_start = first(tspan) in saveat, save_end = last(tspan) in saveat)
    else
        solve(prob, solver; reltol = reltol, abstol = abstol, saveat = saveat)
    end
    successful_retcode(sol) || error("Gaussian moment integration failed with $(sol.retcode)")
    return sol
end

"""Extract (σ,d,ξ) from solution `sol` at index `idx` for N modes."""
function get_σ_d_ξ(sol, N::Int, idx::Int)
    M = 2N
    ps = PackSpec(M)
    y = sol.u[idx]
    σ, d, ξ = unpack_state(y, ps)
    return σ, d, ξ
end

"""Compute ‖ν‖₁ for a unit-trace Gaussian operator in interleaved quadrature order.

The real part of the symmetric covariance must be positive definite. `atol`
controls roundoff near a unit symplectic eigenvalue (a pure Gaussian mode).
"""
function trace_norm_gaussian_nu(
    σ::AbstractMatrix,
    r̄::AbstractVector;
    atol::Real = 1e-12,
)
    return exp(log_trace_norm_gaussian_nu(σ, r̄; atol))
end

# Evaluate the logarithm first so normalization and trace-norm factors can cancel
# without separately overflowing/underflowing in the TSME fidelity.
function log_trace_norm_gaussian_nu(σ::AbstractMatrix, r̄::AbstractVector; atol::Real = 1e-12)
    m, ncol = size(σ)
    m == ncol || throw(DimensionMismatch("σ must be square"))
    m > 0 && iseven(m) || throw(ArgumentError("σ dimension must be positive and even"))
    length(r̄) == m || throw(DimensionMismatch("r̄ must have length matching σ"))
    isfinite(atol) && atol >= 0 || throw(ArgumentError("atol must be finite and nonnegative"))
    all(isfinite, σ) && all(isfinite, r̄) || throw(ArgumentError("Gaussian moments must be finite"))
    isapprox(σ, transpose(σ); atol, rtol = atol) || throw(ArgumentError("σ must be symmetric under plain transpose"))

    n = div(m, 2)

    Ω = symplectic_Ω(n)

    R = real.(σ)
    N = imag.(σ)
    b = imag.(r̄)

    R = 0.5 * (R + transpose(R))
    N = 0.5 * (N + transpose(N))

    cholR = cholesky(Symmetric(R))

    logdetR = 2.0 * sum(log, diag(cholR.U))
    quad = dot(b, cholR \ b)

    if n == 1
        # For one mode, νK²-1 = |det(σ)-1|² / (4det(R)). Computing νK
        # first loses precision near a pure mode: acosh has an infinite slope
        # at 1. The equivalent asinh expression remains accurate there and
        # does not require rounding a nearly unit symplectic eigenvalue.
        # Fused products retain the small determinant excess before adding 1;
        # this also avoids Cholesky-log roundoff at a nearly pure covariance.
        detR_minus_one = fma(-R[1, 2], R[1, 2], fma(R[1, 1], R[2, 2], -1))
        if abs(detR_minus_one) < 0.5
            logdetR = log1p(detR_minus_one)
        end
        real_detσ_minus_one = fma(N[1, 2], N[1, 2],
            fma(-N[1, 1], N[2, 2], detR_minus_one))
        imag_detσ = fma(R[1, 1], N[2, 2],
            fma(R[2, 2], N[1, 1], -2R[1, 2] * N[1, 2]))
        mixedness = hypot(real_detσ_minus_one, imag_detσ) / (2exp(0.5 * logdetR))
        return -0.25 * logdetR + quad + 0.5 * asinh(mixedness)
    end

    A = N + Ω
    σK = 0.5 * (R + A * (cholR \ transpose(A)))
    σK = 0.5 * (σK + transpose(σK))

    # ΩσK is similar to LᵀΩL when σK = LLᵀ. The latter is skew-symmetric,
    # hence iLᵀΩL is Hermitian with eigenvalues ±νₖ. In contrast, -(ΩσK)^2
    # is generally NOT symmetric: symmetrizing it changes its eigenvalues.
    L = cholesky(Symmetric(σK)).L
    spectrum = eigvals(Hermitian(1im .* (transpose(L) * Ω * L)))
    ν = spectrum[(n + 1):end]
    minimum(ν) >= 1 - atol || throw(DomainError(ν, "unphysical symplectic spectrum for ν†ν"))
    ν = map(v -> abs(v - 1) <= atol ? 1.0 : v, ν)
    return -0.25 * logdetR + quad + 0.5 * sum(acosh, ν)
end

end
