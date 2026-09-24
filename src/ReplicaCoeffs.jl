module ReplicaCoeffs

using LinearAlgebra

import ..GaussianMoments: symplectic_Ω

export blockdiag,
    sym,
    skew,
    getfield_nt,
    build_replica_coeffs,
    replica_params,
    build_models_from_theta,
    replicate_initial_moments

"""Access a field of a NamedTuple or model struct."""
getfield_nt(x, s::Symbol) = getproperty(x, s)

"""Dense block-diagonal concatenation, using the complex arithmetic of the solver."""
function blockdiag(mats::AbstractMatrix...)
    out = zeros(ComplexF64, sum(M -> size(M, 1), mats; init = 0),
                sum(M -> size(M, 2), mats; init = 0))
    row, col = 1, 1
    for M in mats
        nr, nc = size(M)
        out[row:(row + nr - 1), col:(col + nc - 1)] .= M
        row += nr
        col += nc
    end
    return out
end

"""Symmetric part with plain transpose: `(M + transpose(M))/2`."""
sym(M) = (M + transpose(M)) / 2

"""Skew part with plain transpose: `(M - transpose(M))/2`."""
skew(M) = (M - transpose(M)) / 2

function _jump_matrix(raw, dimension, label; allow_nothing = false)
    if raw === nothing
        allow_nothing || throw(ArgumentError("$label must be a vector or matrix"))
        return zeros(ComplexF64, 0, dimension)
    end
    # A vector denotes one channel; matrix rows denote separate channels.
    raw isa AbstractVecOrMat || throw(ArgumentError("$label must be a vector or matrix"))
    if size(raw) == (0, 0)
        return zeros(ComplexF64, 0, dimension)
    end
    matrix = raw isa AbstractVector ? reshape(ComplexF64.(raw), 1, :) : ComplexF64.(raw)
    size(matrix, 2) == dimension || throw(DimensionMismatch("$label must have $dimension columns"))
    all(isfinite, matrix) || throw(ArgumentError("$label must be finite"))
    return matrix
end

"""
    build_replica_coeffs(models; periodic=true)

Build Gaussian coefficients for Yang et al.'s replica master equation (Eq. 6,
https://arxiv.org/abs/2504.12400). Each model has fields `H`, `h`, `j`, and `L`.
`H` and `h` use the manuscript convention `Ĥ = rᵀ H r/2 + hᵀ Ω r`;
`j` and `L` contain jump coefficients in their rows. All replicas must have the
same phase-space dimension and the same number of monitored channels.

The phase-space coordinates are grouped by replica, with each replica using
`(q₁,p₁,q₂,p₂,…)`. Periodic boundaries include the closing link for *every*
replica count, including the self-link for one replica, which recovers a
trace-preserving master equation. `periodic=false` constructs an open chain;
its trace is not a waveguide Bargmann invariant.
"""
function build_replica_coeffs(models; periodic::Bool = true)
    isempty(models) && throw(ArgumentError("models must be non-empty"))
    R = length(models)
    first_H = getfield_nt(first(models), :H)
    first_H isa AbstractMatrix || throw(ArgumentError("H[1] must be a matrix"))
    dimension = size(first_H, 1)
    dimension > 0 && iseven(dimension) || throw(ArgumentError("H must have positive even dimension"))

    Hs = Matrix{ComplexF64}[]
    hs = Vector{ComplexF64}[]
    js = Matrix{ComplexF64}[]
    Ls = Matrix{ComplexF64}[]
    for (α, model) in enumerate(models)
        H, h = getfield_nt(model, :H), getfield_nt(model, :h)
        size(H) == (dimension, dimension) || throw(DimensionMismatch("H[$α] must be $dimension × $dimension"))
        h isa AbstractVector && length(h) == dimension ||
            throw(DimensionMismatch("h[$α] must be a vector of length $dimension"))
        all(isfinite, H) && all(isfinite, h) || throw(ArgumentError("H and h must be finite"))
        issymmetric(H) || throw(ArgumentError("H[$α] must be symmetric"))
        isreal(H) && isreal(h) || throw(ArgumentError("H and h must be real for a physical replica model"))
        push!(Hs, ComplexF64.(H))
        push!(hs, ComplexF64.(h))
        push!(js, _jump_matrix(getfield_nt(model, :j), dimension, "j[$α]"))
        push!(Ls, _jump_matrix(getfield_nt(model, :L), dimension, "L[$α]"; allow_nothing = true))
    end
    all(j -> size(j, 1) == size(first(js), 1), js) ||
        throw(DimensionMismatch("all replicas must have the same monitored channels"))

    H_big = blockdiag(Hs...)
    Ω_rep = blockdiag(ntuple(_ -> symplectic_Ω(dimension ÷ 2), R)...)
    M = dimension * R
    K, Aquad = zeros(ComplexF64, M, M), zeros(ComplexF64, M, M)
    block(α) = ((α - 1) * dimension + 1):(α * dimension)

    for α in 1:R
        Lquad, Jquad = Ls[α]' * Ls[α], js[α]' * js[α]
        # Local unmonitored recycling and both local loss anticommutators.
        K[block(α), block(α)] .= transpose(Lquad)
        Aquad[block(α), block(α)] .= -(Lquad + Jquad)
    end
    for α in 1:(periodic ? R : R - 1)
        β = mod1(α + 1, R)
        # J^(β) μ J^(α)†: the transpose is not a Hermitian adjoint.
        K[block(β), block(α)] .+= transpose(js[α]' * js[β])
    end

    Q = sym(K + Aquad)
    D = Ω_rep * sym(K - Aquad) * transpose(Ω_rep)
    W = 1im * skew(K - Aquad)
    A = Ω_rep * (H_big + 1im * skew(K))
    a = zeros(ComplexF64, M)
    c = -1im .* vcat(hs...)
    return (; Q, A, D, a, c, W, Ω = Ω_rep)
end

"""Build coefficients once and return a constant `params(t)` closure."""
function replica_params(models; periodic::Bool = true)
    coeffs = build_replica_coeffs(models; periodic = periodic)
    return _ -> coeffs
end

"""Build replica models from parameter samples and coefficient functions."""
function build_models_from_theta(θs, Hfun, hfun, jfun, Lfun = _ -> nothing)
    isempty(θs) && throw(ArgumentError("θs must be non-empty"))
    return [(H = Hfun(θ), h = hfun(θ), j = jfun(θ), L = Lfun(θ)) for θ in θs]
end

"""Build constant `params(t)` from parameter samples and coefficient functions."""
function replica_params(θs, Hfun, hfun, jfun, Lfun = _ -> nothing; periodic::Bool = true)
    return replica_params(build_models_from_theta(θs, Hfun, hfun, jfun, Lfun); periodic = periodic)
end

"""Replicate single-system covariance and means in replica-block order."""
function replicate_initial_moments(σ0_single::AbstractMatrix, d0_single::AbstractVector, R::Int)
    R >= 1 || throw(ArgumentError("R must be >= 1"))
    dimension = length(d0_single)
    dimension > 0 && iseven(dimension) || throw(ArgumentError("initial moments need positive even dimension"))
    size(σ0_single) == (dimension, dimension) || throw(DimensionMismatch("initial covariance and means disagree"))
    return blockdiag(ntuple(_ -> σ0_single, R)...), repeat(ComplexF64.(d0_single), R)
end

end
