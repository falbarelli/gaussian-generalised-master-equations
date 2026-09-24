module UnravellingQFI

using LinearAlgebra
using Random
using Printf

export get_unravelling_matrices,
    solve_unravelling,
    simulate_unravelling

"""
    get_unravelling_matrices(ω, χ, κ, η, ϕ, meas)

Return `(A, D, B, ∂ωA)` for a single-mode OPO with vacuum covariance `I`.
`meas` is `"hom"` or `"het"`; homodyne measures the quadrature
`x*cos(ϕ) - p*sin(ϕ)`. The innovation convention is
`dy = -√2 B' r dt + dW`, and the filter noise matrix is `B - σ*B`.
"""
function get_unravelling_matrices(ω, χ, κ, η, ϕ, meas::AbstractString)
    all(isfinite, (ω, χ, κ, η, ϕ)) || throw(ArgumentError("model parameters must be finite"))
    κ >= 0 || throw(ArgumentError("κ must be nonnegative"))
    0 <= η <= 1 || throw(ArgumentError("η must lie in [0, 1]"))

    A = [-χ-κ/2 ω; -ω χ-κ/2]
    D = κ * Matrix{Float64}(I, 2, 2)
    ∂A = [0.0 1.0; -1.0 0.0]

    if meas == "hom"
        s = sqrt(η * κ)
        cϕ = cos(ϕ)
        sϕ = sin(ϕ)
        B = [
            -s * cϕ^2 s * cϕ * sϕ
            s * cϕ * sϕ -s * sϕ^2
        ]
    elseif meas == "het"
        s = sqrt(η * κ / 2)
        B = [
            -s 0.0
            0.0 -s
        ]
    else
        throw(ArgumentError("meas must be \"hom\" or \"het\""))
    end

    return A, D, B, ∂A
end

@inline function _format_seconds(total_seconds::Real)
    s = max(0, round(Int, total_seconds))
    h = s ÷ 3600
    m = (s % 3600) ÷ 60
    sec = s % 60
    return @sprintf("%02d:%02d:%02d", h, m, sec)
end

function _print_progress(prefix::AbstractString, i::Int, n::Int, t0::Float64)
    width = 32
    frac = i / n
    nfill = clamp(round(Int, frac * width), 0, width)
    bar = repeat("#", nfill) * repeat("-", width - nfill)
    elapsed = time() - t0
    eta = i > 0 ? elapsed * (n - i) / i : 0.0
    msg = @sprintf(
        "\r%s [%s] %6.2f%% (%d/%d) elapsed=%s eta=%s",
        prefix,
        bar,
        100 * frac,
        i,
        n,
        _format_seconds(elapsed),
        _format_seconds(eta),
    )
    print(stderr, msg)
    flush(stderr)
    if i == n
        println(stderr)
    end
end

# The covariance and its frequency derivative are deterministic: they need only
# one integration, shared by every trajectory. See Fallani et al. (2022), App. B,
# Eqs. (26), (43): https://arxiv.org/abs/2110.15080.
function _covariance_rhs(σ, ∂σ, A, D, B, ∂A)
    M = B - σ * B
    ∂σB = ∂σ * B
    dσ = A * σ + σ * transpose(A) + D - M * transpose(M)
    d∂σ = ∂A * σ + σ * transpose(∂A) + A * ∂σ + ∂σ * transpose(A) +
          ∂σB * transpose(M) + M * transpose(∂σB)
    return dσ, d∂σ
end

function _covariance_step(σ, ∂σ, A, D, B, ∂A, dt)
    k1, l1 = _covariance_rhs(σ, ∂σ, A, D, B, ∂A)
    k2, l2 = _covariance_rhs(σ + (dt/2)*k1, ∂σ + (dt/2)*l1, A, D, B, ∂A)
    k3, l3 = _covariance_rhs(σ + (dt/2)*k2, ∂σ + (dt/2)*l2, A, D, B, ∂A)
    k4, l4 = _covariance_rhs(σ + dt*k3, ∂σ + dt*l3, A, D, B, ∂A)
    σnext = σ + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    ∂σnext = ∂σ + (dt/6) * (l1 + 2*l2 + 2*l3 + l4)
    return (σnext + transpose(σnext))/2, (∂σnext + transpose(∂σnext))/2
end

# Covariance contribution to single-mode Gaussian QFI. The usual purity formula
# is rewritten in terms of det(σ) to avoid subtracting two nearly equal purities.
# A pure family has ∂ω purity = 0 and no purity-derivative contribution; adding an
# arbitrary epsilon to its singular denominator biases nearby mixed states.
function _covariance_qfi(σ, ∂σ; pure::Bool = false)
    determinant = det(σ)
    tolerance = 1e-9
    if !isfinite(determinant) || σ[1, 1] <= 0 || determinant < 1 - tolerance
        throw(DomainError(determinant, "nonphysical covariance; decrease dt"))
    end
    pure && abs(determinant - 1) > tolerance &&
        throw(DomainError(determinant, "pure-state covariance lost purity; decrease dt"))
    invσ = inv(σ)
    X = invσ * ∂σ
    if pure || determinant <= 1
        abs(tr(X)) <= 1e-7 * max(1, norm(X)) ||
            throw(DomainError(tr(X), "inconsistent pure-state derivative; decrease dt"))
        return tr(X * X) / 4, invσ
    end
    covariance_term = tr(X * X) * determinant / (2 * (determinant + 1))
    purity_term = tr(X)^2 * determinant / (2 * (determinant - 1) * (determinant + 1))
    return covariance_term + purity_term, invσ
end

function _unravelling_times(nth, tFinal, dt)
    isfinite(nth) && nth >= 0 || throw(ArgumentError("nth must be finite and nonnegative"))
    isfinite(tFinal) && tFinal >= 0 || throw(ArgumentError("tFinal must be finite and nonnegative"))
    isfinite(dt) && dt > 0 || throw(ArgumentError("dt must be finite and positive"))
    step_ratio = Float64(tFinal / dt)
    nearest_step = round(step_ratio)
    Nstep = isapprox(step_ratio, nearest_step; rtol = 8eps(Float64), atol = 0) ?
            Int(nearest_step) : floor(Int, step_ratio)
    return Float64(dt) .* collect(0:Nstep)
end

# Deterministic moment closure of the fixed-record filter sensitivity:
# z = [r; ∂ωr], dz = F*z*dt + G*dW, V = E[z*z']. The means are zero
# for the frequency-independent thermal initial state used by this API.
# This is the block-matrix form of Sec. IV.A, Eqs. (S11)-(S12), in
# https://arxiv.org/abs/2511.22248 (our B includes −√(ηκ)).
function _ensemble_rhs(u, A, D, B, ∂A)
    σ, ∂σ, V, _ = u
    dσ, d∂σ = _covariance_rhs(σ, ∂σ, A, D, B, ∂A)
    M = B - σ * B
    F = [A zeros(2, 2); ∂A A + M * transpose(B)]
    G = [M; -∂σ * B] / sqrt(2)
    dV = F * V + V * transpose(F) + G * transpose(G)
    P = view(V, 3:4, 3:4)
    dFs = 2tr(transpose(B) * P * B)
    return dσ, d∂σ, dV, dFs
end

function _ensemble_step(u, A, D, B, ∂A, dt)
    k1 = _ensemble_rhs(u, A, D, B, ∂A)
    k2 = _ensemble_rhs(map((v, k) -> v + (dt/2)*k, u, k1), A, D, B, ∂A)
    k3 = _ensemble_rhs(map((v, k) -> v + (dt/2)*k, u, k2), A, D, B, ∂A)
    k4 = _ensemble_rhs(map((v, k) -> v + dt*k, u, k3), A, D, B, ∂A)
    σ, ∂σ, V, Fs = map((v, a, b, c, d) -> v + (dt/6)*(a + 2b + 2c + d),
                       u, k1, k2, k3, k4)
    return (σ + transpose(σ))/2, (∂σ + transpose(∂σ))/2,
           (V + transpose(V))/2, Fs
end

"""
    solve_unravelling(ω, χ, κ, η, ϕ, meas, nth, tFinal, dt; kwargs...)

Compute the signal CFI `Fs`, ensemble-averaged conditional-state QFI `Qc`, and
their sum `Qu` deterministically, returning `(times, Fs, Qc, Qu)`. Parameters,
initial state, and output grid match [`simulate_unravelling`](@ref), without
the `Ntraj` argument. Both `"hom"` and `"het"` are supported.

The conditional covariance `σ` and its fixed-record frequency derivative are
deterministic. The augmented mean `z = [r; ∂ωr]` obeys a linear SDE with
deterministic coefficients `F, G`, so its ensemble second moment satisfies
`dV/dt = F*V + V*F' + G*G'`. If `P = V[3:4, 3:4]`, then
`dFs/dt = 2tr(B' * P * B)` and `Qc = covariance_qfi + 2tr(inv(σ) * P)`.
Thus the same moments give both the CFI and the mean conditional QFI, including
inefficient detection and thermal initial states; no trajectory sampling is
needed. This closure relies on the linear Gaussian model and fixed monitoring
settings, and does not generally extend to nonlinear or adaptive filters.

All coupled equations, including the CFI integral, use fixed-step RK4. There
is no Monte Carlo error; decrease `dt` to check integration error. Finite-`dt`
results differ from the Euler–Maruyama trajectory average by discretization
error, which vanishes as `dt` decreases. Arrays include zero at time zero.

Set `show_progress=true` to print progress every `progress_every` timesteps
(default 1), as well as the first and last steps.
"""
function solve_unravelling(
    ω, χ, κ, η, ϕ, meas::AbstractString, nth, tFinal, dt;
    show_progress::Bool = false,
    progress_every::Int = 1,
    progress_prefix::AbstractString = "solve_unravelling",
)
    progress_every >= 1 || throw(ArgumentError("progress_every must be >= 1"))
    times = _unravelling_times(nth, tFinal, dt)
    A, D, B, ∂A = get_unravelling_matrices(ω, χ, κ, η, ϕ, meas)
    Nstep = length(times) - 1
    Fs, Qc = zeros(Nstep + 1), zeros(Nstep + 1)
    σ = (2.0 * nth + 1.0) * Matrix{Float64}(I, 2, 2)
    u = (σ, zeros(2, 2), zeros(4, 4), 0.0)
    pure = nth == 0 && (η == 1 || κ == 0)
    t0_progress = show_progress ? time() : 0.0

    for step in 1:Nstep
        u = _ensemble_step(u, A, D, B, ∂A, dt)
        σ, ∂σ, V, Fs[step + 1] = u
        covariance_qfi, invσ = _covariance_qfi(σ, ∂σ; pure)
        Qc[step + 1] = covariance_qfi + 2tr(invσ * view(V, 3:4, 3:4))
        if show_progress && (step == 1 || step == Nstep || step % progress_every == 0)
            _print_progress(progress_prefix, step, Nstep, t0_progress)
        end
    end

    return times, Fs, Qc, Fs + Qc
end

"""
    simulate_unravelling(ω, χ, κ, η, ϕ, meas, nth, tFinal, dt, Ntraj; kwargs...)

Estimate the signal CFI `Fs`, mean conditional-state QFI `Qc`, and their sum
`Qu` for frequency estimation, returning `(times, Fs, Qc, Qu)`. The initial state
is a frequency-independent thermal state with mean occupation `nth` and zero
first moments. All arrays include their zero initial value.

Covariances and their derivatives use RK4 once per timestep. Trajectory means
and their derivatives use Euler–Maruyama, with derivatives taken at a **fixed
measurement record**, including the derivative of the innovations. The CFI
increment uses the state at the beginning of the measurement interval.

The final time is the last full step not exceeding `tFinal`; an endpoint within
floating-point roundoff of a whole step is retained. `Ntraj` must be a positive
integer. Monte Carlo and timestep errors should be checked by increasing
`Ntraj` and decreasing `dt`. The default `rng=MersenneTwister(0)` makes repeated
calls reproducible; pass another RNG to choose a different sample.

Use [`solve_unravelling`](@ref) for the deterministic ensemble averages of all
three information quantities, without Monte Carlo sampling.

`show_progress=true` prints at most approximately `Ntraj/progress_every` updates
as the batch of trajectories advances through time.
"""
function simulate_unravelling(
    ω,
    χ,
    κ,
    η,
    ϕ,
    meas::AbstractString,
    nth,
    tFinal,
    dt,
    Ntraj;
    rng = MersenneTwister(0),
    show_progress::Bool = false,
    progress_every::Int = 1,
    progress_prefix::AbstractString = "simulate_unravelling",
)
    progress_every >= 1 || throw(ArgumentError("progress_every must be >= 1"))
    times = _unravelling_times(nth, tFinal, dt)
    Ntraj isa Integer && Ntraj > 0 || throw(ArgumentError("Ntraj must be a positive integer"))

    A, D, B, ∂A = get_unravelling_matrices(ω, χ, κ, η, ϕ, meas)
    Nstep = length(times) - 1
    Fs = zeros(Nstep + 1)
    Qc = zeros(Nstep + 1)

    σ = (2.0 * nth + 1.0) * Matrix{Float64}(I, 2, 2)
    ∂σ = zeros(2, 2)
    pure = nth == 0 && (η == 1 || κ == 0)
    r = zeros(2, Ntraj)
    ∂r = zeros(2, Ntraj)
    dr, d∂r, dW, work = (similar(r) for _ in 1:4)
    noise_scale = sqrt(dt / 2)
    progress_stride = max(1, ceil(Int, Nstep * min(progress_every / Ntraj, 1)))
    t0_progress = show_progress ? time() : 0.0

    for step in 1:Nstep
        mul!(work, transpose(B), ∂r)
        Fs[step + 1] = Fs[step] + (2dt / Ntraj) * sum(abs2, work)

        randn!(rng, dW)
        M = B - σ * B
        mul!(dr, A, r, dt, 0)
        mul!(dr, M, dW, noise_scale, 1)

        # Holding dy fixed gives ∂ωdW = √2 B'∂ωr dt. This accounts
        # for the M*B' term; holding the random noise fixed would be wrong.
        mul!(d∂r, ∂A, r, dt, 0)
        mul!(d∂r, A + M * transpose(B), ∂r, dt, 1)
        mul!(d∂r, ∂σ * B, dW, -noise_scale, 1)
        r .+= dr
        ∂r .+= d∂r

        σ, ∂σ = _covariance_step(σ, ∂σ, A, D, B, ∂A, dt)
        covariance_qfi, invσ = _covariance_qfi(σ, ∂σ; pure)
        mul!(work, invσ, ∂r)
        Qc[step + 1] = covariance_qfi + (2 / Ntraj) * dot(∂r, work)

        if show_progress && (step == 1 || step == Nstep || step % progress_stride == 0)
            _print_progress(progress_prefix, step, Nstep, t0_progress)
        end
    end

    return times, Fs, Qc, Fs + Qc
end

end
