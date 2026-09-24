# Run from the repository root: julia --project=. validation/unravelling_ensemble.jl
# Independent, deterministic benchmark of the Monte Carlo unravelling averages.
# This integrates continuous-time ensemble moments rather than sampled paths.
using LinearAlgebra
using Printf
using OrdinaryDiffEqTsit5: Tsit5
using SciMLBase: ODEProblem, solve, successful_retcode

"""Exact ensemble means and predicted Monte Carlo standard errors at time T."""
function ensemble_information(η, T, Ntraj; ω=0.1, χ=0.45, κ=1.0,
                              reltol=1e-11, abstol=1e-12)
    A = [-χ-κ/2 ω; -ω χ-κ/2]
    D = κ * Matrix{Float64}(I, 2, 2)
    B = [-sqrt(η*κ) 0.0; 0.0 0.0]
    ∂A = [0.0 1.0; -1.0 0.0]
    Q = zeros(4, 4)
    Q[3:4, 3:4] = 2B * transpose(B)

    # z = [r; ∂ωr] is zero-mean Gaussian with covariance V and dynamics
    # dz = F*z*dt + G*dW. All coefficients depend only on σ and ∂ωσ.
    # f = ∫z'Qz dt is the trajectory's accumulated signal information.
    # S = Cov(f, zz') obeys Sdot = F*S + S*F' + 2V*Q*V by Wick's theorem;
    # hence Var(f)dot = 2tr(Q*S). This gives sampling errors without Monte Carlo.
    function ensemble_rhs!(du, u, p, t)
        σ = reshape(view(u, 1:4), 2, 2)
        ∂σ = reshape(view(u, 5:8), 2, 2)
        V = reshape(view(u, 9:24), 4, 4)
        S = reshape(view(u, 26:41), 4, 4)
        M = B - σ * B
        F = [A zeros(2, 2); ∂A A + M * transpose(B)]
        G = [M; -∂σ * B] / sqrt(2)

        view(du, 1:4) .= vec(A * σ + σ * transpose(A) + D - M * transpose(M))
        view(du, 5:8) .= vec(∂A * σ + σ * transpose(∂A) + A * ∂σ +
                            ∂σ * transpose(A) + ∂σ * B * transpose(M) +
                            M * transpose(B) * ∂σ)
        view(du, 9:24) .= vec(F * V + V * transpose(F) + G * transpose(G))
        du[25] = tr(Q * V)
        view(du, 26:41) .= vec(F * S + S * transpose(F) + 2V * Q * V)
        du[42] = 2tr(Q * S)
        return nothing
    end

    initial = zeros(42)
    initial[1] = initial[4] = 1 # Vacuum covariance; all other moments start at 0.
    problem = ODEProblem(ensemble_rhs!, initial, (0.0, Float64(T)))
    solution = solve(problem, Tsit5(); reltol, abstol, saveat=[T],
                     save_everystep=false, dense=false)
    successful_retcode(solution) || error("Ensemble integration failed: $(solution.retcode)")
    u = last(solution.u)
    σ = reshape(u[1:4], 2, 2)
    ∂σ = reshape(u[5:8], 2, 2)
    V = reshape(u[9:24], 4, 4)
    S = reshape(u[26:41], 4, 4)
    invσ = inv(σ)
    X = invσ * ∂σ
    determinant = det(σ)
    covariance_qfi = if η == 1
        tr(X * X) / 4
    else
        tr(X * X) * determinant / (2(determinant + 1)) +
        tr(X)^2 * determinant / (2(determinant - 1) * (determinant + 1))
    end
    R = zeros(4, 4)
    R[3:4, 3:4] = 2invσ
    conditional_qfi = covariance_qfi + tr(R * V)
    conditional_variance = 2tr(R * V * R * V)
    signal_variance = u[42]
    total_variance = signal_variance + conditional_variance + 2tr(R * S)
    return (
        signal_rate=u[25] / T,
        signal_se=sqrt(signal_variance / Ntraj) / T,
        conditional_rate=conditional_qfi / T,
        conditional_se=sqrt(conditional_variance / Ntraj) / T,
        total_rate=(u[25] + conditional_qfi) / T,
        total_se=sqrt(total_variance / Ntraj) / T,
        determinant,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("Continuous-time ensemble benchmarks; SE is for the stated trajectory count.")
    for (η, T, Ntraj) in ((0.5, 20.0, 500), (1.0, 100.0, 1000))
        result = ensemble_information(η, T, Ntraj)
        tightened = ensemble_information(η, T, Ntraj; reltol=2e-13, abstol=2e-14)
        @assert maximum(abs, collect(values(result)) - collect(values(tightened))) < 1e-7
        @printf("η=%.1f, t=%.0f, Ntraj=%d: det(σ)=%.12g\n", η, T, Ntraj, result.determinant)
        @printf("  signal/t      = %.12g ± %.6g (sampling SE)\n", result.signal_rate, result.signal_se)
        @printf("  conditional/t = %.12g ± %.6g (sampling SE)\n", result.conditional_rate, result.conditional_se)
        @printf("  total/t       = %.12g ± %.6g (sampling SE)\n", result.total_rate, result.total_se)
    end
end
