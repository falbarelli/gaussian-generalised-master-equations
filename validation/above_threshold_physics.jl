# Run from the repository root:
#   julia --project=. validation/above_threshold_physics.jl
# Add --extended to inspect the later recovery of the homodyne CFI/time to t=40.
include(joinpath(@__DIR__, "..", "ReplicaME.jl"))
using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "unravelling_ensemble.jl"))

const U = ReplicaME.UnravellingQFI

"""Exact unconditional covariance from vacuum, using the drift eigenbasis."""
function unconditional_covariance(A, D, t)
    eig = eigen(A)
    P = eig.vectors
    invP = inv(P)
    initial = invP * transpose(invP)
    diffusion = invP * D * transpose(invP)
    covariance = similar(initial)
    for j in 1:2, i in 1:2
        rate = eig.values[i] + eig.values[j]
        integral = iszero(rate) ? t : expm1(rate * t) / rate
        covariance[i, j] = initial[i, j] * exp(rate * t) + diffusion[i, j] * integral
    end
    return P * covariance * transpose(P)
end

function check_covariances(A, D, B, ∂A; dt=0.001, T=20.0)
    unconditional = unconditional_covariance.(Ref(A), Ref(D), 0.0:0.02:T)
    min_det = minimum(det, unconditional)
    min_eigenvalue = minimum(σ -> eigmin(Symmetric(σ)), unconditional)
    @assert min_det >= 1 - 1e-10 && min_eigenvalue > 0

    # For our covariance convention, sigma = 2 Var(r). Averaging the filtered
    # state must recover sigma_unconditional = sigma_conditional + 2 Cov(r).
    u = (Matrix{Float64}(I, 2, 2), zeros(2, 2), zeros(4, 4), 0.0)
    min_cond_det = 1.0
    min_cond_eigenvalue = 1.0
    max_reconstruction_error = 0.0
    check_every = round(Int, 1 / dt)
    for step in 1:round(Int, T / dt)
        u = U._ensemble_step(u, A, D, B, ∂A, dt)
        σ, _, V, _ = u
        min_cond_det = min(min_cond_det, det(σ))
        min_cond_eigenvalue = min(min_cond_eigenvalue, eigmin(Symmetric(σ)))
        if step % check_every == 0
            exact = unconditional_covariance(A, D, step * dt)
            reconstructed = σ + 2V[1:2, 1:2]
            max_reconstruction_error = max(max_reconstruction_error,
                                          norm(reconstructed - exact) / norm(exact))
        end
    end
    @assert min_cond_det >= 1 - 1e-10 && min_cond_eigenvalue > 0
    @assert max_reconstruction_error < 1e-9
    σfinal = last(unconditional)
    @printf("Physicality through t=%.0f: min det(unconditional)=%.12g, conditional=%.12g\n",
            T, min_det, min_cond_det)
    @printf("  min covariance eigenvalues: unconditional=%.8g, conditional=%.8g\n",
            min_eigenvalue, min_cond_eigenvalue)
    @printf("  final unconditional det=%.12g, mean occupation=%.12g\n",
            det(σfinal), (tr(σfinal) - 2) / 4)
    @printf("  maximum covariance reconstruction relative error=%.3g\n",
            max_reconstruction_error)
end

function main()
    ω, χ, κ, η = 0.1, 0.7, 1.0, 0.5
    A, D, B, ∂A = get_unravelling_matrices(ω, χ, κ, η, 0.0, "hom")
    analytic_eigenvalues = [-κ/2 - sqrt(χ^2 - ω^2), -κ/2 + sqrt(χ^2 - ω^2)]
    @assert eigvals(A) ≈ analytic_eigenvalues
    @assert maximum(analytic_eigenvalues) > 0
    @printf("chi/kappa=%.1f, omega/kappa=%.1f, eta=%.1f\n", χ/κ, ω/κ, η)
    @printf("Threshold chi/kappa=%.12g; drift eigenvalues=(%.12g, %.12g)\n",
            sqrt(1/4 + (ω/κ)^2), analytic_eigenvalues...)
    println("A positive drift eigenvalue excludes a stationary unconditional covariance.")
    check_covariances(A, D, B, ∂A)

    T = "--extended" in ARGS ? 40.0 : 20.0
    coarse_dt, fine_dt = 0.001, 0.0005
    coarse = solve_unravelling(ω, χ, κ, η, 0.0, "hom", 0.0, T, coarse_dt)
    fine = solve_unravelling(ω, χ, κ, η, 0.0, "hom", 0.0, T, fine_dt)
    # Avoid relative errors at t=0 and extremely small short-time information.
    # t>=0.1 is the saved figure grid; compare every integration point thereafter.
    start = searchsortedfirst(coarse[1], 0.1)
    for (quantity, label) in ((2, "CFI"), (3, "conditional QFI"), (4, "sum"))
        refined = fine[quantity][(2start - 1):2:end]
        error = maximum(abs.((coarse[quantity][start:end] - refined) ./ refined))
        @assert error < 1e-6
        @printf("dt refinement, %s: max relative difference for t>=0.1 is %.3g\n",
                label, error)
    end
    times, Fs, Qc, Qu = fine
    @assert all(diff(Fs) .>= -1e-12) # accumulated information cannot decrease
    last_plot_index = round(Int, 20 / fine_dt) + 1
    peak_index = argmax(Fs[2:last_plot_index] ./ times[2:last_plot_index]) + 1
    @printf("Homodyne CFI/time peak on [0,20]: %.12g at t=%.4f\n",
            Fs[peak_index] / times[peak_index], times[peak_index])
    checkpoints = T > 20 ? (5, 10, 15, 20, 25, 30, 40) : (5, 10, 15, 20)
    for t in checkpoints
        index = round(Int, t / fine_dt) + 1
        @printf("t=%2d: CFI/t=%.12g, conditional QFI/t=%.12g, sum/t=%.12g\n",
                t, Fs[index] / t, Qc[index] / t, Qu[index] / t)
    end

    # Independent adaptive ODE also predicts Monte Carlo error from Gaussian
    # fourth moments. This is separate from the fixed-step production solver.
    reference = ensemble_information(η, 20.0, 500; ω, χ, κ,
                                     reltol=2e-13, abstol=2e-14)
    @assert isapprox(Fs[last_plot_index] / 20, reference.signal_rate; rtol=1e-9)
    @assert isapprox(Qc[last_plot_index] / 20, reference.conditional_rate; rtol=1e-9)
    @printf("Independent endpoint agrees; predicted CFI/t sampling SE for 500 trajectories: %.8g\n",
            reference.signal_se)
    println("CFI bounds output QFI from below; conditional QFI additionally requires a final system measurement.")
    println("All checks passed.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
