@testset "Gaussian unravelling" begin
    UQ = ReplicaME.UnravellingQFI

    @testset "Monitoring matrices and parameter checks" begin
        A, D, B, ∂A = get_unravelling_matrices(0.1, 0.35, 1.0, 0.7, 0.0, "hom")
        @test A ≈ [-0.85 0.1; -0.1 -0.15]
        @test D == Matrix{Float64}(I, 2, 2)
        @test B ≈ [-sqrt(0.7) 0; 0 0]
        @test ∂A == [0.0 1.0; -1.0 0.0]
        _, _, Bphase, _ = get_unravelling_matrices(0.1, 0.35, 1.0, 0.7, π/2, "hom")
        @test Bphase ≈ [0 0; 0 -sqrt(0.7)] atol=1e-15
        _, _, Bhet, _ = get_unravelling_matrices(0.1, 0.35, 1.0, 0.7, 0.0, "het")
        @test Bhet ≈ -sqrt(0.35) * Matrix{Float64}(I, 2, 2)
        @test_throws ArgumentError get_unravelling_matrices(0.1, 0.35, -1.0, 0.5, 0.0, "hom")
        @test_throws ArgumentError get_unravelling_matrices(0.1, 0.35, 1.0, 1.1, 0.0, "hom")
        @test_throws ArgumentError get_unravelling_matrices(0.1, 0.35, 1.0, 0.5, 0.0, "typo")
        for (nth, tfinal, dt, ntraj) in ((-0.1, 1., .01, 2), (0., -1., .01, 2),
                                       (0., 1., 0., 2), (0., 1., .01, 0), (0., 1., .01, 2.5))
            @test_throws ArgumentError simulate_unravelling(.1, .3, 1., .5, 0., "hom", nth, tfinal, dt, ntraj)
        end
    end

    @testset "Pure and nearly pure Gaussian QFI" begin
        for n in (0.2, 1e-6, 1e-10)
            σ = (2n + 1) * Matrix{Float64}(I, 2, 2)
            ∂σ = 2.0 * Matrix{Float64}(I, 2, 2)
            q, _ = UQ._covariance_qfi(σ, ∂σ)
            @test q ≈ 1 / (n * (n + 1)) rtol=1e-6
        end
        σ = Matrix(Diagonal([exp(-0.8), exp(0.8)]))
        ∂σ = Diagonal([-2., 2.]) * σ
        @test first(UQ._covariance_qfi(σ, ∂σ; pure=true)) ≈ 2.0
        @test_throws DomainError UQ._covariance_qfi(0.9 * Matrix{Float64}(I, 2, 2), zeros(2, 2))
    end

    @testset "Vacuum invariance, output grid, and reproducibility" begin
        for meas in ("hom", "het"), η in (0., 0.5, 1.)
            times, Fs, Qc, Qu = simulate_unravelling(.4, 0., 1., η, .3, meas, 0., .3, .1, 3)
            @test length(times) == 4 # 0.3/0.1 lies just below 3 in Float64.
            @test times[end] ≈ .3
            @test Fs == Qc == Qu == zeros(4)
        end
        @test first(simulate_unravelling(.1, .2, 1., 1., 0., "hom", 0., .035, .01, 2)) ≈ [0., .01, .02, .03]
        @test simulate_unravelling(.1, .2, 1., .5, 0., "hom", 0., 0., .01, 2) == ([0.], [0.], [0.], [0.])
        args = (.1, .35, 1., .5, .2, "hom", .1, .2, .002, 8)
        first_run = simulate_unravelling(args...; rng=MersenneTwister(13))
        @test first_run == simulate_unravelling(args...; rng=MersenneTwister(13))
        @test first_run[3] != simulate_unravelling(args...; rng=MersenneTwister(14))[3]
        @test first_run[4] == first_run[2] + first_run[3]
        @test all(>=(0), diff(first_run[2]))
        @test all(>=(0), first_run[3])
    end

    @testset "Unitary evolution agrees with an exact matrix exponential" begin
        ω, χ, t = .6, .2, .8
        δ = 1e-5
        covariance(frequency) = begin
            A, _, _, _ = get_unravelling_matrices(frequency, χ, 0., 0., 0., "hom")
            S = exp(A * t)
            S * transpose(S)
        end
        σ = covariance(ω)
        ∂σ = (covariance(ω + δ) - covariance(ω - δ)) / (2δ)
        expected = tr((σ \ ∂σ)^2) / 4
        times, Fs, Qc, Qu = simulate_unravelling(ω, χ, 0., 0., 0., "hom", 0., t, .005, 2)
        @test Qc[end] ≈ expected rtol=1e-8
        @test iszero(Fs)
        @test Qc == Qu
    end

    @testset "Sensitivities differentiate a fixed measurement record" begin
        # Independently run three filters at ω and ω±δ with the SAME observed
        # increments dy. Finite-difference their states, without differentiating
        # the innovations by hand, and compare both reported information curves.
        ω, χ, κ, η, ϕ, nth, dt, nstep, δ = .3, .35, 1., .6, .4, .2, .005, 80, 1e-5
        for meas in ("hom", "het")
            parameters = [get_unravelling_matrices(w, χ, κ, η, ϕ, meas) for w in (ω, ω+δ, ω-δ)]
            covariances = [(2nth+1) * Matrix{Float64}(I, 2, 2) for _ in 1:3]
            means = [zeros(2) for _ in 1:3]
            rng = MersenneTwister(7)
            noise = zeros(2, 1)
            expected_Fs, expected_Qc = zeros(nstep+1), zeros(nstep+1)
            B = parameters[1][3]
            for step in 1:nstep
                ∂r = (means[2] - means[3]) / (2δ)
                expected_Fs[step+1] = expected_Fs[step] + 2dt * sum(abs2, transpose(B) * ∂r)
                randn!(rng, noise)
                dy = -sqrt(2) * transpose(B) * means[1] * dt + sqrt(dt) * vec(noise)
                for j in 1:3
                    A, D, B, ∂A = parameters[j]
                    σ, r = covariances[j], means[j]
                    innovation = dy + sqrt(2) * transpose(B) * r * dt
                    means[j] = r + A * r * dt + (B - σ * B) * innovation / sqrt(2)
                    covariances[j], _ = UQ._covariance_step(σ, zeros(2, 2), A, D, B, ∂A, dt)
                end
                ∂r = (means[2] - means[3]) / (2δ)
                ∂σ = (covariances[2] - covariances[3]) / (2δ)
                covariance_qfi, invσ = UQ._covariance_qfi(covariances[1], ∂σ)
                expected_Qc[step+1] = covariance_qfi + 2dot(∂r, invσ * ∂r)
            end
            _, Fs, Qc, _ = simulate_unravelling(ω, χ, κ, η, ϕ, meas, nth, nstep*dt, dt, 1; rng=MersenneTwister(7))
            @test Fs ≈ expected_Fs rtol=1e-7 atol=1e-13
            @test Qc ≈ expected_Qc rtol=1e-7 atol=1e-12
        end
    end

    @testset "Deterministic ensemble information" begin
        @testset "Validation, output grid, and vacuum invariance" begin
            for (nth, tfinal, dt) in ((-0.1, 1., .01), (Inf, 1., .01),
                                     (0., -1., .01), (0., Inf, .01),
                                     (0., 1., 0.), (0., 1., NaN))
                @test_throws ArgumentError solve_unravelling(.1, .3, 1., .5, 0., "hom", nth, tfinal, dt)
            end
            @test_throws ArgumentError solve_unravelling(.1, .3, 1., .5, 0., "hom", 0., 1., .01; progress_every=0)
            @test_throws ArgumentError solve_unravelling(.1, .3, 1., .5, 0., "typo", 0., 1., .01)
            @test_throws ArgumentError solve_unravelling(.1, .3, 1., 1.1, 0., "hom", 0., 1., .01)
            for meas in ("hom", "het"), η in (0., .5, 1.)
                times, Fs, Qc, Qu = solve_unravelling(.4, 0., 1., η, .3, meas, 0., .3, .1)
                @test times ≈ [0., .1, .2, .3]
                @test Fs == Qc == Qu == zeros(4)
            end
            @test first(solve_unravelling(.1, .2, 1., 1., 0., "hom", 0., .035, .01)) ≈ [0., .01, .02, .03]
            @test solve_unravelling(.1, .2, 1., .5, 0., "hom", .2, 0., .01) == ([0.], [0.], [0.], [0.])
        end

        @testset "No-noise limits agree with trajectories and unitary dynamics" begin
            for meas in ("hom", "het"), nth in (0., .3), (κ, η) in ((1., 0.), (0., .7))
                args = (.6, .2, κ, η, .4, meas, nth, .8, .005)
                times, Fs, Qc, Qu = solve_unravelling(args...)
                trajectory = simulate_unravelling(args..., 1)
                @test times == trajectory[1]
                @test iszero(Fs)
                @test Qc ≈ trajectory[3] rtol=1e-12 atol=1e-14
                @test Qu == Qc
                if κ == 0
                    covariance(frequency) = begin
                        A, _, _, _ = get_unravelling_matrices(frequency, .2, 0., η, .4, meas)
                        S = exp(.8 * A)
                        (2nth + 1) * S * transpose(S)
                    end
                    σ = covariance(.6)
                    ∂σ = (covariance(.6 + 1e-5) - covariance(.6 - 1e-5)) / 2e-5
                    X = σ \ ∂σ
                    # A unitary orbit preserves purity, so its QFI has only
                    # the covariance-shape term, for pure and thermal inputs.
                    expected = tr(X * X) * det(σ) / (2 * (det(σ) + 1))
                    @test Qc[end] ≈ expected rtol=1e-8
                end
            end
        end

        @testset "Passive thermal heterodyne agrees with an independent Itô kernel" begin
            # With χ=0 and heterodyne, σ=s(t)I and ∂σ=0. Solve the scalar
            # Riccati equation analytically, then evaluate the variance of the
            # mean sensitivity directly from its noise-response kernel. This
            # checks the coupling and diffusion factors without a trajectory
            # sample or a second implementation of the moment ODE.
            κ, η, nth, tfinal = .7, .8, .4, .8
            b2, q = η * κ / 2, η * nth
            h(t) = 1 + q * (-expm1(-κ*t))
            s(t) = 1 + 2nth * exp(-κ*t) / h(t)
            function simpson(f, t; n=128)
                dx = t / n
                total = f(0.) + f(t)
                for j in 1:n-1
                    total += (isodd(j) ? 4 : 2) * f(j * dx)
                end
                return total * dx / 3
            end
            function sensitivity_variance(t)
                simpson(t) do v
                    integrated_h = (1 + q) * (t - v) + q / κ * (exp(-κ*t) - exp(-κ*v))
                    δ = 2nth * exp(-κ*v) / h(v)
                    exp(-κ*(t-v)) * δ^2 * b2 / 2 * (integrated_h / h(t))^2
                end
            end
            expected_Qc = 4sensitivity_variance(tfinal) / s(tfinal)
            expected_Fs = 4b2 * simpson(sensitivity_variance, tfinal)
            for ω in (0., .9)
                _, Fs, Qc, Qu = solve_unravelling(ω, 0., κ, η, .4, "het", nth, tfinal, .005)
                @test Qc[end] ≈ expected_Qc rtol=1e-7
                @test Fs[end] ≈ expected_Fs rtol=1e-7
                @test Qu == Fs + Qc
            end
        end

        @testset "Finite information and fourth-order timestep convergence" begin
            for (meas, η, ϕ, nth) in (("hom", .65, .4, .2), ("hom", 1., -.7, 0.),
                                     ("het", .6, .2, .3), ("het", 1., 0., 0.))
                times, Fs, Qc, Qu = solve_unravelling(.4, .32, 1., η, ϕ, meas, nth, 1., .005)
                @test length(times) == length(Fs) == length(Qc) == length(Qu) == 201
                @test all(isfinite, Fs) && all(isfinite, Qc) && all(isfinite, Qu)
                @test all(>=(0), diff(Fs))
                @test all(>=(0), Qc)
                @test Fs[end] > 0 && Qc[end] > 0
                @test Qu == Fs + Qc
            end
            endpoint(dt) = begin
                _, Fs, Qc, _ = solve_unravelling(.4, .32, 1., .65, .4, "hom", .2, 1., dt)
                [Fs[end], Qc[end]]
            end
            reference = endpoint(.0025)
            errors = [norm(endpoint(dt) - reference) for dt in (.04, .02, .01)]
            @test 8 < errors[1] / errors[2] < 32
            @test 8 < errors[2] / errors[3] < 32
            @test errors[3] < 1e-6 * norm(reference)
        end
    end
end
