@testset "Gaussian moments and TSME" begin
    GM = ReplicaME.GaussianMoments
    TS = ReplicaME.TSME
    Ω = symplectic_Ω(1)
    vacuum = Matrix{Float64}(I, 2, 2)

    @testset "Conventions and Gaussian trace norm" begin
        @test symplectic_Ω(2) == kron(Matrix{Float64}(I, 2, 2), Ω)
        @test_throws ArgumentError symplectic_Ω(0)
        @test_throws ArgumentError PackSpec(3)
        @test trace_norm_gaussian_nu(vacuum, zeros(2)) ≈ 1
        @test trace_norm_gaussian_nu(3vacuum, [1.0, -2.0]) ≈ 1

        # A correlated two-mode thermal state remains normalized under an
        # arbitrary symplectic transformation. This detects both a quadrature
        # ordering mismatch and symmetrization of the non-symmetric (Ωσ)^2.
        generator = [0.6 0.2 0.3 -0.4; 0.2 -0.3 0.1 0.2;
                     0.3 0.1 0.4 -0.2; -0.4 0.2 -0.2 -0.1]
        S = exp(symplectic_Ω(2) * generator)
        covariance = S * Diagonal([1.7, 1.7, 3.2, 3.2]) * transpose(S)
        @test S * symplectic_Ω(2) * transpose(S) ≈ symplectic_Ω(2)
        @test trace_norm_gaussian_nu(covariance, zeros(4)) ≈ 1 atol = 2e-13

        # For a normalized coherent-state dyad, σ=I and log‖ν‖₁=|Im d|².
        displacement = [0.2 + 0.3im, -0.4 + 0.1im]
        @test trace_norm_gaussian_nu(vacuum, displacement) ≈ exp(0.1)
        @test GM.log_trace_norm_gaussian_nu(vacuum, [30im, 0im]) ≈ 900
        @test TS._log_trace_norm(vacuum, [30im, 0im], 900) ≈ 0
        @test_throws ArgumentError trace_norm_gaussian_nu([1.0 1; 0 1], zeros(2))
        @test_throws DimensionMismatch trace_norm_gaussian_nu(vacuum, zeros(4))
    end

    @testset "Stable single-mode trace norm near purity" begin
        # An independent high-precision evaluation constructs σK and obtains
        # its sole symplectic eigenvalue as sqrt(det(σK)). It remains a useful
        # reference near purity when Float64 acosh(sqrt(det(σK))) is unstable.
        function reference_log_norm(σ, d)
            setprecision(256) do
                σbig = Complex{BigFloat}.(σ)
                R, J = real.(σbig), imag.(σbig)
                b = BigFloat.(imag.(d))
                Ωbig = BigFloat.([0 1; -1 0])
                σK = (R + (J + Ωbig) * (R \ transpose(J + Ωbig))) / 2
                ν = sqrt(det(σK))
                return -log(det(R)) / 4 + dot(b, R \ b) + acosh(ν) / 2
            end
        end
        rng = MersenneTwister(918)
        for _ in 1:12
            X, Y = randn(rng, 2, 2), randn(rng, 2, 2)
            R = X * transpose(X) + 0.2I
            σ = R + 1im * (Y + transpose(Y)) / 2
            d = randn(rng, 2) + 1im * randn(rng, 2)
            @test GM.log_trace_norm_gaussian_nu(σ, d) ≈ reference_log_norm(σ, d) rtol = 2e-13
        end
        for ε in (1e-3, 1e-6, 1e-9)
            σ = [exp(0.2) + ε * 1im 0; 0 exp(-0.2)]
            @test GM.log_trace_norm_gaussian_nu(σ, zeros(2)) ≈
                reference_log_norm(σ, zeros(2)) atol = 2e-16
        end

        # At short OPO times σK is almost pure. Its environment information
        # must converge as the finite-difference step changes, even though
        # the eigenvalue-based acosh expression loses many digits here.
        rates = [qfi_from_tsme_fidelity(0.1; ϵ = ε, N = 1,
            tspan = (0.0, 0.1), σ0 = vacuum, d0 = zeros(2),
            Hfun = θ -> [θ -0.45; -0.45 θ],
            Lfun = θ -> sqrt(0.5) .* [1 1im],
            reltol = 2e-13, abstol = 2e-13) / 0.1
            for ε in (0.008, 0.004, 0.002)]
        @test all(>(0), rates)
        @test maximum(rates) - minimum(rates) < 2e-5 * minimum(rates)
    end

    @testset "Independent Fock-space TSME derivative" begin
        cutoff = 18
        annihilation = diagm(1 => sqrt.(1.0:(cutoff - 1)))
        rs = [(annihilation + annihilation') / sqrt(2),
              (annihilation - annihilation') / (sqrt(2) * 1im)]
        α = 0.2 + 0.1im
        ket = ComplexF64[α^n / sqrt(Float64(factorial(n))) for n in 0:(cutoff - 1)]
        ket ./= norm(ket)
        ρ = ket * ket'
        Hfun(θ) = [0.2 + θ 0.08; 0.08 0.3 - θ / 2]
        hfun(θ) = [0.2θ, 0.3 - θ]
        Lfun(θ) = [sqrt((0.8 + 0.3θ) / 2) * exp(0.2im * θ) .* [1 1im];
                   0.2 0.0]
        function fock_hamiltonian(θ)
            H, h = Hfun(θ), hfun(θ)
            quadratic = sum(H[i, j] * rs[i] * rs[j] / 2 for i in 1:2, j in 1:2)
            linear = sum((transpose(h) * Ω)[i] * rs[i] for i in 1:2)
            return quadratic + linear
        end
        jumps(θ) = [sum(Lfun(θ)[k, j] * rs[j] for j in 1:2) for k in 1:2]
        θ1, θ2 = 0.1, 0.4
        J1, J2 = jumps(θ1), jumps(θ2)
        μdot = -1im * (fock_hamiltonian(θ1) * ρ - ρ * fock_hamiltonian(θ2))
        for k in 1:2
            μdot += J1[k] * ρ * J2[k]' - (J1[k]' * J1[k] * ρ + ρ * J2[k]' * J2[k]) / 2
        end
        tracedot = tr(μdot)
        νdot = μdot - ρ * tracedot
        d = [tr(r * ρ) for r in rs]
        dd = [tr(r * νdot) for r in rs]
        σ = [tr((rs[i] * rs[j] + rs[j] * rs[i]) * ρ) - 2d[i] * d[j]
             for i in 1:2, j in 1:2]
        dσ = [tr((rs[i] * rs[j] + rs[j] * rs[i]) * νdot) - 2(dd[i] * d[j] + d[i] * dd[j])
              for i in 1:2, j in 1:2]
        coeffs = build_tsme_coeffs(θ1, θ2; N = 1, Hfun, hfun, Lfun)
        y = vcat(vec(σ), d, 0im)
        dy = similar(y)
        gaussian_rhs_general!(dy, y, (; ps = PackSpec(2), params = t -> coeffs), 0.0)
        @test dy ≈ vcat(vec(dσ), dd, -tracedot) atol = 2e-13
    end

    @testset "Parameter-dependent damping and coherent drive" begin
        Hzero(θ) = zeros(2, 2)
        loss(κ) = sqrt(κ / 2) .* [1 1im]
        κ1, κ2, T = 0.4, 1.1, 0.8
        # Vacuum and its environment stay vacuum for every loss rate.
        stationary = solve_tsme_moments(κ1, κ2; N = 1, tspan = (0.0, T),
            σ0 = vacuum, d0 = zeros(2), Hfun = Hzero, Lfun = loss,
            reltol = 1e-11, abstol = 1e-12)
        σT, dT, ξT = get_σ_d_ξ(stationary, 1, length(stationary.u))
        @test σT ≈ vacuum atol = 1e-12
        @test dT ≈ zeros(2) atol = 1e-12
        @test ξT ≈ 0 atol = 1e-12

        # Independent exact solution: damped coherent system and coherent
        # emitted field, with separate overlaps in those two subsystems.
        α = 0.7 + 0.2im
        sol = solve_tsme_moments(κ1, κ2; N = 1, tspan = (0.0, T),
            σ0 = vacuum, d0 = sqrt(2) .* [real(α), imag(α)],
            Hfun = Hzero, Lfun = loss, reltol = 1e-11, abstol = 1e-12,
            saveat = [0.2, 0.5])
        @test sol.t == [0.2, 0.5]
        env_exponent(t) = abs2(α) / 2 * (2 - exp(-κ1 * t) - exp(-κ2 * t) -
            4sqrt(κ1 * κ2) / (κ1 + κ2) * (1 - exp(-(κ1 + κ2) * t / 2)))
        _, Fenv = fidelity_timecourse_from_solution(sol, 1)
        _, Fjoint = absoverlap_timecourse_from_solution(sol, 1)
        @test Fenv ≈ exp.(-env_exponent.(sol.t)) atol = 5e-11
        @test Fjoint ≈ [exp(-env_exponent(t) - abs2(α) / 2 *
            (exp(-κ1 * t / 2) - exp(-κ2 * t / 2))^2) for t in sol.t] atol = 5e-11

        # With H=hᵀΩr a real, constant h produces mean drift h, not Ωh.
        h = [0.3, -0.4]
        driven = solve_tsme_moments(0.0, 0.0; N = 1, tspan = (0.0, T),
            σ0 = vacuum, d0 = zeros(2), Hfun = Hzero, hfun = θ -> h,
            Lfun = θ -> nothing)
        @test get_σ_d_ξ(driven, 1, length(driven.u))[2] ≈ T .* h atol = 1e-12
        @test fidelity_from_solution(driven, 1) ≈ 1 atol = 1e-12
        @test_throws ArgumentError build_tsme_coeffs(0.0, 1.0; N = 1,
            Hfun = Hzero, Lfun = θ -> θ == 0 ? zeros(1, 2) : zeros(2, 2))
    end

    @testset "Finite-difference QFI conventions" begin
        # A unitary coherent displacement d=θ²t along x has exact joint QFI
        # 2(∂θd)²=8θ²t². Centered and forward extrapolation have different orders.
        θ, T, δ = 0.7, 0.6, 0.04
        kwargs = (; N = 1, tspan = (0.0, T), σ0 = vacuum, d0 = zeros(2),
            Hfun = θ -> zeros(2, 2), hfun = θ -> [θ^2, 0.0],
            Lfun = θ -> nothing, reltol = 1e-12, abstol = 1e-13,
            saveat = [0.0, T])
        _, env_centered, joint_centered = qfi_timecourse_from_tsme_fidelity(θ;
            ϵ = δ, kwargs..., scheme = :centered)
        _, env_forward, joint_forward = qfi_timecourse_from_tsme_fidelity(θ;
            ϵ = δ, kwargs..., scheme = :forward)
        @test maximum(abs, env_centered) < 1e-8
        @test maximum(abs, env_forward) < 1e-8
        @test joint_centered[end] ≈ 8θ^2 * T^2 atol = 1e-8
        @test joint_forward[end] ≈ (8θ^2 - δ^2) * T^2 atol = 1e-8
        _, _, joint_small_step = qfi_timecourse_from_tsme_fidelity(θ;
            ϵ = 1e-8, kwargs..., richardson = false)
        @test joint_small_step[end] ≈ 8θ^2 * T^2 rtol = 1e-7

        # A driven lossy cavity emits a coherent field, whose QFI is obtained
        # independently by integrating the squared derivative of its amplitude.
        κ = 0.9
        expected_env = 8 / κ * (T - 4 / κ * (1 - exp(-κ * T / 2)) +
            (1 - exp(-κ * T)) / κ)
        qfi_env = qfi_from_tsme_fidelity(θ; ϵ = δ, N = 1, tspan = (0.0, T),
            σ0 = vacuum, d0 = zeros(2), Hfun = θ -> zeros(2, 2),
            hfun = θ -> [θ, 0.0], Lfun = θ -> sqrt(κ / 2) .* [1 1im],
            reltol = 1e-12, abstol = 1e-13)
        @test qfi_env ≈ expected_env atol = 1e-9
        @test_throws DomainError TS._qfi_from_fidelity([1.1], δ;
            use_log_formula = true, fidelity_floor = 1e-300)
        @test_throws ArgumentError TS._qfi_from_fidelity([1.0], δ;
            use_log_formula = true, fidelity_floor = 2.0)
    end
end
