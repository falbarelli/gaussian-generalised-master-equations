@testset "Replica coefficients and input contracts" begin
    model = (H = [0.3 0.1; 0.1 0.5], h = [0.2, -0.1],
             j = sqrt(0.4 / 2) .* [1, im], L = sqrt(0.6 / 2) .* [1 im])
    physical = build_replica_coeffs([model])
    @test physical.Q ≈ zeros(2, 2) atol = 1e-15
    @test physical.W ≈ zeros(2, 2) atol = 1e-15
    @test physical.A ≈ symplectic_Ω(1) * model.H - 0.5I
    @test physical.D ≈ Matrix{Float64}(I, 2, 2)
    @test build_replica_coeffs([model]; periodic = false).Q != physical.Q
    params = replica_params([model, model])
    @test params(0.0) === params(2.0)
    @test_throws ArgumentError build_replica_coeffs([])
    @test_throws DimensionMismatch build_replica_coeffs([merge(model, (j = [1.0],))])
    @test_throws DimensionMismatch build_replica_coeffs([model, merge(model, (j = zeros(2, 2),))])
    @test_throws ArgumentError build_replica_coeffs([merge(model, (H = [0.0 1.0; 0.0 0.0],))])
    @test_throws DimensionMismatch replicate_initial_moments(zeros(4, 4), zeros(2), 2)
    @test theta_list_G(1, 2.5, 2, 1) == [1.0, 1.0, 2.5, 2.5]
    @test theta_list_eq20_term(1.5, 0, 2.5, 2, 1) == [1.5, 0.0, 2.5, 0.0]
    @test_throws ArgumentError D_lm(-1, 0)
    @test D_lm(100, 50) == 2binomial(big(100), 50) - 2binomial(big(100), 49)
end

@testset "Replica formulas against a finite density matrix" begin
    # This check has no Gaussian ODEs: differentiate a noncommuting, mixed qubit
    # family and independently evaluate the spectral definition of I_n.
    σx, σz = [0.0 1.0; 1.0 0.0], [1.0 0.0; 0.0 -1.0]
    ρ(θ) = (I + (0.2 + 0.1θ) * σz + 0.3sin(θ) * σx) / 2
    θ = 0.7
    state = ρ(θ)
    derivative = (0.1σz + 0.3cos(θ) * σx) / 2
    Λ = 2sqrt(tr(state^2))
    normalized = Float64[]
    for m in 0:6
        from_powers = sum(Float64(D_lm(m, l)) * mixed_d2_central(
            (a, b) -> tr(ρ(a)^(l + 1) * ρ(b)^(m - l + 1)), θ;
            h = 0.003, richardson = true) for l in 0:m)
        from_insertions = 2sum(binomial(m, l) *
            tr(derivative * state^(m - l) * derivative * state^l) for l in 0:m)
        @test from_powers ≈ from_insertions rtol = 3e-8 atol = 1e-10
        push!(normalized, from_insertions / Λ^(m + 1))
    end
    eigenstate = eigen(Symmetric(state))
    λ = eigenstate.values
    tangent = eigenstate.vectors' * derivative * eigenstate.vectors
    for n in 0:6
        spectral = sum(2abs2(tangent[i, j]) / (λ[i] + λ[j]) *
            (1 - (1 - (λ[i] + λ[j]) / Λ)^(n + 1)) for i in 1:2, j in 1:2)
        series = sum((-1)^m * binomial(n + 1, m + 1) * normalized[m + 1] for m in 0:n)
        @test series ≈ spectral rtol = 2e-12
    end
    @test_throws ArgumentError mixed_d2_central((a, b) -> a * b, θ; h = Inf)
    @test_throws ArgumentError mixed_d2_central((a, b) -> a * b, θ; h = eps(θ) / 10)
    # The logarithms remain resolvable even when exp(logB) rounds to 1.
    h = 1e-9
    logs = (complex(h^2), complex(-h^2), complex(-h^2), complex(h^2))
    @test ReplicaME.BargmannInvariants._mixed_stencil_from_logs(logs..., h) ≈ 1.0
    @test ReplicaME.BargmannInvariants._mixed_stencil_from_logs(logs..., h;
        log_normalization = 20.0) ≈ exp(-20.0)
end

@testset "Replica ODEs against coherent output fields" begin
    κ, η = 1.4, 0.37
    times = [0.2, 0.6, 1.0] # Deliberately omit both endpoints of tspan.
    common = (N = 1, tspan = (0.0, 1.2), σ0_single = Matrix{Float64}(I, 2, 2),
              d0_single = zeros(2), Hfun = θ -> zeros(2, 2),
              hfun = θ -> [θ, 0.0], jfun = θ -> sqrt(η * κ / 2) .* [1, im],
              Lfun = θ -> sqrt((1 - η) * κ / 2) .* [1 im],
              reltol = 2e-12, abstol = 2e-12)
    # Driven passive cavity: β_θ(t) = sqrt(2η/κ) θ (1-exp(-κt/2)).
    # The squared norm of ∂θ β is S(t), hence QFI=4S(t).
    S(t) = 2η / κ * (t - 4 / κ * (1 - exp(-κ * t / 2)) + (1 - exp(-κ * t)) / κ)
    Θ = [-0.2, 0.5, 0.7]
    returned_times, logB = log_bargmann_invariant_timecourse(Θ; times, common...)
    @test returned_times == times
    exact = [-S(t) / 2 * sum((Θ[i] - Θ[mod1(i + 1, length(Θ))])^2 for i in eachindex(Θ)) for t in times]
    @test logB ≈ exact atol = 2e-10
    _, one_replica = log_bargmann_invariant_timecourse([0.7]; times, common...)
    @test one_replica ≈ zeros(length(times)) atol = 2e-12
    @test log_bargmann_invariant(Θ; common...) ≈
        only(last(log_bargmann_invariant_timecourse(Θ; times = [1.2], common...)))

    # Complex coherent amplitudes test orientation, including the nonreal
    # three-replica Bargmann phase and cyclicity.
    complex_model = merge(common, (hfun = θ -> [θ, θ^2],))
    z = Θ .+ im .* Θ.^2
    exact_complex = [S(t) * sum(conj(z[i]) * z[mod1(i + 1, length(z))] - abs2(z[i])
                               for i in eachindex(z)) for t in times]
    _, calculated_complex = log_bargmann_invariant_timecourse(Θ; times, complex_model...)
    @test calculated_complex ≈ exact_complex atol = 2e-10
    _, rotated = log_bargmann_invariant_timecourse(circshift(Θ, 1); times, complex_model...)
    @test rotated ≈ calculated_complex atol = 2e-10

    orders = [0, 1, 3]
    _, mixed = I_n_timecourse(0.3, orders; h = 0.02, times, common...)
    _, inserted = I_n_timecourse(0.3, orders; h = 0.02, times, method = :eq20, common...)
    analytic = [4S(t) * (1 - 2.0^(-n - 1)) for t in times, n in orders]
    @test mixed ≈ analytic atol = 2e-8
    @test inserted ≈ analytic atol = 2e-8
    @test mixed ≈ inserted atol = 2e-8
    # Reusing Richardson samples across orders/steps must agree with a fresh
    # calculation, and an identical call must add no invariant solves.
    order_cache = Dict()
    _, cached = I_n_timecourse(0.3, orders; h = 0.02, times, cache = order_cache, common...)
    @test cached == mixed
    cache_size = length(order_cache)
    _, cached_again = I_n_timecourse(0.3, orders; h = 0.02, times, cache = order_cache, common...)
    @test cached_again == cached
    @test length(order_cache) == cache_size
    _, refined_cached = I_n_timecourse(0.3, orders; h = 0.01, times, cache = order_cache, common...)
    _, refined_fresh = I_n_timecourse(0.3, orders; h = 0.01, times, common...)
    @test refined_cached == refined_fresh
    # The n=20 sum has large alternating coefficients. Check a known exact
    # field state at high order, independently of the OPO plot parameters.
    _, high_order = I_n_timecourse(0.3, [5, 10, 20]; h = 0.02, times = [1.0],
        merge(common, (tspan = (0.0, 1.0),))...)
    @test vec(high_order) ≈ [4S(1.0) * (1 - 2.0^(-n - 1)) for n in [5, 10, 20]] atol = 1e-6
    @test I_n(0.3, 3; h = 0.02, common...) ≈
        only(last(I_n_timecourse(0.3, 3; h = 0.02, times = [1.2], common...)))

    # A shared cache must not reuse an old time grid, initial state or model.
    cache = Dict()
    _, original = f_m_normalized_timecourse(0.3, 0; h = 0.02, times, cache, common...)
    count_before = length(cache)
    _, repeated = f_m_normalized_timecourse(0.3, 0; h = 0.02, times, cache, common...)
    @test repeated == original
    @test length(cache) == count_before
    _, different_grid = f_m_normalized_timecourse(0.3, 0; h = 0.02, times = [0.5], cache, common...)
    @test length(different_grid) == 1
    @test only(different_grid) ≈ 2S(0.5) atol = 2e-8
    stronger = merge(common, (hfun = θ -> [2θ, 0.0],))
    _, changed = f_m_normalized_timecourse(0.3, 0; h = 0.02, times, cache, stronger...)
    @test changed ≈ 4 .* original atol = 2e-8
    @test_throws ArgumentError log_bargmann_invariant_timecourse(Θ; times = [0.7, 0.4], common...)
    @test_throws ArgumentError log_bargmann_invariant_timecourse(Θ; times = [1.3], common...)
    @test_throws ArgumentError log_bargmann_invariant(Θ; periodic = false, common...)
    @test_throws DimensionMismatch normalized_mixed_d2_G_lm_timecourse(0.3, 0, 0;
        h = 0.02, times, logΛθ_vec = [log(2)], common...)
end
