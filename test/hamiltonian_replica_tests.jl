@testset "Hamiltonian replica sensitivities" begin
    H = [0.1 -0.45; -0.45 0.1]
    dH = Matrix{Float64}(I, 2, 2)
    j = 0.5 .* [1.0 im]
    L = copy(j)
    σ0 = Matrix{Float64}(I, 2, 2)
    times = [0.0, 0.2, 0.5, 1.0]
    common = (; H, dH, j, L, σ0_single = σ0, tspan = (0.0, 1.0), times)

    # Independent finite parameter displacements check the insertion locations,
    # ξ-derivative signs, normalization, and the two binomial sums together.
    orders = [0, 2, 4]
    actual_times, actual = I_n_hamiltonian_timecourse(orders; common...)
    fd_times, finite_difference = I_n_timecourse(0.1, orders;
        N = 1, tspan = common.tspan, times, σ0_single = σ0,
        d0_single = zeros(2), Hfun = θ -> [θ -0.45; -0.45 θ],
        hfun = _ -> zeros(2), jfun = _ -> j, Lfun = _ -> L,
        h = 0.04, method = :eq20, reltol = 2e-13, abstol = 2e-14)
    @test actual_times == fd_times == times
    @test actual ≈ finite_difference rtol = 2e-5 atol = 5e-10
    @test actual[1, :] == zeros(length(orders))
    @test maximum(abs, imag.(actual)) < 1e-12
    @test all(diff(real.(actual[end, :])) .> 0)
    _, scalar = I_n_hamiltonian_timecourse(2; common...)
    @test scalar ≈ actual[:, 2] rtol = 1e-11 atol = 1e-14
    _, reordered = I_n_hamiltonian_timecourse([4, 0, 4]; common...)
    @test reordered ≈ actual[:, [3, 1, 3]] rtol = 1e-11 atol = 1e-14

    # A passive oscillator initially in vacuum has a vacuum output for every
    # frequency. Without a monitored channel the detectable output is vacuum
    # even when squeezing changes the system state.
    _, passive = I_n_hamiltonian_timecourse([0, 4];
        common..., H = 0.1 .* dH)
    @test passive == zeros(ComplexF64, length(times), 2)
    _, unmonitored = I_n_hamiltonian_timecourse([0, 4];
        common..., j = zeros(ComplexF64, 1, 2))
    @test unmonitored == zeros(ComplexF64, length(times), 2)
    _, no_parameter = I_n_hamiltonian_timecourse(4;
        common..., dH = zeros(2, 2))
    @test no_parameter == zeros(ComplexF64, length(times))

    # Adding an uncoupled vacuum mode must leave the detectable information
    # unchanged. This also checks the multimode, interleaved block convention.
    H2 = zeros(4, 4); H2[1:2, 1:2] .= H; H2[3:4, 3:4] .= 0.3 .* dH
    dH2 = zeros(4, 4); dH2[1:2, 1:2] .= dH
    _, multimode = I_n_hamiltonian_timecourse(2;
        H = H2, dH = dH2, j = hcat(j, zeros(1, 2)),
        L = hcat(L, zeros(1, 2)), σ0_single = Matrix{Float64}(I, 4, 4),
        tspan = common.tspan, times)
    @test multimode ≈ scalar rtol = 1e-8 atol = 1e-13

    @test_throws ArgumentError I_n_hamiltonian_timecourse(Int[]; common...)
    @test_throws ArgumentError I_n_hamiltonian_timecourse(-1; common...)
    @test_throws DimensionMismatch I_n_hamiltonian_timecourse(0;
        common..., dH = zeros(4, 4))
    @test_throws ArgumentError I_n_hamiltonian_timecourse(0;
        common..., dH = [1.0 1.0; 0.0 1.0])
    @test_throws ArgumentError I_n_hamiltonian_timecourse(0;
        common..., H = H .+ 1im)
    @test_throws ArgumentError I_n_hamiltonian_timecourse(0;
        common..., σ0_single = [1.0 NaN; NaN 1.0])
    @test_throws ArgumentError I_n_hamiltonian_timecourse(0;
        common..., times = [0.5, 0.2])
    @test_throws ArgumentError I_n_hamiltonian_timecourse(0;
        common..., sensitivity_abstol = 0.0)
end
