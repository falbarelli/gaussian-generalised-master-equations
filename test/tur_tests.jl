include(joinpath(@__DIR__, "..", "examples", "tur_common.jl"))

@testset "Stationary OPO counting statistics and TUR" begin
    @test_throws ArgumentError opo_tur_rates(0.5)
    @test_throws ArgumentError opo_tur_rates(0.1; κ=0)
    @test opo_tur_rates(0.0) == (; J=0.0, D=0.0, f=0.0)
    for χ in (0.1, 0.2, 0.4)
        rates = opo_tur_rates(χ)
        # Independently differentiate the scaled cumulant generating function
        # in high precision, rather than comparing two copies of the rates.
        J, D = setprecision(256) do
            x = BigFloat(χ)
            h = big"1e-12"
            C(λ) = (2 - sqrt(1 + 4exp(im * λ) * x + 4x^2) -
                         sqrt(1 - 4exp(im * λ) * x + 4x^2)) / 4
            return real(-im * (C(h) - C(-h)) / (2h)),
                   real(-(C(h) - 2C(big"0") + C(-h)) / h^2)
        end
        @test rates.J ≈ J rtol=2e-13
        @test rates.D ≈ D rtol=2e-13
        @test rates.D / rates.J^2 >= 1 / rates.f

        # Time-rescaling QFI from the full two-sided master equation. Taking
        # a late-time slope removes the initial transient contribution.
        _, _, joint = qfi_timecourse_from_tsme_fidelity(0.0;
            ϵ=0.002, N=1, tspan=(0.0, 200.0),
            σ0=Matrix{Float64}(I, 2, 2), d0=zeros(2),
            Hfun=θ -> (1 + θ) * [0.0 -χ; -χ 0.0],
            Lfun=θ -> sqrt((1 + θ) / 2) * [1 im],
            saveat=[100.0, 200.0], reltol=1e-12, abstol=1e-13)
        @test (joint[2] - joint[1]) / 100 ≈ rates.f rtol=1e-6
    end
end
