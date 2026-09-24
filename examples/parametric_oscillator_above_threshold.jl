include(joinpath(@__DIR__, "opo_common.jl"))

# Above the unconditional stability threshold; retain the same finite time window.
run_oscillator(χ=0.7, n_list=[1, 2, 3, 5],
               ϵ=0.002, reltol=2e-13, abstol=2e-14,
               tsme_steps=((5.0, 0.002), (10.0, 0.000125),
                           (15.0, 0.000015625), (Inf, 0.00000390625)),
               output_dir=joinpath(@__DIR__, "output", "chi_0p7"))
