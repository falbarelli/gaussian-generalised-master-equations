include(joinpath(@__DIR__, "opo_common.jl"))

# Keep the paper's other parameters and sampling, but reduce χ/κ to 0.1.
# Start with lower replica orders to assess their convergence in this regime.
run_oscillator(χ=0.1, n_list=[1, 2, 3, 5, 10, 15, 20],
               output_dir=joinpath(@__DIR__, "output", "chi_0p1"))
