include(joinpath(@__DIR__, "opo_common.jl"))

# Paper parameters are the defaults in run_oscillator. Use --smoke for a quick
# end-to-end run; output/ keeps regenerated data separate from the paper archive.
if "--smoke" in ARGS
    run_oscillator(tFinal=0.5, dt_traj=0.005, dt_eval=0.1, Ntraj=4, n_list=[0, 1, 2],
                   output_dir=joinpath(@__DIR__, "output", "smoke"))
else
    run_oscillator()
end
