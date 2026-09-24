include(joinpath(@__DIR__, "opo_common.jl"))

if "--smoke" in ARGS
    run_oscillator(η=1.0, tFinal=0.5, dt_traj=0.005, dt_eval=0.1,
                   Ntraj=4, n_list=Int[], output_dir=joinpath(@__DIR__, "output", "smoke"))
else
    run_oscillator(η=1.0, tFinal=100.0, dt_traj=0.0005,
                   Ntraj=1000, n_list=Int[])
end
