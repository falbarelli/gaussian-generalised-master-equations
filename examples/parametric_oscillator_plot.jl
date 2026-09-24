include(joinpath(@__DIR__, "opo_plot_common.jl"))

payload = load_plot_data("parametric_oscillator_data.jld2")
fig = paper_figure()
ax = fig.subplots(1, 1)
plot_replica_panel!(ax, payload)

outbase = joinpath(plot_output_dir(), "parametric_oscillator_rates_convergence")
mkpath(dirname(outbase))
fig.savefig(outbase * ".pdf")
fig.savefig(outbase * ".png"; dpi=180)
println("Saved plot to: ", outbase * ".pdf")
