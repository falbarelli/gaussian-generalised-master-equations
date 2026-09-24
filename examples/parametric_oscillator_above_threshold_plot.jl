include(joinpath(@__DIR__, "opo_plot_common.jl"))

payload = load_plot_data(joinpath("chi_0p7", "parametric_oscillator_data.jld2"))
fig = paper_figure()
ax = fig.subplots(1, 1)
# Growing replica curves occupy the lower right; keep the legends above them.
plot_replica_panel!(ax, payload; reference_anchor=(0.53, 1.0),
                    replica_anchor=(0.48, 0.57), replica_location="center")

outbase = joinpath(plot_output_dir(), "chi_0p7", "parametric_oscillator_rates_convergence")
mkpath(dirname(outbase))
fig.savefig(outbase * ".pdf")
fig.savefig(outbase * ".png"; dpi=180)
println("Saved plot to: ", outbase * ".pdf")
