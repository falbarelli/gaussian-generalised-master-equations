include(joinpath(@__DIR__, "opo_plot_common.jl"))

any(arg -> arg in ("--archive", "--smoke"), ARGS) &&
    error("the comparison requires the two regenerated full datasets")

datasets = (
    load_plot_data("parametric_oscillator_data.jld2"),
    load_plot_data(joinpath("chi_0p1", "parametric_oscillator_data.jld2")),
)
common = first(datasets).params
for field in (:ω, :κ, :η, :ϕ, :meas, :nth, :tFinal)
    getproperty(common, field) == getproperty(last(datasets).params, field) ||
        error("the comparison expects matching $field values")
end

fig = plt.figure(figsize=(3.4, 3.6))
axes = fig.subplots(2, 1; sharex=true)
heading = raw"$\omega/\kappa = " * string(round(common.ω / common.κ; sigdigits=4)) *
          raw"$, $\eta = " * string(common.η) * raw"$"
fig.suptitle(heading; fontsize=8)

for (index, payload) in enumerate(datasets)
    plot_replica_panel!(axes[index - 1], payload;
        show_title=false, show_xlabel=index == length(datasets),
        panel_label=index == 1 ? "(a)" : "(b)", headroom=1.15)
end

outbase = joinpath(plot_output_dir(), "parametric_oscillator_rates_comparison")
mkpath(dirname(outbase))
fig.savefig(outbase * ".pdf")
fig.savefig(outbase * ".png"; dpi=180)
println("Saved plot to: ", outbase * ".pdf")
