include(joinpath(@__DIR__, "plot_common.jl"))

payload = load_plot_data("parametric_oscillator_eta=1_data.jld2")
s = payload.series
pms = payload.params
n_list = hasproperty(s, :barg_n_list) ? Int.(s.barg_n_list) : Int[]
plot_nmax = 4000

# --- Plot (tweak freely in this script) ---

fig = paper_figure()
ax = fig.subplots(1, 1)

t_joint_plot, qfi_joint_plot = thin_series_for_plot(s.t_joint_rate, s.qfi_joint_rate; nmax=plot_nmax)
ax.plot(pms.κ .* t_joint_plot, qfi_joint_plot; label="joint TSME-QFI", lw=2.0, color="black")

mask_env = isfinite.(s.qfi_env_rate)
if any(mask_env)
    t_env_plot, qfi_env_plot = thin_series_for_plot(s.t_env_rate[mask_env], s.qfi_env_rate[mask_env]; nmax=plot_nmax)
    ax.plot(
        pms.κ .* t_env_plot,
        qfi_env_plot,
        label = "env. TSME-QFI",
        lw = 2.0,
        color = "tab:orange",
        linestyle="-.",
    )
end

# Not to be removed, I will decide later which notation to use for the labels
# ax.plot(s.t_rate, s.Fs_rate; label=raw"$\mathcal{C}_{\mathrm{sig}}$", lw=2.0, color="tab:blue")
# ax.plot(s.t_rate, s.Qc_rate; label=raw"$\mathcal{Q}_{\mathrm{cond}}$", lw=2.0, color="tab:red")
# ax.plot(s.t_rate, s.Qu_rate; label=raw"$\mathcal{Q}_{\mathrm{unr}}$", lw=2.0, color="tab:green")

t_rate_plot, Fs_rate_plot = thin_series_for_plot(s.t_rate, s.Fs_rate; nmax=plot_nmax)
# _, Qc_rate_plot = thin_series_for_plot(s.t_rate, s.Qc_rate; nmax=plot_nmax)
_, Qu_rate_plot = thin_series_for_plot(s.t_rate, s.Qu_rate; nmax=plot_nmax)
ax.plot(pms.κ .* t_rate_plot, Fs_rate_plot; label=(pms.meas == "hom" ? "homodyne" : "heterodyne") * " sig. CFI", lw=2.0, color="tab:blue", linestyle="--")
# ax.plot(pms.κ .* t_rate_plot, Qc_rate_plot; label=raw"conditional-QFI sys.", lw=2.0, color="tab:red", linestyle=(0, (1, 1)) )
# linestyle=":")
ax.plot(pms.κ .* t_rate_plot, Qu_rate_plot; label=(pms.meas == "hom" ? "homodyne" : "heterodyne") * " unr. QFI", lw=2.0, color="tab:red", linestyle=(0, (1, 1)))

ax.set_yscale("log")
# Match the paper's lower display limit, while retaining every value in data.
# Validation above reports material negative rates before selecting plot limits.
"--smoke" in ARGS || ax.set_ylim(bottom=1e-3)
# ax.set_title(raw" Fisher information rates ($\eta=1$)")
ax.set_title(opo_title(pms))
ax.set_xlabel(raw"time $\kappa t$")
ax.set_ylabel("information / time")

# ax.grid(true, which="both", alpha=0.18, linestyle=":")
# ax.tick_params(direction="in", which="both", top=true, right=true)
# ax.legend(loc="upper left", frameon=false)
ax.legend(loc="lower right", frameon=true)

outbase = joinpath(plot_output_dir(), "parametric_oscillator_QFI_eta=1_paper")
mkpath(dirname(outbase))
fig.savefig(outbase * ".pdf")
fig.savefig(outbase * ".png"; dpi=180)
println("Saved plot to: ", outbase * ".pdf")
