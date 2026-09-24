include(joinpath(@__DIR__, "plot_common.jl"))

"""Draw the saved replica and reference curves with the shared paper style."""
function plot_replica_panel!(ax, payload; show_title=true, show_xlabel=true,
                             panel_label=nothing, headroom=1.1,
                             reference_anchor=(0.45, 1.0),
                             replica_anchor=nothing, replica_location="lower right")
    s = payload.series
    pms = payload.params
    n_list = hasproperty(s, :barg_n_list) ? Int.(s.barg_n_list) : Int[]
    plot_nmax = 4000

    mask_env = isfinite.(s.qfi_env_rate)
    if any(mask_env)
        t_env_plot, qfi_env_plot = thin_series_for_plot(s.t_env_rate[mask_env], s.qfi_env_rate[mask_env]; nmax = plot_nmax)
        ax.plot(pms.κ .* t_env_plot, qfi_env_plot; label = raw"$\mathcal{Q}[\rho_{\mathrm{E}}]$ ($\eta = 1$)", lw = 2.0, color = "tab:orange")
    end

    t_rate_plot, Fs_rate_plot = thin_series_for_plot(s.t_rate, s.Fs_rate; nmax = plot_nmax)
    ax.plot(pms.κ .* t_rate_plot, Fs_rate_plot; label = raw"$\mathcal{C}_{\mathrm{sig}}$", lw = 2.0, color = "tab:blue")
    reference_legend = ax.legend(loc="upper center", bbox_to_anchor=reference_anchor,
                                 frameon=false, handlelength=1.6,
                                 handletextpad=0.4, borderpad=0.2)
    ax.add_artist(reference_legend)

    # Stable styles for the paper orders; arbitrary other orders cycle through both
    # colour and dash patterns. The upper orders remain distinguishable in greyscale.
    barg_styles = Dict(
        1 => ("#0072B2", "--"), 2 => ("#009E73", "-."), 3 => ("#D55E00", ":"),
        5 => ("purple", "--"), 10 => ("magenta", ":"),
        15 => ("saddlebrown", "-."), 20 => ("dimgray", "--"),
        25 => ("#009E73", "-."), 30 => ("#D55E00", ":"),
    )
    fallback_colors = ("#0072B2", "#E69F00", "#CC79A7", "#009E73", "#D55E00", "#555555")
    fallback_lines = ("--", "-.", ":")
    replica_handles = Py[]
    for (idx, n) in enumerate(n_list)
        t_n, rate_n = thin_series_for_plot(s.barg_t[idx], s.barg_rate[idx]; nmax = plot_nmax)
        color, linestyle = get(barg_styles, n,
            (fallback_colors[mod1(idx, length(fallback_colors))],
             fallback_lines[mod1(idx, length(fallback_lines))]))
        line = ax.plot(pms.κ .* t_n, rate_n;
            label=raw"$\mathcal{Q}_{" * string(n) * raw"}$", lw=1.8, color, linestyle)
        push!(replica_handles, line[0])
    end

    show_title && ax.set_title(opo_title(pms; efficiency=true))
    show_xlabel && ax.set_xlabel(raw"time $\kappa t$")
    ax.set_ylabel("information / time")

    # Match the original linear frame: the η=1 reference is drawn in full, then
    # clipped by limits determined only by the signal and replica information.
    focus_max = max(maximum(s.Fs_rate), maximum(maximum, s.barg_rate; init=0.0))
    ymax = focus_max > 0 ? headroom * focus_max : 1.0
    ax.set_ylim(0.0, ymax)
    ax.set_xlim(0.0, pms.κ * pms.tFinal)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))

    if !isempty(replica_handles)
        # Use the band above the signal when it leaves room for the legend below
        # every replica curve in the rightmost part of the frame.
        if replica_anchor === nothing
            legend_y = 0.03
            tail_min = minimum(minimum(rate[t .>= 0.6pms.tFinal])
                               for (t, rate) in zip(s.barg_t, s.barg_rate))
            if tail_min - maximum(s.Fs_rate) > 0.30ymax
                legend_y += maximum(s.Fs_rate) / ymax
                # A fourth row needs a slightly lower anchor in the same empty band.
                length(replica_handles) > 6 && (legend_y -= 0.04)
            end
            replica_anchor = (0.98, legend_y)
        end
        ax.legend(handles=replica_handles, loc=replica_location, bbox_to_anchor=replica_anchor,
                  ncol=2, frameon=false, handlelength=2.0,
                  labelspacing=length(replica_handles) > 6 ? 0.15 : 0.5,
                  handletextpad=0.4, columnspacing=1.0, borderaxespad=0.0)
    end

    if panel_label !== nothing
        annotation = panel_label * raw" $\chi/\kappa = " *
                     string(round(pms.χ / pms.κ; sigdigits=4)) * raw"$"
        ax.text(0.97, 0.96, annotation; transform=ax.transAxes,
                ha="right", va="top", fontsize=8)
    end
    return ax
end
