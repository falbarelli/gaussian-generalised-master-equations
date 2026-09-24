# Analytic Figure 2; no ODE calculation or input dataset is required.
include(joinpath(@__DIR__, "plot_common.jl"))
include(joinpath(@__DIR__, "tur_common.jl"))
using Printf

κ = 1.0
# Keep the original notebook grid, excluding the singular zero-pump endpoint.
χ_values = collect(range(0.0, 0.4999999κ; length=5000))[2:end]
rates = opo_tur_rates.(χ_values; κ)
noise_to_current = [r.D / r.J^2 for r in rates]
inverse_qfi = [1 / r.f for r in rates]

fig = plt.figure(figsize=(3.4, 1.7))
ax = fig.subplots(1, 1)
ax.plot(χ_values ./ κ, noise_to_current;
        label=raw"$\mathtt{D}/\mathtt{J}^2$", lw=2.0, color="black")
ax.plot(χ_values ./ κ, inverse_qfi;
        label=raw"$1/f$", lw=2.0, color="tab:blue", linestyle="--")
ax.set_yscale("log")
ax.set_xlabel(raw"$\chi/\kappa$")
ax.legend(bbox_to_anchor=(0.7, 1), loc="upper right", frameon=true)

outbase = joinpath(plot_output_dir(), "parametric_oscillator_TUR")
mkpath(dirname(outbase))
fig.savefig(outbase * ".pdf")
fig.savefig(outbase * ".png"; dpi=180)
open(outbase * ".tsv", "w") do io
    println(io, "chi_over_kappa\tcurrent\tnoise\tqfi_rate\tnoise_over_current_squared\tinverse_qfi_rate")
    for (χ, r, ratio, inverse) in zip(χ_values, rates, noise_to_current, inverse_qfi)
        @printf(io, "%.16g\t%.16g\t%.16g\t%.16g\t%.16g\t%.16g\n",
                χ / κ, r.J, r.D, r.f, ratio, inverse)
    end
end
println("Saved TUR figure and data to: ", outbase)
