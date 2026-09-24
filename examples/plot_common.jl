ENV["MPLCONFIGDIR"] = joinpath(@__DIR__, ".mplconfig")
ENV["XDG_CACHE_HOME"] = ENV["MPLCONFIGDIR"]
mkpath(ENV["MPLCONFIGDIR"])

using JLD2
using Statistics
using PythonCall
const mpl = pyimport("matplotlib")
const use_tex = "--tex" in ARGS
mpl.use(use_tex ? "pgf" : "Agg")
# Import pyplot directly: a GUI wrapper can override PGF during initialization.
const plt = pyimport("matplotlib.pyplot")

# A fixed journal-column width replaces the rsmf dependency and style monkeypatch.
mpl.rcParams.update(Dict(
    "font.size" => 8, "axes.titlesize" => 8, "legend.fontsize" => 7,
    "font.family" => use_tex ? "serif" : "sans-serif",
    "figure.constrained_layout.use" => true,
    "pgf.texsystem" => "pdflatex", "text.usetex" => use_tex, "pgf.rcfonts" => false,
    "pgf.preamble" => raw"\usepackage{amsmath}\usepackage{amssymb}",
))
paper_figure() = plt.figure(figsize=(3.4, 3.4 * 0.55))

function opo_title(p; efficiency=false)
    title = raw"$\omega/\kappa = " * string(round(p.ω / p.κ; sigdigits=4)) *
            raw"$, $\chi/\kappa = " * string(round(p.χ / p.κ; sigdigits=4)) * raw"$"
    return efficiency ? title * raw", $\eta = " * string(p.η) * raw"$" : title
end

function plot_output_dir()
    directories = filter(startswith("--output-dir="), ARGS)
    length(directories) <= 1 || error("specify --output-dir only once")
    if !isempty(directories)
        any(arg -> arg in ("--smoke", "--archive"), ARGS) &&
            error("--output-dir cannot be combined with --smoke or --archive")
        directory = split(only(directories), '='; limit=2)[2]
        isempty(directory) && error("--output-dir requires a directory")
        return abspath(directory)
    end
    return "--smoke" in ARGS ? joinpath(@__DIR__, "output", "smoke") :
                              joinpath(@__DIR__, "output")
end

"""Block-average a dense curve for vector output; reject mismatched axes."""
function thin_series_for_plot(t::AbstractVector, y::AbstractVector; nmax::Int=4000)
    length(t) == length(y) || throw(DimensionMismatch("times and values differ"))
    nmax > 0 || throw(ArgumentError("nmax must be positive"))
    n = length(t)
    n <= nmax && return Float64.(t), Float64.(y)
    stride = cld(n, nmax)
    ranges = [i:min(i + stride - 1, n) for i in 1:stride:n]
    return [mean(@view t[r]) for r in ranges], [mean(@view y[r]) for r in ranges]
end

function load_plot_data(filename)
    # Choosing the archive must be explicit: its numbers predate this review.
    archive = "--archive" in ARGS
    archive && "--smoke" in ARGS && error("choose either --archive or --smoke")
    path = joinpath(archive ? (@__DIR__) : plot_output_dir(), filename)
    isfile(path) || error("Data file not found: $path. Run the production script first.")
    archive && @warn "Plotting original, uncorrected paper data" path
    println("Reading data from: ", path)
    payload = load(path, "payload")
    for field in (:Fs_rate, :Qc_rate, :Qu_rate, :qfi_env_rate, :qfi_joint_rate)
        values = getproperty(payload.series, field)
        all(isfinite, values) || error("$field contains nonfinite values; check the calculation")
        minimum(real, values) < -1e-7 && @warn "Negative information rate" field minimum=minimum(real, values)
    end
    if hasproperty(payload.series, :barg_rate)
        for (order, values) in zip(payload.series.barg_n_list, payload.series.barg_rate)
            all(isfinite, values) || error("I_$order contains nonfinite values")
            minimum(real, values) < -1e-7 && @warn "Negative replica rate; check convergence" order
        end
    end
    return payload
end
