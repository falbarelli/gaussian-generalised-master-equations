# Load the scientific routines directly from this checkout (no package install).
# The guard permits several example scripts to share one Julia session.
if !isdefined(@__MODULE__, :GaussianGMEs)
    include(joinpath(@__DIR__, "src", "GaussianGMEs.jl"))
end
using .GaussianGMEs
