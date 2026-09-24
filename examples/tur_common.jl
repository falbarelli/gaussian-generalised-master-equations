# Analytic stationary OPO curves from the manuscript's TUR calculation.
# Extracted from TURGaussianMolmerQFI_OPA-plot.ipynb (research notebook).

"""
    opo_tur_rates(χ; κ=1.0)

Return the photon-counting current `J`, noise `D`, and the manuscript's
time-rescaling QFI rate `f` at zero detuning. Requires `κ > 0` and
`0 ≤ χ < κ/2` (the stationary regime). Ratios `D/J^2` and `1/f` diverge
at zero pump, so plots must exclude that endpoint.
"""
function opo_tur_rates(χ::Real; κ::Real=1.0)
    isfinite(κ) && κ > 0 || throw(ArgumentError("κ must be positive and finite"))
    isfinite(χ) && 0 <= χ < κ / 2 ||
        throw(ArgumentError("stationary rates require 0 ≤ χ < κ/2"))
    gap = κ^2 - 4χ^2
    J = 2κ * χ^2 / gap
    D = 4κ * χ^2 * (κ^4 + 2κ^2 * χ^2 + 8χ^4) / gap^3
    f = 2χ^2 / κ
    return (; J, D, f)
end
