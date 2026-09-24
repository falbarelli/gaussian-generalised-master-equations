# Independent oracle: integrate Yang et al.'s operator GRME directly in a
# truncated Fock basis, without using any Gaussian coefficient construction.
using OrdinaryDiffEqTsit5: Tsit5
using SciMLBase: ODEProblem, solve, successful_retcode

function fock_bargmann(Θ, cutoff, tf; Hfun, hfun, jfun, Lfun)
    annihilation = diagm(1 => sqrt.(1.0:(cutoff - 1)))
    quadratures = ((annihilation + annihilation') / sqrt(2),
                   (annihilation - annihilation') / (sqrt(2) * im))
    identity = Matrix{ComplexF64}(I, cutoff, cutoff)
    replicas = length(Θ)
    embed(op, site) = reduce(kron, [i == site ? op : identity for i in 1:replicas])
    dimension = cutoff^replicas
    drift = zeros(ComplexF64, dimension, dimension)
    monitored = Matrix{ComplexF64}[]
    unmonitored = Matrix{ComplexF64}[]
    Ω = [0 1; -1 0]
    for (site, θ) in enumerate(Θ)
        H, h = Hfun(θ), hfun(θ)
        hamiltonian = sum(H[i,j] * quadratures[i] * quadratures[j] / 2
                          for i in 1:2, j in 1:2)
        hamiltonian += sum((transpose(h) * Ω)[j] * quadratures[j] for j in 1:2)
        jump = sum(jfun(θ)[j] * quadratures[j] for j in 1:2)
        loss = sum(Lfun(θ)[j] * quadratures[j] for j in 1:2)
        drift += embed(-im * hamiltonian - (jump' * jump + loss' * loss) / 2, site)
        push!(monitored, embed(jump, site))
        push!(unmonitored, embed(loss, site))
    end
    function rhs!(dρ, ρ, _, t)
        dρ .= drift * ρ + ρ * drift'
        for α in 1:replicas
            next = mod1(α + 1, replicas)
            dρ .+= unmonitored[α] * ρ * unmonitored[α]'
            dρ .+= monitored[next] * ρ * monitored[α]'
        end
    end
    initial = zeros(ComplexF64, dimension, dimension)
    initial[1,1] = 1
    sol = solve(ODEProblem(rhs!, initial, (0.0, tf)), Tsit5();
                reltol=2e-12, abstol=2e-13, save_everystep=false)
    @test successful_retcode(sol)
    return tr(sol.u[end])
end

@testset "Replica trace against the operator GRME" begin
    # Weak squeezing keeps cutoff errors small. Nonzero drive and phase-varying
    # jumps exercise first moments, conjugation, and the oriented replica ring.
    model = (Hfun = θ -> [θ -0.12; -0.12 θ],
             hfun = θ -> [0.08 + 0.02θ, -0.03],
             jfun = θ -> sqrt(0.3 + 0.05θ) .* [1.0, im * exp(im * θ)],
             Lfun = θ -> sqrt(0.2) .* [1.0, im])
    for Θ in ([0.1, 0.25], [0.1, 0.25, -0.15])
        tf = 0.3
        kwargs = (; N=1, tspan=(0.0, tf), σ0_single=Matrix{Float64}(I, 2, 2),
                    d0_single=zeros(2), model..., reltol=2e-12, abstol=2e-13)
        gaussian = exp(log_bargmann_invariant(Θ; kwargs...))
        small = fock_bargmann(Θ, 3, tf; model...)
        large = fock_bargmann(Θ, 6, tf; model...)
        @test abs(large - gaussian) < abs(small - gaussian)
        @test large ≈ gaussian atol=3e-9 rtol=0
    end
end
