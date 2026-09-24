# Validate the added order-25/30 curves at the OPO paper parameters.
# Run: julia --project=. validation/replica_order30.jl
# Add --finite-difference for the independent, slower step-size comparison.
# These checks concern numerical accuracy, not convergence to the exact QFI.
include(joinpath(@__DIR__, "..", "ReplicaME.jl"))
using LinearAlgebra
using Printf

BLAS.set_num_threads(1)
orders = [5, 10, 15, 20, 25, 30]
times = collect(0.0:0.1:20.0)
model = (
    N=1, tspan=(0.0, last(times)), times,
    σ0_single=Matrix{Float64}(I, 2, 2), d0_single=zeros(2),
    Hfun=θ -> [θ -0.45; -0.45 θ], hfun=θ -> zeros(2),
    jfun=θ -> 0.5 .* [1.0 im], Lfun=θ -> 0.5 .* [1.0 im],
    reltol=2e-13, abstol=2e-14,
)
outdir = joinpath(@__DIR__, "output")
mkpath(outdir)

function report(values)
    for t in (0.1, 0.5, 1.0, 5.0, 10.0, 15.0, 20.0)
        row = findfirst(==(t), times)
        @printf("  t=%g: I25/t=%.12g, I30/t=%.12g, Im I30/t=%.3g\n", t,
                real(values[row,end-1])/t, real(values[row,end])/t,
                imag(values[row,end])/t)
    end
    flush(stdout)
end

if "--finite-difference" in ARGS
    open(joinpath(outdir, "replica_order30_late.tsv"), "w") do io
        println(io, "method\th\ttime\torder\tI_n_real\tI_n_imag")
        cache = Dict()
        for h in (0.02, 0.01)
            println("Finite-difference check: h=", h); flush(stdout)
            elapsed = @elapsed _, values = I_n_timecourse(0.1, orders; h, cache, model...)
            @printf("h=%g: %.1f seconds\n", h, elapsed)
            for (column, order) in enumerate(orders), (row, t) in enumerate(times)
                value = values[row, column]
                println(io, join((:mixed, h, t, order, real(value), imag(value)), '\t'))
            end
            flush(io)
            report(values)
        end
    end
else
    tolerances = "--coarse-only" in ARGS ? (2e-12,) : (2e-12, 2e-13)
    open(joinpath(outdir, "replica_order30_sensitivity.tsv"), "w") do io
        println(io, "reltol\tabstol\tsensitivity_abstol\ttime\torder\tI_n_real\tI_n_imag")
        for tolerance in tolerances
            println("Sensitivity check: reltol=", tolerance); flush(stdout)
            elapsed = @elapsed _, values = I_n_hamiltonian_timecourse(orders;
                H=model.Hfun(0.1), dH=Matrix{Float64}(I,2,2),
                j=model.jfun(0.1), L=model.Lfun(0.1), σ0_single=model.σ0_single,
                tspan=model.tspan, times, reltol=tolerance,
                abstol=tolerance/10, sensitivity_abstol=1e-22,
            )
            @printf("reltol=%g: %.1f seconds\n", tolerance, elapsed)
            for (column, order) in enumerate(orders), (row, t) in enumerate(times)
                value = values[row, column]
                println(io, join((tolerance, tolerance/10, 1e-22, t, order,
                                 real(value), imag(value)), '\t'))
            end
            flush(io)
            report(values)
        end
    end
end
