module GaussianGMEs

include("GaussianMoments.jl")
include("ReplicaCoeffs.jl")
include("TSME.jl")
include("BargmannInvariants.jl")
include("HamiltonianReplicaSensitivities.jl")
include("UnravellingQFI.jl")

using .GaussianMoments
using .ReplicaCoeffs
using .TSME
using .BargmannInvariants
using .HamiltonianReplicaSensitivities
using .UnravellingQFI

export PackSpec,
    unpack_state,
    pack_deriv!,
    symplectic_Ω,
    gaussian_rhs_general!,
    solve_gaussian_odes,
    get_σ_d_ξ,
    trace_norm_gaussian_nu,
    blockdiag,
    sym,
    skew,
    getfield_nt,
    build_replica_coeffs,
    replica_params,
    build_models_from_theta,
    replicate_initial_moments,
    build_tsme_coeffs,
    tsme_params,
    solve_tsme_moments,
    overlap_from_solution,
    trace_norm_from_solution,
    fidelity_from_solution,
    fidelity_timecourse_from_solution,
    absoverlap_timecourse_from_solution,
    qfi_timecourse_from_tsme_fidelity,
    qfi_from_tsme_fidelity,
    log_bargmann_invariant,
    log_bargmann_invariant_timecourse,
    log_Lambda,
    log_Lambda_timecourse,
    D_lm,
    theta_list_G,
    mixed_d2_central,
    normalized_mixed_d2_G_lm_timecourse,
    f_m_normalized_timecourse,
    I_n_timecourse,
    I_n_hamiltonian_timecourse,
    normalized_mixed_d2_G_lm,
    f_m_normalized,
    theta_list_eq20_term,
    f_m_normalized_eq20_fd,
    I_n,
    get_unravelling_matrices,
    solve_unravelling,
    simulate_unravelling

end
