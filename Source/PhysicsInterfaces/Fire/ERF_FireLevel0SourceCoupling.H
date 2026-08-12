#ifndef ERF_FIRE_LEVEL0_SOURCE_COUPLING_H_
#define ERF_FIRE_LEVEL0_SOURCE_COUPLING_H_

#include <ERF_FireAtmosphericSource.H>
#include <ERF_FireLevel0Environment.H>
#include <ERF_FireSurfaceFeedback.H>

#include <ERF_DataStruct.H>

#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include <memory>

namespace ERFFire
{

/**
 * Build one native ERF level-0 conserved-variable source MultiFab from a
 * step-integrated Fire surface-feedback increment.
 *
 * The atmosphere state is the already-fillpatched t^n conserved state used for
 * this explicit coupling step. Total pressure is diagnosed from ERF's dry
 * rho_d*theta_d state and water-vapor mixing ratio, then performs the
 * conservative vertical thermodynamic projection.
 *
 * The returned MultiFab has the same BoxArray, DistributionMapping, and
 * component count as conserved_state_tn, with zero ghost cells. Only
 * RhoTheta_comp and RhoQ1_comp are nonzero. The routine does not mutate ERF
 * state and does not encode one-way/two-way policy.
 */
[[nodiscard]] std::unique_ptr<amrex::MultiFab>
make_erf_fire_level0_source_tendency(
    const FireSurfaceFeedbackRaster& feedback,
    const ERFFireLevel0EnvironmentInputs& environment_inputs,
    const amrex::MultiFab& conserved_state_tn,
    MoistureType moisture_type,
    amrex::Real dt_s,
    ERFFireAtmosphericSourceOptions options = {});

} // namespace ERFFire

#endif
