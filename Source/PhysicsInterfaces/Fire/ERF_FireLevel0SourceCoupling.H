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
 * Build one flat-grid native ERF level-0 conserved-variable source MultiFab
 * from a step-integrated Fire surface-feedback increment.
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

/**
 * Build one native ERF level-0 source tendency for static-fitted VariableDz
 * terrain.
 *
 * Each horizontal Fire cell owns one local AGL column. Local vertical faces
 * are the four-corner averages of z_phys_nd at each native vertical face,
 * referenced to that column's bottom face. The exponential deposition shape is
 * normalized independently in every column.
 *
 * Physical cell volume is authoritative from
 *
 *   V_phys = dx * dy * dz_computational * detJ_cc
 *
 * in the current Cartesian/map-plane Fire scope. Fire feedback remains an
 * extensive quantity per horizontal map-plane cell; no terrain-surface area
 * factor is introduced. t^n pressure diagnosis and the native rho_d*theta_d
 * and rho_d*q_v source formulas are identical to the flat projector.
 *
 * The VariableDz two-way driver uses this projector after advancing the
 * coupling-neutral Fire candidate and constructing the step-integrated
 * combustion feedback. one_way coupling never installs this atmospheric
 * source.
 */
[[nodiscard]] std::unique_ptr<amrex::MultiFab>
make_erf_fire_level0_terrain_source_tendency(
    const FireSurfaceFeedbackRaster& feedback,
    const ERFFireLevel0EnvironmentInputs& environment_inputs,
    const amrex::MultiFab& detJ_cc,
    const amrex::MultiFab& conserved_state_tn,
    MoistureType moisture_type,
    amrex::Real dt_s,
    ERFFireAtmosphericSourceOptions options = {});


} // namespace ERFFire

#endif
