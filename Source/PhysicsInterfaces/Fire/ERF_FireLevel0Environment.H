#ifndef ERF_FIRE_LEVEL0_ENVIRONMENT_H_
#define ERF_FIRE_LEVEL0_ENVIRONMENT_H_

#include <Environment/ERF_FireFlatEnvironmentSampler.H>
#include <ERF_DataStruct.H>

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <vector>

namespace ERFFire
{

/**
 * Read-only ERF level-0 fields needed to freeze one flat-grid atmospheric
 * environment snapshot for the fire model.
 *
 * The adapter intentionally receives the relevant solver-scope metadata
 * explicitly instead of owning or depending on an ERF object. Supports
 * only one configured AMR level and physically flat ConstantDz/StretchedDz
 * grids.
 */
struct ERFFireLevel0EnvironmentInputs
{
    const amrex::Geometry& geometry;
    const amrex::MultiFab& x_velocity;
    const amrex::MultiFab& y_velocity;
    const amrex::MultiFab& z_phys_cc;
    const amrex::MultiFab& z_phys_nd;
    MeshType mesh_type{MeshType::ConstantDz};
    TerrainType terrain_type{TerrainType::None};
    BuildingsType buildings_type{BuildingsType::None};
    int configured_max_level{0};
};

/**
 * Freeze the already-fillpatched ERF level-0 horizontal wind at one
 * atmospheric reference height above the flat physical ground.
 *
 * The returned sampler owns a replicated host snapshot and never references
 * the ERF MultiFabs after this function returns. Horizontal interpolation
 * remains on the native x/y face staggering.
 */
FireFlatEnvironmentSampler
freeze_erf_level0_environment (
    const ERFFireLevel0EnvironmentInputs& inputs,
    amrex::Real reference_height_agl_m);

/**
 * Return the common, horizontally flat level-0 nodal vertical faces as
 * heights above physical ground.
 *
 * The returned vector has nz+1 entries, begins at exactly zero AGL, and is
 * strictly increasing. Every nodal z plane is required to be horizontally
 * uniform. This uses the same flat-grid scope/layout contract as the frozen
 * Fire environment sampler.
 */
[[nodiscard]] std::vector<amrex::Real>
erf_fire_level0_flat_vertical_faces_agl (
    const ERFFireLevel0EnvironmentInputs& inputs);

} // namespace ERFFire

#endif
