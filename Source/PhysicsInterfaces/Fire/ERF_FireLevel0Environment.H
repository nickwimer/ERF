#ifndef ERF_FIRE_LEVEL0_ENVIRONMENT_H_
#define ERF_FIRE_LEVEL0_ENVIRONMENT_H_

#include <Environment/ERF_FireFlatEnvironmentSampler.H>
#include <Environment/ERF_FireTerrainSurface.H>
#include <ERF_DataStruct.H>

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>

#include <vector>

namespace ERFFire
{

/**
 * Read-only ERF level-0 fields needed by Fire environment/geometry adapters.
 *
 * The adapter intentionally receives the relevant solver-scope metadata
 * explicitly instead of owning or depending on an ERF object. The existing
 * wind snapshot remains restricted to physically flat ConstantDz/StretchedDz
 * grids. Terrain-surface extraction additionally supports static-fitted
 * VariableDz geometry at level 0.
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
 * Extract the level-0 bottom nodal surface into the coupling-neutral bilinear
 * Fire terrain representation.
 *
 * VariableDz is accepted only for StaticFittedMesh. Moving fitted terrain, EB,
 * immersed forcing, buildings, and configured AMR levels above zero remain
 * outside the current scope.
 */
[[nodiscard]] FireTerrainSurface
make_erf_level0_terrain_surface (
    const ERFFireLevel0EnvironmentInputs& inputs);

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
 * Freeze the already-fillpatched level-0 horizontal wind at one constant
 * local height above static fitted terrain.
 *
 * Each native x/y face owns its own terrain-following vertical column. The
 * adapter derives that face's ground elevation and physical cell-center
 * heights from z_phys_nd, vertically interpolates the native staggered
 * velocity to reference_height_agl_m, and stores only the resulting 2-D
 * staggered reference-wind snapshot.
 *
 * z_phys_nd is authoritative for this terrain reduction; z_phys_cc is not
 * used to construct face heights. The returned sampler owns its host data and
 * never references ERF MultiFabs after this function returns.
 *
 * The VariableDz driver uses this snapshot together with the independently
 * extracted FireTerrainSurface for both supported coupling modes. This routine
 * only freezes/samples the mode-selected local-AGL horizontal reference wind; it
 * applies no WAF, slope forcing, combustion, or atmospheric feedback.
 */
FireFlatEnvironmentSampler
freeze_erf_level0_terrain_reference_wind_environment (
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
