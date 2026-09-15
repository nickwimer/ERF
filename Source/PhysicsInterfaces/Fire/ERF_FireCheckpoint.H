#ifndef ERF_FIRE_CHECKPOINT_H_
#define ERF_FIRE_CHECKPOINT_H_

#include <ERF_FireRuntimeOptions.H>
#include <ERF_FireSpreadOutput.H>

#include <ERF_DataStruct.H>

#include <AMReX_Geometry.H>
#include <AMReX_REAL.H>

#include <iosfwd>
#include <string>

class ERFTerrainSource;

namespace ERFFire
{


/**
 * ERF-owned inputs required to write persistent Fire checkpoint state.
 *
 * Any regular-text terrain source is resolved collectively by the
 * Fire context after the distributed raster write and before IO-rank-only
 * metadata serialization.
 */
struct ERFFireCheckpointWriteInputs
{
    const std::string& checkpoint_directory;
    const amrex::Geometry& level0_geometry;
    const SolverChoice& solver_choices;
    int level0_step{};
    amrex::Real level0_time_s{};
};

/**
 * ERF-owned inputs required to restore persistent Fire checkpoint state.
 *
 * Any regular-text terrain source is resolved collectively by the
 * Fire context after checkpoint-version discovery and before IO-rank-only
 * metadata validation.
 */
struct ERFFireCheckpointRestoreInputs
{
    const std::string& checkpoint_directory;
    const amrex::Geometry& level0_geometry;
    const SolverChoice& solver_choices;
    int level0_step{};
    amrex::Real level0_time_s{};
};

/**
 * Write the Fire terrain-source restart policy.
 *
 * terrain_source must already have been resolved collectively when a
 * regular-text terrain source is active. This routine performs no collective
 * terrain loading.
 */
void write_fire_terrain_source_policy(
    std::ostream& stream,
    const SolverChoice& choices,
    const ERFTerrainSource* terrain_source);

/**
 * Validate the current terrain-source selection against a checkpoint policy.
 *
 * terrain_source must already have been resolved collectively by the
 * caller. This routine performs only metadata/file validation.
 */
void validate_fire_terrain_source_restart_policy(
    const std::string& checkpoint_directory,
    const SolverChoice& choices,
    const ERFTerrainSource* terrain_source);

[[nodiscard]] bool same_fire_spread_config(
    const ERFFireSpreadConfig& lhs,
    const ERFFireSpreadConfig& rhs) noexcept;

void validate_fire_checkpoint_policy(
    const ERFFireCheckpointState& checkpoint,
    const ERFFireRuntimeOptions& options);

} // namespace ERFFire

#endif
