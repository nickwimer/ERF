#ifndef ERF_FIRE_CHECKPOINT_H_
#define ERF_FIRE_CHECKPOINT_H_

#include <ERF_FireRuntimeOptions.H>
#include <ERF_FireSpreadOutput.H>

#include <ERF_DataStruct.H>

#include <iosfwd>
#include <string>

class ERFTerrainSource;

namespace ERFFire
{

/**
 * Write the Fire terrain-source restart policy.
 *
 * terrain_source must already have been prepared collectively by the ERF
 * caller when a regular text terrain source is active. This routine performs
 * no collective terrain loading.
 */
void write_fire_terrain_source_policy(
    std::ostream& stream,
    const SolverChoice& choices,
    const ERFTerrainSource* terrain_source);

/**
 * Validate the current terrain-source selection against a checkpoint policy.
 *
 * terrain_source must already have been prepared collectively by the ERF
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
