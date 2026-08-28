#ifndef ERF_FIRE_SPREAD_OUTPUT_H_
#define ERF_FIRE_SPREAD_OUTPUT_H_

#include <ERF_FireRuntimeOptions.H>
#include <ERF_FireSpreadRuntime.H>

#include <AMReX_MultiFab.H>

#include <cstddef>
#include <iosfwd>
#include <string>

namespace ERFFire
{

class ERFFireSpreadRuntime;

/**
 * Write one deterministic visualization snapshot.
 *
 * The output consists of:
 *   perimeter_NNNNNN.csv -- authoritative vector perimeter
 *   raster_NNNNNN.csv    -- burned, arrival, and combustion history/state
 *   summary.csv           -- manifest with cumulative extensive combustion
 *                            diagnostics for each saved snapshot
 *
 * step_index==0 truncates summary.csv and begins a new manifest. A restart
 * into a new output directory also writes the manifest header when summary.csv
 * does not yet exist. Snapshot filenames are always overwritten individually;
 * unrelated files are never removed from output_dir.
 */
void write_erf_fire_spread_snapshot(
    const ERFFireSpreadRuntime& runtime,
    const std::string& output_dir,
    int step_index);

/**
 * Native ERF checkpoint payload for persistent Fire state.
 *
 * Runtime spread state is kept separate from adapter policy because coupling
 * mode and wind/source policy do not belong to the coupling-neutral spread
 * runtime. Output location/cadence and ignition construction controls are
 * intentionally not persisted: output policy may change across restart, while
 * ignition controls are no longer authoritative after the initial state exists.
 */
struct ERFFireCheckpointState
{
    ERFFireSpreadRuntimeState runtime_state{};
    ERFFireCouplingMode coupling_mode{ERFFireCouplingMode::OneWay};
    ERFFireWindMode wind_mode{ERFFireWindMode::DirectReference};
    amrex::Real reference_height_agl_m{};
    amrex::Real wind_adjustment_factor{amrex::Real(1.0)};
    amrex::Real feedback_extinction_depth_m{amrex::Real(50.0)};
};

void write_erf_fire_checkpoint_state(
    const ERFFireSpreadRuntime& runtime,
    const ERFFireRuntimeOptions& options,
    std::ostream& stream);

/**
 * Serialize one already-packed canonical Fire runtime state.
 *
 * This overload performs no MPI communication. It is the root-side endpoint
 * for collective_snapshot_state_to_io_rank() and preserves checkpoint format
 * version 1 exactly.
 */
void write_erf_fire_checkpoint_state(
    const ERFFireSpreadRuntimeState& state,
    const ERFFireRuntimeOptions& options,
    std::ostream& stream);

[[nodiscard]] ERFFireCheckpointState
read_erf_fire_checkpoint_state(std::istream& stream);

/**
 * Fixed component ordering for the distributed version-2 checkpoint raster.
 */
struct ERFFireCheckpointRasterComponents
{
    static constexpr int burned_fraction = 0;
    static constexpr int arrived = 1;
    static constexpr int first_arrival_time_s = 2;
    static constexpr int ignited_area_fraction = 3;
    static constexpr int remaining_dry_fuel_kg_m2 = 4;
    static constexpr int consumed_dry_fuel_kg_m2 = 5;
    static constexpr int sensible_energy_j_m2 = 6;
    static constexpr int water_released_kg_m2 = 7;
    static constexpr int component_count = 8;
};

/**
 * Root-owned non-raster metadata for checkpoint format version 2.
 *
 * Storage is O(Nperimeter): configuration/scalars are O(1), while the ordered
 * perimeter remains the intentionally replicated authoritative vector state.
 * Canonical raster vectors remain empty; cell-resolved state is stored in
 * FireStateRaster.
 */
struct ERFFireCheckpointV2Metadata
{
    ERFFireCheckpointState checkpoint{};
};

struct ERFFireCheckpointV3Metadata
{
    ERFFireCheckpointState checkpoint{};
};

[[nodiscard]] int
read_erf_fire_checkpoint_version(std::istream& stream);

void write_erf_fire_checkpoint_v2_metadata(
    const ERFFireSpreadRuntime& runtime,
    const ERFFireRuntimeOptions& options,
    std::ostream& stream);

[[nodiscard]] ERFFireCheckpointV2Metadata
read_erf_fire_checkpoint_v2_metadata(std::istream& stream);

void write_erf_fire_checkpoint_v3_metadata(
    const ERFFireSpreadRuntime& runtime,
    const ERFFireRuntimeOptions& options,
    std::ostream& stream);

[[nodiscard]] ERFFireCheckpointV3Metadata
read_erf_fire_checkpoint_v3_metadata(std::istream& stream);

/**
 * Build the distributed version-2 checkpoint payload.
 *
 * The BoxArray is deterministic from raster geometry and independent of MPI
 * rank count; the DistributionMapping is rebuilt for the active run.
 */
[[nodiscard]] amrex::MultiFab
make_erf_fire_checkpoint_v2_raster(
    const ERFFireSpreadRuntime& runtime);

} // namespace ERFFire

#endif
