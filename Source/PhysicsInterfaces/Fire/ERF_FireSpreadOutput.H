#ifndef ERF_FIRE_SPREAD_OUTPUT_H_
#define ERF_FIRE_SPREAD_OUTPUT_H_

#include <ERF_FireRuntimeOptions.H>
#include <ERF_FireSpreadRuntime.H>

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

[[nodiscard]] ERFFireCheckpointState
read_erf_fire_checkpoint_state(std::istream& stream);

} // namespace ERFFire

#endif
