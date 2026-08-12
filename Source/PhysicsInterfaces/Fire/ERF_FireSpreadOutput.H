#ifndef ERF_FIRE_SPREAD_OUTPUT_H_
#define ERF_FIRE_SPREAD_OUTPUT_H_

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
 * step_index==0 truncates summary.csv and begins a new manifest. Snapshot
 * filenames are always overwritten individually; unrelated files are never
 * removed from output_dir.
 */
void write_erf_fire_spread_snapshot(
    const ERFFireSpreadRuntime& runtime,
    const std::string& output_dir,
    int step_index);

} // namespace ERFFire

#endif
