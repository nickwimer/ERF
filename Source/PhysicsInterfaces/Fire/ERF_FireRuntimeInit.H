#ifndef ERF_FIRE_RUNTIME_INIT_H_
#define ERF_FIRE_RUNTIME_INIT_H_

#include <ERF_FireRuntimeOptions.H>

#include <AMReX_Geometry.H>

#include <memory>

namespace ERFFire
{

class ERFFireSpreadRuntime;
struct ERFFireSpreadConfig;

[[nodiscard]] ERFFireSpreadConfig
make_erf_fire_spread_config(
    const ERFFireRuntimeOptions& options,
    const amrex::Geometry& geometry);

/**
 * Build the initial flat-grid fire state from ERF runtime options.
 *
 * The level-0 ERF horizontal cell geometry is also the fire history raster
 * geometry. The initial ignition is one circular, counter-clockwise perimeter.
 */
[[nodiscard]] std::unique_ptr<ERFFireSpreadRuntime>
make_erf_fire_spread_runtime(
    const ERFFireRuntimeOptions& options,
    const amrex::Geometry& geometry,
    amrex::Real initial_time_s);

} // namespace ERFFire

#endif
