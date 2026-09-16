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
 * Build the initial map-plane Fire state from ERF runtime options.
 *
 * The Fire history raster spans the same physical horizontal domain as ERF
 * level 0. When fire.n_cell is absent it inherits the level-0 horizontal cell
 * counts; otherwise the configured integer-commensurate Fire counts define an
 * independent raster. The initial ignition is one circular,
 * counter-clockwise perimeter.
 */
[[nodiscard]] std::unique_ptr<ERFFireSpreadRuntime>
make_erf_fire_spread_runtime(
    const ERFFireRuntimeOptions& options,
    const amrex::Geometry& geometry,
    amrex::Real initial_time_s);

} // namespace ERFFire

#endif
