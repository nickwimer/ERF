#ifndef ERF_FIRE_PERIMETER_SWEEP_H_
#define ERF_FIRE_PERIMETER_SWEEP_H_

#include <ERF_FirePerimeter.H>

#include <AMReX_REAL.H>

namespace ERFFire
{

/**
 * Linear dense-output interpolation for one fixed-topology physical sweep.
 *
 * Vertex i is interpolated between matching start/end vertices at alpha in
 * [0,1]. This is a geometry convention shared by arrival and combustion
 * timing; it is not a fire-spread law. Remeshing belongs after the physical
 * sweep represented here.
 */
[[nodiscard]] FirePerimeter
interpolate_fire_perimeter_linear_sweep(
    const FirePerimeter& start_perimeter,
    const FirePerimeter& end_perimeter,
    amrex::Real alpha);

} // namespace ERFFire

#endif
