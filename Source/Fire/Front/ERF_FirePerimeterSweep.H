#ifndef ERF_FIRE_PERIMETER_SWEEP_H_
#define ERF_FIRE_PERIMETER_SWEEP_H_

#include <ERF_FireFront.H>
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

/**
 * Linear dense-output interpolation for one fixed-topology FireFront sweep.
 *
 * Start/end fronts must have identical component ordering, roles, and vertex
 * counts. Matching vertices are interpolated linearly at alpha in [0,1].
 */
[[nodiscard]] FireFront
interpolate_fire_front_linear_sweep(
    const FireFront& start_front,
    const FireFront& end_front,
    amrex::Real alpha);

/**
 * Linear interpolation from a valid pre-event FireFront toward raw terminal
 * component vertices at a topology event.
 *
 * event_vertices_m must preserve start-front component ordering and vertex
 * correspondence. alpha lies in [0,1). The exact alpha == 1 geometry may be
 * degenerate or self-contacting and therefore cannot necessarily be
 * represented as a FireFront; callers must use the resolved post-event front
 * at the exact event instant.
 */
[[nodiscard]] FireFront
interpolate_fire_front_topology_event_sweep(
    const FireFront& start_front,
    const std::vector<std::vector<FireVec2>>& event_vertices_m,
    amrex::Real alpha);

} // namespace ERFFire

#endif
