#ifndef ERF_FIRE_CELL_ARRIVAL_H
#define ERF_FIRE_CELL_ARRIVAL_H

#include <ERF_FireCellCoverage.H>

#include <AMReX_REAL.H>

#include <vector>

namespace ERFFire
{

/**
 * Result of one first-arrival query.
 *
 * arrival_time_s is meaningful only when arrived is true.
 */
struct FireCellArrivalResult
{
    bool arrived{};
    amrex::Real arrival_time_s{};
};

/**
 * First positive-area arrival during a linearly interpolated perimeter sweep.
 *
 * The start and end perimeters must have the same vertex count and matching
 * vertex ordering. For alpha in [0,1],
 *
 *   p_i(alpha) = (1-alpha) p_i(start) + alpha p_i(end)
 *
 * and time is interpolated over [start_time_s, end_time_s].
 *
 * This is a numerical dense-output convention for one fixed-topology physical
 * propagation step, not a fire-spread law. Remeshing must occur after the
 * physical sweep used for this query.
 *
 * First arrival is the infimum of times at which the cell has strictly
 * positive intersection area with the fire region. Boundary-only point or
 * edge contact has zero area, so the infimum can occur at a zero-area contact
 * instant even though every later time in the local sweep has positive area.
 *
 * The initial implementation assumes the physical sweep is outward/monotone
 * for this cell: if the end perimeter has zero positive-area coverage, the
 * result is no arrival.
 *
 * If the cell already has positive-area coverage at start_time_s, that time is
 * returned exactly. Otherwise the crossing is bracketed by bisection and the
 * returned time is the first known positive-area upper bound, with a bracket
 * width no greater than time_tolerance_s unless floating-point time resolution
 * is exhausted first.
 */
[[nodiscard]] FireCellArrivalResult
fire_cell_first_arrival_time_linear_sweep(
    const FirePerimeter& start_perimeter,
    const FirePerimeter& end_perimeter,
    const FireCartesianCell2D& cell,
    amrex::Real start_time_s,
    amrex::Real end_time_s,
    amrex::Real time_tolerance_s);

/**
 * First positive-area arrival during motion from one simple perimeter to its
 * first topology-changing contact.
 *
 * event_vertices_m preserves the start-perimeter vertex correspondence at
 * the contact instant. For every alpha < 1 the boundary is the ordinary
 * linearly interpolated simple perimeter. The exact alpha == 1 endpoint is
 * represented by event_front rather than by the raw self-contact boundary.
 */
[[nodiscard]] FireCellArrivalResult
fire_cell_first_arrival_time_topology_event_sweep(
    const FirePerimeter& start_perimeter,
    const std::vector<FireVec2>& event_vertices_m,
    const FireFront& event_front,
    const FireCartesianCell2D& cell,
    amrex::Real start_time_s,
    amrex::Real event_time_s,
    amrex::Real time_tolerance_s);

} // namespace ERFFire

#endif // ERF_FIRE_CELL_ARRIVAL_H
