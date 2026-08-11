#ifndef ERF_FIRE_PERIMETER_REMESHER_H
#define ERF_FIRE_PERIMETER_REMESHER_H

#include <ERF_FirePerimeter.H>

#include <cstddef>

namespace ERFFire
{

/**
 * Physical-space controls for local fire-perimeter remeshing.
 *
 * min_edge_length_m must be no greater than half max_edge_length_m. This
 * hysteresis guarantees that splitting an overlong straight edge cannot
 * immediately create edges that the coarsener considers undersampled.
 */
struct FirePerimeterRemeshOptions
{
    amrex::Real min_edge_length_m{0.5};
    amrex::Real max_edge_length_m{1.5};
    amrex::Real max_chord_error_m{0.05};
};

struct FirePerimeterRemeshStats
{
    std::size_t vertices_removed{};
    std::size_t vertices_added{};
};

struct FirePerimeterRemeshResult
{
    FirePerimeter perimeter;
    FirePerimeterRemeshStats stats;
};

/**
 * Locally adapt the vertex representation without smoothing the perimeter.
 *
 * Coarsening removes an undersampled vertex only when replacing its two
 * incident edges by the neighbor-to-neighbor chord:
 *   - stays within max_edge_length_m,
 *   - moves the removed vertex by at most max_chord_error_m relative to that
 *     chord,
 *   - preserves a positive signed area, and
 *   - does not introduce a new nonlocal segment intersection.
 *
 * Refinement then subdivides every overlong edge by linear interpolation.
 * Straight-edge subdivision exactly preserves the represented polygon.
 * Existing vertices are never smoothed or projected.
 */
[[nodiscard]] FirePerimeterRemeshResult remesh_perimeter(
    const FirePerimeter& perimeter,
    const FirePerimeterRemeshOptions& options);

} // namespace ERFFire

#endif // ERF_FIRE_PERIMETER_REMESHER_H
