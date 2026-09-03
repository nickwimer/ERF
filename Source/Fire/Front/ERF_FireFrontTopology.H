#ifndef ERF_FIRE_FRONT_TOPOLOGY_H
#define ERF_FIRE_FRONT_TOPOLOGY_H

#include <ERF_FireFront.H>

#include <cstddef>
#include <vector>

namespace ERFFire
{

/**
 * Two explicit locations on the same simple perimeter where a pinch closes.
 *
 * Each location is identified by a zero-based edge index and a fraction in
 * [0,1] measured from that edge's start vertex toward its end vertex.
 */
struct FirePerimeterPinch
{
    std::size_t first_edge_index{};
    amrex::Real first_edge_fraction{};
    std::size_t second_edge_index{};
    amrex::Real second_edge_fraction{};
};

/**
 * Split one simple counter-clockwise perimeter at an explicitly identified
 * pinch into one outer boundary and one hole boundary.
 *
 * The operation changes topology but preserves represented burned area.
 * Detection/localization of a pinch is intentionally outside this primitive.
 */
[[nodiscard]] FireFront split_perimeter_at_pinch(
    const FirePerimeter& perimeter,
    const FirePerimeterPinch& pinch);

/**
 * Split raw perimeter vertices at an explicitly identified pinch.
 *
 * This overload is for the topology-change instant itself, where the raw
 * boundary can contain the nonadjacent self-contact represented by pinch and
 * therefore cannot first be constructed as a FirePerimeter.
 *
 * The two resulting component paths must individually form valid simple
 * FirePerimeters.
 */
[[nodiscard]] FireFront split_perimeter_at_pinch(
    const std::vector<FireVec2>& vertices_m,
    const FirePerimeterPinch& pinch);

/**
 * Split a raw self-contacting Hole boundary into two positive-area Holes.
 *
 * This is the Hole analogue of an Outer pinch. At the topology-change
 * instant the raw boundary cannot be constructed as a FirePerimeter.
 * Both resulting paths must be finite, simple, counter-clockwise
 * FirePerimeters, and their areas must partition the original Hole area.
 */
[[nodiscard]] std::vector<FirePerimeter>
split_hole_perimeter_at_pinch(
    const std::vector<FireVec2>& vertices_m,
    const FirePerimeterPinch& pinch);


/**
 * Return true only when two simple perimeters have strictly disjoint closed
 * polygonal regions.
 *
 * Boundary crossing, boundary touching, and containment all return false.
 * This predicate is intended as the exact narrow-phase test after any cheaper
 * bounding-box rejection.
 */
[[nodiscard]] bool perimeters_are_strictly_disjoint(
    const FirePerimeter& first,
    const FirePerimeter& second);


} // namespace ERFFire

#endif // ERF_FIRE_FRONT_TOPOLOGY_H
