#ifndef ERF_FIRE_FRONT_COLLISION_H
#define ERF_FIRE_FRONT_COLLISION_H

#include <ERF_FireFrontTopology.H>

#include <optional>
#include <vector>

namespace ERFFire
{

struct FirePerimeterCollision
{
    amrex::Real motion_fraction{};
    FirePerimeterPinch pinch{};
    FireVec2 contact_point_m{};
};

/**
 * Locate the first nonadjacent-boundary contact while corresponding perimeter
 * vertices move linearly from start to end_vertices_m.
 *
 * start must be a valid simple perimeter. end_vertices_m may be
 * self-intersecting, but must contain exactly one position for every start
 * vertex. motion_fraction lies in [0,1].
 *
 * Returns std::nullopt when no topology-changing contact occurs.
 */
[[nodiscard]] std::optional<FirePerimeterCollision>
locate_first_perimeter_collision(
    const FirePerimeter& start,
    const std::vector<FireVec2>& end_vertices_m);

/**
 * Locate the first positive motion fraction at which the signed area of a
 * perimeter becomes exactly zero while corresponding vertices move linearly
 * from start to end_vertices_m.
 *
 * The represented signed area is quadratic in motion fraction. start must be
 * a valid counter-clockwise perimeter and end_vertices_m must preserve vertex
 * correspondence. Returns std::nullopt when no zero-area event occurs in
 * (0,1].
 */
[[nodiscard]] std::optional<amrex::Real>
locate_first_perimeter_area_collapse(
    const FirePerimeter& start,
    const std::vector<FireVec2>& end_vertices_m);

} // namespace ERFFire

#endif // ERF_FIRE_FRONT_COLLISION_H
