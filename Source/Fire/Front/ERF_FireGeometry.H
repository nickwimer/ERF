#ifndef ERF_FIRE_GEOMETRY_H
#define ERF_FIRE_GEOMETRY_H

#include <ERF_FireTypes.H>

#include <cstddef>
#include <vector>

namespace ERFFire::detail
{

/**
 * Signed twice-area primitive for two-dimensional vectors.
 */
inline amrex::Real
cross_2d (const FireVec2& a, const FireVec2& b) noexcept
{
    return a.x * b.y - a.y * b.x;
}

/**
 * Signed area of an ordered polygon vertex sequence.
 *
 * Positive area denotes counter-clockwise orientation. The closing edge from
 * the final vertex to the first is implicit. This routine
 * operates on raw vertex vectors so callers can validate candidate geometry
 * before constructing FirePerimeter, which normalizes orientation.
 */
inline amrex::Real
signed_polygon_area_m2 (const std::vector<FireVec2>& vertices_m) noexcept
{
    if (vertices_m.size() < 3) {
        return amrex::Real(0.0);
    }

    const FireVec2& origin = vertices_m.front();

    amrex::Real twice_area = 0.0;
    for (std::size_t i = 1; i + 1 < vertices_m.size(); ++i) {
        const FireVec2 current{
            vertices_m[i].x - origin.x,
            vertices_m[i].y - origin.y
        };
        const FireVec2 next{
            vertices_m[i + 1].x - origin.x,
            vertices_m[i + 1].y - origin.y
        };

        twice_area += cross_2d(current, next);
    }

    return amrex::Real(0.5) * twice_area;
}

} // namespace ERFFire::detail

#endif // ERF_FIRE_GEOMETRY_H
