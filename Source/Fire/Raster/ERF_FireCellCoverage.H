#ifndef ERF_FIRE_CELL_COVERAGE_H
#define ERF_FIRE_CELL_COVERAGE_H

#include <ERF_FireFront.H>
#include <ERF_FirePerimeter.H>
#include <ERF_FireRasterGeometry.H>

#include <AMReX_REAL.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ERFFire
{

/**
 * Exact polygon-cell intersection area for the current piecewise-linear
 * FirePerimeter and an axis-aligned Cartesian cell.
 *
 * The perimeter is authoritative. The raster cell is only a geometric query.
 * Boundary-only contact has zero area. Subject perimeters may be simple and
 * concave; self-intersecting/complex perimeter topology remains outside the
 * initial ERF-Fire scope.
 */
[[nodiscard]] amrex::Real fire_perimeter_cell_intersection_area_m2(
    const FirePerimeter& perimeter,
    const FireCartesianCell2D& cell);

/**
 * Fraction of cell area covered by the FirePerimeter interior.
 *
 * The result is guaranteed to lie in [0,1]. Only roundoff-scale excursions
 * outside that interval are clamped back to the nearest endpoint; legitimate
 * small positive or near-unit coverage values are preserved. A larger bounds
 * violation is treated as an internal geometry error.
 */
[[nodiscard]] amrex::Real fire_perimeter_cell_coverage_fraction(
    const FirePerimeter& perimeter,
    const FireCartesianCell2D& cell);

/**
 * Burned-region intersection area represented by a topology-capable front.
 *
 * Outer components contribute area and Hole components subtract area.
 */
[[nodiscard]] inline amrex::Real fire_front_cell_intersection_area_m2(
    const FireFront& front,
    const FireCartesianCell2D& cell)
{
    amrex::Real area_m2 = 0.0;

    for (const FireFrontComponent& component : front.components()) {
        const amrex::Real component_area_m2 =
            fire_perimeter_cell_intersection_area_m2(
                component.perimeter,
                cell);

        if (component.role == FireFrontRole::Outer) {
            area_m2 += component_area_m2;
        } else {
            area_m2 -= component_area_m2;
        }
    }

    const amrex::Real cell_area_m2 =
        (cell.xhi_m - cell.xlo_m)
        * (cell.yhi_m - cell.ylo_m);
    const amrex::Real tolerance =
        amrex::Real(1024.0)
        * std::numeric_limits<amrex::Real>::epsilon();

    if (!std::isfinite(area_m2)
        || !std::isfinite(cell_area_m2)
        || !(cell_area_m2 > amrex::Real(0.0))
        || area_m2 < -tolerance * cell_area_m2
        || area_m2
            > cell_area_m2 * (amrex::Real(1.0) + tolerance)) {
        throw std::runtime_error(
            "Fire front-cell intersection area is outside cell bounds");
    }

    return std::clamp(
        area_m2,
        amrex::Real(0.0),
        cell_area_m2);
}

/**
 * Fraction of cell area covered by the burned region represented by front.
 */
[[nodiscard]] inline amrex::Real fire_front_cell_coverage_fraction(
    const FireFront& front,
    const FireCartesianCell2D& cell)
{
    const amrex::Real cell_area_m2 =
        (cell.xhi_m - cell.xlo_m)
        * (cell.yhi_m - cell.ylo_m);
    const amrex::Real area_m2 =
        fire_front_cell_intersection_area_m2(front, cell);

    return area_m2 / cell_area_m2;
}

} // namespace ERFFire

#endif // ERF_FIRE_CELL_COVERAGE_H
