#ifndef ERF_FIRE_CELL_COVERAGE_H
#define ERF_FIRE_CELL_COVERAGE_H

#include <ERF_FirePerimeter.H>

#include <AMReX_REAL.H>

namespace ERFFire
{

/**
 * Physical bounds of one two-dimensional Cartesian raster cell.
 *
 * All coordinates are SI meters.
 * A raster adapter is needed to construct these physical bounds 
 * from the host grid.
 */
struct FireCartesianCell2D
{
    amrex::Real xlo_m{};
    amrex::Real xhi_m{};
    amrex::Real ylo_m{};
    amrex::Real yhi_m{};
};

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

} // namespace ERFFire

#endif // ERF_FIRE_CELL_COVERAGE_H
