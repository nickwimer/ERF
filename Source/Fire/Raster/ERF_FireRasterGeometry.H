#ifndef ERF_FIRE_RASTER_GEOMETRY_H
#define ERF_FIRE_RASTER_GEOMETRY_H

#include <AMReX_REAL.H>

#include <cstddef>

namespace ERFFire
{

/**
 * Dependency-light uniform Cartesian raster geometry.
 *
 * The represented raster spans physical boundaries generated from
 *
 *   x_i = xlo_m + i*dx_m
 *   y_j = ylo_m + j*dy_m
 *
 * so adjacent cells share bitwise-identical represented faces.
 */
struct FireCartesianRasterGeometry2D
{
    std::size_t nx{};
    std::size_t ny{};
    amrex::Real xlo_m{};
    amrex::Real ylo_m{};
    amrex::Real dx_m{};
    amrex::Real dy_m{};
};

/**
 * Physical bounds of one two-dimensional Cartesian raster cell.
 *
 * All coordinates are SI meters.
 */
struct FireCartesianCell2D
{
    amrex::Real xlo_m{};
    amrex::Real xhi_m{};
    amrex::Real ylo_m{};
    amrex::Real yhi_m{};
};

namespace detail
{

/**
 * Validate raster dimensions, coordinate representability, and finite
 * represented area. Returns nx*ny after overflow checking.
 */
[[nodiscard]] std::size_t validate_fire_cartesian_raster_geometry(
    const FireCartesianRasterGeometry2D& geometry);

/**
 * Row-major cell index for an already validated raster geometry.
 * Throws std::out_of_range when i or j is outside the raster.
 */
[[nodiscard]] std::size_t fire_cartesian_raster_flat_index(
    const FireCartesianRasterGeometry2D& geometry,
    std::size_t i,
    std::size_t j);

/**
 * Physical bounds using the single shared-face representation above.
 * The geometry must already have passed validation.
 */
[[nodiscard]] FireCartesianCell2D fire_cartesian_raster_cell_bounds(
    const FireCartesianRasterGeometry2D& geometry,
    std::size_t i,
    std::size_t j);

/**
 * Area of one already validated physical Cartesian cell.
 */
[[nodiscard]] amrex::Real fire_cartesian_cell_area_m2(
    const FireCartesianCell2D& cell) noexcept;

} // namespace detail
} // namespace ERFFire

#endif // ERF_FIRE_RASTER_GEOMETRY_H
