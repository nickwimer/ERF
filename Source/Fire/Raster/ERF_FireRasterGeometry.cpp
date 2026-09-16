#include <ERF_FireRasterGeometry.H>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace ERFFire
{
namespace
{

amrex::Real
axis_boundary_m (
    amrex::Real origin_m,
    amrex::Real spacing_m,
    std::size_t index) noexcept
{
    return origin_m
        + static_cast<amrex::Real>(index) * spacing_m;
}

} // namespace

namespace detail
{

std::size_t
validate_fire_cartesian_raster_geometry (
    const FireCartesianRasterGeometry2D& geometry)
{
    if (geometry.nx == 0 || geometry.ny == 0) {
        throw std::invalid_argument(
            "Fire raster dimensions must be positive");
    }

    if (!std::isfinite(geometry.xlo_m)
        || !std::isfinite(geometry.ylo_m)
        || !std::isfinite(geometry.dx_m)
        || !std::isfinite(geometry.dy_m)) {
        throw std::invalid_argument(
            "Fire raster origin and spacing must be finite");
    }

    if (geometry.dx_m <= amrex::Real(0.0)
        || geometry.dy_m <= amrex::Real(0.0)) {
        throw std::invalid_argument(
            "Fire raster spacing must be strictly positive");
    }

    if (geometry.nx
        > std::numeric_limits<std::size_t>::max() / geometry.ny) {
        throw std::overflow_error(
            "Fire raster cell count overflows size_t");
    }

    const std::size_t cell_count = geometry.nx * geometry.ny;
    const amrex::Real nominal_cell_area_m2 =
        geometry.dx_m * geometry.dy_m;

    if (!std::isfinite(nominal_cell_area_m2)
        || nominal_cell_area_m2 <= amrex::Real(0.0)) {
        throw std::overflow_error(
            "Fire raster cell area is not finite");
    }

    const amrex::Real x_first_hi_m =
        axis_boundary_m(geometry.xlo_m, geometry.dx_m, 1);
    const amrex::Real y_first_hi_m =
        axis_boundary_m(geometry.ylo_m, geometry.dy_m, 1);
    const amrex::Real x_last_lo_m =
        axis_boundary_m(
            geometry.xlo_m, geometry.dx_m, geometry.nx - 1);
    const amrex::Real y_last_lo_m =
        axis_boundary_m(
            geometry.ylo_m, geometry.dy_m, geometry.ny - 1);
    const amrex::Real xhi_m =
        axis_boundary_m(
            geometry.xlo_m, geometry.dx_m, geometry.nx);
    const amrex::Real yhi_m =
        axis_boundary_m(
            geometry.ylo_m, geometry.dy_m, geometry.ny);

    if (!std::isfinite(xhi_m) || !std::isfinite(yhi_m)) {
        throw std::overflow_error(
            "Fire raster physical extent is not finite");
    }

    if (!(x_first_hi_m > geometry.xlo_m)
        || !(y_first_hi_m > geometry.ylo_m)
        || !(xhi_m > x_last_lo_m)
        || !(yhi_m > y_last_lo_m)) {
        throw std::invalid_argument(
            "Fire raster spacing is not representable at its physical coordinates");
    }

    const amrex::Real width_m = xhi_m - geometry.xlo_m;
    const amrex::Real height_m = yhi_m - geometry.ylo_m;
    const amrex::Real raster_area_m2 = width_m * height_m;

    if (!std::isfinite(width_m)
        || !std::isfinite(height_m)
        || width_m <= amrex::Real(0.0)
        || height_m <= amrex::Real(0.0)
        || !std::isfinite(raster_area_m2)
        || raster_area_m2 <= amrex::Real(0.0)) {
        throw std::overflow_error(
            "Fire raster represented physical area is not finite");
    }

    return cell_count;
}

std::size_t
fire_cartesian_raster_flat_index (
    const FireCartesianRasterGeometry2D& geometry,
    std::size_t i,
    std::size_t j)
{
    if (i >= geometry.nx || j >= geometry.ny) {
        throw std::out_of_range(
            "Fire raster cell index is outside the raster");
    }

    return j * geometry.nx + i;
}

FireCartesianCell2D
fire_cartesian_raster_cell_bounds (
    const FireCartesianRasterGeometry2D& geometry,
    std::size_t i,
    std::size_t j)
{
    (void)fire_cartesian_raster_flat_index(
        geometry, i, j);

    return {
        axis_boundary_m(geometry.xlo_m, geometry.dx_m, i),
        axis_boundary_m(geometry.xlo_m, geometry.dx_m, i + 1),
        axis_boundary_m(geometry.ylo_m, geometry.dy_m, j),
        axis_boundary_m(geometry.ylo_m, geometry.dy_m, j + 1)
    };
}

amrex::Real
fire_cartesian_cell_area_m2 (
    const FireCartesianCell2D& cell) noexcept
{
    return (cell.xhi_m - cell.xlo_m)
        * (cell.yhi_m - cell.ylo_m);
}

} // namespace detail
} // namespace ERFFire
