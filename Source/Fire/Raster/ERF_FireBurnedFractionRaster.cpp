#include <ERF_FireBurnedFractionRaster.H>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace ERFFire
{
namespace
{

struct ValidatedRasterGeometry
{
    std::size_t cell_count;
};

amrex::Real
axis_boundary_m (
    amrex::Real origin_m,
    amrex::Real spacing_m,
    std::size_t index) noexcept
{
    return origin_m
        + static_cast<amrex::Real>(index) * spacing_m;
}

amrex::Real
cell_area_m2 (
    const FireCartesianCell2D& cell) noexcept
{
    return (cell.xhi_m - cell.xlo_m)
        * (cell.yhi_m - cell.ylo_m);
}

amrex::Real
represented_cell_area_m2 (
    const FireCartesianRasterGeometry2D& geometry,
    std::size_t i,
    std::size_t j) noexcept
{
    const FireCartesianCell2D cell{
        axis_boundary_m(geometry.xlo_m, geometry.dx_m, i),
        axis_boundary_m(geometry.xlo_m, geometry.dx_m, i + 1),
        axis_boundary_m(geometry.ylo_m, geometry.dy_m, j),
        axis_boundary_m(geometry.ylo_m, geometry.dy_m, j + 1)
    };

    return cell_area_m2(cell);
}

ValidatedRasterGeometry
validate_geometry (
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

    return {cell_count};
}

} // namespace

FireBurnedFractionRaster::FireBurnedFractionRaster (
    const FireCartesianRasterGeometry2D& geometry)
    : geometry_(geometry)
{
    const ValidatedRasterGeometry validated =
        validate_geometry(geometry_);

    burned_fraction_.assign(
        validated.cell_count,
        amrex::Real(0.0));
}

std::size_t
FireBurnedFractionRaster::flat_index (
    std::size_t i,
    std::size_t j) const
{
    if (i >= geometry_.nx || j >= geometry_.ny) {
        throw std::out_of_range(
            "Fire raster cell index is outside the raster");
    }

    return j * geometry_.nx + i;
}

FireCartesianCell2D
FireBurnedFractionRaster::cell_bounds (
    std::size_t i,
    std::size_t j) const
{
    (void)flat_index(i, j);

    return {
        axis_boundary_m(geometry_.xlo_m, geometry_.dx_m, i),
        axis_boundary_m(geometry_.xlo_m, geometry_.dx_m, i + 1),
        axis_boundary_m(geometry_.ylo_m, geometry_.dy_m, j),
        axis_boundary_m(geometry_.ylo_m, geometry_.dy_m, j + 1)
    };
}

amrex::Real
FireBurnedFractionRaster::burned_fraction (
    std::size_t i,
    std::size_t j) const
{
    return burned_fraction_[flat_index(i, j)];
}

amrex::Real
FireBurnedFractionRaster::burned_area_m2 () const noexcept
{
    amrex::Real area_m2 = 0.0;

    for (std::size_t j = 0; j < geometry_.ny; ++j) {
        for (std::size_t i = 0; i < geometry_.nx; ++i) {
            const std::size_t index =
                j * geometry_.nx + i;
            area_m2 += burned_fraction_[index]
                * represented_cell_area_m2(geometry_, i, j);
        }
    }

    return area_m2;
}

FireRasterBurnedAreaUpdate
FireBurnedFractionRaster::update_from_perimeter (
    const FirePerimeter& perimeter)
{
    std::vector<amrex::Real> next_burned_fraction =
        burned_fraction_;

    amrex::Real newly_burned_area_m2 = 0.0;
    amrex::Real burned_area_m2 = 0.0;

    for (std::size_t j = 0; j < geometry_.ny; ++j) {
        for (std::size_t i = 0; i < geometry_.nx; ++i) {
            const std::size_t index = flat_index(i, j);
            const FireCartesianCell2D cell =
                cell_bounds(i, j);
            const amrex::Real coverage =
                fire_perimeter_cell_coverage_fraction(
                    perimeter,
                    cell);

            const FireBurnedFractionUpdate update =
                update_fire_burned_fraction(
                    burned_fraction_[index],
                    coverage);
            const amrex::Real represented_area_m2 =
                cell_area_m2(cell);

            next_burned_fraction[index] =
                update.burned_fraction;
            newly_burned_area_m2 +=
                update.newly_burned_fraction * represented_area_m2;
            burned_area_m2 +=
                update.burned_fraction * represented_area_m2;
        }
    }

    if (!std::isfinite(newly_burned_area_m2)
        || !std::isfinite(burned_area_m2)) {
        throw std::overflow_error(
            "Fire raster burned-area accounting is not finite");
    }

    burned_fraction_.swap(next_burned_fraction);

    return {
        burned_area_m2,
        newly_burned_area_m2
    };
}

} // namespace ERFFire
