#include <ERF_FireBurnedFractionRaster.H>
#include <ERF_FireCellCoverage.H>

#include <cmath>
#include <stdexcept>

namespace ERFFire
{

FireBurnedFractionRaster::FireBurnedFractionRaster (
    const FireCartesianRasterGeometry2D& geometry)
    : geometry_(geometry)
{
    const std::size_t cell_count =
        detail::validate_fire_cartesian_raster_geometry(
            geometry_);

    burned_fraction_.assign(
        cell_count,
        amrex::Real(0.0));
}

std::size_t
FireBurnedFractionRaster::flat_index (
    std::size_t i,
    std::size_t j) const
{
    return detail::fire_cartesian_raster_flat_index(
        geometry_, i, j);
}

FireCartesianCell2D
FireBurnedFractionRaster::cell_bounds (
    std::size_t i,
    std::size_t j) const
{
    return detail::fire_cartesian_raster_cell_bounds(
        geometry_, i, j);
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
                flat_index(i, j);
            const FireCartesianCell2D cell =
                cell_bounds(i, j);
            area_m2 += burned_fraction_[index]
                * detail::fire_cartesian_cell_area_m2(cell);
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
                detail::fire_cartesian_cell_area_m2(cell);

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
