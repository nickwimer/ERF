#ifndef ERF_FIRE_BURNED_FRACTION_RASTER_H
#define ERF_FIRE_BURNED_FRACTION_RASTER_H

#include <ERF_FireBurnedFraction.H>
#include <ERF_FireRasterGeometry.H>

#include <AMReX_REAL.H>

#include <cstddef>
#include <vector>

namespace ERFFire
{

class FirePerimeter;

/**
 * Aggregate result of updating the persistent burned-fraction raster.
 *
 * burned_area_m2 is the total persistent burned area after the update.
 * newly_burned_area_m2 is the extensive area increment created by this update.
 */
struct FireRasterBurnedAreaUpdate
{
    amrex::Real burned_area_m2{};
    amrex::Real newly_burned_area_m2{};
};

/**
 * Persistent burned-area fraction on a uniform Cartesian raster.
 *
 * FirePerimeter remains authoritative. This raster stores only monotone burn
 * history derived from exact polygon-cell coverage. It deliberately stores no
 * arrival time and no copy of instantaneous coverage.
 */
class FireBurnedFractionRaster
{
public:
    explicit FireBurnedFractionRaster(
        const FireCartesianRasterGeometry2D& geometry);

    [[nodiscard]] const FireCartesianRasterGeometry2D&
    geometry () const noexcept
    {
        return geometry_;
    }

    [[nodiscard]] std::size_t cell_count () const noexcept
    {
        return burned_fraction_.size();
    }

    [[nodiscard]] FireCartesianCell2D cell_bounds(
        std::size_t i,
        std::size_t j) const;

    [[nodiscard]] amrex::Real burned_fraction(
        std::size_t i,
        std::size_t j) const;

    [[nodiscard]] amrex::Real burned_area_m2() const noexcept;

    [[nodiscard]] FireRasterBurnedAreaUpdate update_from_perimeter(
        const FirePerimeter& perimeter);

private:
    [[nodiscard]] std::size_t flat_index(
        std::size_t i,
        std::size_t j) const;

    FireCartesianRasterGeometry2D geometry_{};
    std::vector<amrex::Real> burned_fraction_;
};

} // namespace ERFFire

#endif // ERF_FIRE_BURNED_FRACTION_RASTER_H
