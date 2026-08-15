#ifndef ERF_FIRE_BURNED_FRACTION_RASTER_H
#define ERF_FIRE_BURNED_FRACTION_RASTER_H

#include <ERF_FireBurnedFraction.H>
#include <ERF_FireRasterGeometry.H>
#include <ERF_FireSurfaceLayout.H>

#include <AMReX_MultiFab.H>
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

struct FireBurnedFractionRasterState
{
    std::vector<amrex::Real> burned_fraction;
};

/**
 * Persistent burned-area fraction on a uniform Cartesian raster.
 *
 * FirePerimeter remains authoritative. This raster stores only monotone burn
 * history derived from exact polygon-cell coverage. It deliberately stores no
 * arrival time and no copy of instantaneous coverage.
 *
 * Expensive perimeter/cell coverage work is distributed over an AMReX
 * MultiFab surface layout. During this migration stage, a canonical row-major
 * vector is retained as a synchronized compatibility representation for the
 * existing checkpoint/output/combustion APIs. The MultiFab and canonical
 * vector are committed atomically after a successful distributed update.
 */
class FireBurnedFractionRaster
{
public:
    explicit FireBurnedFractionRaster(
        const FireCartesianRasterGeometry2D& geometry);

    FireBurnedFractionRaster(
        const FireCartesianRasterGeometry2D& geometry,
        FireBurnedFractionRasterState state);

    FireBurnedFractionRaster(
        const FireBurnedFractionRaster& other);
    FireBurnedFractionRaster& operator=(
        const FireBurnedFractionRaster& other);
    FireBurnedFractionRaster(
        FireBurnedFractionRaster&&) noexcept = default;
    FireBurnedFractionRaster& operator=(
        FireBurnedFractionRaster&&) noexcept = default;

    [[nodiscard]] FireBurnedFractionRasterState
    snapshot_state() const
    {
        return {burned_fraction_};
    }

    [[nodiscard]] const FireCartesianRasterGeometry2D&
    geometry () const noexcept
    {
        return geometry_;
    }

    [[nodiscard]] const FireSurfaceLayout&
    surface_layout() const noexcept
    {
        return surface_layout_;
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

    void scatter_canonical_to_distributed(
        const std::vector<amrex::Real>& canonical,
        amrex::MultiFab& distributed) const;

    [[nodiscard]] std::vector<amrex::Real>
    gather_distributed_to_canonical(
        const amrex::MultiFab& distributed) const;

    FireCartesianRasterGeometry2D geometry_{};
    FireSurfaceLayout surface_layout_;
    amrex::MultiFab burned_fraction_mf_;
    std::vector<amrex::Real> burned_fraction_;
};

} // namespace ERFFire

#endif // ERF_FIRE_BURNED_FRACTION_RASTER_H
