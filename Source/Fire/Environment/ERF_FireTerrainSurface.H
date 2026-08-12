#ifndef ERF_FIRE_TERRAIN_SURFACE_H_
#define ERF_FIRE_TERRAIN_SURFACE_H_

#include <ERF_FireRasterGeometry.H>
#include <ERF_FireTypes.H>

#include <AMReX_REAL.H>

#include <cstddef>
#include <vector>

namespace ERFFire
{

/**
 * Coupling-neutral terrain surface on the same uniform map-plane x/y geometry
 * used by the Fire raster.
 *
 * Ground elevation is stored at horizontal cell corners:
 *
 *   z_{i,j},  i = 0..nx,  j = 0..ny.
 *
 * Inside each horizontal cell the represented surface is bilinear. Height and
 * terrain gradient are therefore evaluated from the same continuous surface
 * representation. The Fire perimeter remains a map-plane x/y object; the
 * gradient is dh/dx, dh/dy in physical horizontal coordinates.
 */
class FireTerrainSurface
{
public:
    FireTerrainSurface(
        FireCartesianRasterGeometry2D horizontal_geometry,
        std::vector<amrex::Real> nodal_ground_height_m);

    [[nodiscard]] const FireCartesianRasterGeometry2D&
    geometry() const noexcept
    {
        return geometry_;
    }

    [[nodiscard]] std::size_t nodal_count() const noexcept
    {
        return nodal_ground_height_m_.size();
    }

    [[nodiscard]] bool contains_physical_point(
        amrex::Real x_m,
        amrex::Real y_m) const noexcept;

    [[nodiscard]] amrex::Real nodal_height_m(
        std::size_t i,
        std::size_t j) const;

    [[nodiscard]] amrex::Real ground_height_m(
        amrex::Real x_m,
        amrex::Real y_m) const;

    /**
     * Return (dh/dx, dh/dy), dimensionless m/m.
     *
     * At an internal cell boundary the containing-cell convention is
     * deterministic: the cell on the positive-index side is used. At the
     * physical upper domain boundary the last cell is used with local
     * coordinate equal to one.
     */
    [[nodiscard]] FireVec2 terrain_gradient_m_per_m(
        amrex::Real x_m,
        amrex::Real y_m) const;

private:
    struct HorizontalLocation
    {
        std::size_t i{};
        std::size_t j{};
        amrex::Real alpha{};
        amrex::Real beta{};
    };

    [[nodiscard]] HorizontalLocation locate(
        amrex::Real x_m,
        amrex::Real y_m) const;

    [[nodiscard]] std::size_t nodal_index(
        std::size_t i,
        std::size_t j) const;

    FireCartesianRasterGeometry2D geometry_{};
    std::vector<amrex::Real> nodal_ground_height_m_;
};

} // namespace ERFFire

#endif
