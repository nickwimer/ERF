#ifndef ERF_FIRE_SURFACE_LAYOUT_H_
#define ERF_FIRE_SURFACE_LAYOUT_H_

#include <ERF_FireRasterGeometry.H>

#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>

namespace ERFFire
{

/**
 * AMReX-native ownership layout for the level-0 Fire map plane.
 *
 * The physical Fire raster remains a two-dimensional horizontal map. AMReX is
 * three-dimensional in ERF, so the distributed storage uses a one-cell-thick
 * k=0 slab. The layout is runtime/decomposition state only: it is deliberately
 * not part of Fire physics configuration or checkpoint state.
 *
 * The BoxArray is split into at least enough horizontal boxes to use all MPI
 * ranks when the raster has at least that many cells. The split is derived only
 * from raster dimensions and the active rank count, so all Fire raster fields
 * constructed on one run obtain identical layouts without depending on ERF's
 * vertical BoxArray decomposition.
 */
class FireSurfaceLayout
{
public:
    explicit FireSurfaceLayout(
        const FireCartesianRasterGeometry2D& geometry);

    [[nodiscard]] const FireCartesianRasterGeometry2D&
    geometry() const noexcept
    {
        return geometry_;
    }

    [[nodiscard]] const amrex::Box& cell_domain() const noexcept
    {
        return cell_domain_;
    }

    [[nodiscard]] const amrex::BoxArray& box_array() const noexcept
    {
        return box_array_;
    }

    [[nodiscard]] const amrex::DistributionMapping&
    distribution_map() const noexcept
    {
        return distribution_map_;
    }

private:
    FireCartesianRasterGeometry2D geometry_{};
    amrex::Box cell_domain_{};
    amrex::BoxArray box_array_{};
    amrex::DistributionMapping distribution_map_{};
};

} // namespace ERFFire

#endif // ERF_FIRE_SURFACE_LAYOUT_H_
