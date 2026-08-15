#include <ERF_FireBurnedFractionRaster.H>
#include <ERF_FireCellCoverage.H>

#include <AMReX_Arena.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace ERFFire
{
namespace
{

amrex::MFInfo
fire_surface_mf_info()
{
    amrex::MFInfo info;
    info.SetArena(amrex::The_Pinned_Arena());
    return info;
}

} // namespace

FireBurnedFractionRaster::FireBurnedFractionRaster (
    const FireCartesianRasterGeometry2D& geometry)
    : geometry_(geometry),
      surface_layout_(geometry_),
      burned_fraction_mf_(
          surface_layout_.box_array(),
          surface_layout_.distribution_map(),
          1,
          0,
          fire_surface_mf_info())
{
    const std::size_t cell_count =
        detail::validate_fire_cartesian_raster_geometry(
            geometry_);

    burned_fraction_.assign(
        cell_count,
        amrex::Real(0.0));
    scatter_canonical_to_distributed(
        burned_fraction_,
        burned_fraction_mf_);
}

FireBurnedFractionRaster::FireBurnedFractionRaster (
    const FireCartesianRasterGeometry2D& geometry,
    FireBurnedFractionRasterState state)
    : geometry_(geometry),
      surface_layout_(geometry_),
      burned_fraction_mf_(
          surface_layout_.box_array(),
          surface_layout_.distribution_map(),
          1,
          0,
          fire_surface_mf_info())
{
    const std::size_t cell_count =
        detail::validate_fire_cartesian_raster_geometry(
            geometry_);

    if (state.burned_fraction.size() != cell_count) {
        throw std::invalid_argument(
            "restored fire burned-fraction state has the wrong cell count");
    }

    for (const amrex::Real value : state.burned_fraction) {
        if (!std::isfinite(value)
            || value < amrex::Real(0.0)
            || value > amrex::Real(1.0)) {
            throw std::invalid_argument(
                "restored fire burned fraction must be finite in [0,1]");
        }
    }

    burned_fraction_ = std::move(state.burned_fraction);
    scatter_canonical_to_distributed(
        burned_fraction_,
        burned_fraction_mf_);
}

FireBurnedFractionRaster::FireBurnedFractionRaster(
    const FireBurnedFractionRaster& other)
    : geometry_(other.geometry_),
      surface_layout_(other.surface_layout_),
      burned_fraction_mf_(
          surface_layout_.box_array(),
          surface_layout_.distribution_map(),
          1,
          0,
          fire_surface_mf_info()),
      burned_fraction_(other.burned_fraction_)
{
    scatter_canonical_to_distributed(
        burned_fraction_,
        burned_fraction_mf_);
}

FireBurnedFractionRaster&
FireBurnedFractionRaster::operator=(
    const FireBurnedFractionRaster& other)
{
    if (this != &other) {
        FireBurnedFractionRaster copy(other);
        *this = std::move(copy);
    }
    return *this;
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

void
FireBurnedFractionRaster::scatter_canonical_to_distributed(
    const std::vector<amrex::Real>& canonical,
    amrex::MultiFab& distributed) const
{
    if (canonical.size() != geometry_.nx * geometry_.ny) {
        throw std::invalid_argument(
            "Fire burned-fraction canonical state has the wrong cell count");
    }
    if (distributed.boxArray() != surface_layout_.box_array()
        || distributed.DistributionMap()
            != surface_layout_.distribution_map()
        || distributed.nComp() != 1
        || distributed.nGrow() != 0) {
        throw std::invalid_argument(
            "Fire burned-fraction MultiFab does not match its surface layout");
    }

    for (amrex::MFIter mfi(distributed); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.validbox();
        const auto values = distributed.array(mfi);

        for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
            for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                values(i, j, 0) =
                    canonical[
                        flat_index(
                            static_cast<std::size_t>(i),
                            static_cast<std::size_t>(j))];
            }
        }
    }
}

std::vector<amrex::Real>
FireBurnedFractionRaster::gather_distributed_to_canonical(
    const amrex::MultiFab& distributed) const
{
    if (distributed.boxArray() != surface_layout_.box_array()
        || distributed.DistributionMap()
            != surface_layout_.distribution_map()
        || distributed.nComp() != 1
        || distributed.nGrow() != 0) {
        throw std::invalid_argument(
            "Fire burned-fraction MultiFab does not match its surface layout");
    }

    amrex::FArrayBox gathered(
        surface_layout_.cell_domain(),
        1,
        amrex::The_Pinned_Arena());
    distributed.copyTo(
        gathered,
        0,
        0,
        1,
        0);
    const auto values = gathered.const_array();

    std::vector<amrex::Real> canonical(
        geometry_.nx * geometry_.ny,
        amrex::Real(0.0));
    for (std::size_t j = 0; j < geometry_.ny; ++j) {
        for (std::size_t i = 0; i < geometry_.nx; ++i) {
            canonical[flat_index(i, j)] =
                values(
                    static_cast<int>(i),
                    static_cast<int>(j),
                    0);
        }
    }
    return canonical;
}

FireRasterBurnedAreaUpdate
FireBurnedFractionRaster::update_from_perimeter (
    const FirePerimeter& perimeter)
{
    amrex::MultiFab next_burned_fraction(
        surface_layout_.box_array(),
        surface_layout_.distribution_map(),
        1,
        0,
        fire_surface_mf_info());

    std::string local_error;
    try {
        for (amrex::MFIter mfi(next_burned_fraction);
             mfi.isValid(); ++mfi) {
            const amrex::Box& box = mfi.validbox();
            const auto next = next_burned_fraction.array(mfi);

            for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
                for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                    const std::size_t ii =
                        static_cast<std::size_t>(i);
                    const std::size_t jj =
                        static_cast<std::size_t>(j);
                    const std::size_t index =
                        flat_index(ii, jj);
                    const FireCartesianCell2D cell =
                        cell_bounds(ii, jj);
                    const amrex::Real coverage =
                        fire_perimeter_cell_coverage_fraction(
                            perimeter,
                            cell);

                    const FireBurnedFractionUpdate update =
                        update_fire_burned_fraction(
                            burned_fraction_[index],
                            coverage);
                    next(i, j, 0) = update.burned_fraction;
                }
            }
        }
    } catch (const std::exception& error) {
        local_error = error.what();
    } catch (...) {
        local_error = "unknown local Fire burned-fraction error";
    }

    int failed = local_error.empty() ? 0 : 1;
    amrex::ParallelDescriptor::ReduceIntMax(failed);
    if (failed != 0) {
        if (!local_error.empty()) {
            throw std::runtime_error(
                "distributed Fire burned-fraction update failed: "
                + local_error);
        }
        throw std::runtime_error(
            "distributed Fire burned-fraction update failed on another MPI rank");
    }

    std::vector<amrex::Real> next_canonical =
        gather_distributed_to_canonical(
            next_burned_fraction);

    amrex::Real newly_burned_area_m2 = 0.0;
    amrex::Real burned_area_m2 = 0.0;

    // Preserve the canonical row-major accumulation order used by the serial
    // implementation. This keeps extensive diagnostics decomposition-neutral
    // while cell geometry work itself is distributed.
    for (std::size_t j = 0; j < geometry_.ny; ++j) {
        for (std::size_t i = 0; i < geometry_.nx; ++i) {
            const std::size_t index = flat_index(i, j);
            const FireCartesianCell2D cell =
                cell_bounds(i, j);
            const amrex::Real represented_area_m2 =
                detail::fire_cartesian_cell_area_m2(cell);
            const amrex::Real newly_burned_fraction =
                next_canonical[index] - burned_fraction_[index];

            newly_burned_area_m2 +=
                newly_burned_fraction * represented_area_m2;
            burned_area_m2 +=
                next_canonical[index] * represented_area_m2;
        }
    }

    if (!std::isfinite(newly_burned_area_m2)
        || !std::isfinite(burned_area_m2)) {
        throw std::overflow_error(
            "Fire raster burned-area accounting is not finite");
    }

    burned_fraction_mf_ = std::move(next_burned_fraction);
    burned_fraction_ = std::move(next_canonical);

    return {
        burned_area_m2,
        newly_burned_area_m2
    };
}

} // namespace ERFFire
