#include <ERF_FireBurnedFractionRaster.H>
#include <ERF_FireCellCoverage.H>
#include <ERF_FirePerimeterSweep.H>

#include <AMReX_Arena.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <cmath>
#include <limits>
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
    (void)detail::validate_fire_cartesian_raster_geometry(
        geometry_);

    burned_fraction_mf_.setVal(amrex::Real(0.0));
    burned_area_m2_ = amrex::Real(0.0);
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

    burned_area_m2_ = amrex::Real(0.0);
    for (std::size_t j = 0; j < geometry_.ny; ++j) {
        for (std::size_t i = 0; i < geometry_.nx; ++i) {
            const std::size_t index =
                flat_index(i, j);
            const FireCartesianCell2D cell =
                cell_bounds(i, j);
            burned_area_m2_ +=
                state.burned_fraction[index]
                * detail::fire_cartesian_cell_area_m2(cell);
        }
    }
    if (!std::isfinite(burned_area_m2_)) {
        throw std::overflow_error(
            "restored fire burned area is not finite");
    }

    scatter_canonical_to_distributed(
        state.burned_fraction,
        burned_fraction_mf_);
}

FireBurnedFractionRaster
FireBurnedFractionRaster::collective_restore_from_io_rank_state(
    const FireCartesianRasterGeometry2D& geometry,
    const FireBurnedFractionRasterState& state)
{
    FireBurnedFractionRaster result(geometry);
    const int io_rank =
        amrex::ParallelDescriptor::IOProcessorNumber();

    int invalid_state = 0;
    amrex::Real burned_area_m2 = amrex::Real(0.0);
    if (amrex::ParallelDescriptor::IOProcessor()) {
        try {
            const std::size_t cell_count =
                geometry.nx * geometry.ny;
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

            for (std::size_t j = 0; j < geometry.ny; ++j) {
                for (std::size_t i = 0; i < geometry.nx; ++i) {
                    const std::size_t index =
                        result.flat_index(i, j);
                    burned_area_m2 +=
                        state.burned_fraction[index]
                        * detail::fire_cartesian_cell_area_m2(
                            result.cell_bounds(i, j));
                }
            }
            if (!std::isfinite(burned_area_m2)) {
                throw std::overflow_error(
                    "restored fire burned area is not finite");
            }
        } catch (...) {
            invalid_state = 1;
        }
    }

    amrex::ParallelDescriptor::Bcast(
        &invalid_state, 1, io_rank);
    if (invalid_state != 0) {
        throw std::invalid_argument(
            "collective Fire burned-fraction restore rejected IO-rank state");
    }
    amrex::ParallelDescriptor::Bcast(
        &burned_area_m2, 1, io_rank);

    amrex::BoxArray io_boxes{
        result.surface_layout_.cell_domain()};
    amrex::Vector<int> processor_map(1, io_rank);
    const amrex::DistributionMapping io_dm(
        std::move(processor_map));
    amrex::MultiFab io_state(
        io_boxes, io_dm, 1, 0, fire_surface_mf_info());

    if (amrex::ParallelDescriptor::IOProcessor()) {
        for (amrex::MFIter mfi(io_state); mfi.isValid(); ++mfi) {
            const amrex::Box& box = mfi.validbox();
            const auto values = io_state.array(mfi);
            for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
                for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                    values(i, j, 0) =
                        state.burned_fraction[
                            result.flat_index(
                                static_cast<std::size_t>(i),
                                static_cast<std::size_t>(j))];
                }
            }
        }
    }

    result.burned_fraction_mf_.ParallelCopy(
        io_state, 0, 0, 1, 0, 0);
    result.burned_area_m2_ = burned_area_m2;
    return result;
}

FireBurnedFractionRaster
FireBurnedFractionRaster::collective_restore_from_checkpoint_raster(
    const FireCartesianRasterGeometry2D& geometry,
    const amrex::MultiFab& checkpoint,
    int source_comp)
{
    FireBurnedFractionRaster result(geometry);

    if (source_comp < 0
        || source_comp >= checkpoint.nComp()
        || checkpoint.nGrow() != 0) {
        throw std::invalid_argument(
            "Fire burned-fraction checkpoint MultiFab is incompatible");
    }

    result.burned_fraction_mf_.setVal(
        std::numeric_limits<amrex::Real>::quiet_NaN());
    result.burned_fraction_mf_.ParallelCopy(
        checkpoint,
        source_comp,
        0,
        1,
        0,
        0);

    int invalid_state = 0;
    amrex::Real burned_area_m2 = amrex::Real(0.0);
    const amrex::Real cell_area_m2 =
        geometry.dx_m * geometry.dy_m;

    for (amrex::MFIter mfi(result.burned_fraction_mf_);
         mfi.isValid();
         ++mfi) {
        const amrex::Box& box = mfi.validbox();
        const auto values =
            result.burned_fraction_mf_.const_array(mfi);
        for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
            for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                const amrex::Real value = values(i, j, 0);
                if (!std::isfinite(value)
                    || value < amrex::Real(0.0)
                    || value > amrex::Real(1.0)) {
                    invalid_state = 1;
                    continue;
                }
                burned_area_m2 += value * cell_area_m2;
            }
        }
    }

    amrex::ParallelDescriptor::ReduceIntMax(invalid_state);
    if (invalid_state != 0) {
        throw std::invalid_argument(
            "distributed Fire burned-fraction checkpoint data are invalid");
    }

    amrex::ParallelDescriptor::ReduceRealSum(burned_area_m2);
    if (!std::isfinite(burned_area_m2)
        || burned_area_m2 < amrex::Real(0.0)) {
        throw std::overflow_error(
            "restored Fire burned area is not finite and nonnegative");
    }

    result.burned_area_m2_ = burned_area_m2;
    return result;
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
      burned_area_m2_(other.burned_area_m2_)
{
    copy_distributed_state(
        other.burned_fraction_mf_,
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

FireBurnedFractionRasterState
FireBurnedFractionRaster::snapshot_state() const
{
    if (amrex::ParallelDescriptor::NProcs() != 1) {
        throw std::logic_error(
            "Fire burned-fraction snapshot_state is single-rank only; "
            "use collective_snapshot_state_to_io_rank in parallel");
    }

    FireBurnedFractionRasterState state;
    state.burned_fraction.assign(
        geometry_.nx * geometry_.ny,
        amrex::Real(0.0));

    for (amrex::MFIter mfi(burned_fraction_mf_);
         mfi.isValid();
         ++mfi) {
        const amrex::Box& box = mfi.validbox();
        const auto values =
            burned_fraction_mf_.const_array(mfi);

        for (int j = box.smallEnd(1);
             j <= box.bigEnd(1);
             ++j) {
            for (int i = box.smallEnd(0);
                 i <= box.bigEnd(0);
                 ++i) {
                state.burned_fraction[
                    flat_index(
                        static_cast<std::size_t>(i),
                        static_cast<std::size_t>(j))] =
                            values(i, j, 0);
            }
        }
    }

    return state;
}

FireBurnedFractionRasterState
FireBurnedFractionRaster::
collective_snapshot_state_to_io_rank() const
{
    amrex::BoxArray io_boxes{
        surface_layout_.cell_domain()};
    amrex::Vector<int> processor_map(
        1,
        amrex::ParallelDescriptor::IOProcessorNumber());
    const amrex::DistributionMapping io_dm(
        std::move(processor_map));

    amrex::MultiFab gathered(
        io_boxes,
        io_dm,
        1,
        0,
        fire_surface_mf_info());
    gathered.ParallelCopy(
        burned_fraction_mf_,
        0,
        0,
        1,
        0,
        0);

    FireBurnedFractionRasterState state;
    if (!amrex::ParallelDescriptor::IOProcessor()) {
        return state;
    }

    state.burned_fraction.assign(
        geometry_.nx * geometry_.ny,
        amrex::Real(0.0));

    for (amrex::MFIter mfi(gathered);
         mfi.isValid();
         ++mfi) {
        const amrex::Box& box = mfi.validbox();
        const auto values = gathered.const_array(mfi);

        for (int j = box.smallEnd(1);
             j <= box.bigEnd(1);
             ++j) {
            for (int i = box.smallEnd(0);
                 i <= box.bigEnd(0);
                 ++i) {
                state.burned_fraction[
                    flat_index(
                        static_cast<std::size_t>(i),
                        static_cast<std::size_t>(j))] =
                            values(i, j, 0);
            }
        }
    }

    return state;
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
    (void)flat_index(i, j);

    if (amrex::ParallelDescriptor::NProcs() != 1) {
        throw std::logic_error(
            "Fire burned-fraction scalar cell access is single-rank only; "
            "parallel Fire physics must use distributed_burned_fraction");
    }

    const amrex::IntVect cell(
        static_cast<int>(i),
        static_cast<int>(j),
        0);
    for (amrex::MFIter mfi(burned_fraction_mf_);
         mfi.isValid();
         ++mfi) {
        if (mfi.validbox().contains(cell)) {
            return
                burned_fraction_mf_.const_array(mfi)(
                    cell[0],
                    cell[1],
                    cell[2]);
        }
    }

    throw std::logic_error(
        "single-rank Fire burned-fraction cell is not locally represented");
}

amrex::Real
FireBurnedFractionRaster::burned_area_m2 () const noexcept
{
    return burned_area_m2_;
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

void
FireBurnedFractionRaster::copy_distributed_state(
    const amrex::MultiFab& source,
    amrex::MultiFab& destination) const
{
    const auto matches_layout =
        [this](const amrex::MultiFab& field) {
            return field.boxArray()
                    == surface_layout_.box_array()
                && field.DistributionMap()
                    == surface_layout_.distribution_map()
                && field.nComp() == 1
                && field.nGrow() == 0;
        };
    if (!matches_layout(source)
        || !matches_layout(destination)) {
        throw std::invalid_argument(
            "Fire burned-fraction MultiFab does not match its surface layout");
    }

    for (amrex::MFIter mfi(destination);
         mfi.isValid();
         ++mfi) {
        const amrex::Box& box = mfi.validbox();
        const auto src = source.const_array(mfi);
        const auto dst = destination.array(mfi);

        for (int j = box.smallEnd(1);
             j <= box.bigEnd(1);
             ++j) {
            for (int i = box.smallEnd(0);
                 i <= box.bigEnd(0);
                 ++i) {
                dst(i, j, 0) = src(i, j, 0);
            }
        }
    }
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

    amrex::Real local_newly_burned_area_m2 =
        amrex::Real(0.0);
    amrex::Real local_burned_area_m2 =
        amrex::Real(0.0);

    std::string local_error;
    try {
        for (amrex::MFIter mfi(next_burned_fraction);
             mfi.isValid(); ++mfi) {
            const amrex::Box& box = mfi.validbox();
            const auto current =
                burned_fraction_mf_.const_array(mfi);
            const auto next = next_burned_fraction.array(mfi);

            for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
                for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                    const std::size_t ii =
                        static_cast<std::size_t>(i);
                    const std::size_t jj =
                        static_cast<std::size_t>(j);
                    const FireCartesianCell2D cell =
                        cell_bounds(ii, jj);
                    const amrex::Real coverage =
                        fire_perimeter_cell_coverage_fraction(
                            perimeter,
                            cell);

                    const amrex::Real previous =
                        current(i, j, 0);
                    const FireBurnedFractionUpdate update =
                        update_fire_burned_fraction(
                            previous,
                            coverage);
                    next(i, j, 0) = update.burned_fraction;

                    const amrex::Real represented_area_m2 =
                        detail::fire_cartesian_cell_area_m2(cell);
                    local_newly_burned_area_m2 +=
                        (update.burned_fraction - previous)
                        * represented_area_m2;
                    local_burned_area_m2 +=
                        update.burned_fraction
                        * represented_area_m2;
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

    amrex::Real newly_burned_area_m2 =
        local_newly_burned_area_m2;
    amrex::Real burned_area_m2 =
        local_burned_area_m2;
    amrex::ParallelDescriptor::ReduceRealSum(
        newly_burned_area_m2);
    amrex::ParallelDescriptor::ReduceRealSum(
        burned_area_m2);

    if (!std::isfinite(newly_burned_area_m2)
        || !std::isfinite(burned_area_m2)) {
        throw std::overflow_error(
            "Fire raster burned-area accounting is not finite");
    }

    burned_fraction_mf_ = std::move(next_burned_fraction);
    burned_area_m2_ = burned_area_m2;

    return {
        burned_area_m2,
        newly_burned_area_m2
    };
}


FireRasterBurnedAreaUpdate
FireBurnedFractionRaster::update_from_linear_sweep (
    const FirePerimeter& start_perimeter,
    const FirePerimeter& end_perimeter,
    std::size_t temporal_substeps)
{
    if (start_perimeter.size() != end_perimeter.size()) {
        throw std::invalid_argument(
            "Fire burned-fraction sweep requires matching perimeter vertex counts");
    }

    if (temporal_substeps == 0) {
        throw std::invalid_argument(
            "Fire burned-fraction sweep temporal_substeps must be positive");
    }

    amrex::MultiFab next_burned_fraction(
        surface_layout_.box_array(),
        surface_layout_.distribution_map(),
        1,
        0,
        fire_surface_mf_info());

    copy_distributed_state(
        burned_fraction_mf_,
        next_burned_fraction);

    amrex::Real local_newly_burned_area_m2 =
        amrex::Real(0.0);
    amrex::Real local_burned_area_m2 =
        amrex::Real(0.0);

    std::string local_error;

    try {
        for (std::size_t substep = 0;
             substep < temporal_substeps;
             ++substep) {
            const amrex::Real alpha =
                static_cast<amrex::Real>(substep + 1)
                / static_cast<amrex::Real>(
                    temporal_substeps);

            const FirePerimeter sample_perimeter =
                interpolate_fire_perimeter_linear_sweep(
                    start_perimeter,
                    end_perimeter,
                    alpha);

            const auto& sample_vertices =
                sample_perimeter.vertices_m();

            amrex::Real sample_xlo_m =
                sample_vertices.front().x;
            amrex::Real sample_xhi_m =
                sample_vertices.front().x;
            amrex::Real sample_ylo_m =
                sample_vertices.front().y;
            amrex::Real sample_yhi_m =
                sample_vertices.front().y;

            for (const FireVec2& vertex : sample_vertices) {
                sample_xlo_m =
                    std::min(sample_xlo_m, vertex.x);
                sample_xhi_m =
                    std::max(sample_xhi_m, vertex.x);
                sample_ylo_m =
                    std::min(sample_ylo_m, vertex.y);
                sample_yhi_m =
                    std::max(sample_yhi_m, vertex.y);
            }

            for (amrex::MFIter mfi(next_burned_fraction);
                 mfi.isValid();
                 ++mfi) {
                const amrex::Box& box =
                    mfi.validbox();
                const auto next =
                    next_burned_fraction.array(mfi);

                for (int j = box.smallEnd(1);
                     j <= box.bigEnd(1);
                     ++j) {
                    for (int i = box.smallEnd(0);
                         i <= box.bigEnd(0);
                         ++i) {
                        const amrex::Real previous =
                            next(i, j, 0);

                        if (previous == amrex::Real(1.0)) {
                            continue;
                        }

                        const FireCartesianCell2D cell =
                            cell_bounds(
                                static_cast<std::size_t>(i),
                                static_cast<std::size_t>(j));

                        if (cell.xhi_m < sample_xlo_m
                            || cell.xlo_m > sample_xhi_m
                            || cell.yhi_m < sample_ylo_m
                            || cell.ylo_m > sample_yhi_m) {
                            continue;
                        }

                        const amrex::Real coverage =
                            fire_perimeter_cell_coverage_fraction(
                                sample_perimeter,
                                cell);

                        next(i, j, 0) =
                            update_fire_burned_fraction(
                                previous,
                                coverage)
                                .burned_fraction;
                    }
                }
            }
        }

        for (amrex::MFIter mfi(next_burned_fraction);
             mfi.isValid();
             ++mfi) {
            const amrex::Box& box =
                mfi.validbox();
            const auto previous =
                burned_fraction_mf_.const_array(mfi);
            const auto next =
                next_burned_fraction.const_array(mfi);

            for (int j = box.smallEnd(1);
                 j <= box.bigEnd(1);
                 ++j) {
                for (int i = box.smallEnd(0);
                     i <= box.bigEnd(0);
                     ++i) {
                    const FireCartesianCell2D cell =
                        cell_bounds(
                            static_cast<std::size_t>(i),
                            static_cast<std::size_t>(j));

                    const amrex::Real represented_area_m2 =
                        detail::fire_cartesian_cell_area_m2(
                            cell);

                    local_newly_burned_area_m2 +=
                        (next(i, j, 0)
                         - previous(i, j, 0))
                        * represented_area_m2;

                    local_burned_area_m2 +=
                        next(i, j, 0)
                        * represented_area_m2;
                }
            }
        }
    } catch (const std::exception& error) {
        local_error = error.what();
    } catch (...) {
        local_error =
            "unknown local Fire burned-fraction sweep error";
    }

    int failed =
        local_error.empty() ? 0 : 1;
    amrex::ParallelDescriptor::ReduceIntMax(failed);

    if (failed != 0) {
        if (!local_error.empty()) {
            throw std::runtime_error(
                "distributed Fire burned-fraction sweep failed: "
                + local_error);
        }

        throw std::runtime_error(
            "distributed Fire burned-fraction sweep failed on another MPI rank");
    }

    amrex::Real newly_burned_area_m2 =
        local_newly_burned_area_m2;
    amrex::Real burned_area_m2 =
        local_burned_area_m2;

    amrex::ParallelDescriptor::ReduceRealSum(
        newly_burned_area_m2);
    amrex::ParallelDescriptor::ReduceRealSum(
        burned_area_m2);

    if (!std::isfinite(newly_burned_area_m2)
        || !std::isfinite(burned_area_m2)) {
        throw std::overflow_error(
            "Fire raster swept burned-area accounting is not finite");
    }

    burned_fraction_mf_ =
        std::move(next_burned_fraction);
    burned_area_m2_ =
        burned_area_m2;

    return {
        burned_area_m2,
        newly_burned_area_m2
    };
}


} // namespace ERFFire
