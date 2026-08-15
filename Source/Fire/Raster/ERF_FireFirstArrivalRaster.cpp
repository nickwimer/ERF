#include <ERF_FireFirstArrivalRaster.H>

#include <ERF_FireCellArrival.H>
#include <ERF_FireCellCoverage.H>

#include <AMReX_Arena.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_IArrayBox.H>
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

FireFirstArrivalRaster::FireFirstArrivalRaster (
    const FireCartesianRasterGeometry2D& geometry)
    : geometry_(geometry),
      surface_layout_(geometry_),
      arrived_mf_(
          surface_layout_.box_array(),
          surface_layout_.distribution_map(),
          1,
          0,
          fire_surface_mf_info()),
      first_arrival_time_mf_(
          surface_layout_.box_array(),
          surface_layout_.distribution_map(),
          1,
          0,
          fire_surface_mf_info())
{
    (void)detail::validate_fire_cartesian_raster_geometry(
        geometry_);

    arrived_mf_.setVal(0);
    first_arrival_time_mf_.setVal(amrex::Real(0.0));
    arrived_cell_count_ = 0;
}

FireFirstArrivalRaster::FireFirstArrivalRaster (
    const FireCartesianRasterGeometry2D& geometry,
    FireFirstArrivalRasterState state)
    : geometry_(geometry),
      surface_layout_(geometry_),
      arrived_mf_(
          surface_layout_.box_array(),
          surface_layout_.distribution_map(),
          1,
          0,
          fire_surface_mf_info()),
      first_arrival_time_mf_(
          surface_layout_.box_array(),
          surface_layout_.distribution_map(),
          1,
          0,
          fire_surface_mf_info())
{
    const std::size_t cell_count =
        detail::validate_fire_cartesian_raster_geometry(
            geometry_);

    if (state.arrived.size() != cell_count
        || state.first_arrival_time_s.size() != cell_count) {
        throw std::invalid_argument(
            "restored fire first-arrival state has the wrong cell count");
    }

    if (state.has_initial_condition) {
        if (!std::isfinite(state.initial_condition_time_s)
            || state.initial_condition_time_s < amrex::Real(0.0)) {
            throw std::invalid_argument(
                "restored fire first-arrival initial time must be finite and nonnegative");
        }
    } else if (state.has_committed_sweep) {
        throw std::invalid_argument(
            "restored fire first-arrival state cannot have sweeps without an initial condition");
    }

    if (state.has_committed_sweep) {
        if (!std::isfinite(state.last_sweep_end_time_s)
            || state.last_sweep_end_time_s
                < state.initial_condition_time_s) {
            throw std::invalid_argument(
                "restored fire first-arrival sweep time is invalid");
        }
    } else if (!std::isfinite(state.last_sweep_end_time_s)) {
        throw std::invalid_argument(
            "restored fire first-arrival stored sweep time must be finite");
    }

    std::size_t arrived_count = 0;
    for (std::size_t index = 0; index < cell_count; ++index) {
        const std::uint8_t mask = state.arrived[index];
        const amrex::Real arrival_time =
            state.first_arrival_time_s[index];

        if (mask != std::uint8_t(0)
            && mask != std::uint8_t(1)) {
            throw std::invalid_argument(
                "restored fire first-arrival mask must contain only 0 or 1");
        }
        if (!std::isfinite(arrival_time)) {
            throw std::invalid_argument(
                "restored fire first-arrival times must be finite");
        }

        if (mask == std::uint8_t(0)) {
            continue;
        }

        ++arrived_count;
        if (!state.has_initial_condition
            || arrival_time < state.initial_condition_time_s) {
            throw std::invalid_argument(
                "restored fire arrival precedes the initial condition");
        }

        if (state.has_committed_sweep) {
            if (arrival_time > state.last_sweep_end_time_s) {
                throw std::invalid_argument(
                    "restored fire arrival lies after the committed sweep");
            }
        } else if (arrival_time
                   != state.initial_condition_time_s) {
            throw std::invalid_argument(
                "restored fire arrival without a sweep must equal the initial time");
        }
    }

    if (!state.has_initial_condition && arrived_count != 0) {
        throw std::invalid_argument(
            "restored fire first-arrival state has arrived cells without initialization");
    }

    arrived_cell_count_ = arrived_count;
    has_initial_condition_ =
        state.has_initial_condition;
    initial_condition_time_s_ =
        state.initial_condition_time_s;
    has_committed_sweep_ =
        state.has_committed_sweep;
    last_sweep_end_time_s_ =
        state.last_sweep_end_time_s;

    scatter_canonical_to_distributed(
        state.arrived,
        state.first_arrival_time_s,
        arrived_mf_,
        first_arrival_time_mf_);
}

FireFirstArrivalRaster::FireFirstArrivalRaster(
    const FireFirstArrivalRaster& other)
    : geometry_(other.geometry_),
      surface_layout_(other.surface_layout_),
      arrived_mf_(
          surface_layout_.box_array(),
          surface_layout_.distribution_map(),
          1,
          0,
          fire_surface_mf_info()),
      first_arrival_time_mf_(
          surface_layout_.box_array(),
          surface_layout_.distribution_map(),
          1,
          0,
          fire_surface_mf_info()),
      arrived_cell_count_(other.arrived_cell_count_),
      has_initial_condition_(other.has_initial_condition_),
      initial_condition_time_s_(other.initial_condition_time_s_),
      has_committed_sweep_(other.has_committed_sweep_),
      last_sweep_end_time_s_(other.last_sweep_end_time_s_)
{
    copy_distributed_state(
        other.arrived_mf_,
        other.first_arrival_time_mf_,
        arrived_mf_,
        first_arrival_time_mf_);
}

FireFirstArrivalRaster&
FireFirstArrivalRaster::operator=(
    const FireFirstArrivalRaster& other)
{
    if (this != &other) {
        FireFirstArrivalRaster copy(other);
        *this = std::move(copy);
    }
    return *this;
}

FireFirstArrivalRasterState
FireFirstArrivalRaster::snapshot_state() const
{
    if (amrex::ParallelDescriptor::NProcs() != 1) {
        throw std::logic_error(
            "Fire first-arrival snapshot_state is single-rank only; "
            "use collective_snapshot_state_to_io_rank in parallel");
    }

    FireFirstArrivalRasterState state;
    state.arrived.assign(
        geometry_.nx * geometry_.ny,
        std::uint8_t(0));
    state.first_arrival_time_s.assign(
        geometry_.nx * geometry_.ny,
        amrex::Real(0.0));
    state.has_initial_condition =
        has_initial_condition_;
    state.initial_condition_time_s =
        initial_condition_time_s_;
    state.has_committed_sweep =
        has_committed_sweep_;
    state.last_sweep_end_time_s =
        last_sweep_end_time_s_;

    for (amrex::MFIter mfi(arrived_mf_);
         mfi.isValid();
         ++mfi) {
        const amrex::Box& box = mfi.validbox();
        const auto mask = arrived_mf_.const_array(mfi);
        const auto times =
            first_arrival_time_mf_.const_array(mfi);

        for (int j = box.smallEnd(1);
             j <= box.bigEnd(1);
             ++j) {
            for (int i = box.smallEnd(0);
                 i <= box.bigEnd(0);
                 ++i) {
                const int value = mask(i, j, 0);
                if (value != 0 && value != 1) {
                    throw std::logic_error(
                        "distributed Fire first-arrival mask is not 0 or 1");
                }
                const std::size_t index =
                    flat_index(
                        static_cast<std::size_t>(i),
                        static_cast<std::size_t>(j));
                state.arrived[index] =
                    static_cast<std::uint8_t>(value);
                state.first_arrival_time_s[index] =
                    times(i, j, 0);
            }
        }
    }

    return state;
}

FireFirstArrivalRasterState
FireFirstArrivalRaster::
collective_snapshot_state_to_io_rank() const
{
    amrex::BoxArray io_boxes{
        surface_layout_.cell_domain()};
    amrex::Vector<int> processor_map(
        1,
        amrex::ParallelDescriptor::IOProcessorNumber());
    const amrex::DistributionMapping io_dm(
        std::move(processor_map));

    amrex::iMultiFab gathered_arrived(
        io_boxes,
        io_dm,
        1,
        0,
        fire_surface_mf_info());
    amrex::MultiFab gathered_times(
        io_boxes,
        io_dm,
        1,
        0,
        fire_surface_mf_info());

    gathered_arrived.ParallelCopy(
        arrived_mf_,
        0,
        0,
        1,
        0,
        0);
    gathered_times.ParallelCopy(
        first_arrival_time_mf_,
        0,
        0,
        1,
        0,
        0);

    FireFirstArrivalRasterState state;
    state.has_initial_condition =
        has_initial_condition_;
    state.initial_condition_time_s =
        initial_condition_time_s_;
    state.has_committed_sweep =
        has_committed_sweep_;
    state.last_sweep_end_time_s =
        last_sweep_end_time_s_;

    if (!amrex::ParallelDescriptor::IOProcessor()) {
        return state;
    }

    const std::size_t cell_count =
        geometry_.nx * geometry_.ny;
    state.arrived.assign(
        cell_count,
        std::uint8_t(0));
    state.first_arrival_time_s.assign(
        cell_count,
        amrex::Real(0.0));

    for (amrex::MFIter mfi(gathered_arrived);
         mfi.isValid();
         ++mfi) {
        const amrex::Box& box = mfi.validbox();
        const auto mask =
            gathered_arrived.const_array(mfi);
        const auto times =
            gathered_times.const_array(mfi);

        for (int j = box.smallEnd(1);
             j <= box.bigEnd(1);
             ++j) {
            for (int i = box.smallEnd(0);
                 i <= box.bigEnd(0);
                 ++i) {
                const int value = mask(i, j, 0);

                const std::size_t index =
                    flat_index(
                        static_cast<std::size_t>(i),
                        static_cast<std::size_t>(j));
                state.arrived[index] =
                    static_cast<std::uint8_t>(value);
                state.first_arrival_time_s[index] =
                    times(i, j, 0);
            }
        }
    }

    return state;
}

std::size_t
FireFirstArrivalRaster::flat_index (
    std::size_t i,
    std::size_t j) const
{
    return detail::fire_cartesian_raster_flat_index(
        geometry_, i, j);
}

FireCartesianCell2D
FireFirstArrivalRaster::cell_bounds (
    std::size_t i,
    std::size_t j) const
{
    return detail::fire_cartesian_raster_cell_bounds(
        geometry_, i, j);
}

bool
FireFirstArrivalRaster::has_arrived (
    std::size_t i,
    std::size_t j) const
{
    (void)flat_index(i, j);

    if (amrex::ParallelDescriptor::NProcs() != 1) {
        throw std::logic_error(
            "Fire first-arrival scalar cell access is single-rank only; "
            "parallel code must use distributed state or collective packing");
    }

    const amrex::IntVect cell(
        static_cast<int>(i),
        static_cast<int>(j),
        0);
    for (amrex::MFIter mfi(arrived_mf_);
         mfi.isValid();
         ++mfi) {
        if (mfi.validbox().contains(cell)) {
            return arrived_mf_.const_array(mfi)(
                       cell[0], cell[1], cell[2])
                != 0;
        }
    }

    throw std::logic_error(
        "single-rank Fire first-arrival cell is not locally represented");
}

amrex::Real
FireFirstArrivalRaster::first_arrival_time_s (
    std::size_t i,
    std::size_t j) const
{
    (void)flat_index(i, j);

    if (amrex::ParallelDescriptor::NProcs() != 1) {
        throw std::logic_error(
            "Fire first-arrival scalar cell access is single-rank only; "
            "parallel code must use distributed state or collective packing");
    }

    const amrex::IntVect cell(
        static_cast<int>(i),
        static_cast<int>(j),
        0);
    for (amrex::MFIter mfi(arrived_mf_);
         mfi.isValid();
         ++mfi) {
        if (!mfi.validbox().contains(cell)) {
            continue;
        }

        if (arrived_mf_.const_array(mfi)(
                cell[0], cell[1], cell[2])
            == 0) {
            throw std::logic_error(
                "Fire first-arrival time requested for a cell that has not arrived");
        }

        return first_arrival_time_mf_.const_array(mfi)(
            cell[0], cell[1], cell[2]);
    }

    throw std::logic_error(
        "single-rank Fire first-arrival cell is not locally represented");
}

std::size_t
FireFirstArrivalRaster::arrived_cell_count () const noexcept
{
    return arrived_cell_count_;
}

void
FireFirstArrivalRaster::scatter_canonical_to_distributed(
    const std::vector<std::uint8_t>& arrived,
    const std::vector<amrex::Real>& first_arrival_time_s,
    amrex::iMultiFab& arrived_mf,
    amrex::MultiFab& first_arrival_time_mf) const
{
    const std::size_t cell_count = geometry_.nx * geometry_.ny;
    if (arrived.size() != cell_count
        || first_arrival_time_s.size() != cell_count) {
        throw std::invalid_argument(
            "Fire first-arrival canonical state has the wrong cell count");
    }
    if (arrived_mf.boxArray() != surface_layout_.box_array()
        || arrived_mf.DistributionMap()
            != surface_layout_.distribution_map()
        || arrived_mf.nComp() != 1
        || arrived_mf.nGrow() != 0
        || first_arrival_time_mf.boxArray()
            != surface_layout_.box_array()
        || first_arrival_time_mf.DistributionMap()
            != surface_layout_.distribution_map()
        || first_arrival_time_mf.nComp() != 1
        || first_arrival_time_mf.nGrow() != 0) {
        throw std::invalid_argument(
            "Fire first-arrival MultiFabs do not match the surface layout");
    }

    for (amrex::MFIter mfi(arrived_mf); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.validbox();
        const auto mask = arrived_mf.array(mfi);
        const auto times = first_arrival_time_mf.array(mfi);

        for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
            for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                const std::size_t index =
                    flat_index(
                        static_cast<std::size_t>(i),
                        static_cast<std::size_t>(j));
                mask(i, j, 0) =
                    static_cast<int>(arrived[index]);
                times(i, j, 0) = first_arrival_time_s[index];
            }
        }
    }
}

void
FireFirstArrivalRaster::copy_distributed_state(
    const amrex::iMultiFab& source_arrived,
    const amrex::MultiFab& source_times,
    amrex::iMultiFab& destination_arrived,
    amrex::MultiFab& destination_times) const
{
    const auto matches_arrived =
        [this](const amrex::iMultiFab& field) {
            return field.boxArray()
                    == surface_layout_.box_array()
                && field.DistributionMap()
                    == surface_layout_.distribution_map()
                && field.nComp() == 1
                && field.nGrow() == 0;
        };
    const auto matches_times =
        [this](const amrex::MultiFab& field) {
            return field.boxArray()
                    == surface_layout_.box_array()
                && field.DistributionMap()
                    == surface_layout_.distribution_map()
                && field.nComp() == 1
                && field.nGrow() == 0;
        };

    if (!matches_arrived(source_arrived)
        || !matches_arrived(destination_arrived)
        || !matches_times(source_times)
        || !matches_times(destination_times)) {
        throw std::invalid_argument(
            "Fire first-arrival MultiFabs do not match the surface layout");
    }

    for (amrex::MFIter mfi(destination_arrived);
         mfi.isValid();
         ++mfi) {
        const amrex::Box& box = mfi.validbox();
        const auto source_mask =
            source_arrived.const_array(mfi);
        const auto source_time =
            source_times.const_array(mfi);
        const auto destination_mask =
            destination_arrived.array(mfi);
        const auto destination_time =
            destination_times.array(mfi);

        for (int j = box.smallEnd(1);
             j <= box.bigEnd(1);
             ++j) {
            for (int i = box.smallEnd(0);
                 i <= box.bigEnd(0);
                 ++i) {
                destination_mask(i, j, 0) =
                    source_mask(i, j, 0);
                destination_time(i, j, 0) =
                    source_time(i, j, 0);
            }
        }
    }
}

FireFirstArrivalRasterUpdate
FireFirstArrivalRaster::initialize_from_perimeter (
    const FirePerimeter& perimeter,
    amrex::Real time_s)
{
    if (!std::isfinite(time_s)) {
        throw std::invalid_argument(
            "Fire first-arrival initial-condition time must be finite");
    }

    if (has_initial_condition_ || has_committed_sweep_) {
        throw std::logic_error(
            "Fire first-arrival initial condition may be set only once before sweeps");
    }

    amrex::iMultiFab next_arrived(
        surface_layout_.box_array(),
        surface_layout_.distribution_map(),
        1,
        0,
        fire_surface_mf_info());
    amrex::MultiFab next_first_arrival_time_s(
        surface_layout_.box_array(),
        surface_layout_.distribution_map(),
        1,
        0,
        fire_surface_mf_info());
    copy_distributed_state(
        arrived_mf_,
        first_arrival_time_mf_,
        next_arrived,
        next_first_arrival_time_s);

    amrex::Long local_newly_arrived_cell_count = 0;
    std::string local_error;
    try {
        for (amrex::MFIter mfi(next_arrived);
             mfi.isValid(); ++mfi) {
            const amrex::Box& box = mfi.validbox();
            const auto next_mask = next_arrived.array(mfi);
            const auto next_times =
                next_first_arrival_time_s.array(mfi);

            for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
                for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                    const std::size_t ii =
                        static_cast<std::size_t>(i);
                    const std::size_t jj =
                        static_cast<std::size_t>(j);
                    if (fire_perimeter_cell_intersection_area_m2(
                            perimeter,
                            cell_bounds(ii, jj))
                        > amrex::Real(0.0)) {
                        if (next_mask(i, j, 0) == 0) {
                            ++local_newly_arrived_cell_count;
                        }
                        next_mask(i, j, 0) = 1;
                        next_times(i, j, 0) = time_s;
                    }
                }
            }
        }
    } catch (const std::exception& error) {
        local_error = error.what();
    } catch (...) {
        local_error = "unknown local Fire first-arrival initialization error";
    }

    int failed = local_error.empty() ? 0 : 1;
    amrex::ParallelDescriptor::ReduceIntMax(failed);
    if (failed != 0) {
        if (!local_error.empty()) {
            throw std::runtime_error(
                "distributed Fire first-arrival initialization failed: "
                + local_error);
        }
        throw std::runtime_error(
            "distributed Fire first-arrival initialization failed on another MPI rank");
    }

    amrex::ParallelDescriptor::ReduceLongSum(
        local_newly_arrived_cell_count);
    const std::size_t newly_arrived_cell_count =
        static_cast<std::size_t>(
            local_newly_arrived_cell_count);
    if (arrived_cell_count_ > cell_count()
        || newly_arrived_cell_count
            > cell_count() - arrived_cell_count_) {
        throw std::logic_error(
            "Fire first-arrival count exceeds raster cell count");
    }

    arrived_mf_ = std::move(next_arrived);
    first_arrival_time_mf_ =
        std::move(next_first_arrival_time_s);
    arrived_cell_count_ += newly_arrived_cell_count;
    has_initial_condition_ = true;
    initial_condition_time_s_ = time_s;

    return {
        arrived_cell_count_,
        newly_arrived_cell_count
    };
}

FireFirstArrivalRasterUpdate
FireFirstArrivalRaster::update_from_sweep (
    const FirePerimeter& start_perimeter,
    const FirePerimeter& end_perimeter,
    amrex::Real start_time_s,
    amrex::Real end_time_s,
    amrex::Real time_tolerance_s)
{
    if (has_committed_sweep_
        && start_time_s != last_sweep_end_time_s_) {
        throw std::invalid_argument(
            "Fire first-arrival raster sweeps must be contiguous in time");
    }
    if (!has_committed_sweep_
        && has_initial_condition_
        && start_time_s != initial_condition_time_s_) {
        throw std::invalid_argument(
            "Fire first-arrival first sweep must start at the initial-condition time");
    }

    // Preserve the serial validation/query order for one known-valid cell.
    // Every rank owns the same replicated perimeter and evaluates this query.
    const FireCartesianCell2D first_cell =
        cell_bounds(0, 0);
    const FireCellArrivalResult first_query =
        fire_cell_first_arrival_time_linear_sweep(
            start_perimeter,
            end_perimeter,
            first_cell,
            start_time_s,
            end_time_s,
            time_tolerance_s);

    amrex::iMultiFab next_arrived(
        surface_layout_.box_array(),
        surface_layout_.distribution_map(),
        1,
        0,
        fire_surface_mf_info());
    amrex::MultiFab next_first_arrival_time_s(
        surface_layout_.box_array(),
        surface_layout_.distribution_map(),
        1,
        0,
        fire_surface_mf_info());
    copy_distributed_state(
        arrived_mf_,
        first_arrival_time_mf_,
        next_arrived,
        next_first_arrival_time_s);

    amrex::Long local_newly_arrived_cell_count = 0;
    std::string local_error;
    try {
        for (amrex::MFIter mfi(next_arrived);
             mfi.isValid(); ++mfi) {
            const amrex::Box& box = mfi.validbox();
            const auto next_mask = next_arrived.array(mfi);
            const auto next_times =
                next_first_arrival_time_s.array(mfi);

            for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
                for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                    const std::size_t ii =
                        static_cast<std::size_t>(i);
                    const std::size_t jj =
                        static_cast<std::size_t>(j);

                    if (next_mask(i, j, 0) != 0) {
                        continue;
                    }

                    const FireCellArrivalResult result =
                        (i == 0 && j == 0)
                        ? first_query
                        : fire_cell_first_arrival_time_linear_sweep(
                            start_perimeter,
                            end_perimeter,
                            cell_bounds(ii, jj),
                            start_time_s,
                            end_time_s,
                            time_tolerance_s);

                    if (result.arrived) {
                        next_mask(i, j, 0) = 1;
                        next_times(i, j, 0) = result.arrival_time_s;
                        ++local_newly_arrived_cell_count;
                    }
                }
            }
        }
    } catch (const std::exception& error) {
        local_error = error.what();
    } catch (...) {
        local_error = "unknown local Fire first-arrival sweep error";
    }

    int failed = local_error.empty() ? 0 : 1;
    amrex::ParallelDescriptor::ReduceIntMax(failed);
    if (failed != 0) {
        if (!local_error.empty()) {
            throw std::runtime_error(
                "distributed Fire first-arrival sweep failed: "
                + local_error);
        }
        throw std::runtime_error(
            "distributed Fire first-arrival sweep failed on another MPI rank");
    }

    amrex::ParallelDescriptor::ReduceLongSum(
        local_newly_arrived_cell_count);
    const std::size_t newly_arrived_cell_count =
        static_cast<std::size_t>(
            local_newly_arrived_cell_count);
    if (arrived_cell_count_ > cell_count()
        || newly_arrived_cell_count
            > cell_count() - arrived_cell_count_) {
        throw std::logic_error(
            "Fire first-arrival count exceeds raster cell count");
    }

    arrived_mf_ = std::move(next_arrived);
    first_arrival_time_mf_ =
        std::move(next_first_arrival_time_s);
    arrived_cell_count_ += newly_arrived_cell_count;
    has_committed_sweep_ = true;
    last_sweep_end_time_s_ = end_time_s;

    return {
        arrived_cell_count_,
        newly_arrived_cell_count
    };
}

} // namespace ERFFire
