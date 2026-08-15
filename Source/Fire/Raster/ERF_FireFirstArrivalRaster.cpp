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
    const std::size_t cell_count =
        detail::validate_fire_cartesian_raster_geometry(
            geometry_);

    arrived_.assign(
        cell_count,
        std::uint8_t(0));
    first_arrival_time_s_.assign(
        cell_count,
        amrex::Real(0.0));
    scatter_canonical_to_distributed(
        arrived_,
        first_arrival_time_s_,
        arrived_mf_,
        first_arrival_time_mf_);
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

    arrived_ = std::move(state.arrived);
    first_arrival_time_s_ =
        std::move(state.first_arrival_time_s);
    has_initial_condition_ =
        state.has_initial_condition;
    initial_condition_time_s_ =
        state.initial_condition_time_s;
    has_committed_sweep_ =
        state.has_committed_sweep;
    last_sweep_end_time_s_ =
        state.last_sweep_end_time_s;

    scatter_canonical_to_distributed(
        arrived_,
        first_arrival_time_s_,
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
      arrived_(other.arrived_),
      first_arrival_time_s_(other.first_arrival_time_s_),
      has_initial_condition_(other.has_initial_condition_),
      initial_condition_time_s_(other.initial_condition_time_s_),
      has_committed_sweep_(other.has_committed_sweep_),
      last_sweep_end_time_s_(other.last_sweep_end_time_s_)
{
    scatter_canonical_to_distributed(
        arrived_,
        first_arrival_time_s_,
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
    return arrived_[flat_index(i, j)] != std::uint8_t(0);
}

amrex::Real
FireFirstArrivalRaster::first_arrival_time_s (
    std::size_t i,
    std::size_t j) const
{
    const std::size_t index = flat_index(i, j);

    if (arrived_[index] == std::uint8_t(0)) {
        throw std::logic_error(
            "Fire first-arrival time requested for a cell that has not arrived");
    }

    return first_arrival_time_s_[index];
}

std::size_t
FireFirstArrivalRaster::arrived_cell_count () const noexcept
{
    std::size_t count = 0;

    for (const std::uint8_t arrived : arrived_) {
        if (arrived != std::uint8_t(0)) {
            ++count;
        }
    }

    return count;
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

std::pair<
    std::vector<std::uint8_t>,
    std::vector<amrex::Real>>
FireFirstArrivalRaster::gather_distributed_to_canonical(
    const amrex::iMultiFab& arrived_mf,
    const amrex::MultiFab& first_arrival_time_mf) const
{
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

    amrex::IArrayBox gathered_arrived(
        surface_layout_.cell_domain(),
        1,
        amrex::The_Pinned_Arena());
    amrex::FArrayBox gathered_times(
        surface_layout_.cell_domain(),
        1,
        amrex::The_Pinned_Arena());
    arrived_mf.copyTo(
        gathered_arrived,
        0,
        0,
        1,
        0);
    first_arrival_time_mf.copyTo(
        gathered_times,
        0,
        0,
        1,
        0);

    const auto mask = gathered_arrived.const_array();
    const auto times = gathered_times.const_array();
    std::vector<std::uint8_t> canonical_arrived(
        geometry_.nx * geometry_.ny,
        std::uint8_t(0));
    std::vector<amrex::Real> canonical_times(
        geometry_.nx * geometry_.ny,
        amrex::Real(0.0));

    for (std::size_t j = 0; j < geometry_.ny; ++j) {
        for (std::size_t i = 0; i < geometry_.nx; ++i) {
            const int value = mask(
                static_cast<int>(i),
                static_cast<int>(j),
                0);
            if (value != 0 && value != 1) {
                throw std::logic_error(
                    "distributed Fire first-arrival mask is not 0 or 1");
            }
            const std::size_t index = flat_index(i, j);
            canonical_arrived[index] =
                static_cast<std::uint8_t>(value);
            canonical_times[index] = times(
                static_cast<int>(i),
                static_cast<int>(j),
                0);
        }
    }

    return {
        std::move(canonical_arrived),
        std::move(canonical_times)};
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
    scatter_canonical_to_distributed(
        arrived_,
        first_arrival_time_s_,
        next_arrived,
        next_first_arrival_time_s);

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

    auto [next_arrived_canonical, next_times_canonical] =
        gather_distributed_to_canonical(
            next_arrived,
            next_first_arrival_time_s);

    std::size_t newly_arrived_cell_count = 0;
    for (std::size_t index = 0;
         index < next_arrived_canonical.size();
         ++index) {
        if (arrived_[index] == std::uint8_t(0)
            && next_arrived_canonical[index] != std::uint8_t(0)) {
            ++newly_arrived_cell_count;
        }
    }

    arrived_mf_ = std::move(next_arrived);
    first_arrival_time_mf_ =
        std::move(next_first_arrival_time_s);
    arrived_ = std::move(next_arrived_canonical);
    first_arrival_time_s_ = std::move(next_times_canonical);
    has_initial_condition_ = true;
    initial_condition_time_s_ = time_s;

    return {
        arrived_cell_count(),
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
    scatter_canonical_to_distributed(
        arrived_,
        first_arrival_time_s_,
        next_arrived,
        next_first_arrival_time_s);

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
                    const std::size_t index = flat_index(ii, jj);

                    if (arrived_[index] != std::uint8_t(0)) {
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

    auto [next_arrived_canonical, next_times_canonical] =
        gather_distributed_to_canonical(
            next_arrived,
            next_first_arrival_time_s);

    std::size_t newly_arrived_cell_count = 0;
    for (std::size_t index = 0;
         index < next_arrived_canonical.size();
         ++index) {
        if (arrived_[index] == std::uint8_t(0)
            && next_arrived_canonical[index] != std::uint8_t(0)) {
            ++newly_arrived_cell_count;
        }
    }

    arrived_mf_ = std::move(next_arrived);
    first_arrival_time_mf_ =
        std::move(next_first_arrival_time_s);
    arrived_ = std::move(next_arrived_canonical);
    first_arrival_time_s_ = std::move(next_times_canonical);
    has_committed_sweep_ = true;
    last_sweep_end_time_s_ = end_time_s;

    return {
        arrived_cell_count(),
        newly_arrived_cell_count
    };
}

} // namespace ERFFire
