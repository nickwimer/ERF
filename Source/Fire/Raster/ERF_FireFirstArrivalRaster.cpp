#include <ERF_FireFirstArrivalRaster.H>

#include <ERF_FireCellArrival.H>
#include <ERF_FireCellCoverage.H>

#include <cmath>
#include <stdexcept>
#include <utility>

namespace ERFFire
{

FireFirstArrivalRaster::FireFirstArrivalRaster (
    const FireCartesianRasterGeometry2D& geometry)
    : geometry_(geometry)
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
}

FireFirstArrivalRaster::FireFirstArrivalRaster (
    const FireCartesianRasterGeometry2D& geometry,
    FireFirstArrivalRasterState state)
    : geometry_(geometry)
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

    std::vector<std::uint8_t> next_arrived = arrived_;
    std::vector<amrex::Real> next_first_arrival_time_s =
        first_arrival_time_s_;

    std::size_t newly_arrived_cell_count = 0;

    for (std::size_t j = 0; j < geometry_.ny; ++j) {
        for (std::size_t i = 0; i < geometry_.nx; ++i) {
            const std::size_t index = flat_index(i, j);
            if (fire_perimeter_cell_intersection_area_m2(
                    perimeter, cell_bounds(i, j))
                > amrex::Real(0.0)) {
                next_arrived[index] = std::uint8_t(1);
                next_first_arrival_time_s[index] = time_s;
                ++newly_arrived_cell_count;
            }
        }
    }

    arrived_.swap(next_arrived);
    first_arrival_time_s_.swap(next_first_arrival_time_s);
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

    // Query one known-valid raster cell unconditionally.
    // Reuse the result below if that first cell is unresolved.
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

    std::vector<std::uint8_t> next_arrived =
        arrived_;
    std::vector<amrex::Real> next_first_arrival_time_s =
        first_arrival_time_s_;

    std::size_t newly_arrived_cell_count = 0;

    for (std::size_t j = 0; j < geometry_.ny; ++j) {
        for (std::size_t i = 0; i < geometry_.nx; ++i) {
            const std::size_t index = flat_index(i, j);

            if (arrived_[index] != std::uint8_t(0)) {
                continue;
            }

            const FireCellArrivalResult result =
                (i == 0 && j == 0)
                ? first_query
                : fire_cell_first_arrival_time_linear_sweep(
                    start_perimeter,
                    end_perimeter,
                    cell_bounds(i, j),
                    start_time_s,
                    end_time_s,
                    time_tolerance_s);

            if (result.arrived) {
                next_arrived[index] = std::uint8_t(1);
                next_first_arrival_time_s[index] =
                    result.arrival_time_s;
                ++newly_arrived_cell_count;
            }
        }
    }

    arrived_.swap(next_arrived);
    first_arrival_time_s_.swap(
        next_first_arrival_time_s);
    has_committed_sweep_ = true;
    last_sweep_end_time_s_ = end_time_s;

    return {
        arrived_cell_count(),
        newly_arrived_cell_count
    };
}

} // namespace ERFFire
