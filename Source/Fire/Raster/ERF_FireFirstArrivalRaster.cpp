#include <ERF_FireFirstArrivalRaster.H>

#include <ERF_FireCellArrival.H>

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
FireFirstArrivalRaster::update_from_sweep (
    const FirePerimeter& start_perimeter,
    const FirePerimeter& end_perimeter,
    amrex::Real start_time_s,
    amrex::Real end_time_s,
    amrex::Real time_tolerance_s)
{
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

    if (has_committed_sweep_
        && start_time_s != last_sweep_end_time_s_) {
        throw std::invalid_argument(
            "Fire first-arrival raster sweeps must be contiguous in time");
    }

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
