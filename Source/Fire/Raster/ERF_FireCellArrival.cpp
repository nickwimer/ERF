#include <ERF_FireCellArrival.H>
#include <ERF_FirePerimeterSweep.H>

#include <cmath>
#include <stdexcept>

namespace ERFFire
{
namespace
{

void
validate_sweep_inputs (
    const FirePerimeter& start_perimeter,
    const FirePerimeter& end_perimeter,
    amrex::Real start_time_s,
    amrex::Real end_time_s,
    amrex::Real time_tolerance_s)
{
    if (start_perimeter.size() != end_perimeter.size()) {
        throw std::invalid_argument(
            "Fire arrival sweep requires matching perimeter vertex counts");
    }

    if (!std::isfinite(start_time_s)
        || !std::isfinite(end_time_s)
        || !std::isfinite(time_tolerance_s)) {
        throw std::invalid_argument(
            "Fire arrival sweep times and tolerance must be finite");
    }

    if (!(end_time_s > start_time_s)) {
        throw std::invalid_argument(
            "Fire arrival sweep end time must be greater than start time");
    }

    const amrex::Real duration_s =
        end_time_s - start_time_s;

    if (!std::isfinite(duration_s)
        || !(duration_s > amrex::Real(0.0))) {
        throw std::overflow_error(
            "Fire arrival sweep duration must be finite and positive");
    }

    if (!(time_tolerance_s > amrex::Real(0.0))
        || time_tolerance_s > duration_s) {
        throw std::invalid_argument(
            "Fire arrival time tolerance must lie in (0, sweep duration]");
    }

    if (!(start_time_s + time_tolerance_s > start_time_s)) {
        throw std::invalid_argument(
            "Fire arrival time tolerance is not representable at the supplied time");
    }
}

void
validate_topology_event_sweep_inputs (
    const FirePerimeter& start_perimeter,
    const std::vector<FireVec2>& event_vertices_m,
    amrex::Real start_time_s,
    amrex::Real event_time_s,
    amrex::Real time_tolerance_s)
{
    if (start_perimeter.size()
        != event_vertices_m.size()) {
        throw std::invalid_argument(
            "Fire topology-event arrival sweep requires matching vertex counts");
    }

    for (const FireVec2& vertex : event_vertices_m) {
        if (!std::isfinite(vertex.x)
            || !std::isfinite(vertex.y)) {
            throw std::invalid_argument(
                "Fire topology-event arrival vertices must be finite");
        }
    }

    if (!std::isfinite(start_time_s)
        || !std::isfinite(event_time_s)
        || !std::isfinite(time_tolerance_s)) {
        throw std::invalid_argument(
            "Fire topology-event arrival sweep times and tolerance must be finite");
    }

    if (!(event_time_s > start_time_s)) {
        throw std::invalid_argument(
            "Fire topology-event arrival time must be greater than start time");
    }

    const amrex::Real duration_s =
        event_time_s - start_time_s;

    if (!std::isfinite(duration_s)
        || !(duration_s > amrex::Real(0.0))) {
        throw std::overflow_error(
            "Fire topology-event arrival sweep duration must be finite and positive");
    }

    if (!(time_tolerance_s > amrex::Real(0.0))
        || time_tolerance_s > duration_s) {
        throw std::invalid_argument(
            "Fire topology-event arrival tolerance must lie in "
            "(0, sweep duration]");
    }

    if (!(start_time_s + time_tolerance_s
          > start_time_s)) {
        throw std::invalid_argument(
            "Fire topology-event arrival tolerance is not representable "
            "at the supplied time");
    }
}

bool
has_positive_cell_area (
    const FirePerimeter& perimeter,
    const FireCartesianCell2D& cell)
{
    return fire_perimeter_cell_intersection_area_m2(
        perimeter, cell) > amrex::Real(0.0);
}

bool
has_positive_cell_area (
    const FireFront& front,
    const FireCartesianCell2D& cell)
{
    return fire_front_cell_intersection_area_m2(
        front, cell) > amrex::Real(0.0);
}

} // namespace

FireCellArrivalResult
fire_cell_first_arrival_time_linear_sweep(
    const FirePerimeter& start_perimeter,
    const FirePerimeter& end_perimeter,
    const FireCartesianCell2D& cell,
    amrex::Real start_time_s,
    amrex::Real end_time_s,
    amrex::Real time_tolerance_s)
{
    validate_sweep_inputs(
        start_perimeter,
        end_perimeter,
        start_time_s,
        end_time_s,
        time_tolerance_s);

    if (has_positive_cell_area(start_perimeter, cell)) {
        return {true, start_time_s};
    }

    if (!has_positive_cell_area(end_perimeter, cell)) {
        return {false, amrex::Real(0.0)};
    }

    amrex::Real lower_alpha = 0.0;
    amrex::Real upper_alpha = 1.0;
    amrex::Real lower_time_s = start_time_s;
    amrex::Real upper_time_s = end_time_s;

    for (;;) {
        if (upper_time_s - lower_time_s <= time_tolerance_s) {
            break;
        }

        const amrex::Real middle_alpha =
            amrex::Real(0.5) * (lower_alpha + upper_alpha);

        if (middle_alpha == lower_alpha
            || middle_alpha == upper_alpha) {
            break;
        }

        const amrex::Real middle_time_s =
            start_time_s
            + middle_alpha * (end_time_s - start_time_s);

        if (middle_time_s == lower_time_s
            || middle_time_s == upper_time_s) {
            break;
        }

        const FirePerimeter middle_perimeter =
            interpolate_fire_perimeter_linear_sweep(
                start_perimeter,
                end_perimeter,
                middle_alpha);

        if (has_positive_cell_area(middle_perimeter, cell)) {
            upper_alpha = middle_alpha;
            upper_time_s = middle_time_s;
        } else {
            lower_alpha = middle_alpha;
            lower_time_s = middle_time_s;
        }
    }

    return {true, upper_time_s};
}

FireCellArrivalResult
fire_cell_first_arrival_time_topology_event_sweep(
    const FirePerimeter& start_perimeter,
    const std::vector<FireVec2>& event_vertices_m,
    const FireFront& event_front,
    const FireCartesianCell2D& cell,
    amrex::Real start_time_s,
    amrex::Real event_time_s,
    amrex::Real time_tolerance_s)
{
    validate_topology_event_sweep_inputs(
        start_perimeter,
        event_vertices_m,
        start_time_s,
        event_time_s,
        time_tolerance_s);

    if (has_positive_cell_area(
            start_perimeter,
            cell)) {
        return {
            true,
            start_time_s
        };
    }

    if (!has_positive_cell_area(
            event_front,
            cell)) {
        return {
            false,
            amrex::Real(0.0)
        };
    }

    const auto& start_vertices =
        start_perimeter.vertices_m();

    amrex::Real lower_alpha =
        amrex::Real(0.0);
    amrex::Real upper_alpha =
        amrex::Real(1.0);
    amrex::Real lower_time_s =
        start_time_s;
    amrex::Real upper_time_s =
        event_time_s;

    for (;;) {
        if (upper_time_s - lower_time_s
            <= time_tolerance_s) {
            break;
        }

        const amrex::Real middle_alpha =
            amrex::Real(0.5)
            * (lower_alpha + upper_alpha);

        if (middle_alpha == lower_alpha
            || middle_alpha == upper_alpha) {
            break;
        }

        const amrex::Real middle_time_s =
            start_time_s
            + middle_alpha
                * (event_time_s
                   - start_time_s);

        if (middle_time_s == lower_time_s
            || middle_time_s == upper_time_s) {
            break;
        }

        std::vector<FireVec2> middle_vertices;
        middle_vertices.reserve(
            start_vertices.size());

        for (std::size_t index = 0;
             index < start_vertices.size();
             ++index) {
            middle_vertices.push_back(
                start_vertices[index]
                + middle_alpha
                    * (event_vertices_m[index]
                       - start_vertices[index]));
        }

        const FirePerimeter middle_perimeter(
            std::move(middle_vertices));

        if (has_positive_cell_area(
                middle_perimeter,
                cell)) {
            upper_alpha = middle_alpha;
            upper_time_s = middle_time_s;
        } else {
            lower_alpha = middle_alpha;
            lower_time_s = middle_time_s;
        }
    }

    return {
        true,
        upper_time_s
    };
}

} // namespace ERFFire
