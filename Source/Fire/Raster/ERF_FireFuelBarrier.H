#ifndef ERF_FIRE_FUEL_BARRIER_H_
#define ERF_FIRE_FUEL_BARRIER_H_

#include <ERF_FireFuelRaster.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace ERFFire
{

/**
 * First entry of one map-plane motion segment into the interior of the
 * NonBurnable material set.
 *
 * motion_fraction is measured along start + f * (end - start), with f in
 * [0,1). A value of zero is valid when the motion starts inside, or points
 * immediately into, NonBurnable material. The reported cell is one
 * NonBurnable cell participating in the contacted material set.
 */
struct FireFuelBarrierContact
{
    amrex::Real motion_fraction{};
    FireVec2 position_m{};
    FireCartesianRasterIndex2D cell{};
};

namespace detail
{

inline amrex::Real
fire_barrier_fraction_tolerance(amrex::Real scale) noexcept
{
    return amrex::Real(256)
        * std::numeric_limits<amrex::Real>::epsilon()
        * std::max(amrex::Real(1), std::abs(scale));
}

inline bool
fire_barrier_fraction_equal(amrex::Real lhs, amrex::Real rhs) noexcept
{
    return std::abs(lhs - rhs)
        <= fire_barrier_fraction_tolerance(
            std::max(std::abs(lhs), std::abs(rhs)));
}

inline amrex::Real
fire_barrier_axis_lower_face(
    const FireCartesianRasterGeometry2D& geometry,
    bool x_axis,
    std::size_t index)
{
    if (x_axis) {
        return fire_cartesian_raster_cell_bounds(
            geometry, index, 0).xlo_m;
    }
    return fire_cartesian_raster_cell_bounds(
        geometry, 0, index).ylo_m;
}

inline amrex::Real
fire_barrier_axis_upper_face(
    const FireCartesianRasterGeometry2D& geometry,
    bool x_axis,
    std::size_t index)
{
    if (x_axis) {
        return fire_cartesian_raster_cell_bounds(
            geometry, index, 0).xhi_m;
    }
    return fire_cartesian_raster_cell_bounds(
        geometry, 0, index).yhi_m;
}

struct FireFuelBarrierAxisTraversal
{
    std::array<std::size_t, 2> cells{};
    std::size_t cell_count{1};
    bool moving{false};
    int step{0};
    std::size_t current_cell{};
    amrex::Real next_fraction{
        std::numeric_limits<amrex::Real>::infinity()};
};

inline amrex::Real
fire_barrier_next_axis_fraction(
    const FireCartesianRasterGeometry2D& geometry,
    bool x_axis,
    std::size_t current_cell,
    int step,
    amrex::Real start_coordinate,
    amrex::Real delta)
{
    const amrex::Real face =
        step > 0
        ? fire_barrier_axis_upper_face(
            geometry, x_axis, current_cell)
        : fire_barrier_axis_lower_face(
            geometry, x_axis, current_cell);

    const amrex::Real fraction =
        (face - start_coordinate) / delta;
    if (!std::isfinite(fraction)) {
        throw std::overflow_error(
            "Fire fuel barrier traversal produced a non-finite face fraction");
    }
    return fraction;
}

inline FireFuelBarrierAxisTraversal
make_fire_fuel_barrier_axis_traversal(
    const FireCartesianRasterGeometry2D& geometry,
    bool x_axis,
    amrex::Real start_coordinate,
    amrex::Real delta,
    std::size_t located_cell)
{
    FireFuelBarrierAxisTraversal result;
    const std::size_t axis_count =
        x_axis ? geometry.nx : geometry.ny;

    if (located_cell >= axis_count) {
        throw std::logic_error(
            "Fire fuel barrier traversal received an invalid located cell");
    }

    const amrex::Real lower_face =
        fire_barrier_axis_lower_face(
            geometry, x_axis, located_cell);

    if (delta == amrex::Real(0)) {
        // A path coincident with an internal raster face lies on the material
        // boundary. Keep both adjacent cells in the interval cross-section:
        // one NonBurnable side and one burnable side is a traversable contact
        // surface, while NonBurnable on both sides is interior to the union.
        if (start_coordinate == lower_face
            && located_cell > 0) {
            result.cells[0] = located_cell - 1;
            result.cells[1] = located_cell;
            result.cell_count = 2;
        } else {
            result.cells[0] = located_cell;
            result.cell_count = 1;
        }
        return result;
    }

    result.moving = true;
    result.step = delta > amrex::Real(0) ? 1 : -1;
    result.current_cell = located_cell;

    // Internal faces use positive-cell point ownership. A trajectory moving
    // toward the negative side immediately occupies the negative cell instead.
    if (result.step < 0
        && start_coordinate == lower_face
        && located_cell > 0) {
        result.current_cell = located_cell - 1;
    }

    result.cells[0] = result.current_cell;
    result.cell_count = 1;
    result.next_fraction =
        fire_barrier_next_axis_fraction(
            geometry,
            x_axis,
            result.current_cell,
            result.step,
            start_coordinate,
            delta);

    if (!(result.next_fraction > amrex::Real(0))) {
        throw std::logic_error(
            "Fire fuel barrier traversal did not advance from its initial cell");
    }

    return result;
}

inline void
advance_fire_fuel_barrier_axis(
    FireFuelBarrierAxisTraversal& axis,
    const FireCartesianRasterGeometry2D& geometry,
    bool x_axis,
    amrex::Real start_coordinate,
    amrex::Real delta)
{
    if (!axis.moving || axis.step == 0) {
        throw std::logic_error(
            "attempted to advance a stationary Fire fuel barrier axis");
    }

    const std::size_t axis_count =
        x_axis ? geometry.nx : geometry.ny;

    if (axis.step > 0) {
        if (axis.current_cell + 1 >= axis_count) {
            throw std::out_of_range(
                "Fire fuel barrier segment leaves the raster before its endpoint");
        }
        ++axis.current_cell;
    } else {
        if (axis.current_cell == 0) {
            throw std::out_of_range(
                "Fire fuel barrier segment leaves the raster before its endpoint");
        }
        --axis.current_cell;
    }

    axis.cells[0] = axis.current_cell;
    axis.cell_count = 1;
    axis.next_fraction =
        fire_barrier_next_axis_fraction(
            geometry,
            x_axis,
            axis.current_cell,
            axis.step,
            start_coordinate,
            delta);
}

struct FireFuelBarrierInterval
{
    std::size_t segment_index{};
    amrex::Real entry_fraction{};
    std::size_t sample_begin{};
    std::size_t sample_count{};
    FireCartesianRasterIndex2D representative_cell{};
};

inline void
append_fire_fuel_barrier_interval(
    const FireCartesianRasterGeometry2D& geometry,
    std::size_t segment_index,
    amrex::Real entry_fraction,
    const FireFuelBarrierAxisTraversal& x_axis,
    const FireFuelBarrierAxisTraversal& y_axis,
    std::vector<FireVec2>& sample_positions,
    std::vector<FireFuelBarrierInterval>& intervals)
{
    FireFuelBarrierInterval interval;
    interval.segment_index = segment_index;
    interval.entry_fraction = entry_fraction;
    interval.sample_begin = sample_positions.size();

    bool have_representative = false;
    for (std::size_t y_index = 0;
         y_index < y_axis.cell_count;
         ++y_index) {
        for (std::size_t x_index = 0;
             x_index < x_axis.cell_count;
             ++x_index) {
            const FireCartesianRasterIndex2D cell{
                x_axis.cells[x_index],
                y_axis.cells[y_index]};
            const FireCartesianCell2D bounds =
                fire_cartesian_raster_cell_bounds(
                    geometry,
                    cell.i,
                    cell.j);

            // The lower-left represented face belongs to this exact cell under
            // the authoritative positive-face ownership convention. This lets
            // the existing distributed point sampler serve as an exact ordered
            // cell sampler without adding a second MPI ownership mechanism.
            sample_positions.push_back({
                bounds.xlo_m,
                bounds.ylo_m});

            if (!have_representative) {
                interval.representative_cell = cell;
                have_representative = true;
            }
        }
    }

    interval.sample_count =
        sample_positions.size() - interval.sample_begin;
    if (!have_representative || interval.sample_count == 0) {
        throw std::logic_error(
            "Fire fuel barrier traversal produced an empty material interval");
    }

    intervals.push_back(interval);
}

inline void
append_fire_fuel_barrier_segment_intervals(
    const FireCartesianRasterGeometry2D& geometry,
    std::size_t segment_index,
    const FireVec2& start,
    const FireVec2& end,
    std::vector<FireVec2>& sample_positions,
    std::vector<FireFuelBarrierInterval>& intervals)
{
    if (!std::isfinite(start.x)
        || !std::isfinite(start.y)
        || !std::isfinite(end.x)
        || !std::isfinite(end.y)) {
        throw std::invalid_argument(
            "Fire fuel barrier segment endpoints must be finite");
    }

    const FireCartesianRasterIndex2D start_cell =
        fire_cartesian_raster_locate_cell(
            geometry, start.x, start.y);
    // Endpoint lookup is also the closed-domain validation for the complete
    // segment; the raster domain is convex, so two valid endpoints imply the
    // whole straight segment stays inside the physical raster rectangle.
    (void)fire_cartesian_raster_locate_cell(
        geometry, end.x, end.y);

    const amrex::Real dx = end.x - start.x;
    const amrex::Real dy = end.y - start.y;
    if (!std::isfinite(dx) || !std::isfinite(dy)) {
        throw std::overflow_error(
            "Fire fuel barrier segment displacement is not finite");
    }

    FireFuelBarrierAxisTraversal x_axis =
        make_fire_fuel_barrier_axis_traversal(
            geometry,
            true,
            start.x,
            dx,
            start_cell.i);
    FireFuelBarrierAxisTraversal y_axis =
        make_fire_fuel_barrier_axis_traversal(
            geometry,
            false,
            start.y,
            dy,
            start_cell.j);

    amrex::Real entry_fraction = amrex::Real(0);

    while (true) {
        append_fire_fuel_barrier_interval(
            geometry,
            segment_index,
            entry_fraction,
            x_axis,
            y_axis,
            sample_positions,
            intervals);

        const amrex::Real next_fraction =
            std::min(
                amrex::Real(1),
                std::min(
                    x_axis.next_fraction,
                    y_axis.next_fraction));

        if (next_fraction >= amrex::Real(1)) {
            break;
        }

        if (!(next_fraction > entry_fraction)) {
            throw std::logic_error(
                "Fire fuel barrier traversal made no forward progress");
        }

        const bool cross_x =
            x_axis.moving
            && fire_barrier_fraction_equal(
                x_axis.next_fraction,
                next_fraction);
        const bool cross_y =
            y_axis.moving
            && fire_barrier_fraction_equal(
                y_axis.next_fraction,
                next_fraction);

        if (!cross_x && !cross_y) {
            throw std::logic_error(
                "Fire fuel barrier traversal lost its next raster face");
        }

        if (cross_x) {
            advance_fire_fuel_barrier_axis(
                x_axis,
                geometry,
                true,
                start.x,
                dx);
        }
        if (cross_y) {
            advance_fire_fuel_barrier_axis(
                y_axis,
                geometry,
                false,
                start.y,
                dy);
        }

        entry_fraction = next_fraction;
    }
}

} // namespace detail

/**
 * Collectively locate first NonBurnable contacts for ordered motion segments.
 *
 * Every rank must supply the same ordered starts/ends. Raster traversal is
 * deterministic and host-local; all material cells touched by every segment
 * are sampled in one ordered sparse collective through FireFuelRaster. Thus a
 * batch performs one material all-reduce, never one collective per vertex or
 * crossed cell.
 *
 * The contact set is the interior of the union of NonBurnable cells. Motion
 * along a shared burnable/NonBurnable raster face is therefore allowed, while
 * a raster-face seam flanked by NonBurnable cells on both sides is blocked.
 * This distinction is required for later barrier-contact/sliding dynamics and
 * prevents zero-width grid seams from becoming tunneling paths through a
 * multi-cell barrier.
 */
[[nodiscard]] inline std::vector<std::optional<FireFuelBarrierContact>>
collective_first_nonburnable_contacts(
    const FireFuelRaster& raster,
    const std::vector<FireVec2>& starts_m,
    const std::vector<FireVec2>& ends_m)
{
    if (starts_m.size() != ends_m.size()) {
        throw std::invalid_argument(
            "Fire fuel barrier start/end segment batches must have equal size");
    }
    if (starts_m.empty()) {
        return {};
    }

    const auto& geometry = raster.geometry();
    (void)detail::validate_fire_cartesian_raster_geometry(
        geometry);

    std::vector<FireVec2> sample_positions;
    std::vector<detail::FireFuelBarrierInterval> intervals;

    for (std::size_t segment = 0;
         segment < starts_m.size();
         ++segment) {
        detail::append_fire_fuel_barrier_segment_intervals(
            geometry,
            segment,
            starts_m[segment],
            ends_m[segment],
            sample_positions,
            intervals);
    }

    const std::vector<FireFuelRasterCell> materials =
        raster.collective_sample_points(
            sample_positions);
    if (materials.size() != sample_positions.size()) {
        throw std::logic_error(
            "Fire fuel barrier material sampler returned the wrong size");
    }

    std::vector<std::optional<FireFuelBarrierContact>> result(
        starts_m.size());

    for (const detail::FireFuelBarrierInterval& interval :
         intervals) {
        if (result[interval.segment_index].has_value()) {
            continue;
        }

        bool all_nonburnable = true;
        for (std::size_t offset = 0;
             offset < interval.sample_count;
             ++offset) {
            if (materials[interval.sample_begin + offset].model_id
                != FireFuelModelId::NonBurnable) {
                all_nonburnable = false;
                break;
            }
        }

        if (!all_nonburnable) {
            continue;
        }

        const amrex::Real fraction =
            std::clamp(
                interval.entry_fraction,
                amrex::Real(0),
                amrex::Real(1));
        const FireVec2& start =
            starts_m[interval.segment_index];
        const FireVec2& end =
            ends_m[interval.segment_index];

        result[interval.segment_index] =
            FireFuelBarrierContact{
                fraction,
                {
                    start.x + fraction * (end.x - start.x),
                    start.y + fraction * (end.y - start.y)
                },
                interval.representative_cell};
    }

    return result;
}

} // namespace ERFFire

#endif // ERF_FIRE_FUEL_BARRIER_H_
