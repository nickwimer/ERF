#ifndef ERF_FIRE_FUEL_BARRIER_SWEEP_H_
#define ERF_FIRE_FUEL_BARRIER_SWEEP_H_

#include <ERF_FireCellCoverage.H>
#include <ERF_FireFuelBarrierAdvance.H>
#include <ERF_FireFuelBarrierFront.H>
#include <ERF_FireFuelBarrierTopology.H>
#include <ERF_FireGeometry.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ERFFire
{

/**
 * Conservative continuous contact between one moving front edge and a
 * NonBurnable raster cell.
 *
 * overlap_area_m2 is the positive-area overlap between the cell and the convex
 * hull of the edge's four endpoint states. For linearly moving endpoints the
 * complete ruled edge sweep is contained in that hull, so this test cannot
 * miss a true swept-edge penetration. A strongly folded/sheared edge motion
 * can make the hull larger than the exact ruled sweep; such a case is rejected
 * conservatively rather than allowing possible barrier tunneling.
 */
struct FireFuelBarrierSweptEdgeContact
{
    std::size_t component_index{};
    std::size_t edge_index{};
    FireFrontRole role{FireFrontRole::Outer};
    FireCartesianRasterIndex2D cell{};
    amrex::Real overlap_area_m2{};
};

namespace detail
{

struct FireFuelBarrierSweptEdgeRef
{
    std::size_t component_index{};
    std::size_t edge_index{};
    FireFrontRole role{FireFrontRole::Outer};
};

struct FireFuelBarrierSweptEnvelope
{
    FireFuelBarrierSweptEdgeRef edge{};
    FirePerimeter perimeter;
    std::size_t sample_begin{};
    std::size_t sample_count{};
};

inline bool
fire_barrier_sweep_point_less(
    const FireVec2& lhs,
    const FireVec2& rhs) noexcept
{
    return lhs.x < rhs.x
        || (lhs.x == rhs.x && lhs.y < rhs.y);
}

inline bool
fire_barrier_sweep_point_equal(
    const FireVec2& lhs,
    const FireVec2& rhs) noexcept
{
    return lhs.x == rhs.x && lhs.y == rhs.y;
}

inline amrex::Real
fire_barrier_sweep_turn(
    const FireVec2& origin,
    const FireVec2& first,
    const FireVec2& second) noexcept
{
    return cross_2d(
        first - origin,
        second - origin);
}

/** Convex hull of the four represented endpoint states, CCW without repeats. */
inline std::vector<FireVec2>
fire_barrier_swept_edge_hull(
    const FireVec2& start_first,
    const FireVec2& start_second,
    const FireVec2& end_first,
    const FireVec2& end_second)
{
    std::vector<FireVec2> points{
        start_first,
        start_second,
        end_first,
        end_second};

    for (const FireVec2& point : points) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
            throw std::invalid_argument(
                "Fire barrier swept-edge endpoints must be finite");
        }
    }

    std::sort(
        points.begin(),
        points.end(),
        fire_barrier_sweep_point_less);
    points.erase(
        std::unique(
            points.begin(),
            points.end(),
            fire_barrier_sweep_point_equal),
        points.end());

    if (points.size() < 3U) {
        return {};
    }

    std::vector<FireVec2> hull;
    hull.resize(2U * points.size());
    std::size_t count = 0;

    for (const FireVec2& point : points) {
        while (count >= 2U
               && fire_barrier_sweep_turn(
                      hull[count - 2U],
                      hull[count - 1U],
                      point)
                      <= amrex::Real(0)) {
            --count;
        }
        hull[count++] = point;
    }

    const std::size_t lower_count = count;
    for (std::size_t reverse = points.size() - 1U;
         reverse > 0U;
         --reverse) {
        const FireVec2& point = points[reverse - 1U];
        while (count > lower_count
               && fire_barrier_sweep_turn(
                      hull[count - 2U],
                      hull[count - 1U],
                      point)
                      <= amrex::Real(0)) {
            --count;
        }
        hull[count++] = point;
    }

    if (count == 0U) {
        return {};
    }
    --count; // final point repeats the first one
    hull.resize(count);

    if (hull.size() < 3U
        || signed_polygon_area_m2(hull) <= amrex::Real(0)) {
        return {};
    }

    return hull;
}

inline void
append_fire_barrier_swept_envelope_samples(
    const FireFuelRaster& raster,
    FireFuelBarrierSweptEdgeRef edge,
    const FireVec2& start_first,
    const FireVec2& start_second,
    const FireVec2& end_first,
    const FireVec2& end_second,
    std::vector<FireVec2>& sample_positions,
    std::vector<FireCartesianRasterIndex2D>& sample_cells,
    std::vector<FireFuelBarrierSweptEnvelope>& envelopes)
{
    std::vector<FireVec2> hull_vertices =
        fire_barrier_swept_edge_hull(
            start_first,
            start_second,
            end_first,
            end_second);

    // A line/point envelope has zero swept area. Endpoint trajectories and
    // endpoint edge states are guarded by the lower barrier layers.
    if (hull_vertices.empty()) {
        return;
    }

    FirePerimeter hull(std::move(hull_vertices));
    const auto& vertices = hull.vertices_m();

    amrex::Real xmin = vertices.front().x;
    amrex::Real xmax = vertices.front().x;
    amrex::Real ymin = vertices.front().y;
    amrex::Real ymax = vertices.front().y;
    for (const FireVec2& vertex : vertices) {
        xmin = std::min(xmin, vertex.x);
        xmax = std::max(xmax, vertex.x);
        ymin = std::min(ymin, vertex.y);
        ymax = std::max(ymax, vertex.y);
    }

    const auto& geometry = raster.geometry();
    const FireCartesianRasterIndex2D lower =
        fire_cartesian_raster_locate_cell(
            geometry, xmin, ymin);
    const FireCartesianRasterIndex2D upper =
        fire_cartesian_raster_locate_cell(
            geometry, xmax, ymax);

    std::size_t ilo = std::min(lower.i, upper.i);
    std::size_t ihi = std::max(lower.i, upper.i);
    std::size_t jlo = std::min(lower.j, upper.j);
    std::size_t jhi = std::max(lower.j, upper.j);

    // Exact face ownership assigns an internal face to its positive-index
    // cell. Expanding by one cell makes the bbox candidate set symmetric at
    // represented faces; boundary-only overlap still evaluates to zero area.
    if (ilo > 0U) {
        --ilo;
    }
    if (jlo > 0U) {
        --jlo;
    }
    if (ihi + 1U < geometry.nx) {
        ++ihi;
    }
    if (jhi + 1U < geometry.ny) {
        ++jhi;
    }

    FireFuelBarrierSweptEnvelope envelope{
        edge,
        std::move(hull),
        sample_positions.size(),
        0U};

    for (std::size_t j = jlo; j <= jhi; ++j) {
        for (std::size_t i = ilo; i <= ihi; ++i) {
            const FireCartesianCell2D cell =
                fire_cartesian_raster_cell_bounds(
                    geometry, i, j);
            sample_positions.push_back({
                cell.xlo_m,
                cell.ylo_m});
            sample_cells.push_back({i, j});
            ++envelope.sample_count;
        }
    }

    if (envelope.sample_count == 0U) {
        throw std::logic_error(
            "Fire barrier swept-edge envelope produced no candidate cells");
    }

    envelopes.push_back(std::move(envelope));
}

inline void
require_fire_barrier_sweep_correspondence(
    const std::vector<FireFrontRole>& roles,
    const std::vector<std::vector<FireVec2>>& start_vertices_m,
    const std::vector<std::vector<FireVec2>>& end_vertices_m)
{
    if (roles.empty()
        || roles.size() != start_vertices_m.size()
        || roles.size() != end_vertices_m.size()) {
        throw std::invalid_argument(
            "Fire barrier swept-edge role/component counts must match and be nonzero");
    }

    for (std::size_t component = 0;
         component < roles.size();
         ++component) {
        if (start_vertices_m[component].size() < 2U
            || start_vertices_m[component].size()
                != end_vertices_m[component].size()) {
            throw std::invalid_argument(
                "Fire barrier swept-edge vertex correspondence is invalid");
        }
    }
}

inline void
throw_first_nonburnable_swept_edge_contact(
    const std::vector<FireFuelBarrierSweptEdgeContact>& contacts,
    const char* stage)
{
    if (stage == nullptr) {
        throw std::invalid_argument(
            "Fire barrier swept-edge stage must not be null");
    }
    if (contacts.empty()) {
        return;
    }

    const auto& first = contacts.front();
    throw std::runtime_error(
        std::string("Fire front swept edge enters NonBurnable material at ")
        + stage
        + " (component "
        + std::to_string(first.component_index)
        + ", edge "
        + std::to_string(first.edge_index)
        + ")");
}

} // namespace detail

/**
 * Conservatively detect continuous edge sweeps entering NonBurnable material.
 *
 * Matching vertices move linearly from start to end. For each corresponding
 * closed front edge, the ruled continuous sweep is contained by the convex
 * hull of its four endpoint states. Candidate raster cells inside that small
 * envelope are sampled in one ordered sparse collective for the complete
 * front. A NonBurnable cell is reported only when it has positive-area overlap
 * with the envelope above a roundoff-scale cell-area tolerance.
 *
 * Endpoint edge legality and individual vertex trajectories are intentionally
 * separate invariants, handled by the existing barrier-front and motion-clip
 * layers. This function closes the remaining between-state edge-tunneling gap.
 */
[[nodiscard]] inline std::vector<FireFuelBarrierSweptEdgeContact>
collective_nonburnable_raw_front_swept_edge_contacts(
    const std::vector<FireFrontRole>& roles,
    const std::vector<std::vector<FireVec2>>& start_vertices_m,
    const std::vector<std::vector<FireVec2>>& end_vertices_m,
    const FireFuelRaster& raster)
{
    detail::require_fire_barrier_sweep_correspondence(
        roles,
        start_vertices_m,
        end_vertices_m);

    const auto& geometry = raster.geometry();
    (void)detail::validate_fire_cartesian_raster_geometry(geometry);

    std::vector<FireVec2> sample_positions;
    std::vector<FireCartesianRasterIndex2D> sample_cells;
    std::vector<detail::FireFuelBarrierSweptEnvelope> envelopes;

    for (std::size_t component = 0;
         component < roles.size();
         ++component) {
        const auto& start = start_vertices_m[component];
        const auto& end = end_vertices_m[component];

        for (std::size_t edge = 0;
             edge < start.size();
             ++edge) {
            const std::size_t next =
                (edge + 1U) % start.size();
            detail::append_fire_barrier_swept_envelope_samples(
                raster,
                detail::FireFuelBarrierSweptEdgeRef{
                    component,
                    edge,
                    roles[component]},
                start[edge],
                start[next],
                end[edge],
                end[next],
                sample_positions,
                sample_cells,
                envelopes);
        }
    }

    if (envelopes.empty()) {
        return {};
    }

    const std::vector<FireFuelRasterCell> materials =
        raster.collective_sample_points(sample_positions);
    if (materials.size() != sample_positions.size()
        || sample_cells.size() != sample_positions.size()) {
        throw std::logic_error(
            "Fire barrier swept-edge material batch returned the wrong size");
    }

    std::vector<FireFuelBarrierSweptEdgeContact> result;
    for (const detail::FireFuelBarrierSweptEnvelope& envelope :
         envelopes) {
        for (std::size_t offset = 0;
             offset < envelope.sample_count;
             ++offset) {
            const std::size_t sample =
                envelope.sample_begin + offset;
            if (materials[sample].model_id
                != FireFuelModelId::NonBurnable) {
                continue;
            }

            const FireCartesianRasterIndex2D cell_index =
                sample_cells[sample];
            const FireCartesianCell2D cell =
                detail::fire_cartesian_raster_cell_bounds(
                    geometry,
                    cell_index.i,
                    cell_index.j);
            const amrex::Real overlap_area_m2 =
                fire_perimeter_cell_intersection_area_m2(
                    envelope.perimeter,
                    cell);
            const amrex::Real cell_area_m2 =
                detail::fire_cartesian_cell_area_m2(cell);
            const amrex::Real area_tolerance_m2 =
                amrex::Real(1024)
                * std::numeric_limits<amrex::Real>::epsilon()
                * cell_area_m2;

            if (overlap_area_m2 <= area_tolerance_m2) {
                continue;
            }

            result.push_back({
                envelope.edge.component_index,
                envelope.edge.edge_index,
                envelope.edge.role,
                cell_index,
                overlap_area_m2});
        }
    }

    return result;
}

inline void
collective_require_raw_fire_front_swept_edges_avoid_nonburnable(
    const std::vector<FireFrontRole>& roles,
    const std::vector<std::vector<FireVec2>>& start_vertices_m,
    const std::vector<std::vector<FireVec2>>& end_vertices_m,
    const FireFuelRaster& raster,
    const char* stage)
{
    detail::throw_first_nonburnable_swept_edge_contact(
        collective_nonburnable_raw_front_swept_edge_contacts(
            roles,
            start_vertices_m,
            end_vertices_m,
            raster),
        stage);
}

[[nodiscard]] inline std::vector<FireFuelBarrierSweptEdgeContact>
collective_nonburnable_front_swept_edge_contacts(
    const FireFront& start_front,
    const FireFront& end_front,
    const FireFuelRaster& raster)
{
    if (start_front.components().size()
        != end_front.components().size()) {
        throw std::invalid_argument(
            "Fire barrier swept-front component counts must match");
    }

    std::vector<FireFrontRole> roles;
    std::vector<std::vector<FireVec2>> start_vertices_m;
    std::vector<std::vector<FireVec2>> end_vertices_m;
    roles.reserve(start_front.components().size());
    start_vertices_m.reserve(start_front.components().size());
    end_vertices_m.reserve(end_front.components().size());

    for (std::size_t component = 0;
         component < start_front.components().size();
         ++component) {
        if (start_front.components()[component].role
            != end_front.components()[component].role) {
            throw std::invalid_argument(
                "Fire barrier swept-front component roles must match");
        }

        roles.push_back(start_front.components()[component].role);
        start_vertices_m.push_back(
            start_front.components()[component]
                .perimeter.vertices_m());
        end_vertices_m.push_back(
            end_front.components()[component]
                .perimeter.vertices_m());
    }

    return collective_nonburnable_raw_front_swept_edge_contacts(
        roles,
        start_vertices_m,
        end_vertices_m,
        raster);
}

inline void
collective_require_fire_front_swept_edges_avoid_nonburnable(
    const FireFront& start_front,
    const FireFront& end_front,
    const FireFuelRaster& raster,
    const char* stage)
{
    detail::throw_first_nonburnable_swept_edge_contact(
        collective_nonburnable_front_swept_edge_contacts(
            start_front,
            end_front,
            raster),
        stage);
}

/**
 * Final fixed-topology NonBurnable RK2 contract.
 *
 * The lower layer clips vertex trajectories and validates initial/midpoint/end
 * edge states. This wrapper additionally validates the continuous linear
 * physical sweep from the committed start front to the RK2 end front, matching
 * the dense-output sweep convention used by arrival and combustion history.
 */
[[nodiscard]] inline FireFront
advance_front_rk2_batched_constrained_by_nonburnable(
    const FireFront& front,
    amrex::Real time_s,
    amrex::Real dt_s,
    const NormalSpeedBatchFunction& normal_speed_mps,
    const FireFuelRaster& fuel_raster)
{
    FireFront result =
        advance_front_rk2_batched_clipped_to_nonburnable(
            front,
            time_s,
            dt_s,
            normal_speed_mps,
            fuel_raster);

    collective_require_fire_front_swept_edges_avoid_nonburnable(
        front,
        result,
        fuel_raster,
        "RK2 physical sweep");

    return result;
}

/**
 * Final topology-aware NonBurnable RK2 contract through event localization.
 *
 * The lower topology layer clips vertex trajectories and validates all discrete
 * raw/resolved edge states. This wrapper additionally guards the continuous
 * physical sweep from the committed start state to either the localized event
 * state or the completed end state. It intentionally checks only to the first
 * localized topology event, never geometry that would occur after that event.
 */
[[nodiscard]] inline FireFrontTopologyAdvanceResult
advance_front_rk2_batched_until_topology_event_constrained_by_nonburnable(
    const FireFront& front,
    amrex::Real time_s,
    amrex::Real dt_s,
    const NormalSpeedBatchFunction& normal_speed_mps,
    const FireFuelRaster& fuel_raster)
{
    FireFrontTopologyAdvanceResult result =
        advance_front_rk2_batched_until_topology_event_avoiding_nonburnable(
            front,
            time_s,
            dt_s,
            normal_speed_mps,
            fuel_raster);

    if (result.topology_event.has_value()) {
        std::vector<FireFrontRole> roles;
        std::vector<std::vector<FireVec2>> start_vertices_m;
        roles.reserve(front.components().size());
        start_vertices_m.reserve(front.components().size());
        for (const FireFrontComponent& component : front.components()) {
            roles.push_back(component.role);
            start_vertices_m.push_back(
                component.perimeter.vertices_m());
        }

        collective_require_raw_fire_front_swept_edges_avoid_nonburnable(
            roles,
            start_vertices_m,
            result.terminal_vertices_m,
            fuel_raster,
            "topology-event physical sweep");
        return result;
    }

    if (!result.completed_front.has_value()) {
        throw std::logic_error(
            "Fire barrier topology sweep guard received no completed front or event");
    }

    collective_require_fire_front_swept_edges_avoid_nonburnable(
        front,
        *result.completed_front,
        fuel_raster,
        "topology-aware RK2 physical sweep");

    return result;
}

} // namespace ERFFire

#endif // ERF_FIRE_FUEL_BARRIER_SWEEP_H_
