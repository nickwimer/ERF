#ifndef ERF_FIRE_FUEL_BARRIER_TOPOLOGY_H_
#define ERF_FIRE_FUEL_BARRIER_TOPOLOGY_H_

#include <ERF_FireFuelBarrierAdvance.H>
#include <ERF_FireFuelBarrierFront.H>
#include <ERF_FireFrontCollision.H>
#include <ERF_FireGeometry.H>

#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ERFFire
{
namespace detail
{

inline std::vector<FireFrontRole>
fire_barrier_topology_roles(const FireFront& front)
{
    std::vector<FireFrontRole> roles;
    roles.reserve(front.components().size());
    for (const FireFrontComponent& component : front.components()) {
        roles.push_back(component.role);
    }
    return roles;
}

inline std::vector<std::vector<FireVec2>>
fire_barrier_topology_vertices(const FireFront& front)
{
    std::vector<std::vector<FireVec2>> vertices_m;
    vertices_m.reserve(front.components().size());
    for (const FireFrontComponent& component : front.components()) {
        vertices_m.push_back(component.perimeter.vertices_m());
    }
    return vertices_m;
}

inline std::vector<FireVec2>
fire_barrier_flatten_component_vertices(
    const std::vector<std::vector<FireVec2>>& component_vertices_m)
{
    std::size_t count = 0;
    for (const auto& vertices : component_vertices_m) {
        count += vertices.size();
    }

    std::vector<FireVec2> flat;
    flat.reserve(count);
    for (const auto& vertices : component_vertices_m) {
        flat.insert(flat.end(), vertices.begin(), vertices.end());
    }
    return flat;
}

inline void
fire_barrier_scatter_component_vertices(
    const std::vector<FireVec2>& flat,
    std::vector<std::vector<FireVec2>>& component_vertices_m)
{
    std::size_t expected = 0;
    for (const auto& vertices : component_vertices_m) {
        expected += vertices.size();
    }
    if (flat.size() != expected) {
        throw std::logic_error(
            "Fire barrier topology clip lost vertex correspondence");
    }

    std::size_t offset = 0;
    for (auto& vertices : component_vertices_m) {
        for (FireVec2& vertex : vertices) {
            vertex = flat[offset++];
        }
    }
}

inline std::vector<std::optional<FireFuelBarrierContact>>
collective_clip_fire_barrier_topology_candidates(
    const FireFuelRaster& raster,
    const std::vector<std::vector<FireVec2>>& starts_m,
    std::vector<std::vector<FireVec2>>& candidates_m)
{
    if (starts_m.size() != candidates_m.size()) {
        throw std::logic_error(
            "Fire barrier topology start/candidate component counts differ");
    }
    for (std::size_t component = 0;
         component < starts_m.size();
         ++component) {
        if (starts_m[component].size()
            != candidates_m[component].size()) {
            throw std::logic_error(
                "Fire barrier topology start/candidate vertex counts differ");
        }
    }

    const std::vector<FireVec2> flat_starts =
        fire_barrier_flatten_component_vertices(starts_m);
    const std::vector<FireVec2> flat_candidates =
        fire_barrier_flatten_component_vertices(candidates_m);

    const FireFuelBarrierMotionClipResult clipped =
        collective_clip_fire_fuel_motion_to_nonburnable(
            raster,
            flat_starts,
            flat_candidates);

    fire_barrier_scatter_component_vertices(
        clipped.positions_m,
        candidates_m);
    return clipped.contacts;
}

inline void
require_fire_barrier_topology_counter_clockwise(
    const std::vector<FireVec2>& vertices_m,
    const char* stage)
{
    if (detail::signed_polygon_area_m2(vertices_m)
        <= amrex::Real(0)) {
        throw std::runtime_error(
            std::string(
                "Fire barrier topology component lost counter-clockwise orientation at ")
            + stage);
    }
}

inline std::vector<std::vector<FireVec2>>
interpolate_fire_barrier_topology_vertices(
    const std::vector<std::vector<FireVec2>>& start_vertices_m,
    const std::vector<std::vector<FireVec2>>& end_vertices_m,
    amrex::Real fraction)
{
    if (start_vertices_m.size() != end_vertices_m.size()) {
        throw std::logic_error(
            "Fire barrier topology interpolation lost component correspondence");
    }

    std::vector<std::vector<FireVec2>> result;
    result.reserve(start_vertices_m.size());

    for (std::size_t component = 0;
         component < start_vertices_m.size();
         ++component) {
        if (start_vertices_m[component].size()
            != end_vertices_m[component].size()) {
            throw std::logic_error(
                "Fire barrier topology interpolation lost vertex correspondence");
        }

        std::vector<FireVec2> vertices;
        vertices.reserve(start_vertices_m[component].size());
        for (std::size_t vertex = 0;
             vertex < start_vertices_m[component].size();
             ++vertex) {
            vertices.push_back(
                start_vertices_m[component][vertex]
                + fraction
                    * (end_vertices_m[component][vertex]
                       - start_vertices_m[component][vertex]));
        }
        result.push_back(std::move(vertices));
    }

    return result;
}

struct FireBarrierTopologyStageEvent
{
    std::optional<FirePerimeterCollision> collision;
    std::size_t collision_component{};
    std::optional<amrex::Real> hole_extinction;
    std::size_t hole_extinction_component{};
};

inline FireBarrierTopologyStageEvent
locate_fire_barrier_topology_stage_event(
    const FireFront& start_front,
    const std::vector<std::vector<FireVec2>>& candidate_vertices_m)
{
    if (start_front.components().size()
        != candidate_vertices_m.size()) {
        throw std::logic_error(
            "Fire barrier topology event search lost component correspondence");
    }

    FireBarrierTopologyStageEvent result;

    for (std::size_t component = 0;
         component < start_front.components().size();
         ++component) {
        const FireFrontComponent& start_component =
            start_front.components()[component];

        const auto collision =
            locate_first_perimeter_collision(
                start_component.perimeter,
                candidate_vertices_m[component]);
        if (collision.has_value()
            && (!result.collision.has_value()
                || collision->motion_fraction
                    < result.collision->motion_fraction)) {
            result.collision = *collision;
            result.collision_component = component;
        }

        if (start_component.role != FireFrontRole::Hole) {
            continue;
        }

        const auto extinction =
            locate_first_perimeter_area_collapse(
                start_component.perimeter,
                candidate_vertices_m[component]);
        if (extinction.has_value()
            && (!result.hole_extinction.has_value()
                || *extinction < *result.hole_extinction)) {
            result.hole_extinction = *extinction;
            result.hole_extinction_component = component;
        }
    }

    return result;
}

inline bool
fire_barrier_topology_extinction_precedes(
    const FireBarrierTopologyStageEvent& event) noexcept
{
    return event.hole_extinction.has_value()
        && (!event.collision.has_value()
            || *event.hole_extinction
                < event.collision->motion_fraction);
}

inline FireFrontTopologyAdvanceResult
make_fire_barrier_topology_event_result(
    const FireFront& start_front,
    const std::vector<FireFrontRole>& roles,
    const std::vector<std::vector<FireVec2>>& start_vertices_m,
    const std::vector<std::vector<FireVec2>>& candidate_vertices_m,
    const FireBarrierTopologyStageEvent& stage_event,
    amrex::Real requested_dt_s,
    amrex::Real stage_fraction,
    const FireFuelRaster& fuel_raster)
{
    const bool extinction_precedes =
        fire_barrier_topology_extinction_precedes(
            stage_event);

    amrex::Real local_fraction{};
    std::size_t component_index{};
    FireFrontComponentTopologyEventType type{
        FireFrontComponentTopologyEventType::SelfContact};
    FirePerimeterCollision collision{};

    if (extinction_precedes) {
        local_fraction = *stage_event.hole_extinction;
        component_index =
            stage_event.hole_extinction_component;
        type = FireFrontComponentTopologyEventType::HoleExtinction;
    } else if (stage_event.collision.has_value()) {
        local_fraction =
            stage_event.collision->motion_fraction;
        component_index =
            stage_event.collision_component;
        collision = *stage_event.collision;
        type = FireFrontComponentTopologyEventType::SelfContact;
    } else {
        throw std::logic_error(
            "Fire barrier topology event result requested without an event");
    }

    const amrex::Real full_step_fraction =
        stage_fraction * local_fraction;

    std::vector<std::vector<FireVec2>> event_vertices_m =
        interpolate_fire_barrier_topology_vertices(
            start_vertices_m,
            candidate_vertices_m,
            local_fraction);

    collective_require_raw_fire_front_edges_avoid_nonburnable(
        roles,
        event_vertices_m,
        fuel_raster,
        "topology event state");

    if (type == FireFrontComponentTopologyEventType::SelfContact) {
        collision.motion_fraction = full_step_fraction;
    }

    return {
        std::nullopt,
        requested_dt_s * full_step_fraction,
        {},
        std::move(event_vertices_m),
        FireFrontComponentTopologyEvent{
            component_index,
            start_front.components()[component_index].role,
            std::move(collision),
            type,
            full_step_fraction
        }
    };
}

inline FireFront
make_fire_barrier_topology_front(
    const FireFront& reference,
    const std::vector<std::vector<FireVec2>>& vertices_m,
    const char* stage)
{
    if (reference.components().size() != vertices_m.size()) {
        throw std::logic_error(
            "Fire barrier topology front rebuild lost component correspondence");
    }

    std::vector<FireFrontComponent> components;
    components.reserve(reference.components().size());

    for (std::size_t component = 0;
         component < reference.components().size();
         ++component) {
        require_fire_barrier_topology_counter_clockwise(
            vertices_m[component],
            stage);
        components.push_back({
            reference.components()[component].role,
            FirePerimeter(vertices_m[component])
        });
    }

    return FireFront(std::move(components));
}

} // namespace detail

/**
 * Topology-aware explicit-midpoint/RK2 advance constrained by NonBurnable.
 *
 * Vertex trajectories are clipped at first NonBurnable contact at both RK
 * stages before topology events are localized. A midpoint contact is persistent
 * for the rest of the RK2 step when no earlier topology event intervenes: with
 * no represented sliding law, a zero stage-two material speed on the exact
 * barrier face must not algebraically move the final explicit-midpoint state
 * back into the burnable region. If a self-contact or Hole extinction occurs,
 * every raw component edge at the common event time is required to remain
 * outside NonBurnable interior before the event is returned. When no topology
 * event occurs, midpoint/final fronts satisfy the same edge invariant before
 * constructing the next stage.
 *
 * This routine preserves the existing topology-event ordering: Hole
 * extinction wins only when it occurs strictly before self-contact, so a tie
 * remains a self-contact event. With an all-burnable raster its vertex
 * arithmetic and event localization match the existing topology-aware RK2
 * advance.
 *
 * The returned topology event is intentionally not resolved here. The caller
 * must apply the existing topology resolver and then validate the resolved
 * FireFront with collective_require_resolved_fire_topology_avoids_nonburnable
 * before committing history.
 */
[[nodiscard]] inline FireFrontTopologyAdvanceResult
advance_front_rk2_batched_until_topology_event_avoiding_nonburnable(
    const FireFront& front,
    amrex::Real time_s,
    amrex::Real dt_s,
    const NormalSpeedBatchFunction& normal_speed_mps,
    const FireFuelRaster& fuel_raster)
{
    if (!std::isfinite(time_s)) {
        throw std::invalid_argument(
            "Fire barrier topology time must be finite");
    }
    if (!std::isfinite(dt_s) || dt_s < amrex::Real(0)) {
        throw std::invalid_argument(
            "Fire barrier topology timestep must be finite and nonnegative");
    }
    if (!normal_speed_mps) {
        throw std::invalid_argument(
            "Fire barrier topology normal-speed callback must be set");
    }

    collective_require_fire_front_edges_avoid_nonburnable(
        front,
        fuel_raster,
        "initial topology-aware RK2 state");

    const std::vector<FireFrontRole> roles =
        detail::fire_barrier_topology_roles(front);
    const std::vector<std::vector<FireVec2>> initial_vertices_m =
        detail::fire_barrier_topology_vertices(front);

    if (dt_s == amrex::Real(0)) {
        return {
            front,
            amrex::Real(0),
            initial_vertices_m,
            initial_vertices_m,
            std::nullopt
        };
    }

    std::vector<FireVec2> initial_positions;
    std::vector<FireVec2> initial_normals;
    detail::fire_barrier_flatten_front(
        front,
        initial_positions,
        initial_normals);

    const std::vector<amrex::Real> initial_speeds =
        detail::fire_barrier_checked_normal_speeds(
            normal_speed_mps,
            initial_positions,
            initial_normals,
            time_s);

    std::vector<std::vector<FireVec2>> midpoint_vertices_m =
        initial_vertices_m;
    std::size_t flat_index = 0;
    for (auto& component : midpoint_vertices_m) {
        for (FireVec2& vertex : component) {
            vertex =
                initial_positions[flat_index]
                + initial_normals[flat_index]
                    * (amrex::Real(0.5)
                       * dt_s
                       * initial_speeds[flat_index]);
            ++flat_index;
        }
    }

    const std::vector<std::optional<FireFuelBarrierContact>>
        midpoint_contacts =
            detail::collective_clip_fire_barrier_topology_candidates(
                fuel_raster,
                initial_vertices_m,
                midpoint_vertices_m);

    const detail::FireBarrierTopologyStageEvent midpoint_event =
        detail::locate_fire_barrier_topology_stage_event(
            front,
            midpoint_vertices_m);

    if (midpoint_event.collision.has_value()
        || midpoint_event.hole_extinction.has_value()) {
        return detail::make_fire_barrier_topology_event_result(
            front,
            roles,
            initial_vertices_m,
            midpoint_vertices_m,
            midpoint_event,
            dt_s,
            amrex::Real(0.5),
            fuel_raster);
    }

    collective_require_raw_fire_front_edges_avoid_nonburnable(
        roles,
        midpoint_vertices_m,
        fuel_raster,
        "topology-aware RK2 midpoint");

    FireFront midpoint_front =
        detail::make_fire_barrier_topology_front(
            front,
            midpoint_vertices_m,
            "topology-aware RK2 midpoint");

    std::vector<FireVec2> midpoint_positions;
    std::vector<FireVec2> midpoint_normals;
    detail::fire_barrier_flatten_front(
        midpoint_front,
        midpoint_positions,
        midpoint_normals);

    const std::vector<amrex::Real> midpoint_speeds =
        detail::fire_barrier_checked_normal_speeds(
            normal_speed_mps,
            midpoint_positions,
            midpoint_normals,
            time_s + amrex::Real(0.5) * dt_s);

    std::vector<std::vector<FireVec2>> final_vertices_m =
        initial_vertices_m;
    flat_index = 0;
    for (auto& component : final_vertices_m) {
        for (FireVec2& vertex : component) {
            vertex =
                initial_positions[flat_index]
                + midpoint_normals[flat_index]
                    * (dt_s * midpoint_speeds[flat_index]);
            ++flat_index;
        }
    }

    (void)detail::collective_clip_fire_barrier_topology_candidates(
        fuel_raster,
        initial_vertices_m,
        final_vertices_m);

    if (midpoint_contacts.size() != initial_positions.size()) {
        throw std::logic_error(
            "Fire barrier topology midpoint contacts lost vertex correspondence");
    }

    flat_index = 0;
    for (auto& component : final_vertices_m) {
        for (FireVec2& vertex : component) {
            if (midpoint_contacts[flat_index].has_value()) {
                vertex =
                    midpoint_contacts[flat_index]->position_m;
            }
            ++flat_index;
        }
    }

    const detail::FireBarrierTopologyStageEvent final_event =
        detail::locate_fire_barrier_topology_stage_event(
            front,
            final_vertices_m);

    if (final_event.collision.has_value()
        || final_event.hole_extinction.has_value()) {
        return detail::make_fire_barrier_topology_event_result(
            front,
            roles,
            initial_vertices_m,
            final_vertices_m,
            final_event,
            dt_s,
            amrex::Real(1),
            fuel_raster);
    }

    collective_require_raw_fire_front_edges_avoid_nonburnable(
        roles,
        final_vertices_m,
        fuel_raster,
        "topology-aware RK2 final state");

    FireFront final_front =
        detail::make_fire_barrier_topology_front(
            front,
            final_vertices_m,
            "topology-aware RK2 final state");

    return {
        std::move(final_front),
        dt_s,
        std::move(midpoint_vertices_m),
        std::move(final_vertices_m),
        std::nullopt
    };
}

/**
 * Validate the FireFront created by resolving a topology event.
 *
 * Topology split/extinction operations may change component count and edge
 * connectivity. The resolved front must therefore be checked independently
 * even when the raw event-time edge loops were barrier-legal.
 */
inline void
collective_require_resolved_fire_topology_avoids_nonburnable(
    const FireFront& resolved_front,
    const FireFuelRaster& fuel_raster)
{
    collective_require_fire_front_edges_avoid_nonburnable(
        resolved_front,
        fuel_raster,
        "resolved topology state");
}

} // namespace ERFFire

#endif // ERF_FIRE_FUEL_BARRIER_TOPOLOGY_H_
