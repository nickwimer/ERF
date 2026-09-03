#ifndef ERF_FIRE_FRONT_PROPAGATOR_H
#define ERF_FIRE_FRONT_PROPAGATOR_H

#include <ERF_FireFrontCollision.H>
#include <ERF_VectorPerimeterPropagator.H>

#include <optional>
#include <vector>

namespace ERFFire
{

struct FireFrontAdvanceResult
{
    FireFront front;
    amrex::Real advanced_dt_s{};
    std::vector<FireVec2> terminal_vertices_m;
    std::optional<FirePerimeterCollision> topology_event;
};

enum class FireFrontComponentTopologyEventType
{
    SelfContact,
    HoleExtinction
};

struct FireFrontComponentTopologyEvent
{
    std::size_t component_index{};
    FireFrontRole role{FireFrontRole::Outer};
    // Valid geometry only for SelfContact. HoleExtinction is represented by
    // raw terminal component vertices because the event perimeter has zero
    // area and cannot be constructed as a FirePerimeter.
    FirePerimeterCollision collision{};
    FireFrontComponentTopologyEventType type{
        FireFrontComponentTopologyEventType::SelfContact};
    amrex::Real motion_fraction{};
};

struct FireFrontTopologyAdvanceResult
{
    std::optional<FireFront> completed_front;
    amrex::Real advanced_dt_s{};
    std::vector<std::vector<FireVec2>> terminal_vertices_m;
    std::optional<FireFrontComponentTopologyEvent> topology_event;
};

/**
 * Advance one simple perimeter with batched explicit-midpoint/RK2 dynamics
 * until either the requested dt is complete or the first topology-changing
 * self-contact is reached.
 *
 * When no event occurs, front contains one Outer component, advanced_dt_s
 * equals dt_s, terminal_vertices_m is the ordinary RK2 endpoint, and
 * topology_event is empty.
 *
 * When an event occurs, front is the Outer+Hole split at the event,
 * advanced_dt_s is the consumed fraction of dt_s, terminal_vertices_m
 * preserves the original perimeter vertex correspondence at the event, and
 * topology_event describes the localized contact.
 */
[[nodiscard]] FireFrontAdvanceResult
advance_perimeter_rk2_batched_until_topology_event(
    const FirePerimeter& perimeter,
    amrex::Real time_s,
    amrex::Real dt_s,
    const NormalSpeedBatchFunction& normal_speed_mps);

/**
 * Advance every component of a fixed-topology FireFront with one batched
 * explicit-midpoint/RK2 step.
 *
 * All component vertices are flattened into one normal-speed batch at each
 * RK2 stage. Spread normals point from burned material toward available fuel:
 * geometrically outward for Outer components and inward for Hole components.
 *
 * Component count, ordering, roles, and vertex correspondence are preserved.
 * This primitive does not resolve additional topology changes.
 */
[[nodiscard]] FireFront
advance_front_rk2_batched(
    const FireFront& front,
    amrex::Real time_s,
    amrex::Real dt_s,
    const NormalSpeedBatchFunction& normal_speed_mps);

/**
 * Advance a fixed-topology FireFront until either dt is complete or the
 * earliest supported topology event of any boundary component is reached.
 *
 * Supported events are component self-contact and continuous extinction of a
 * Hole when its represented signed area reaches exactly zero.
 *
 * On normal completion, completed_front is populated, topology_event is
 * empty, advanced_dt_s equals dt_s, and terminal_vertices_m contains the
 * endpoint vertices for every component in original component order.
 *
 * On a topology event, completed_front is empty, advanced_dt_s is the
 * consumed part of dt_s, terminal_vertices_m contains every component at the
 * common localized event time, and topology_event identifies the component,
 * role, event type, and full-step motion fraction.
 */
[[nodiscard]] FireFrontTopologyAdvanceResult
advance_front_rk2_batched_until_topology_event(
    const FireFront& front,
    amrex::Real time_s,
    amrex::Real dt_s,
    const NormalSpeedBatchFunction& normal_speed_mps);

} // namespace ERFFire

#endif // ERF_FIRE_FRONT_PROPAGATOR_H
