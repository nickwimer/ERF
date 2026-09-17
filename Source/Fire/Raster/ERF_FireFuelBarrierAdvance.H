#ifndef ERF_FIRE_FUEL_BARRIER_ADVANCE_H_
#define ERF_FIRE_FUEL_BARRIER_ADVANCE_H_

#include <ERF_FireFuelBarrier.H>
#include <ERF_FireFuelBarrierFront.H>
#include <ERF_FireFrontPropagator.H>

#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ERFFire
{

/** Result of clipping an ordered batch of candidate motions to NonBurnable. */
struct FireFuelBarrierMotionClipResult
{
    std::vector<FireVec2> positions_m;
    std::vector<std::optional<FireFuelBarrierContact>> contacts;
};

/**
 * Clip ordered candidate motions at their first NonBurnable contact.
 *
 * Every rank must provide the same starts/candidates. Candidates with no
 * contact are preserved bit-for-bit. Contacted candidates are replaced by the
 * exact raster-face contact returned by collective_first_nonburnable_contacts.
 */
[[nodiscard]] inline FireFuelBarrierMotionClipResult
collective_clip_fire_fuel_motion_to_nonburnable(
    const FireFuelRaster& raster,
    const std::vector<FireVec2>& starts_m,
    const std::vector<FireVec2>& candidates_m)
{
    if (starts_m.size() != candidates_m.size()) {
        throw std::invalid_argument(
            "Fire fuel barrier motion clip start/candidate batches must have equal size");
    }

    FireFuelBarrierMotionClipResult result;
    result.positions_m = candidates_m;
    result.contacts =
        collective_first_nonburnable_contacts(
            raster,
            starts_m,
            candidates_m);

    if (result.contacts.size() != candidates_m.size()) {
        throw std::logic_error(
            "Fire fuel barrier contact batch returned the wrong size");
    }

    for (std::size_t index = 0;
         index < result.contacts.size();
         ++index) {
        if (result.contacts[index].has_value()) {
            result.positions_m[index] =
                result.contacts[index]->position_m;
        }
    }

    return result;
}

namespace detail
{

inline std::vector<amrex::Real>
fire_barrier_checked_normal_speeds(
    const NormalSpeedBatchFunction& normal_speed_mps,
    const std::vector<FireVec2>& positions_m,
    const std::vector<FireVec2>& normals,
    amrex::Real time_s)
{
    if (positions_m.size() != normals.size()) {
        throw std::logic_error(
            "Fire barrier RK2 positions/normals size mismatch");
    }

    std::vector<amrex::Real> speeds =
        normal_speed_mps(
            positions_m,
            normals,
            time_s);

    if (speeds.size() != positions_m.size()) {
        throw std::invalid_argument(
            "Fire barrier RK2 normal-speed callback returned the wrong size");
    }

    for (const amrex::Real speed : speeds) {
        if (!std::isfinite(speed)
            || speed < amrex::Real(0)) {
            throw std::invalid_argument(
                "Fire barrier RK2 normal speed must be finite and nonnegative");
        }
    }

    return speeds;
}

inline std::size_t
fire_barrier_front_vertex_count(const FireFront& front)
{
    std::size_t count = 0;
    for (const FireFrontComponent& component : front.components()) {
        count += component.perimeter.size();
    }
    return count;
}

inline void
fire_barrier_flatten_front(
    const FireFront& front,
    std::vector<FireVec2>& positions_m,
    std::vector<FireVec2>& normals)
{
    const std::size_t count =
        fire_barrier_front_vertex_count(front);
    positions_m.clear();
    normals.clear();
    positions_m.reserve(count);
    normals.reserve(count);

    const auto& components = front.components();
    for (std::size_t component_index = 0;
         component_index < components.size();
         ++component_index) {
        const FirePerimeter& perimeter =
            components[component_index].perimeter;
        for (std::size_t vertex_index = 0;
             vertex_index < perimeter.size();
             ++vertex_index) {
            positions_m.push_back(
                perimeter.vertices_m()[vertex_index]);
            normals.push_back(
                front.spread_normal_unit(
                    component_index,
                    vertex_index));
        }
    }
}

inline FireFront
fire_barrier_rebuild_front(
    const FireFront& reference,
    const std::vector<FireVec2>& flat_vertices_m)
{
    if (flat_vertices_m.size()
        != fire_barrier_front_vertex_count(reference)) {
        throw std::logic_error(
            "Fire barrier RK2 lost front vertex correspondence");
    }

    std::vector<FireFrontComponent> components;
    components.reserve(reference.components().size());

    std::size_t offset = 0;
    for (const FireFrontComponent& component : reference.components()) {
        std::vector<FireVec2> vertices;
        vertices.reserve(component.perimeter.size());
        for (std::size_t vertex = 0;
             vertex < component.perimeter.size();
             ++vertex, ++offset) {
            vertices.push_back(flat_vertices_m[offset]);
        }
        components.push_back({
            component.role,
            FirePerimeter(std::move(vertices))
        });
    }

    return FireFront(std::move(components));
}

} // namespace detail

/**
 * Fixed-topology explicit-midpoint/RK2 advance with NonBurnable trajectory
 * clipping and edge-exclusion checks at both RK stages.
 *
 * Every candidate vertex trajectory is clipped at first entry into the
 * interior of the NonBurnable cell union before the next RK geometry is
 * constructed. The reconstructed initial, midpoint, and final fronts are also
 * required to have no edge entering NonBurnable interior, so two individually
 * legal vertex trajectories cannot create an illegal chord through a barrier.
 * With no contacts the arithmetic/order of vertex evolution matches
 * advance_front_rk2_batched exactly.
 *
 * This primitive deliberately does not yet resolve front self-contact,
 * barrier sliding/rerouting, topology events interacting with barriers, or
 * remeshing near a barrier. Edge penetration is rejected rather than repaired
 * because choosing a detour here would silently change represented topology.
 */
[[nodiscard]] inline FireFront
advance_front_rk2_batched_clipped_to_nonburnable(
    const FireFront& front,
    amrex::Real time_s,
    amrex::Real dt_s,
    const NormalSpeedBatchFunction& normal_speed_mps,
    const FireFuelRaster& fuel_raster)
{
    if (!std::isfinite(time_s)) {
        throw std::invalid_argument(
            "Fire barrier RK2 time must be finite");
    }
    if (!std::isfinite(dt_s) || dt_s < amrex::Real(0)) {
        throw std::invalid_argument(
            "Fire barrier RK2 timestep must be finite and nonnegative");
    }
    if (!normal_speed_mps) {
        throw std::invalid_argument(
            "Fire barrier RK2 normal-speed callback must be set");
    }

    collective_require_fire_front_edges_avoid_nonburnable(
        front,
        fuel_raster,
        "initial RK2 state");

    if (dt_s == amrex::Real(0)) {
        return front;
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

    std::vector<FireVec2> midpoint_candidates;
    midpoint_candidates.reserve(initial_positions.size());
    for (std::size_t index = 0;
         index < initial_positions.size();
         ++index) {
        midpoint_candidates.push_back(
            initial_positions[index]
            + initial_normals[index]
                * (amrex::Real(0.5)
                   * dt_s
                   * initial_speeds[index]));
    }

    FireFuelBarrierMotionClipResult midpoint_clip =
        collective_clip_fire_fuel_motion_to_nonburnable(
            fuel_raster,
            initial_positions,
            midpoint_candidates);

    FireFront midpoint_front =
        detail::fire_barrier_rebuild_front(
            front,
            midpoint_clip.positions_m);

    collective_require_fire_front_edges_avoid_nonburnable(
        midpoint_front,
        fuel_raster,
        "RK2 midpoint");

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

    std::vector<FireVec2> final_candidates;
    final_candidates.reserve(initial_positions.size());
    for (std::size_t index = 0;
         index < initial_positions.size();
         ++index) {
        final_candidates.push_back(
            initial_positions[index]
            + midpoint_normals[index]
                * (dt_s * midpoint_speeds[index]));
    }

    FireFuelBarrierMotionClipResult final_clip =
        collective_clip_fire_fuel_motion_to_nonburnable(
            fuel_raster,
            initial_positions,
            final_candidates);

    FireFront final_front =
        detail::fire_barrier_rebuild_front(
            front,
            final_clip.positions_m);

    collective_require_fire_front_edges_avoid_nonburnable(
        final_front,
        fuel_raster,
        "RK2 final state");

    return final_front;
}

} // namespace ERFFire

#endif // ERF_FIRE_FUEL_BARRIER_ADVANCE_H_
