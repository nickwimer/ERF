#ifndef ERF_FIRE_FUEL_BARRIER_ADVANCE_H_
#define ERF_FIRE_FUEL_BARRIER_ADVANCE_H_

#include <ERF_FireFuelBarrier.H>
#include <ERF_FireFuelBarrierFront.H>
#include <ERF_FireFrontPropagator.H>

#include <algorithm>
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
 *
 * An RK trial point may lie outside the represented Fire rectangle even when a
 * NonBurnable barrier is encountered first. The underlying raster traversal
 * intentionally accepts only closed-domain endpoints, so such trial segments
 * are truncated to their first domain-face intersection for the material
 * query. If no barrier is found before that face, the original out-of-domain
 * motion is rejected rather than silently turning the Fire-domain boundary
 * into an implicit barrier.
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

    const auto& geometry = raster.geometry();
    (void)detail::validate_fire_cartesian_raster_geometry(
        geometry);

    const FireCartesianCell2D upper_cell =
        detail::fire_cartesian_raster_cell_bounds(
            geometry,
            geometry.nx - 1,
            geometry.ny - 1);
    const amrex::Real xlo_m = geometry.xlo_m;
    const amrex::Real ylo_m = geometry.ylo_m;
    const amrex::Real xhi_m = upper_cell.xhi_m;
    const amrex::Real yhi_m = upper_cell.yhi_m;

    std::vector<FireVec2> query_candidates =
        candidates_m;
    std::vector<unsigned char> domain_truncated(
        candidates_m.size(),
        static_cast<unsigned char>(0));

    for (std::size_t index = 0;
         index < candidates_m.size();
         ++index) {
        const FireVec2& start = starts_m[index];
        const FireVec2& candidate = candidates_m[index];

        // The committed state must already be represented by the Fire raster.
        (void)detail::fire_cartesian_raster_locate_cell(
            geometry,
            start.x,
            start.y);

        if (!std::isfinite(candidate.x)
            || !std::isfinite(candidate.y)) {
            throw std::invalid_argument(
                "Fire fuel barrier motion candidate must be finite");
        }

        const bool candidate_inside =
            candidate.x >= xlo_m
            && candidate.x <= xhi_m
            && candidate.y >= ylo_m
            && candidate.y <= yhi_m;
        if (candidate_inside) {
            continue;
        }

        const amrex::Real dx =
            candidate.x - start.x;
        const amrex::Real dy =
            candidate.y - start.y;
        if (!std::isfinite(dx) || !std::isfinite(dy)) {
            throw std::overflow_error(
                "Fire fuel barrier motion displacement is not finite");
        }

        amrex::Real domain_fraction =
            amrex::Real(1);

        if (candidate.x < xlo_m) {
            if (!(dx < amrex::Real(0))) {
                throw std::logic_error(
                    "Fire fuel barrier motion cannot reach the lower x domain face");
            }
            domain_fraction =
                std::min(
                    domain_fraction,
                    (xlo_m - start.x) / dx);
        } else if (candidate.x > xhi_m) {
            if (!(dx > amrex::Real(0))) {
                throw std::logic_error(
                    "Fire fuel barrier motion cannot reach the upper x domain face");
            }
            domain_fraction =
                std::min(
                    domain_fraction,
                    (xhi_m - start.x) / dx);
        }

        if (candidate.y < ylo_m) {
            if (!(dy < amrex::Real(0))) {
                throw std::logic_error(
                    "Fire fuel barrier motion cannot reach the lower y domain face");
            }
            domain_fraction =
                std::min(
                    domain_fraction,
                    (ylo_m - start.y) / dy);
        } else if (candidate.y > yhi_m) {
            if (!(dy > amrex::Real(0))) {
                throw std::logic_error(
                    "Fire fuel barrier motion cannot reach the upper y domain face");
            }
            domain_fraction =
                std::min(
                    domain_fraction,
                    (yhi_m - start.y) / dy);
        }

        if (!std::isfinite(domain_fraction)
            || domain_fraction < amrex::Real(0)
            || domain_fraction > amrex::Real(1)) {
            throw std::logic_error(
                "Fire fuel barrier motion produced an invalid domain intersection");
        }

        const amrex::Real clipped_fraction =
            std::clamp(
                domain_fraction,
                amrex::Real(0),
                amrex::Real(1));
        query_candidates[index] = {
            std::clamp(
                start.x + clipped_fraction * dx,
                xlo_m,
                xhi_m),
            std::clamp(
                start.y + clipped_fraction * dy,
                ylo_m,
                yhi_m)
        };
        domain_truncated[index] =
            static_cast<unsigned char>(1);
    }

    FireFuelBarrierMotionClipResult result;
    result.positions_m = candidates_m;
    result.contacts =
        collective_first_nonburnable_contacts(
            raster,
            starts_m,
            query_candidates);

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
            continue;
        }

        if (domain_truncated[index] != 0) {
            throw std::out_of_range(
                "Fire fuel barrier motion leaves the raster before reaching NonBurnable material");
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
 * A contact reached by the midpoint trial is persistent for the remainder of
 * the RK2 step: without an explicit sliding law the final vertex is pinned at
 * that first contact instead of allowing a zero midpoint speed to algebraically
 * retreat the explicit-midpoint result back into the burnable region.
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

    if (midpoint_clip.contacts.size()
        != final_clip.positions_m.size()) {
        throw std::logic_error(
            "Fire barrier RK2 midpoint contact batch lost vertex correspondence");
    }
    for (std::size_t index = 0;
         index < midpoint_clip.contacts.size();
         ++index) {
        if (!midpoint_clip.contacts[index].has_value()) {
            continue;
        }

        // No sliding/rerouting law is represented. Once the first half-stage
        // reaches an impermeable barrier, preserve that physical contact for
        // the rest of this RK2 step even if the stage-two material sample on
        // the exact raster face reports zero propagation speed.
        final_clip.positions_m[index] =
            midpoint_clip.contacts[index]->position_m;
        final_clip.contacts[index] =
            midpoint_clip.contacts[index];
    }

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
