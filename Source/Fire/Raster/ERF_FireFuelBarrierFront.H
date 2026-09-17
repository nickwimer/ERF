#ifndef ERF_FIRE_FUEL_BARRIER_FRONT_H_
#define ERF_FIRE_FUEL_BARRIER_FRONT_H_

#include <ERF_FireFuelBarrier.H>
#include <ERF_FireFront.H>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace ERFFire
{

/** One FireFront edge that enters the interior of NonBurnable material. */
struct FireFuelBarrierFrontEdgeContact
{
    std::size_t component_index{};
    std::size_t edge_index{};
    FireFrontRole role{FireFrontRole::Outer};
    FireFuelBarrierContact contact{};
};

namespace detail
{

struct FireFuelBarrierFrontEdgeRef
{
    std::size_t component_index{};
    std::size_t edge_index{};
    FireFrontRole role{FireFrontRole::Outer};
};

} // namespace detail

/**
 * Collectively find every front edge that enters NonBurnable interior.
 *
 * Edges are flattened deterministically in component order, then perimeter
 * edge order. Each closed perimeter contributes one segment from vertex i to
 * vertex (i+1)%n. All edge segments are evaluated through one ordered barrier
 * contact batch, so this adds one sparse material collective regardless of
 * front vertex count.
 *
 * The same union-interior semantics as the motion-contact primitive apply:
 * an edge may lie tangentially on a one-sided burnable/NonBurnable face, while
 * an edge through cell interior or along a seam with NonBurnable material on
 * both sides is reported as a barrier penetration.
 */
[[nodiscard]] inline std::vector<FireFuelBarrierFrontEdgeContact>
collective_nonburnable_front_edge_contacts(
    const FireFront& front,
    const FireFuelRaster& raster)
{
    std::vector<FireVec2> starts_m;
    std::vector<FireVec2> ends_m;
    std::vector<detail::FireFuelBarrierFrontEdgeRef> edge_refs;

    std::size_t edge_count = 0;
    for (const FireFrontComponent& component : front.components()) {
        edge_count += component.perimeter.size();
    }

    starts_m.reserve(edge_count);
    ends_m.reserve(edge_count);
    edge_refs.reserve(edge_count);

    const auto& components = front.components();
    for (std::size_t component_index = 0;
         component_index < components.size();
         ++component_index) {
        const FireFrontComponent& component =
            components[component_index];
        const auto& vertices =
            component.perimeter.vertices_m();

        if (vertices.empty()) {
            throw std::logic_error(
                "Fire barrier edge validation received an empty front component");
        }

        for (std::size_t edge_index = 0;
             edge_index < vertices.size();
             ++edge_index) {
            starts_m.push_back(vertices[edge_index]);
            ends_m.push_back(
                vertices[(edge_index + 1U) % vertices.size()]);
            edge_refs.push_back({
                component_index,
                edge_index,
                component.role});
        }
    }

    if (starts_m.empty()) {
        throw std::logic_error(
            "Fire barrier edge validation received a front with no edges");
    }

    const auto contacts =
        collective_first_nonburnable_contacts(
            raster,
            starts_m,
            ends_m);

    if (contacts.size() != edge_refs.size()) {
        throw std::logic_error(
            "Fire barrier edge contact batch returned the wrong size");
    }

    std::vector<FireFuelBarrierFrontEdgeContact> result;
    for (std::size_t flat_edge = 0;
         flat_edge < contacts.size();
         ++flat_edge) {
        if (!contacts[flat_edge].has_value()) {
            continue;
        }

        const auto& edge = edge_refs[flat_edge];
        result.push_back({
            edge.component_index,
            edge.edge_index,
            edge.role,
            *contacts[flat_edge]});
    }

    return result;
}

/**
 * Require every front edge to remain outside NonBurnable interior.
 *
 * This is intentionally a hard invariant rather than a repair operation.
 * Re-routing an edge around a barrier changes front topology and requires a
 * dedicated contact/sliding resolver; silently inserting an arbitrary path
 * would make the represented burned region mesh-dependent.
 */
inline void
collective_require_fire_front_edges_avoid_nonburnable(
    const FireFront& front,
    const FireFuelRaster& raster,
    const char* stage)
{
    if (stage == nullptr) {
        throw std::invalid_argument(
            "Fire barrier edge-validation stage must not be null");
    }

    const auto contacts =
        collective_nonburnable_front_edge_contacts(
            front,
            raster);

    if (contacts.empty()) {
        return;
    }

    const auto& first = contacts.front();
    throw std::runtime_error(
        std::string("Fire front edge enters NonBurnable material at ")
        + stage
        + " (component "
        + std::to_string(first.component_index)
        + ", edge "
        + std::to_string(first.edge_index)
        + ")");
}

} // namespace ERFFire

#endif // ERF_FIRE_FUEL_BARRIER_FRONT_H_
