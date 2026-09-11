#include "ERF_FirePerimeterSweep.H"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ERFFire
{

FirePerimeter
interpolate_fire_perimeter_linear_sweep(
    const FirePerimeter& start_perimeter,
    const FirePerimeter& end_perimeter,
    amrex::Real alpha)
{
    if (start_perimeter.size() != end_perimeter.size()) {
        throw std::invalid_argument(
            "Fire linear perimeter sweep requires matching vertex counts");
    }

    if (!std::isfinite(alpha)
        || alpha < amrex::Real(0)
        || alpha > amrex::Real(1)) {
        throw std::invalid_argument(
            "Fire linear perimeter sweep alpha must be finite in [0,1]");
    }

    if (alpha == amrex::Real(0)) {
        return start_perimeter;
    }
    if (alpha == amrex::Real(1)) {
        return end_perimeter;
    }

    std::vector<FireVec2> vertices;
    vertices.reserve(start_perimeter.size());

    const auto& start = start_perimeter.vertices_m();
    const auto& end = end_perimeter.vertices_m();

    for (std::size_t i = 0; i < start.size(); ++i) {
        vertices.push_back(
            start[i] + (end[i] - start[i]) * alpha);
    }

    return FirePerimeter(std::move(vertices));
}

FireFront
interpolate_fire_front_linear_sweep(
    const FireFront& start_front,
    const FireFront& end_front,
    amrex::Real alpha)
{
    const auto& start_components =
        start_front.components();
    const auto& end_components =
        end_front.components();

    if (start_components.size()
        != end_components.size()) {
        throw std::invalid_argument(
            "Fire linear front sweep requires matching component counts");
    }

    for (std::size_t component_index = 0;
         component_index < start_components.size();
         ++component_index) {
        if (start_components[component_index].role
            != end_components[component_index].role) {
            throw std::invalid_argument(
                "Fire linear front sweep requires matching component roles");
        }

        if (start_components[component_index]
                .perimeter.size()
            != end_components[component_index]
                .perimeter.size()) {
            throw std::invalid_argument(
                "Fire linear front sweep requires matching component "
                "vertex counts");
        }
    }

    if (!std::isfinite(alpha)
        || alpha < amrex::Real(0)
        || alpha > amrex::Real(1)) {
        throw std::invalid_argument(
            "Fire linear front sweep alpha must be finite in [0,1]");
    }

    if (alpha == amrex::Real(0)) {
        return start_front;
    }
    if (alpha == amrex::Real(1)) {
        return end_front;
    }

    std::vector<FireFrontComponent> components;
    components.reserve(start_components.size());

    for (std::size_t component_index = 0;
         component_index < start_components.size();
         ++component_index) {
        components.push_back({
            start_components[component_index].role,
            interpolate_fire_perimeter_linear_sweep(
                start_components[component_index].perimeter,
                end_components[component_index].perimeter,
                alpha)
        });
    }

    return FireFront(std::move(components));
}

FireFront
interpolate_fire_front_topology_event_sweep(
    const FireFront& start_front,
    const std::vector<std::vector<FireVec2>>& event_vertices_m,
    amrex::Real alpha)
{
    const auto& start_components =
        start_front.components();

    if (start_components.size()
        != event_vertices_m.size()) {
        throw std::invalid_argument(
            "Fire topology-event front sweep requires matching component counts");
    }

    for (std::size_t component_index = 0;
         component_index < start_components.size();
         ++component_index) {
        if (start_components[component_index]
                .perimeter.size()
            != event_vertices_m[component_index]
                .size()) {
            throw std::invalid_argument(
                "Fire topology-event front sweep requires matching component "
                "vertex counts");
        }

        for (const FireVec2& vertex :
             event_vertices_m[component_index]) {
            if (!std::isfinite(vertex.x)
                || !std::isfinite(vertex.y)) {
                throw std::invalid_argument(
                    "Fire topology-event front vertices must be finite");
            }
        }
    }

    if (!std::isfinite(alpha)
        || alpha < amrex::Real(0)
        || alpha >= amrex::Real(1)) {
        throw std::invalid_argument(
            "Fire topology-event front sweep alpha must be finite in [0,1)");
    }

    if (alpha == amrex::Real(0)) {
        return start_front;
    }

    std::vector<FireFrontComponent> components;
    components.reserve(start_components.size());

    for (std::size_t component_index = 0;
         component_index < start_components.size();
         ++component_index) {
        const auto& start_vertices =
            start_components[component_index]
                .perimeter.vertices_m();
        const auto& event_vertices =
            event_vertices_m[component_index];

        std::vector<FireVec2> sample_vertices;
        sample_vertices.reserve(
            start_vertices.size());

        for (std::size_t vertex_index = 0;
             vertex_index < start_vertices.size();
             ++vertex_index) {
            sample_vertices.push_back(
                start_vertices[vertex_index]
                + alpha
                    * (event_vertices[vertex_index]
                       - start_vertices[vertex_index]));
        }

        components.push_back({
            start_components[component_index].role,
            FirePerimeter(std::move(sample_vertices))
        });
    }

    return FireFront(std::move(components));
}

} // namespace ERFFire
