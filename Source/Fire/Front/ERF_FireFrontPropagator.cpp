#include <ERF_FireFrontPropagator.H>

#include <ERF_FireGeometry.H>
#include <ERF_FireFrontTopology.H>

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ERFFire
{
namespace
{

std::vector<amrex::Real>
checked_speeds (
    const NormalSpeedBatchFunction& normal_speed_mps,
    const std::vector<FireVec2>& positions_m,
    const std::vector<FireVec2>& normals,
    amrex::Real time_s)
{
    if (positions_m.size() != normals.size()) {
        throw std::logic_error(
            "Fire batched normal-speed positions/normals size mismatch");
    }

    std::vector<amrex::Real> speeds =
        normal_speed_mps(
            positions_m,
            normals,
            time_s);

    if (speeds.size() != positions_m.size()) {
        throw std::invalid_argument(
            "Fire batched normal spread speed returned the wrong size");
    }

    for (const amrex::Real speed : speeds) {
        if (!std::isfinite(speed)
            || speed < amrex::Real(0.0)) {
            throw std::invalid_argument(
                "Fire normal spread speed must be finite and non-negative");
        }
    }

    return speeds;
}

void
require_counter_clockwise (
    const std::vector<FireVec2>& vertices_m,
    const char* stage)
{
    if (detail::signed_polygon_area_m2(vertices_m)
        <= amrex::Real(0.0)) {
        throw std::runtime_error(
            std::string(
                "Fire perimeter lost counter-clockwise orientation at ")
            + stage);
    }
}

FireFront
single_outer_front (FirePerimeter perimeter)
{
    std::vector<FireFrontComponent> components;
    components.reserve(1);
    components.push_back({
        FireFrontRole::Outer,
        std::move(perimeter)
    });

    return FireFront(std::move(components));
}

std::vector<FireVec2>
interpolate_vertices (
    const std::vector<FireVec2>& start_vertices_m,
    const std::vector<FireVec2>& end_vertices_m,
    amrex::Real fraction)
{
    if (start_vertices_m.size()
        != end_vertices_m.size()) {
        throw std::logic_error(
            "Fire topology-event interpolation size mismatch");
    }

    std::vector<FireVec2> vertices_m;
    vertices_m.reserve(start_vertices_m.size());

    for (std::size_t i = 0;
         i < start_vertices_m.size();
         ++i) {
        vertices_m.push_back(
            start_vertices_m[i]
            + fraction
                * (end_vertices_m[i]
                   - start_vertices_m[i]));
    }

    return vertices_m;
}

FireFrontAdvanceResult
make_event_result (
    const std::vector<FireVec2>& initial_vertices_m,
    const std::vector<FireVec2>& candidate_vertices_m,
    FirePerimeterCollision collision,
    amrex::Real requested_dt_s,
    amrex::Real full_step_fraction)
{
    std::vector<FireVec2> event_vertices_m =
        interpolate_vertices(
            initial_vertices_m,
            candidate_vertices_m,
            collision.motion_fraction);

    collision.motion_fraction =
        full_step_fraction
        * collision.motion_fraction;

    FireFront front =
        split_perimeter_at_pinch(
            event_vertices_m,
            collision.pinch);

    return {
        std::move(front),
        requested_dt_s
            * collision.motion_fraction,
        std::move(event_vertices_m),
        std::move(collision)
    };
}

} // namespace

FireFrontAdvanceResult
advance_perimeter_rk2_batched_until_topology_event(
    const FirePerimeter& perimeter,
    amrex::Real time_s,
    amrex::Real dt_s,
    const NormalSpeedBatchFunction& normal_speed_mps)
{
    if (!std::isfinite(time_s)) {
        throw std::invalid_argument(
            "Fire time must be finite");
    }

    if (!std::isfinite(dt_s)
        || dt_s < amrex::Real(0.0)) {
        throw std::invalid_argument(
            "Fire timestep must be finite and non-negative");
    }

    if (!normal_speed_mps) {
        throw std::invalid_argument(
            "Fire batched normal spread speed callback must be set");
    }

    const auto& initial_vertices =
        perimeter.vertices_m();

    if (dt_s == amrex::Real(0.0)) {
        return {
            single_outer_front(perimeter),
            amrex::Real(0.0),
            initial_vertices,
            std::nullopt
        };
    }

    std::vector<FireVec2> initial_normals;
    initial_normals.reserve(initial_vertices.size());

    for (std::size_t i = 0;
         i < initial_vertices.size();
         ++i) {
        initial_normals.push_back(
            perimeter.outward_normal_unit(i));
    }

    const std::vector<amrex::Real> initial_speeds =
        checked_speeds(
            normal_speed_mps,
            initial_vertices,
            initial_normals,
            time_s);

    std::vector<FireVec2> midpoint_vertices;
    midpoint_vertices.reserve(initial_vertices.size());

    for (std::size_t i = 0;
         i < initial_vertices.size();
         ++i) {
        midpoint_vertices.push_back(
            initial_vertices[i]
            + initial_normals[i]
                * (amrex::Real(0.5)
                   * dt_s
                   * initial_speeds[i]));
    }

    const auto midpoint_collision =
        locate_first_perimeter_collision(
            perimeter,
            midpoint_vertices);

    if (midpoint_collision.has_value()) {
        return make_event_result(
            initial_vertices,
            midpoint_vertices,
            *midpoint_collision,
            dt_s,
            amrex::Real(0.5));
    }

    require_counter_clockwise(
        midpoint_vertices,
        "RK2 midpoint");

    const FirePerimeter midpoint(
        midpoint_vertices);

    const auto& midpoint_positions =
        midpoint.vertices_m();

    std::vector<FireVec2> midpoint_normals;
    midpoint_normals.reserve(
        midpoint_positions.size());

    for (std::size_t i = 0;
         i < midpoint_positions.size();
         ++i) {
        midpoint_normals.push_back(
            midpoint.outward_normal_unit(i));
    }

    const std::vector<amrex::Real> midpoint_speeds =
        checked_speeds(
            normal_speed_mps,
            midpoint_positions,
            midpoint_normals,
            time_s
                + amrex::Real(0.5) * dt_s);

    std::vector<FireVec2> final_vertices;
    final_vertices.reserve(initial_vertices.size());

    for (std::size_t i = 0;
         i < initial_vertices.size();
         ++i) {
        final_vertices.push_back(
            initial_vertices[i]
            + midpoint_normals[i]
                * (dt_s * midpoint_speeds[i]));
    }

    const auto final_collision =
        locate_first_perimeter_collision(
            perimeter,
            final_vertices);

    if (final_collision.has_value()) {
        return make_event_result(
            initial_vertices,
            final_vertices,
            *final_collision,
            dt_s,
            amrex::Real(1.0));
    }

    require_counter_clockwise(
        final_vertices,
        "RK2 final state");

    FirePerimeter final_perimeter(
        final_vertices);

    return {
        single_outer_front(
            std::move(final_perimeter)),
        dt_s,
        std::move(final_vertices),
        std::nullopt
    };
}

FireFront
advance_front_rk2_batched(
    const FireFront& front,
    amrex::Real time_s,
    amrex::Real dt_s,
    const NormalSpeedBatchFunction& normal_speed_mps)
{
    if (!std::isfinite(time_s)) {
        throw std::invalid_argument(
            "Fire time must be finite");
    }

    if (!std::isfinite(dt_s)
        || dt_s < amrex::Real(0.0)) {
        throw std::invalid_argument(
            "Fire timestep must be finite and non-negative");
    }

    if (!normal_speed_mps) {
        throw std::invalid_argument(
            "Fire batched normal spread speed callback must be set");
    }

    if (dt_s == amrex::Real(0.0)) {
        return front;
    }

    const auto& initial_components =
        front.components();

    std::size_t total_vertices = 0;
    for (const FireFrontComponent& component :
         initial_components) {
        total_vertices +=
            component.perimeter.size();
    }

    std::vector<FireVec2> initial_positions;
    std::vector<FireVec2> initial_normals;
    initial_positions.reserve(total_vertices);
    initial_normals.reserve(total_vertices);

    for (std::size_t component_index = 0;
         component_index < initial_components.size();
         ++component_index) {
        const FirePerimeter& perimeter =
            initial_components[component_index]
                .perimeter;
        const auto& vertices =
            perimeter.vertices_m();

        for (std::size_t vertex_index = 0;
             vertex_index < vertices.size();
             ++vertex_index) {
            initial_positions.push_back(
                vertices[vertex_index]);
            initial_normals.push_back(
                front.spread_normal_unit(
                    component_index,
                    vertex_index));
        }
    }

    const std::vector<amrex::Real>
        initial_speeds =
            checked_speeds(
                normal_speed_mps,
                initial_positions,
                initial_normals,
                time_s);

    std::vector<FireFrontComponent>
        midpoint_components;
    midpoint_components.reserve(
        initial_components.size());

    std::size_t flat_index = 0;
    for (const FireFrontComponent& component :
         initial_components) {
        std::vector<FireVec2>
            midpoint_vertices;
        midpoint_vertices.reserve(
            component.perimeter.size());

        for (std::size_t vertex_index = 0;
             vertex_index
                 < component.perimeter.size();
             ++vertex_index, ++flat_index) {
            midpoint_vertices.push_back(
                initial_positions[flat_index]
                + initial_normals[flat_index]
                    * (amrex::Real(0.5)
                       * dt_s
                       * initial_speeds[
                           flat_index]));
        }

        require_counter_clockwise(
            midpoint_vertices,
            "FireFront RK2 midpoint");

        midpoint_components.push_back({
            component.role,
            FirePerimeter(
                std::move(midpoint_vertices))
        });
    }

    FireFront midpoint_front(
        std::move(midpoint_components));

    std::vector<FireVec2> midpoint_positions;
    std::vector<FireVec2> midpoint_normals;
    midpoint_positions.reserve(total_vertices);
    midpoint_normals.reserve(total_vertices);

    const auto& midpoint_front_components =
        midpoint_front.components();

    for (std::size_t component_index = 0;
         component_index
             < midpoint_front_components.size();
         ++component_index) {
        const FirePerimeter& perimeter =
            midpoint_front_components[
                component_index]
                .perimeter;
        const auto& vertices =
            perimeter.vertices_m();

        for (std::size_t vertex_index = 0;
             vertex_index < vertices.size();
             ++vertex_index) {
            midpoint_positions.push_back(
                vertices[vertex_index]);
            midpoint_normals.push_back(
                midpoint_front.spread_normal_unit(
                    component_index,
                    vertex_index));
        }
    }

    const std::vector<amrex::Real>
        midpoint_speeds =
            checked_speeds(
                normal_speed_mps,
                midpoint_positions,
                midpoint_normals,
                time_s
                    + amrex::Real(0.5)
                        * dt_s);

    std::vector<FireFrontComponent>
        final_components;
    final_components.reserve(
        initial_components.size());

    flat_index = 0;
    for (const FireFrontComponent& component :
         initial_components) {
        std::vector<FireVec2> final_vertices;
        final_vertices.reserve(
            component.perimeter.size());

        for (std::size_t vertex_index = 0;
             vertex_index
                 < component.perimeter.size();
             ++vertex_index, ++flat_index) {
            final_vertices.push_back(
                initial_positions[flat_index]
                + midpoint_normals[flat_index]
                    * (dt_s
                       * midpoint_speeds[
                           flat_index]));
        }

        require_counter_clockwise(
            final_vertices,
            "FireFront RK2 final state");

        final_components.push_back({
            component.role,
            FirePerimeter(
                std::move(final_vertices))
        });
    }

    return FireFront(
        std::move(final_components));
}

FireFrontTopologyAdvanceResult
advance_front_rk2_batched_until_topology_event(
    const FireFront& front,
    amrex::Real time_s,
    amrex::Real dt_s,
    const NormalSpeedBatchFunction& normal_speed_mps)
{
    if (!std::isfinite(time_s)) {
        throw std::invalid_argument(
            "Fire time must be finite");
    }

    if (!std::isfinite(dt_s)
        || dt_s < amrex::Real(0.0)) {
        throw std::invalid_argument(
            "Fire timestep must be finite and non-negative");
    }

    if (!normal_speed_mps) {
        throw std::invalid_argument(
            "Fire batched normal spread speed callback must be set");
    }

    const auto& initial_components =
        front.components();

    std::vector<std::vector<FireVec2>>
        initial_vertices_m;
    initial_vertices_m.reserve(
        initial_components.size());

    std::size_t total_vertices = 0;
    for (const FireFrontComponent& component :
         initial_components) {
        initial_vertices_m.push_back(
            component.perimeter.vertices_m());
        total_vertices +=
            component.perimeter.size();
    }

    if (dt_s == amrex::Real(0.0)) {
        return {
            front,
            amrex::Real(0.0),
            std::move(initial_vertices_m),
            std::nullopt
        };
    }

    std::vector<FireVec2> initial_positions;
    std::vector<FireVec2> initial_normals;
    initial_positions.reserve(total_vertices);
    initial_normals.reserve(total_vertices);

    for (std::size_t component_index = 0;
         component_index < initial_components.size();
         ++component_index) {
        const FirePerimeter& perimeter =
            initial_components[component_index]
                .perimeter;

        for (std::size_t vertex_index = 0;
             vertex_index < perimeter.size();
             ++vertex_index) {
            initial_positions.push_back(
                perimeter.vertices_m()[
                    vertex_index]);
            initial_normals.push_back(
                front.spread_normal_unit(
                    component_index,
                    vertex_index));
        }
    }

    const std::vector<amrex::Real>
        initial_speeds =
            checked_speeds(
                normal_speed_mps,
                initial_positions,
                initial_normals,
                time_s);

    std::vector<std::vector<FireVec2>>
        midpoint_vertices_m;
    midpoint_vertices_m.reserve(
        initial_components.size());

    std::size_t flat_index = 0;
    for (const FireFrontComponent& component :
         initial_components) {
        std::vector<FireVec2> vertices;
        vertices.reserve(
            component.perimeter.size());

        for (std::size_t vertex_index = 0;
             vertex_index
                 < component.perimeter.size();
             ++vertex_index, ++flat_index) {
            vertices.push_back(
                initial_positions[flat_index]
                + initial_normals[flat_index]
                    * (amrex::Real(0.5)
                       * dt_s
                       * initial_speeds[
                           flat_index]));
        }

        midpoint_vertices_m.push_back(
            std::move(vertices));
    }

    std::optional<FirePerimeterCollision>
        midpoint_collision;
    std::size_t midpoint_component_index = 0;

    for (std::size_t component_index = 0;
         component_index < initial_components.size();
         ++component_index) {
        const auto collision =
            locate_first_perimeter_collision(
                initial_components[
                    component_index]
                    .perimeter,
                midpoint_vertices_m[
                    component_index]);

        if (collision.has_value()
            && (!midpoint_collision.has_value()
                || collision->motion_fraction
                    < midpoint_collision
                        ->motion_fraction)) {
            midpoint_collision =
                *collision;
            midpoint_component_index =
                component_index;
        }
    }

    if (midpoint_collision.has_value()) {
        const amrex::Real local_fraction =
            midpoint_collision->motion_fraction;
        const amrex::Real full_step_fraction =
            amrex::Real(0.5)
            * local_fraction;

        std::vector<std::vector<FireVec2>>
            event_vertices_m;
        event_vertices_m.reserve(
            initial_components.size());

        for (std::size_t component_index = 0;
             component_index
                 < initial_components.size();
             ++component_index) {
            event_vertices_m.push_back(
                interpolate_vertices(
                    initial_vertices_m[
                        component_index],
                    midpoint_vertices_m[
                        component_index],
                    local_fraction));
        }

        FirePerimeterCollision collision =
            *midpoint_collision;
        collision.motion_fraction =
            full_step_fraction;

        return {
            std::nullopt,
            dt_s * full_step_fraction,
            std::move(event_vertices_m),
            FireFrontComponentTopologyEvent{
                midpoint_component_index,
                initial_components[
                    midpoint_component_index]
                    .role,
                std::move(collision)
            }
        };
    }

    std::vector<FireFrontComponent>
        midpoint_components;
    midpoint_components.reserve(
        initial_components.size());

    for (std::size_t component_index = 0;
         component_index < initial_components.size();
         ++component_index) {
        require_counter_clockwise(
            midpoint_vertices_m[
                component_index],
            "FireFront topology-aware RK2 midpoint");

        midpoint_components.push_back({
            initial_components[
                component_index]
                .role,
            FirePerimeter(
                midpoint_vertices_m[
                    component_index])
        });
    }

    FireFront midpoint_front(
        std::move(midpoint_components));

    std::vector<FireVec2> midpoint_positions;
    std::vector<FireVec2> midpoint_normals;
    midpoint_positions.reserve(total_vertices);
    midpoint_normals.reserve(total_vertices);

    const auto& midpoint_components_ref =
        midpoint_front.components();

    for (std::size_t component_index = 0;
         component_index
             < midpoint_components_ref.size();
         ++component_index) {
        const FirePerimeter& perimeter =
            midpoint_components_ref[
                component_index]
                .perimeter;

        for (std::size_t vertex_index = 0;
             vertex_index < perimeter.size();
             ++vertex_index) {
            midpoint_positions.push_back(
                perimeter.vertices_m()[
                    vertex_index]);
            midpoint_normals.push_back(
                midpoint_front.spread_normal_unit(
                    component_index,
                    vertex_index));
        }
    }

    const std::vector<amrex::Real>
        midpoint_speeds =
            checked_speeds(
                normal_speed_mps,
                midpoint_positions,
                midpoint_normals,
                time_s
                    + amrex::Real(0.5)
                        * dt_s);

    std::vector<std::vector<FireVec2>>
        final_vertices_m;
    final_vertices_m.reserve(
        initial_components.size());

    flat_index = 0;
    for (const FireFrontComponent& component :
         initial_components) {
        std::vector<FireVec2> vertices;
        vertices.reserve(
            component.perimeter.size());

        for (std::size_t vertex_index = 0;
             vertex_index
                 < component.perimeter.size();
             ++vertex_index, ++flat_index) {
            vertices.push_back(
                initial_positions[flat_index]
                + midpoint_normals[flat_index]
                    * (dt_s
                       * midpoint_speeds[
                           flat_index]));
        }

        final_vertices_m.push_back(
            std::move(vertices));
    }

    std::optional<FirePerimeterCollision>
        final_collision;
    std::size_t final_component_index = 0;

    for (std::size_t component_index = 0;
         component_index < initial_components.size();
         ++component_index) {
        const auto collision =
            locate_first_perimeter_collision(
                initial_components[
                    component_index]
                    .perimeter,
                final_vertices_m[
                    component_index]);

        if (collision.has_value()
            && (!final_collision.has_value()
                || collision->motion_fraction
                    < final_collision
                        ->motion_fraction)) {
            final_collision =
                *collision;
            final_component_index =
                component_index;
        }
    }

    if (final_collision.has_value()) {
        const amrex::Real full_step_fraction =
            final_collision->motion_fraction;

        std::vector<std::vector<FireVec2>>
            event_vertices_m;
        event_vertices_m.reserve(
            initial_components.size());

        for (std::size_t component_index = 0;
             component_index
                 < initial_components.size();
             ++component_index) {
            event_vertices_m.push_back(
                interpolate_vertices(
                    initial_vertices_m[
                        component_index],
                    final_vertices_m[
                        component_index],
                    full_step_fraction));
        }

        return {
            std::nullopt,
            dt_s * full_step_fraction,
            std::move(event_vertices_m),
            FireFrontComponentTopologyEvent{
                final_component_index,
                initial_components[
                    final_component_index]
                    .role,
                *final_collision
            }
        };
    }

    std::vector<FireFrontComponent>
        final_components;
    final_components.reserve(
        initial_components.size());

    for (std::size_t component_index = 0;
         component_index < initial_components.size();
         ++component_index) {
        require_counter_clockwise(
            final_vertices_m[
                component_index],
            "FireFront topology-aware RK2 final state");

        final_components.push_back({
            initial_components[
                component_index]
                .role,
            FirePerimeter(
                final_vertices_m[
                    component_index])
        });
    }

    return {
        FireFront(
            std::move(final_components)),
        dt_s,
        std::move(final_vertices_m),
        std::nullopt
    };
}

} // namespace ERFFire
