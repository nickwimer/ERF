#include <ERF_VectorPerimeterPropagator.H>
#include <ERF_FireGeometry.H>

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ERFFire
{
namespace
{
amrex::Real
checked_speed (
    const NormalSpeedFunction& normal_speed_mps,
    const FireVec2& position_m,
    const FireVec2& outward_normal,
    amrex::Real time_s)
{
    const amrex::Real speed =
        normal_speed_mps(position_m, outward_normal, time_s);

    if (!std::isfinite(speed) || speed < amrex::Real(0.0)) {
        throw std::invalid_argument(
            "Fire normal spread speed must be finite and non-negative");
    }

    return speed;
}

std::vector<amrex::Real>
checked_speeds (
    const NormalSpeedBatchFunction& normal_speed_mps,
    const std::vector<FireVec2>& positions_m,
    const std::vector<FireVec2>& outward_normals,
    amrex::Real time_s)
{
    if (positions_m.size() != outward_normals.size()) {
        throw std::logic_error(
            "Fire batched normal-speed positions/normals size mismatch");
    }

    std::vector<amrex::Real> speeds =
        normal_speed_mps(
            positions_m,
            outward_normals,
            time_s);
    if (speeds.size() != positions_m.size()) {
        throw std::invalid_argument(
            "Fire batched normal spread speed returned the wrong size");
    }

    for (const amrex::Real speed : speeds) {
        if (!std::isfinite(speed) || speed < amrex::Real(0.0)) {
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
    if (detail::signed_polygon_area_m2(vertices_m) <= amrex::Real(0.0)) {
        throw std::runtime_error(
            std::string("Fire perimeter lost counter-clockwise orientation at ")
            + stage);
    }
}

} // namespace

FirePerimeter
advance_perimeter_rk2 (
    const FirePerimeter& perimeter,
    amrex::Real time_s,
    amrex::Real dt_s,
    const NormalSpeedFunction& normal_speed_mps)
{
    if (!std::isfinite(time_s)) {
        throw std::invalid_argument("Fire time must be finite");
    }

    if (!std::isfinite(dt_s) || dt_s < amrex::Real(0.0)) {
        throw std::invalid_argument(
            "Fire timestep must be finite and non-negative");
    }

    if (!normal_speed_mps) {
        throw std::invalid_argument(
            "Fire normal spread speed callback must be set");
    }

    if (dt_s == amrex::Real(0.0)) {
        return perimeter;
    }

    const auto& initial_vertices = perimeter.vertices_m();
    std::vector<FireVec2> midpoint_vertices;
    midpoint_vertices.reserve(initial_vertices.size());

    for (std::size_t i = 0; i < initial_vertices.size(); ++i) {
        const FireVec2 normal = perimeter.outward_normal_unit(i);
        const amrex::Real speed = checked_speed(
            normal_speed_mps, initial_vertices[i], normal, time_s);

        midpoint_vertices.push_back(
            initial_vertices[i]
            + normal * (amrex::Real(0.5) * dt_s * speed));
    }

    require_counter_clockwise(midpoint_vertices, "RK2 midpoint");
    const FirePerimeter midpoint(
        std::move(midpoint_vertices));

    std::vector<FireVec2> final_vertices;
    final_vertices.reserve(initial_vertices.size());

    for (std::size_t i = 0; i < initial_vertices.size(); ++i) {
        const FireVec2 normal = midpoint.outward_normal_unit(i);
        const amrex::Real speed = checked_speed(
            normal_speed_mps,
            midpoint.vertices_m()[i],
            normal,
            time_s + amrex::Real(0.5) * dt_s);

        final_vertices.push_back(
            initial_vertices[i] + normal * (dt_s * speed));
    }

    require_counter_clockwise(final_vertices, "RK2 final state");
    return FirePerimeter(std::move(final_vertices));
}

FirePerimeter
advance_perimeter_rk2_batched (
    const FirePerimeter& perimeter,
    amrex::Real time_s,
    amrex::Real dt_s,
    const NormalSpeedBatchFunction& normal_speed_mps)
{
    if (!std::isfinite(time_s)) {
        throw std::invalid_argument("Fire time must be finite");
    }

    if (!std::isfinite(dt_s) || dt_s < amrex::Real(0.0)) {
        throw std::invalid_argument(
            "Fire timestep must be finite and non-negative");
    }

    if (!normal_speed_mps) {
        throw std::invalid_argument(
            "Fire batched normal spread speed callback must be set");
    }

    if (dt_s == amrex::Real(0.0)) {
        return perimeter;
    }

    const auto& initial_vertices = perimeter.vertices_m();

    std::vector<FireVec2> initial_normals;
    initial_normals.reserve(initial_vertices.size());
    for (std::size_t i = 0; i < initial_vertices.size(); ++i) {
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
    for (std::size_t i = 0; i < initial_vertices.size(); ++i) {
        midpoint_vertices.push_back(
            initial_vertices[i]
            + initial_normals[i]
                * (amrex::Real(0.5) * dt_s * initial_speeds[i]));
    }

    require_counter_clockwise(midpoint_vertices, "RK2 midpoint");
    const FirePerimeter midpoint(
        std::move(midpoint_vertices));

    const auto& midpoint_positions = midpoint.vertices_m();
    std::vector<FireVec2> midpoint_normals;
    midpoint_normals.reserve(midpoint_positions.size());
    for (std::size_t i = 0; i < midpoint_positions.size(); ++i) {
        midpoint_normals.push_back(
            midpoint.outward_normal_unit(i));
    }

    const std::vector<amrex::Real> midpoint_speeds =
        checked_speeds(
            normal_speed_mps,
            midpoint_positions,
            midpoint_normals,
            time_s + amrex::Real(0.5) * dt_s);

    std::vector<FireVec2> final_vertices;
    final_vertices.reserve(initial_vertices.size());
    for (std::size_t i = 0; i < initial_vertices.size(); ++i) {
        final_vertices.push_back(
            initial_vertices[i]
            + midpoint_normals[i] * (dt_s * midpoint_speeds[i]));
    }

    require_counter_clockwise(final_vertices, "RK2 final state");
    return FirePerimeter(std::move(final_vertices));
}

} // namespace ERFFire
