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
    const FirePerimeter midpoint(std::move(midpoint_vertices));

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

} // namespace ERFFire
