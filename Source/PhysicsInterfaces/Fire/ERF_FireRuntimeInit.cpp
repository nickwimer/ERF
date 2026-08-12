#include "ERF_FireRuntimeInit.H"

#include <ERF_FirePerimeter.H>
#include <ERF_FireSpreadRuntime.H>
#include <ERF_RothermelFuel.H>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ERFFire
{
namespace
{

constexpr amrex::Real pi =
    amrex::Real(3.141592653589793238462643383279502884L);

void
require(bool condition, const char* message)
{
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

FirePerimeter
make_circular_ignition(
    const ERFFireRuntimeOptions& options,
    const amrex::Geometry& geometry)
{
    const auto prob_lo = geometry.ProbLoArray();
    const auto prob_hi = geometry.ProbHiArray();

    require(
        std::isfinite(options.ignition_center_x_m) &&
            std::isfinite(options.ignition_center_y_m),
        "fire ignition center must be finite");
    require(
        std::isfinite(options.ignition_radius_m) &&
            options.ignition_radius_m > amrex::Real(0.0),
        "fire ignition radius must be finite and positive");
    require(
        options.ignition_vertex_count >= 8,
        "fire circular ignition requires at least 8 vertices");

    if (options.ignition_center_x_m - options.ignition_radius_m
            < prob_lo[0] ||
        options.ignition_center_x_m + options.ignition_radius_m
            > prob_hi[0] ||
        options.ignition_center_y_m - options.ignition_radius_m
            < prob_lo[1] ||
        options.ignition_center_y_m + options.ignition_radius_m
            > prob_hi[1]) {
        throw std::out_of_range(
            "fire initial ignition perimeter lies outside the level-0 horizontal domain");
    }

    std::vector<FireVec2> vertices;
    vertices.reserve(
        static_cast<std::size_t>(options.ignition_vertex_count));

    for (int i = 0; i < options.ignition_vertex_count; ++i) {
        const amrex::Real angle =
            amrex::Real(2.0) * pi
            * static_cast<amrex::Real>(i)
            / static_cast<amrex::Real>(
                options.ignition_vertex_count);

        vertices.push_back({
            options.ignition_center_x_m
                + options.ignition_radius_m * std::cos(angle),
            options.ignition_center_y_m
                + options.ignition_radius_m * std::sin(angle)});
    }

    return FirePerimeter(std::move(vertices));
}

FireCartesianRasterGeometry2D
make_level0_raster_geometry(const amrex::Geometry& geometry)
{
    const auto& domain = geometry.Domain();
    const auto prob_lo = geometry.ProbLoArray();
    const auto cell_size = geometry.CellSizeArray();

    return {
        static_cast<std::size_t>(domain.length(0)),
        static_cast<std::size_t>(domain.length(1)),
        prob_lo[0],
        prob_lo[1],
        cell_size[0],
        cell_size[1]
    };
}

} // namespace

std::unique_ptr<ERFFireSpreadRuntime>
make_erf_fire_spread_runtime(
    const ERFFireRuntimeOptions& options,
    const amrex::Geometry& geometry,
    amrex::Real initial_time_s)
{
    require(options.enabled, "cannot initialize disabled fire runtime");
    require(
        options.fuel_model == "FM1",
        "M7e supports only fire.fuel_model = FM1");

    ERFFireSpreadConfig config{
        make_fm1_fuel_parameters(),
        options.dead_fuel_moisture_fraction,
        FirePerimeterRemeshOptions{
            options.remesh_min_edge_length_m,
            options.remesh_max_edge_length_m,
            options.remesh_max_chord_error_m},
        make_level0_raster_geometry(geometry),
        options.arrival_time_tolerance_s};

    return std::make_unique<ERFFireSpreadRuntime>(
        make_circular_ignition(options, geometry),
        initial_time_s,
        std::move(config));
}

} // namespace ERFFire
