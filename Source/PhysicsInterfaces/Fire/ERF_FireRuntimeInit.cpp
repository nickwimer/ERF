#include "ERF_FireRuntimeInit.H"

#include <ERF_FireCombustion.H>
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

bool
integer_commensurate(int lhs, int rhs) noexcept
{
    return lhs % rhs == 0 || rhs % lhs == 0;
}

FireCartesianRasterGeometry2D
make_fire_raster_geometry(
    const ERFFireRuntimeOptions& options,
    const amrex::Geometry& geometry)
{
    const auto& domain = geometry.Domain();
    const int atmosphere_nx = domain.length(0);
    const int atmosphere_ny = domain.length(1);

    require(
        atmosphere_nx > 0 && atmosphere_ny > 0,
        "ERF level-0 horizontal cell counts must be positive");

    const bool use_level0_counts =
        options.n_cell_x == 0 && options.n_cell_y == 0;

    require(
        use_level0_counts
            || (options.n_cell_x > 0 && options.n_cell_y > 0),
        "fire.n_cell must contain two positive horizontal cell counts");

    const int fire_nx =
        use_level0_counts ? atmosphere_nx : options.n_cell_x;
    const int fire_ny =
        use_level0_counts ? atmosphere_ny : options.n_cell_y;

    require(
        integer_commensurate(fire_nx, atmosphere_nx)
            && integer_commensurate(fire_ny, atmosphere_ny),
        "fire.n_cell must be integer-commensurate with the level-0 atmospheric grid");

    const auto prob_lo = geometry.ProbLoArray();
    const auto prob_hi = geometry.ProbHiArray();

    const amrex::Real extent_x_m =
        prob_hi[0] - prob_lo[0];
    const amrex::Real extent_y_m =
        prob_hi[1] - prob_lo[1];

    require(
        std::isfinite(extent_x_m)
            && std::isfinite(extent_y_m)
            && extent_x_m > amrex::Real(0.0)
            && extent_y_m > amrex::Real(0.0),
        "ERF level-0 horizontal physical extents must be finite and positive");

    return {
        static_cast<std::size_t>(fire_nx),
        static_cast<std::size_t>(fire_ny),
        prob_lo[0],
        prob_lo[1],
        extent_x_m / static_cast<amrex::Real>(fire_nx),
        extent_y_m / static_cast<amrex::Real>(fire_ny)
    };
}

} // namespace

ERFFireSpreadConfig
make_erf_fire_spread_config(
    const ERFFireRuntimeOptions& options,
    const amrex::Geometry& geometry)
{
    require(options.enabled, "cannot configure disabled fire runtime");
    require(
        options.fuel_model == "FM1",
        "ERF-Fire currently supports only fire.fuel_model = FM1");

    const FireCartesianRasterGeometry2D raster_geometry =
        make_fire_raster_geometry(options, geometry);

    if (options.coupling_mode
        == ERFFireCouplingMode::TwoWay) {
        const auto& domain = geometry.Domain();
        const std::size_t atmosphere_nx =
            static_cast<std::size_t>(domain.length(0));
        const std::size_t atmosphere_ny =
            static_cast<std::size_t>(domain.length(1));

        const bool fire_finer_or_equal =
            raster_geometry.nx >= atmosphere_nx
            && raster_geometry.ny >= atmosphere_ny;

        const bool fire_coarser_or_equal =
            raster_geometry.nx <= atmosphere_nx
            && raster_geometry.ny <= atmosphere_ny;

        require(
            fire_finer_or_equal || fire_coarser_or_equal,
            "two-way fire.n_cell cannot mix finer and coarser axes relative to level 0");
    }

    return {
        make_fm1_fuel_parameters(),
        options.dead_fuel_moisture_fraction,
        make_fm1_combustion_parameters(
            options.dead_fuel_moisture_fraction),
        FireCombustionRasterOptions{
            static_cast<std::size_t>(
                options.combustion_temporal_substeps)},
        FirePerimeterRemeshOptions{
            options.remesh_min_edge_length_m,
            options.remesh_max_edge_length_m,
            options.remesh_max_chord_error_m},
        raster_geometry,
        options.arrival_time_tolerance_s};
}

std::unique_ptr<ERFFireSpreadRuntime>
make_erf_fire_spread_runtime(
    const ERFFireRuntimeOptions& options,
    const amrex::Geometry& geometry,
    amrex::Real initial_time_s)
{
    ERFFireSpreadConfig config =
        make_erf_fire_spread_config(options, geometry);

    return std::make_unique<ERFFireSpreadRuntime>(
        make_circular_ignition(options, geometry),
        initial_time_s,
        std::move(config));
}

} // namespace ERFFire
