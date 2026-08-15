#include <ERF_FireBurnedFractionRaster.H>
#include <ERF_FireFirstArrivalRaster.H>
#include <ERF_FirePerimeterRemesher.H>
#include <ERF_RichardsDirectionalSpread.H>
#include <ERF_RothermelFuel.H>
#include <ERF_VectorPerimeterPropagator.H>

#include "ERF_FireTestUtils.H"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace
{

using ERFFire::FireBurnedFractionRaster;
using ERFFire::FireCartesianRasterGeometry2D;
using ERFFire::FireFirstArrivalRaster;
using ERFFire::FirePerimeter;
using ERFFire::FirePerimeterRemeshOptions;
using ERFFire::FireVec2;
using ERFFire::RichardsDirectionalSpread;
using ERFFire::RothermelInputs;

constexpr amrex::Real pi =
    amrex::Real(3.141592653589793238462643383279502884L);

// Independent fixed fixture for FM1, 8% dead-fuel moisture, 1 m/s
// wind pushing east, and slope_tangent=0.20 directed north.
constexpr amrex::Real reference_heading_x =
    amrex::Real(0.9264182564780811);
constexpr amrex::Real reference_heading_y =
    amrex::Real(0.37649596819104475);
constexpr amrex::Real reference_head_ros_mps =
    amrex::Real(0.10852086861083189);
constexpr amrex::Real reference_back_ros_mps =
    amrex::Real(0.029717476468763306);
constexpr amrex::Real reference_flank_ros_mps =
    amrex::Real(0.056788787267489274);
constexpr amrex::Real reference_semi_major_rate_mps =
    amrex::Real(0.0691191725397976);
constexpr amrex::Real reference_center_translation_rate_mps =
    amrex::Real(0.03940169607103429);

// Independent wind-only FM1 fixture: 8% moisture, 1 m/s wind pushing east,
// zero slope. The ellipse rates are
// independently derived from the FARSITE length-to-breadth relation at 1 m/s.
constexpr amrex::Real reference_wind_only_head_ros_mps =
    amrex::Real(0.10202231168335671);
constexpr amrex::Real reference_wind_only_flank_ros_mps =
    amrex::Real(0.05412799019404888);
constexpr amrex::Real reference_wind_only_semi_major_rate_mps =
    amrex::Real(0.06536997242848683);
constexpr amrex::Real reference_wind_only_center_translation_rate_mps =
    amrex::Real(0.03665233925486989);

constexpr amrex::Real initial_wavelet_age_s = amrex::Real(200.0);
constexpr amrex::Real integration_duration_s = amrex::Real(100.0);
constexpr amrex::Real dt_s = amrex::Real(1.0);
constexpr int integration_steps = 100;

constexpr amrex::Real reference_raster_xlo_m = amrex::Real(-20.0);
constexpr amrex::Real reference_raster_ylo_m = amrex::Real(-20.0);
constexpr amrex::Real reference_raster_extent_m = amrex::Real(60.0);

using RemeshedGrowthObserver =
    std::function<void(amrex::Real, const FirePerimeter&)>;

using RemeshedGrowthSweepObserver =
    std::function<void(
        amrex::Real,
        amrex::Real,
        const FirePerimeter&,
        const FirePerimeter&)>;

struct FireArrivalTimelineRow
{
    amrex::Real time_s{};
    std::size_t pre_remesh_vertices{};
    std::size_t post_remesh_vertices{};
    std::size_t newly_arrived_cells{};
    std::size_t arrived_cells{};
};

[[nodiscard]] const char*
fire_test_output_dir () noexcept
{
    const char* output_dir =
        std::getenv("ERF_FIRE_TEST_OUTPUT_DIR");
    if (output_dir == nullptr || output_dir[0] == '\0') {
        return nullptr;
    }
    return output_dir;
}

[[nodiscard]] std::ofstream
open_fire_test_csv (const std::string& filename)
{
    const char* output_dir = fire_test_output_dir();
    if (output_dir == nullptr) {
        return {};
    }

    std::ofstream stream(
        std::string(output_dir) + "/" + filename);
    if (!stream.is_open()) {
        throw std::runtime_error(
            "Unable to open ERF-Fire visualization CSV output");
    }

    stream << std::setprecision(17);
    return stream;
}

void
maybe_write_fire_arrival_timeline (
    const std::string& filename,
    const std::vector<FireArrivalTimelineRow>& rows)
{
    std::ofstream stream = open_fire_test_csv(filename);
    if (!stream.is_open()) {
        return;
    }

    stream
        << "time_s,pre_remesh_vertices,post_remesh_vertices,"
        << "newly_arrived_cells,arrived_cells\n";

    for (const auto& row : rows) {
        stream
            << row.time_s << ","
            << row.pre_remesh_vertices << ","
            << row.post_remesh_vertices << ","
            << row.newly_arrived_cells << ","
            << row.arrived_cells << "\n";
    }
}

void
maybe_write_fire_arrival_wind_history (
    const FireFirstArrivalRaster& arrival,
    const FireBurnedFractionRaster& burned)
{
    std::ofstream stream =
        open_fire_test_csv(
            "fire_arrival_wind_arrival_history.csv");
    if (!stream.is_open()) {
        return;
    }

    stream
        << "cell_i,xlo_m,xhi_m,ylo_m,yhi_m,has_arrived,"
        << "numerical_arrival_s,analytic_arrival_s,error_s,"
        << "burned_fraction\n";

    const auto& geometry = arrival.geometry();
    for (std::size_t i = 0; i < geometry.nx; ++i) {
        const auto cell = arrival.cell_bounds(i, 0);
        const bool has_arrived =
            arrival.has_arrived(i, 0);
        const amrex::Real analytic_arrival_s =
            cell.xlo_m
            / reference_wind_only_head_ros_mps
            - initial_wavelet_age_s;

        stream
            << i << ","
            << cell.xlo_m << ","
            << cell.xhi_m << ","
            << cell.ylo_m << ","
            << cell.yhi_m << ","
            << (has_arrived ? 1 : 0) << ",";

        if (has_arrived) {
            const amrex::Real numerical_arrival_s =
                arrival.first_arrival_time_s(i, 0);
            stream
                << numerical_arrival_s << ","
                << analytic_arrival_s << ","
                << numerical_arrival_s
                    - analytic_arrival_s << ",";
        } else {
            stream
                << ","
                << analytic_arrival_s << ","
                << ",";
        }

        stream
            << burned.burned_fraction(i, 0)
            << "\n";
    }
}

void
maybe_write_fire_arrival_extinction_history (
    const FireFirstArrivalRaster& arrival)
{
    std::ofstream stream =
        open_fire_test_csv(
            "fire_arrival_extinction_arrival_history.csv");
    if (!stream.is_open()) {
        return;
    }

    stream
        << "cell_i,xlo_m,xhi_m,ylo_m,yhi_m,has_arrived,"
        << "arrival_time_s\n";

    const auto& geometry = arrival.geometry();
    for (std::size_t i = 0; i < geometry.nx; ++i) {
        const auto cell = arrival.cell_bounds(i, 0);
        const bool has_arrived =
            arrival.has_arrived(i, 0);

        stream
            << i << ","
            << cell.xlo_m << ","
            << cell.xhi_m << ","
            << cell.ylo_m << ","
            << cell.yhi_m << ","
            << (has_arrived ? 1 : 0) << ",";

        if (has_arrived) {
            stream
                << arrival.first_arrival_time_s(i, 0);
        }

        stream << "\n";
    }
}

FireVec2
reference_heading (bool mirrored = false) noexcept
{
    return {
        reference_heading_x,
        mirrored ? -reference_heading_y : reference_heading_y
    };
}

FireVec2
left_perpendicular (const FireVec2& direction) noexcept
{
    return {-direction.y, direction.x};
}

FirePerimeter
make_reference_wavelet_with_rates (
    std::size_t vertex_count,
    amrex::Real age_s,
    const FireVec2& heading,
    amrex::Real semi_major_rate_mps,
    amrex::Real semi_minor_rate_mps,
    amrex::Real center_translation_rate_mps)
{
    std::vector<FireVec2> vertices;
    vertices.reserve(vertex_count);

    const FireVec2 flank_direction = left_perpendicular(heading);
    const FireVec2 center =
        heading * (center_translation_rate_mps * age_s);

    const amrex::Real semi_major_m =
        semi_major_rate_mps * age_s;
    const amrex::Real semi_minor_m =
        semi_minor_rate_mps * age_s;

    for (std::size_t i = 0; i < vertex_count; ++i) {
        const amrex::Real angle =
            amrex::Real(2.0) * pi
            * static_cast<amrex::Real>(i)
            / static_cast<amrex::Real>(vertex_count);

        vertices.push_back(
            center
            + heading * (semi_major_m * std::cos(angle))
            + flank_direction * (semi_minor_m * std::sin(angle)));
    }

    return FirePerimeter(std::move(vertices));
}

FirePerimeter
make_exact_reference_wavelet (
    std::size_t vertex_count,
    amrex::Real age_s,
    const FireVec2& heading)
{
    return make_reference_wavelet_with_rates(
        vertex_count,
        age_s,
        heading,
        reference_semi_major_rate_mps,
        reference_flank_ros_mps,
        reference_center_translation_rate_mps);
}

FirePerimeter
make_wind_only_reference_wavelet (
    std::size_t vertex_count,
    amrex::Real age_s)
{
    return make_reference_wavelet_with_rates(
        vertex_count,
        age_s,
        {1.0, 0.0},
        reference_wind_only_semi_major_rate_mps,
        reference_wind_only_flank_ros_mps,
        reference_wind_only_center_translation_rate_mps);
}

amrex::Real
exact_reference_area_m2 (amrex::Real age_s) noexcept
{
    return pi
        * (reference_semi_major_rate_mps * age_s)
        * (reference_flank_ros_mps * age_s);
}

FireCartesianRasterGeometry2D
make_reference_raster_geometry (amrex::Real spacing_m)
{
    const auto cells_per_side =
        static_cast<std::size_t>(
            std::llround(reference_raster_extent_m / spacing_m));

    return {
        cells_per_side,
        cells_per_side,
        reference_raster_xlo_m,
        reference_raster_ylo_m,
        spacing_m,
        spacing_m
    };
}

std::vector<amrex::Real>
capture_burned_fraction (
    const FireBurnedFractionRaster& raster)
{
    std::vector<amrex::Real> values;
    values.reserve(raster.cell_count());

    const auto& geometry = raster.geometry();
    for (std::size_t j = 0; j < geometry.ny; ++j) {
        for (std::size_t i = 0; i < geometry.nx; ++i) {
            values.push_back(raster.burned_fraction(i, j));
        }
    }

    return values;
}

std::size_t
partial_cell_count (const FireBurnedFractionRaster& raster)
{
    std::size_t count = 0;
    const auto& geometry = raster.geometry();

    for (std::size_t j = 0; j < geometry.ny; ++j) {
        for (std::size_t i = 0; i < geometry.nx; ++i) {
            const amrex::Real fraction =
                raster.burned_fraction(i, j);
            if (fraction > amrex::Real(0.0)
                && fraction < amrex::Real(1.0)) {
                ++count;
            }
        }
    }

    return count;
}

amrex::Real
polygon_support_m (
    const FirePerimeter& perimeter,
    const FireVec2& unit_direction) noexcept
{
    amrex::Real support =
        -std::numeric_limits<amrex::Real>::infinity();

    for (const auto& vertex : perimeter.vertices_m()) {
        support = std::max(support, ERFFire::dot(vertex, unit_direction));
    }

    return support;
}

amrex::Real
exact_reference_support_m (
    amrex::Real age_s,
    const FireVec2& heading,
    const FireVec2& unit_direction)
{
    const FireVec2 flank_direction = left_perpendicular(heading);
    const amrex::Real heading_projection =
        ERFFire::dot(heading, unit_direction);
    const amrex::Real flank_projection =
        ERFFire::dot(flank_direction, unit_direction);

    const amrex::Real center_support_m =
        reference_center_translation_rate_mps
        * age_s * heading_projection;
    const amrex::Real major_support_m =
        reference_semi_major_rate_mps
        * age_s * heading_projection;
    const amrex::Real minor_support_m =
        reference_flank_ros_mps
        * age_s * flank_projection;

    return center_support_m
        + std::sqrt(
            major_support_m * major_support_m
            + minor_support_m * minor_support_m);
}

amrex::Real
maximum_support_error_m (
    const FirePerimeter& perimeter,
    amrex::Real age_s,
    const FireVec2& heading,
    std::size_t direction_count = 1024)
{
    amrex::Real maximum_error_m = 0.0;

    for (std::size_t i = 0; i < direction_count; ++i) {
        const amrex::Real angle =
            amrex::Real(2.0) * pi
            * static_cast<amrex::Real>(i)
            / static_cast<amrex::Real>(direction_count);
        const FireVec2 direction{
            std::cos(angle),
            std::sin(angle)
        };

        maximum_error_m = std::max(
            maximum_error_m,
            std::abs(
                polygon_support_m(perimeter, direction)
                - exact_reference_support_m(age_s, heading, direction)));
    }

    return maximum_error_m;
}

RichardsDirectionalSpread
make_oblique_fm1_spread (bool mirrored = false)
{
    const auto behavior = ERFFire::evaluate_rothermel(
        ERFFire::make_fm1_fuel_parameters(),
        RothermelInputs{0.08, 1.0, 0.20});

    return ERFFire::make_richards_directional_spread(
        behavior,
        {1.0, 0.0},
        {0.0, mirrored ? -1.0 : 1.0});
}

RichardsDirectionalSpread
make_wind_only_fm1_spread ()
{
    const auto behavior = ERFFire::evaluate_rothermel(
        ERFFire::make_fm1_fuel_parameters(),
        RothermelInputs{0.08, 1.0, 0.0});

    return ERFFire::make_richards_directional_spread(
        behavior,
        {1.0, 0.0},
        {0.0, 1.0});
}

FirePerimeter
advance_fixed_topology (
    FirePerimeter perimeter,
    const RichardsDirectionalSpread& spread)
{
    const auto normal_speed = [&spread] (
        const FireVec2&,
        const FireVec2& outward_normal,
        amrex::Real) -> amrex::Real
    {
        return ERFFire::richards_normal_speed_mps(
            spread.ellipse, outward_normal);
    };

    amrex::Real time_s = 0.0;
    for (int step = 1; step <= integration_steps; ++step) {
        perimeter = ERFFire::advance_perimeter_rk2(
            perimeter, time_s, dt_s, normal_speed);
        time_s += dt_s;
    }

    return perimeter;
}

amrex::Real
minimum_edge_length_m (const FirePerimeter& perimeter)
{
    amrex::Real minimum =
        std::numeric_limits<amrex::Real>::infinity();
    const auto& vertices = perimeter.vertices_m();

    for (std::size_t i = 0; i < vertices.size(); ++i) {
        minimum = std::min(
            minimum,
            ERFFire::norm(
                vertices[(i + 1) % vertices.size()] - vertices[i]));
    }

    return minimum;
}

amrex::Real
maximum_edge_length_m (const FirePerimeter& perimeter)
{
    amrex::Real maximum = 0.0;
    const auto& vertices = perimeter.vertices_m();

    for (std::size_t i = 0; i < vertices.size(); ++i) {
        maximum = std::max(
            maximum,
            ERFFire::norm(
                vertices[(i + 1) % vertices.size()] - vertices[i]));
    }

    return maximum;
}

struct RemeshedGrowthRun
{
    FirePerimeter perimeter;
    std::size_t initial_remeshed_vertex_count;
    std::size_t vertices_removed;
    std::size_t vertices_added;
};

RemeshedGrowthRun
advance_with_remeshing (
    FirePerimeter perimeter,
    const RichardsDirectionalSpread& spread,
    const FireVec2& reference_heading,
    const FirePerimeterRemeshOptions& options,
    bool write_reference_snapshots = false,
    const RemeshedGrowthObserver& observer = {},
    const RemeshedGrowthSweepObserver& sweep_observer = {})
{
    std::size_t vertices_removed = 0;
    std::size_t vertices_added = 0;

    auto remeshed = ERFFire::remesh_perimeter(perimeter, options);
    perimeter = std::move(remeshed.perimeter);
    vertices_removed += remeshed.stats.vertices_removed;
    vertices_added += remeshed.stats.vertices_added;

    const std::size_t initial_remeshed_vertex_count = perimeter.size();

    if (write_reference_snapshots) {
        ERFFireTest::maybe_write_snapshot(
            "standalone_oblique_remeshed", 0.0, perimeter);
        ERFFireTest::maybe_write_snapshot(
            "standalone_oblique_remeshed_exact",
            0.0,
            make_exact_reference_wavelet(
                2048, initial_wavelet_age_s, reference_heading));
    }

    if (observer) {
        observer(amrex::Real(0.0), perimeter);
    }

    const auto normal_speed = [&spread] (
        const FireVec2&,
        const FireVec2& outward_normal,
        amrex::Real) -> amrex::Real
    {
        return ERFFire::richards_normal_speed_mps(
            spread.ellipse, outward_normal);
    };

    amrex::Real time_s = 0.0;
    for (int step = 1; step <= integration_steps; ++step) {
        const amrex::Real next_time_s = time_s + dt_s;
        FirePerimeter advanced =
            ERFFire::advance_perimeter_rk2(
                perimeter, time_s, dt_s, normal_speed);

        if (sweep_observer) {
            sweep_observer(
                time_s,
                next_time_s,
                perimeter,
                advanced);
        }

        perimeter = std::move(advanced);
        time_s = next_time_s;

        remeshed = ERFFire::remesh_perimeter(perimeter, options);
        perimeter = std::move(remeshed.perimeter);
        vertices_removed += remeshed.stats.vertices_removed;
        vertices_added += remeshed.stats.vertices_added;

        if (observer) {
            observer(time_s, perimeter);
        }

        if (write_reference_snapshots && step % 25 == 0) {
            ERFFireTest::maybe_write_snapshot(
                "standalone_oblique_remeshed",
                time_s,
                perimeter);
            ERFFireTest::maybe_write_snapshot(
                "standalone_oblique_remeshed_exact",
                time_s,
                make_exact_reference_wavelet(
                    2048,
                    initial_wavelet_age_s + time_s,
                    reference_heading));
        }
    }

    return {
        std::move(perimeter),
        initial_remeshed_vertex_count,
        vertices_removed,
        vertices_added
    };
}

TEST(FireStandaloneGrowth, HomogeneousObliqueFM1TracksExactWavelet)
{
    constexpr std::size_t vertex_count = 256;
    const FireVec2 heading = reference_heading();
    const auto spread = make_oblique_fm1_spread();

    FirePerimeter perimeter = make_exact_reference_wavelet(
        vertex_count, initial_wavelet_age_s, heading);

    ERFFireTest::maybe_write_snapshot(
        "standalone_oblique", 0.0, perimeter);
    ERFFireTest::maybe_write_snapshot(
        "standalone_oblique_exact",
        0.0,
        make_exact_reference_wavelet(
            2048, initial_wavelet_age_s, heading));

    const auto normal_speed = [&spread] (
        const FireVec2&,
        const FireVec2& outward_normal,
        amrex::Real) -> amrex::Real
    {
        return ERFFire::richards_normal_speed_mps(
            spread.ellipse, outward_normal);
    };

    amrex::Real time_s = 0.0;
    for (int step = 1; step <= integration_steps; ++step) {
        perimeter = ERFFire::advance_perimeter_rk2(
            perimeter, time_s, dt_s, normal_speed);
        time_s += dt_s;

        if (step % 25 == 0) {
            ERFFireTest::maybe_write_snapshot(
                "standalone_oblique", time_s, perimeter);
            ERFFireTest::maybe_write_snapshot(
                "standalone_oblique_exact",
                time_s,
                make_exact_reference_wavelet(
                    2048,
                    initial_wavelet_age_s + time_s,
                    heading));
        }
    }

    const amrex::Real final_age_s =
        initial_wavelet_age_s + integration_duration_s;
    const FireVec2 backing{-heading.x, -heading.y};
    const FireVec2 flank = left_perpendicular(heading);
    const FireVec2 opposite_flank{-flank.x, -flank.y};

    const amrex::Real head_support_m =
        polygon_support_m(perimeter, heading);
    const amrex::Real back_support_m =
        polygon_support_m(perimeter, backing);
    const amrex::Real flank_support_m =
        polygon_support_m(perimeter, flank);
    const amrex::Real opposite_flank_support_m =
        polygon_support_m(perimeter, opposite_flank);

    EXPECT_NEAR(
        static_cast<double>(head_support_m),
        static_cast<double>(reference_head_ros_mps * final_age_s),
        5.0e-9);
    EXPECT_NEAR(
        static_cast<double>(back_support_m),
        static_cast<double>(reference_back_ros_mps * final_age_s),
        5.0e-9);

    EXPECT_NEAR(
        static_cast<double>(
            amrex::Real(0.5) * (head_support_m - back_support_m)),
        static_cast<double>(
            reference_center_translation_rate_mps * final_age_s),
        5.0e-9);
    EXPECT_NEAR(
        static_cast<double>(
            amrex::Real(0.5) * (head_support_m + back_support_m)),
        static_cast<double>(
            reference_semi_major_rate_mps * final_age_s),
        5.0e-9);

    EXPECT_NEAR(
        static_cast<double>(
            amrex::Real(0.5)
            * (flank_support_m - opposite_flank_support_m)),
        0.0,
        1.0e-3);
    EXPECT_NEAR(
        static_cast<double>(
            amrex::Real(0.5)
            * (flank_support_m + opposite_flank_support_m)),
        static_cast<double>(reference_flank_ros_mps * final_age_s),
        1.0e-3);

    const amrex::Real exact_area_m2 =
        pi
        * (reference_flank_ros_mps * final_age_s)
        * (reference_semi_major_rate_mps * final_age_s);
    const amrex::Real relative_area_error =
        std::abs(perimeter.area_m2() - exact_area_m2) / exact_area_m2;

    EXPECT_LT(
        static_cast<double>(relative_area_error),
        2.0e-4);

    EXPECT_LT(
        static_cast<double>(
            maximum_support_error_m(perimeter, final_age_s, heading)),
        6.0e-3);
}

TEST(FireStandaloneGrowth, FixedTopologyConvergesGeometricallyWithResolution)
{
    const FireVec2 heading = reference_heading();
    const auto spread = make_oblique_fm1_spread();
    const amrex::Real final_age_s =
        initial_wavelet_age_s + integration_duration_s;

    const auto run = [&] (std::size_t vertex_count) {
        FirePerimeter perimeter = make_exact_reference_wavelet(
            vertex_count, initial_wavelet_age_s, heading);
        perimeter = advance_fixed_topology(std::move(perimeter), spread);
        return maximum_support_error_m(
            perimeter, final_age_s, heading);
    };

    const amrex::Real error_64 = run(64);
    const amrex::Real error_128 = run(128);
    const amrex::Real error_256 = run(256);

    EXPECT_LT(static_cast<double>(error_64), 8.0e-2);
    EXPECT_LT(static_cast<double>(error_128), 2.5e-2);
    EXPECT_LT(static_cast<double>(error_256), 6.0e-3);

    EXPECT_LT(
        static_cast<double>(error_128),
        0.30 * static_cast<double>(error_64));
    EXPECT_LT(
        static_cast<double>(error_256),
        0.30 * static_cast<double>(error_128));
}

TEST(FireStandaloneGrowth, MirroredSlopeProducesMirroredFront)
{
    constexpr std::size_t vertex_count = 256;
    const FireVec2 north_heading = reference_heading(false);
    const FireVec2 south_heading = reference_heading(true);

    FirePerimeter north = make_exact_reference_wavelet(
        vertex_count, initial_wavelet_age_s, north_heading);
    FirePerimeter south = make_exact_reference_wavelet(
        vertex_count, initial_wavelet_age_s, south_heading);

    north = advance_fixed_topology(
        std::move(north), make_oblique_fm1_spread(false));
    south = advance_fixed_topology(
        std::move(south), make_oblique_fm1_spread(true));

    amrex::Real maximum_mirror_error_m = 0.0;
    constexpr std::size_t direction_count = 1024;

    for (std::size_t i = 0; i < direction_count; ++i) {
        const amrex::Real angle =
            amrex::Real(2.0) * pi
            * static_cast<amrex::Real>(i)
            / static_cast<amrex::Real>(direction_count);
        const FireVec2 north_direction{
            std::cos(angle),
            std::sin(angle)
        };
        const FireVec2 south_direction{
            north_direction.x,
            -north_direction.y
        };

        maximum_mirror_error_m = std::max(
            maximum_mirror_error_m,
            std::abs(
                polygon_support_m(north, north_direction)
                - polygon_support_m(south, south_direction)));
    }

    EXPECT_LT(
        static_cast<double>(maximum_mirror_error_m),
        1.0e-10);

    const amrex::Real final_age_s =
        initial_wavelet_age_s + integration_duration_s;
    EXPECT_LT(
        static_cast<double>(
            maximum_support_error_m(
                north, final_age_s, north_heading)),
        6.0e-3);
    EXPECT_LT(
        static_cast<double>(
            maximum_support_error_m(
                south, final_age_s, south_heading)),
        6.0e-3);
}


TEST(FireStandaloneGrowth, HomogeneousObliqueFM1WithRemeshingTracksExactWavelet)
{
    constexpr std::size_t initial_vertex_count = 512;
    const FireVec2 heading = reference_heading();
    const auto spread = make_oblique_fm1_spread();

    const FirePerimeterRemeshOptions options{
        0.25,
        0.75,
        0.0125
    };

    auto run = advance_with_remeshing(
        make_exact_reference_wavelet(
            initial_vertex_count,
            initial_wavelet_age_s,
            heading),
        spread,
        heading,
        options,
        true);

    const amrex::Real final_age_s =
        initial_wavelet_age_s + integration_duration_s;

    EXPECT_LT(run.initial_remeshed_vertex_count, initial_vertex_count);
    EXPECT_GT(run.vertices_removed, 0U);
    EXPECT_GT(run.vertices_added, 0U);
    EXPECT_GT(run.perimeter.size(), run.initial_remeshed_vertex_count);

    EXPECT_GE(
        static_cast<double>(minimum_edge_length_m(run.perimeter)),
        0.25 - 1.0e-12);
    EXPECT_LE(
        static_cast<double>(maximum_edge_length_m(run.perimeter)),
        0.75 + 1.0e-12);

    const amrex::Real support_error_m =
        maximum_support_error_m(
            run.perimeter, final_age_s, heading);

    EXPECT_LT(
        static_cast<double>(support_error_m),
        1.5e-2);

    const amrex::Real exact_area_m2 =
        pi
        * (reference_flank_ros_mps * final_age_s)
        * (reference_semi_major_rate_mps * final_age_s);
    const amrex::Real relative_area_error =
        std::abs(run.perimeter.area_m2() - exact_area_m2)
        / exact_area_m2;

    EXPECT_LT(
        static_cast<double>(relative_area_error),
        1.0e-3);
}

TEST(FireStandaloneGrowth, TighterRemeshSpacingReducesGeometryError)
{
    constexpr std::size_t initial_vertex_count = 512;
    const FireVec2 heading = reference_heading();
    const auto spread = make_oblique_fm1_spread();

    const FirePerimeterRemeshOptions coarse_options{
        0.50,
        1.50,
        0.0500
    };
    const FirePerimeterRemeshOptions medium_options{
        0.25,
        0.75,
        0.0125
    };

    auto coarse = advance_with_remeshing(
        make_exact_reference_wavelet(
            initial_vertex_count,
            initial_wavelet_age_s,
            heading),
        spread,
        heading,
        coarse_options);

    auto medium = advance_with_remeshing(
        make_exact_reference_wavelet(
            initial_vertex_count,
            initial_wavelet_age_s,
            heading),
        spread,
        heading,
        medium_options);

    const amrex::Real final_age_s =
        initial_wavelet_age_s + integration_duration_s;
    const amrex::Real coarse_error_m =
        maximum_support_error_m(
            coarse.perimeter, final_age_s, heading);
    const amrex::Real medium_error_m =
        maximum_support_error_m(
            medium.perimeter, final_age_s, heading);

    EXPECT_GT(coarse.vertices_removed, 0U);
    EXPECT_GT(coarse.vertices_added, 0U);
    EXPECT_GT(medium.vertices_removed, 0U);
    EXPECT_GT(medium.vertices_added, 0U);

    EXPECT_LT(
        static_cast<double>(coarse_error_m),
        6.0e-2);
    EXPECT_LT(
        static_cast<double>(medium_error_m),
        1.5e-2);
    EXPECT_LT(
        static_cast<double>(medium_error_m),
        0.40 * static_cast<double>(coarse_error_m));

    EXPECT_LE(
        static_cast<double>(maximum_edge_length_m(coarse.perimeter)),
        1.50 + 1.0e-12);
    EXPECT_LE(
        static_cast<double>(maximum_edge_length_m(medium.perimeter)),
        0.75 + 1.0e-12);
}

TEST(FireStandaloneGrowth, MirroredSlopeWithRemeshingPreservesSymmetry)
{
    constexpr std::size_t initial_vertex_count = 512;
    const FireVec2 north_heading = reference_heading(false);
    const FireVec2 south_heading = reference_heading(true);

    const FirePerimeterRemeshOptions options{
        0.25,
        0.75,
        0.0125
    };

    auto north = advance_with_remeshing(
        make_exact_reference_wavelet(
            initial_vertex_count,
            initial_wavelet_age_s,
            north_heading),
        make_oblique_fm1_spread(false),
        north_heading,
        options);

    auto south = advance_with_remeshing(
        make_exact_reference_wavelet(
            initial_vertex_count,
            initial_wavelet_age_s,
            south_heading),
        make_oblique_fm1_spread(true),
        south_heading,
        options);

    ERFFireTest::maybe_write_snapshot(
        "standalone_oblique_remeshed_north",
        integration_duration_s,
        north.perimeter);
    ERFFireTest::maybe_write_snapshot(
        "standalone_oblique_remeshed_south",
        integration_duration_s,
        south.perimeter);

    amrex::Real maximum_mirror_error_m = 0.0;
    constexpr std::size_t direction_count = 1024;

    for (std::size_t i = 0; i < direction_count; ++i) {
        const amrex::Real angle =
            amrex::Real(2.0) * pi
            * static_cast<amrex::Real>(i)
            / static_cast<amrex::Real>(direction_count);
        const FireVec2 north_direction{
            std::cos(angle),
            std::sin(angle)
        };
        const FireVec2 south_direction{
            north_direction.x,
            -north_direction.y
        };

        maximum_mirror_error_m = std::max(
            maximum_mirror_error_m,
            std::abs(
                polygon_support_m(
                    north.perimeter, north_direction)
                - polygon_support_m(
                    south.perimeter, south_direction)));
    }

    const amrex::Real final_age_s =
        initial_wavelet_age_s + integration_duration_s;
    const amrex::Real north_error_m =
        maximum_support_error_m(
            north.perimeter, final_age_s, north_heading);
    const amrex::Real south_error_m =
        maximum_support_error_m(
            south.perimeter, final_age_s, south_heading);

    EXPECT_LT(
        static_cast<double>(north_error_m),
        1.5e-2);
    EXPECT_LT(
        static_cast<double>(south_error_m),
        1.5e-2);
    EXPECT_LT(
        static_cast<double>(
            std::abs(north_error_m - south_error_m)),
        1.0e-3);
    EXPECT_LT(
        static_cast<double>(maximum_mirror_error_m),
        1.5e-2);
}

TEST(FireRasterIntegration, RemeshedObliqueFM1MaintainsConservativeMonotoneHistory)
{
    constexpr std::size_t initial_vertex_count = 512;
    const FireVec2 heading = reference_heading();
    const auto spread = make_oblique_fm1_spread();

    const FirePerimeterRemeshOptions options{
        0.25,
        0.75,
        0.0125
    };

    const auto raster_geometry =
        make_reference_raster_geometry(amrex::Real(0.5));
    FireBurnedFractionRaster raster(raster_geometry);

    std::vector<amrex::Real> previous_fraction(
        raster.cell_count(), amrex::Real(0.0));

    amrex::Real previous_burned_area_m2 = 0.0;
    amrex::Real cumulative_newly_burned_area_m2 = 0.0;
    std::size_t sampled_front_count = 0;

    const RemeshedGrowthObserver observer =
        [&] (amrex::Real time_s, const FirePerimeter& perimeter)
    {
        const long long whole_seconds = std::llround(time_s);
        if (whole_seconds % 25 != 0) {
            return;
        }

        FireBurnedFractionRaster instantaneous(raster_geometry);
        const auto instantaneous_update =
            instantaneous.update_from_perimeter(perimeter);
        const auto persistent_update =
            raster.update_from_perimeter(perimeter);

        const amrex::Real exact_area_m2 =
            exact_reference_area_m2(initial_wavelet_age_s + time_s);
        const amrex::Real relative_vector_area_error =
            std::abs(perimeter.area_m2() - exact_area_m2)
            / exact_area_m2;

        EXPECT_LT(
            static_cast<double>(relative_vector_area_error),
            1.0e-3);

        EXPECT_NEAR(
            static_cast<double>(instantaneous_update.burned_area_m2),
            static_cast<double>(perimeter.area_m2()),
            5.0e-8);

        EXPECT_NEAR(
            static_cast<double>(persistent_update.burned_area_m2),
            static_cast<double>(instantaneous_update.burned_area_m2),
            5.0e-8);

        EXPECT_NEAR(
            static_cast<double>(
                previous_burned_area_m2
                + persistent_update.newly_burned_area_m2),
            static_cast<double>(persistent_update.burned_area_m2),
            5.0e-8);

        EXPECT_GE(
            static_cast<double>(persistent_update.burned_area_m2),
            static_cast<double>(previous_burned_area_m2));

        if (sampled_front_count > 0) {
            EXPECT_GT(
                static_cast<double>(persistent_update.newly_burned_area_m2),
                0.0);
        }

        const auto current_fraction = capture_burned_fraction(raster);
        ASSERT_EQ(current_fraction.size(), previous_fraction.size());

        for (std::size_t i = 0; i < current_fraction.size(); ++i) {
            EXPECT_GE(
                static_cast<double>(current_fraction[i]),
                static_cast<double>(previous_fraction[i]));
        }

        previous_fraction = current_fraction;
        previous_burned_area_m2 = persistent_update.burned_area_m2;
        cumulative_newly_burned_area_m2 +=
            persistent_update.newly_burned_area_m2;
        ++sampled_front_count;
    };

    auto run = advance_with_remeshing(
        make_exact_reference_wavelet(
            initial_vertex_count,
            initial_wavelet_age_s,
            heading),
        spread,
        heading,
        options,
        false,
        observer);

    EXPECT_EQ(sampled_front_count, 5U);

    EXPECT_NEAR(
        static_cast<double>(cumulative_newly_burned_area_m2),
        static_cast<double>(raster.burned_area_m2()),
        5.0e-8);

    const auto repeated = raster.update_from_perimeter(run.perimeter);

    EXPECT_DOUBLE_EQ(
        static_cast<double>(repeated.newly_burned_area_m2),
        0.0);
    EXPECT_NEAR(
        static_cast<double>(repeated.burned_area_m2),
        static_cast<double>(previous_burned_area_m2),
        5.0e-8);
}

TEST(FireRasterIntegration, FinalRemeshedPerimeterAreaIsResolutionInvariant)
{
    constexpr std::size_t initial_vertex_count = 512;
    const FireVec2 heading = reference_heading();
    const auto spread = make_oblique_fm1_spread();

    const FirePerimeterRemeshOptions options{
        0.25,
        0.75,
        0.0125
    };

    auto run = advance_with_remeshing(
        make_exact_reference_wavelet(
            initial_vertex_count,
            initial_wavelet_age_s,
            heading),
        spread,
        heading,
        options);

    std::vector<std::size_t> partial_counts;

    for (const amrex::Real spacing_m : {
            amrex::Real(1.0),
            amrex::Real(0.5),
            amrex::Real(0.25)}) {
        FireBurnedFractionRaster raster(
            make_reference_raster_geometry(spacing_m));

        const auto update = raster.update_from_perimeter(run.perimeter);

        EXPECT_NEAR(
            static_cast<double>(update.burned_area_m2),
            static_cast<double>(run.perimeter.area_m2()),
            1.0e-7);

        EXPECT_NEAR(
            static_cast<double>(update.newly_burned_area_m2),
            static_cast<double>(run.perimeter.area_m2()),
            1.0e-7);

        const std::size_t partial = partial_cell_count(raster);
        EXPECT_GT(partial, 0U);
        partial_counts.push_back(partial);
    }

    ASSERT_EQ(partial_counts.size(), 3U);
    EXPECT_LT(partial_counts[0], partial_counts[1]);
    EXPECT_LT(partial_counts[1], partial_counts[2]);
}

TEST(FireRasterIntegration, ExtinguishedRemeshedFrontCreatesNoAdditionalBurnHistory)
{
    const auto behavior = ERFFire::evaluate_rothermel(
        ERFFire::make_fm1_fuel_parameters(),
        RothermelInputs{0.12, 1.0, 0.20});
    const auto spread = ERFFire::make_richards_directional_spread(
        behavior,
        {1.0, 0.0},
        {0.0, 1.0});

    const FirePerimeterRemeshOptions options{
        0.25,
        0.75,
        0.0125
    };

    FireBurnedFractionRaster raster(
        make_reference_raster_geometry(amrex::Real(0.5)));

    amrex::Real initial_burned_area_m2 = 0.0;
    std::size_t sampled_front_count = 0;

    const RemeshedGrowthObserver observer =
        [&] (amrex::Real time_s, const FirePerimeter& perimeter)
    {
        const long long whole_seconds = std::llround(time_s);
        if (whole_seconds % 25 != 0) {
            return;
        }

        const auto update = raster.update_from_perimeter(perimeter);

        if (sampled_front_count == 0) {
            initial_burned_area_m2 = update.burned_area_m2;
            EXPECT_GT(
                static_cast<double>(update.newly_burned_area_m2),
                0.0);
        } else {
            EXPECT_DOUBLE_EQ(
                static_cast<double>(update.newly_burned_area_m2),
                0.0);
            EXPECT_DOUBLE_EQ(
                static_cast<double>(update.burned_area_m2),
                static_cast<double>(initial_burned_area_m2));
        }

        ++sampled_front_count;
    };

    auto run = advance_with_remeshing(
        ERFFireTest::make_circle(256, 10.0),
        spread,
        {1.0, 0.0},
        options,
        false,
        observer);

    EXPECT_EQ(sampled_front_count, 5U);

    const auto repeated = raster.update_from_perimeter(run.perimeter);

    EXPECT_DOUBLE_EQ(
        static_cast<double>(repeated.newly_burned_area_m2),
        0.0);
    EXPECT_DOUBLE_EQ(
        static_cast<double>(repeated.burned_area_m2),
        static_cast<double>(initial_burned_area_m2));
}

TEST(FireArrivalIntegration, WindOnlyHeadStripTracksAnalyticArrivalThroughRemeshing)
{
    constexpr std::size_t initial_vertex_count = 512;
    const auto spread = make_wind_only_fm1_spread();

    const FirePerimeterRemeshOptions options{
        0.25,
        0.75,
        0.0125
    };

    FireFirstArrivalRaster arrival({
        5,
        1,
        22.0,
        -0.25,
        2.0,
        0.5
    });
    FireBurnedFractionRaster burned(
        arrival.geometry());

    std::size_t sweep_count = 0;
    std::size_t total_newly_arrived = 0;
    std::size_t topology_change_steps = 0;
    std::size_t last_pre_remesh_vertex_count = 0;
    std::size_t last_newly_arrived_cell_count = 0;
    std::size_t last_arrived_cell_count = 0;
    bool sweep_observed = false;
    std::vector<FireArrivalTimelineRow> timeline;

    const RemeshedGrowthSweepObserver sweep_observer =
        [&] (
            amrex::Real start_time_s,
            amrex::Real end_time_s,
            const FirePerimeter& start_perimeter,
            const FirePerimeter& end_pre_remesh)
    {
        EXPECT_EQ(
            start_perimeter.size(),
            end_pre_remesh.size());

        const auto arrival_update =
            arrival.update_from_sweep(
                start_perimeter,
                end_pre_remesh,
                start_time_s,
                end_time_s,
                1.0e-7);

        (void)burned.update_from_perimeter(
            end_pre_remesh);

        total_newly_arrived +=
            arrival_update.newly_arrived_cell_count;
        last_newly_arrived_cell_count =
            arrival_update.newly_arrived_cell_count;
        last_arrived_cell_count =
            arrival_update.arrived_cell_count;

        const long long whole_end_seconds =
            std::llround(end_time_s);
        if (whole_end_seconds % 20 == 0) {
            ERFFireTest::maybe_write_snapshot(
                "fire_arrival_wind_pre_remesh",
                end_time_s,
                end_pre_remesh);
        }

        for (std::size_t i = 0; i < 5; ++i) {
            if (burned.burned_fraction(i, 0)
                > amrex::Real(0.0)) {
                EXPECT_TRUE(arrival.has_arrived(i, 0));
                if (arrival.has_arrived(i, 0)) {
                    EXPECT_LE(
                        static_cast<double>(
                            arrival.first_arrival_time_s(i, 0)),
                        static_cast<double>(
                            end_time_s + amrex::Real(1.0e-7)));
                }
            }
        }

        last_pre_remesh_vertex_count =
            end_pre_remesh.size();
        sweep_observed = true;
        ++sweep_count;
    };

    const RemeshedGrowthObserver post_remesh_observer =
        [&] (
            amrex::Real time_s,
            const FirePerimeter& perimeter)
    {
        const long long whole_seconds =
            std::llround(time_s);
        if (whole_seconds % 20 == 0) {
            ERFFireTest::maybe_write_snapshot(
                "fire_arrival_wind_post_remesh",
                time_s,
                perimeter);
            ERFFireTest::maybe_write_snapshot(
                "fire_arrival_wind_exact",
                time_s,
                make_wind_only_reference_wavelet(
                    2048,
                    initial_wavelet_age_s + time_s));
        }

        if (time_s == amrex::Real(0.0)) {
            timeline.push_back({
                time_s,
                perimeter.size(),
                perimeter.size(),
                0U,
                0U
            });
            return;
        }

        EXPECT_TRUE(sweep_observed);
        if (perimeter.size()
            != last_pre_remesh_vertex_count) {
            ++topology_change_steps;
        }

        timeline.push_back({
            time_s,
            last_pre_remesh_vertex_count,
            perimeter.size(),
            last_newly_arrived_cell_count,
            last_arrived_cell_count
        });
        sweep_observed = false;
    };

    (void)advance_with_remeshing(
        make_wind_only_reference_wavelet(
            initial_vertex_count,
            initial_wavelet_age_s),
        spread,
        {1.0, 0.0},
        options,
        false,
        post_remesh_observer,
        sweep_observer);

    EXPECT_EQ(sweep_count, 100U);
    EXPECT_FALSE(sweep_observed);
    EXPECT_GT(topology_change_steps, 0U);
    EXPECT_EQ(arrival.arrived_cell_count(), 5U);
    EXPECT_EQ(total_newly_arrived, 5U);

    maybe_write_fire_arrival_timeline(
        "fire_arrival_wind_timeline.csv",
        timeline);
    maybe_write_fire_arrival_wind_history(
        arrival,
        burned);

    for (std::size_t i = 0; i < 5; ++i) {
        ASSERT_TRUE(arrival.has_arrived(i, 0));
        EXPECT_GT(
            static_cast<double>(
                burned.burned_fraction(i, 0)),
            0.0);

        const amrex::Real cell_xlo_m =
            amrex::Real(22.0)
            + amrex::Real(2.0)
            * static_cast<amrex::Real>(i);
        const amrex::Real exact_arrival_time_s =
            cell_xlo_m
            / reference_wind_only_head_ros_mps
            - initial_wavelet_age_s;
        const amrex::Real stored_arrival_time_s =
            arrival.first_arrival_time_s(i, 0);

        EXPECT_NEAR(
            static_cast<double>(stored_arrival_time_s),
            static_cast<double>(exact_arrival_time_s),
            0.25);

        if (i > 0) {
            EXPECT_GT(
                static_cast<double>(stored_arrival_time_s),
                static_cast<double>(
                    arrival.first_arrival_time_s(i - 1, 0)));
        }
    }
}

TEST(FireArrivalIntegration, ExtinguishedRemeshedSweepsDoNotAdvanceArrivalHistory)
{
    const auto behavior = ERFFire::evaluate_rothermel(
        ERFFire::make_fm1_fuel_parameters(),
        RothermelInputs{0.12, 1.0, 0.20});
    const auto spread =
        ERFFire::make_richards_directional_spread(
            behavior,
            {1.0, 0.0},
            {0.0, 1.0});

    const FirePerimeterRemeshOptions options{
        0.25,
        0.75,
        0.0125
    };

    FireFirstArrivalRaster arrival({
        6,
        1,
        8.0,
        -0.25,
        1.0,
        0.5
    });

    std::size_t sweep_count = 0;
    std::size_t total_newly_arrived = 0;
    std::size_t last_pre_remesh_vertex_count = 0;
    std::size_t last_newly_arrived_cell_count = 0;
    std::size_t last_arrived_cell_count = 0;
    bool sweep_observed = false;
    std::vector<FireArrivalTimelineRow> timeline;

    const RemeshedGrowthSweepObserver sweep_observer =
        [&] (
            amrex::Real start_time_s,
            amrex::Real end_time_s,
            const FirePerimeter& start_perimeter,
            const FirePerimeter& end_pre_remesh)
    {
        const auto update =
            arrival.update_from_sweep(
                start_perimeter,
                end_pre_remesh,
                start_time_s,
                end_time_s,
                1.0e-7);

        if (sweep_count == 0) {
            EXPECT_EQ(
                update.newly_arrived_cell_count,
                2U);
        } else {
            EXPECT_EQ(
                update.newly_arrived_cell_count,
                0U);
        }

        total_newly_arrived +=
            update.newly_arrived_cell_count;
        last_pre_remesh_vertex_count =
            end_pre_remesh.size();
        last_newly_arrived_cell_count =
            update.newly_arrived_cell_count;
        last_arrived_cell_count =
            update.arrived_cell_count;
        sweep_observed = true;

        const long long whole_end_seconds =
            std::llround(end_time_s);
        if (whole_end_seconds % 25 == 0) {
            ERFFireTest::maybe_write_snapshot(
                "fire_arrival_extinction_pre_remesh",
                end_time_s,
                end_pre_remesh);
        }

        ++sweep_count;
    };

    const RemeshedGrowthObserver post_remesh_observer =
        [&] (
            amrex::Real time_s,
            const FirePerimeter& perimeter)
    {
        const long long whole_seconds =
            std::llround(time_s);
        if (whole_seconds % 25 == 0) {
            ERFFireTest::maybe_write_snapshot(
                "fire_arrival_extinction_post_remesh",
                time_s,
                perimeter);
        }

        if (time_s == amrex::Real(0.0)) {
            timeline.push_back({
                time_s,
                perimeter.size(),
                perimeter.size(),
                0U,
                0U
            });
            return;
        }

        EXPECT_TRUE(sweep_observed);
        timeline.push_back({
            time_s,
            last_pre_remesh_vertex_count,
            perimeter.size(),
            last_newly_arrived_cell_count,
            last_arrived_cell_count
        });
        sweep_observed = false;
    };

    (void)advance_with_remeshing(
        ERFFireTest::make_circle(256, 10.0),
        spread,
        {1.0, 0.0},
        options,
        false,
        post_remesh_observer,
        sweep_observer);

    EXPECT_EQ(sweep_count, 100U);
    EXPECT_FALSE(sweep_observed);
    EXPECT_EQ(total_newly_arrived, 2U);
    EXPECT_EQ(arrival.arrived_cell_count(), 2U);

    maybe_write_fire_arrival_timeline(
        "fire_arrival_extinction_timeline.csv",
        timeline);
    maybe_write_fire_arrival_extinction_history(
        arrival);

    ASSERT_TRUE(arrival.has_arrived(0, 0));
    ASSERT_TRUE(arrival.has_arrived(1, 0));
    EXPECT_FALSE(arrival.has_arrived(2, 0));

    EXPECT_DOUBLE_EQ(
        static_cast<double>(
            arrival.first_arrival_time_s(0, 0)),
        0.0);
    EXPECT_DOUBLE_EQ(
        static_cast<double>(
            arrival.first_arrival_time_s(1, 0)),
        0.0);
}

TEST(FireStandaloneGrowth, ExtinctionWithRemeshingIsIdempotentlyStationary)
{
    const auto behavior = ERFFire::evaluate_rothermel(
        ERFFire::make_fm1_fuel_parameters(),
        RothermelInputs{0.12, 1.0, 0.20});
    const auto spread = ERFFire::make_richards_directional_spread(
        behavior, {1.0, 0.0}, {0.0, 1.0});

    const FirePerimeterRemeshOptions options{
        0.25,
        0.75,
        0.0125
    };

    auto initial_remesh = ERFFire::remesh_perimeter(
        ERFFireTest::make_circle(256, 10.0),
        options);

    EXPECT_GT(initial_remesh.stats.vertices_removed, 0U);

    FirePerimeter perimeter = std::move(initial_remesh.perimeter);
    const std::vector<FireVec2> canonical_vertices =
        perimeter.vertices_m();

    const auto normal_speed = [&spread] (
        const FireVec2&,
        const FireVec2& outward_normal,
        amrex::Real) -> amrex::Real
    {
        return ERFFire::richards_normal_speed_mps(
            spread.ellipse, outward_normal);
    };

    amrex::Real time_s = 0.0;
    for (int step = 0; step < 20; ++step) {
        perimeter = ERFFire::advance_perimeter_rk2(
            perimeter, time_s, 1.0, normal_speed);
        time_s += 1.0;

        auto remeshed = ERFFire::remesh_perimeter(
            perimeter, options);

        EXPECT_EQ(remeshed.stats.vertices_removed, 0U);
        EXPECT_EQ(remeshed.stats.vertices_added, 0U);

        perimeter = std::move(remeshed.perimeter);
    }

    ASSERT_EQ(perimeter.size(), canonical_vertices.size());
    for (std::size_t i = 0; i < perimeter.size(); ++i) {
        EXPECT_DOUBLE_EQ(
            static_cast<double>(perimeter.vertices_m()[i].x),
            static_cast<double>(canonical_vertices[i].x));
        EXPECT_DOUBLE_EQ(
            static_cast<double>(perimeter.vertices_m()[i].y),
            static_cast<double>(canonical_vertices[i].y));
    }
}

TEST(FireStandaloneGrowth, ExtinctionLeavesPerimeterExactlyStationary)
{
    const auto behavior = ERFFire::evaluate_rothermel(
        ERFFire::make_fm1_fuel_parameters(),
        RothermelInputs{0.12, 1.0, 0.20});
    const auto spread = ERFFire::make_richards_directional_spread(
        behavior, {1.0, 0.0}, {0.0, 1.0});

    FirePerimeter perimeter = ERFFireTest::make_circle(64, 10.0);
    const std::vector<FireVec2> initial_vertices = perimeter.vertices_m();

    const auto normal_speed = [&spread] (
        const FireVec2&,
        const FireVec2& outward_normal,
        amrex::Real) -> amrex::Real
    {
        return ERFFire::richards_normal_speed_mps(
            spread.ellipse, outward_normal);
    };

    amrex::Real time_s = 0.0;
    for (int step = 0; step < 20; ++step) {
        perimeter = ERFFire::advance_perimeter_rk2(
            perimeter, time_s, 1.0, normal_speed);
        time_s += 1.0;
    }

    ASSERT_EQ(perimeter.size(), initial_vertices.size());
    for (std::size_t i = 0; i < perimeter.size(); ++i) {
        EXPECT_DOUBLE_EQ(
            static_cast<double>(perimeter.vertices_m()[i].x),
            static_cast<double>(initial_vertices[i].x));
        EXPECT_DOUBLE_EQ(
            static_cast<double>(perimeter.vertices_m()[i].y),
            static_cast<double>(initial_vertices[i].y));
    }
}

} // namespace
