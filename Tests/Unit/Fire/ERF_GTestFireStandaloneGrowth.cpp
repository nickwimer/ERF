#include <ERF_FirePerimeterRemesher.H>
#include <ERF_RichardsDirectionalSpread.H>
#include <ERF_RothermelFuel.H>
#include <ERF_VectorPerimeterPropagator.H>

#include "ERF_FireTestUtils.H"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace
{

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

constexpr amrex::Real initial_wavelet_age_s = amrex::Real(200.0);
constexpr amrex::Real integration_duration_s = amrex::Real(100.0);
constexpr amrex::Real dt_s = amrex::Real(1.0);
constexpr int integration_steps = 100;

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
make_exact_reference_wavelet (
    std::size_t vertex_count,
    amrex::Real age_s,
    const FireVec2& heading)
{
    std::vector<FireVec2> vertices;
    vertices.reserve(vertex_count);

    const FireVec2 flank_direction = left_perpendicular(heading);
    const FireVec2 center =
        heading * (reference_center_translation_rate_mps * age_s);

    const amrex::Real semi_major_m =
        reference_semi_major_rate_mps * age_s;
    const amrex::Real semi_minor_m =
        reference_flank_ros_mps * age_s;

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
    bool write_reference_snapshots = false)
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

        remeshed = ERFFire::remesh_perimeter(perimeter, options);
        perimeter = std::move(remeshed.perimeter);
        vertices_removed += remeshed.stats.vertices_removed;
        vertices_added += remeshed.stats.vertices_added;

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
