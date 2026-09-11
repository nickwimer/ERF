#include <ERF_FirePerimeterRemesher.H>
#include <ERF_VectorPerimeterPropagator.H>
#include "ERF_FireTestUtils.H"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace
{

using ERFFire::FireFront;
using ERFFire::FireFrontComponent;
using ERFFire::FireFrontRole;
using ERFFire::FirePerimeter;
using ERFFire::FirePerimeterRemeshOptions;
using ERFFire::FireVec2;
amrex::Real
minimum_edge_length_m (const FirePerimeter& perimeter)
{
    amrex::Real minimum = std::numeric_limits<amrex::Real>::max();
    const auto& vertices = perimeter.vertices_m();

    for (std::size_t i = 0; i < vertices.size(); ++i) {
        minimum = std::min(
            minimum,
            ERFFire::norm(vertices[(i + 1) % vertices.size()] - vertices[i]));
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
            ERFFire::norm(vertices[(i + 1) % vertices.size()] - vertices[i]));
    }

    return maximum;
}

amrex::Real
mean_radius_m (const FirePerimeter& perimeter)
{
    amrex::Real radius_sum = 0.0;
    for (const auto& vertex : perimeter.vertices_m()) {
        radius_sum += ERFFire::norm(vertex);
    }
    return radius_sum / static_cast<amrex::Real>(perimeter.size());
}

amrex::Real
maximum_radial_error_m (
    const FirePerimeter& perimeter,
    amrex::Real expected_radius_m)
{
    amrex::Real maximum_error = 0.0;
    for (const auto& vertex : perimeter.vertices_m()) {
        maximum_error = std::max(
            maximum_error,
            std::abs(ERFFire::norm(vertex) - expected_radius_m));
    }
    return maximum_error;
}

amrex::Real
point_to_segment_distance_m (
    const FireVec2& point,
    const FireVec2& start,
    const FireVec2& end)
{
    const FireVec2 segment = end - start;
    const amrex::Real length_squared = ERFFire::norm_squared(segment);
    const amrex::Real fraction = std::clamp(
        ERFFire::dot(point - start, segment) / length_squared,
        amrex::Real(0.0),
        amrex::Real(1.0));
    return ERFFire::norm(point - (start + fraction * segment));
}

amrex::Real
directed_vertex_to_segments_distance_m (
    const FirePerimeter& source,
    const FirePerimeter& target)
{
    amrex::Real maximum_distance = 0.0;
    const auto& target_vertices = target.vertices_m();

    for (const auto& point : source.vertices_m()) {
        amrex::Real minimum_distance =
            std::numeric_limits<amrex::Real>::max();

        for (std::size_t edge = 0; edge < target_vertices.size(); ++edge) {
            minimum_distance = std::min(
                minimum_distance,
                point_to_segment_distance_m(
                    point,
                    target_vertices[edge],
                    target_vertices[(edge + 1) % target_vertices.size()]));
        }

        maximum_distance = std::max(maximum_distance, minimum_distance);
    }

    return maximum_distance;
}

amrex::Real
symmetric_vertex_to_segments_distance_m (
    const FirePerimeter& a,
    const FirePerimeter& b)
{
    return std::max(
        directed_vertex_to_segments_distance_m(a, b),
        directed_vertex_to_segments_distance_m(b, a));
}

FireVec2
rotate_translate (
    const FireVec2& point,
    amrex::Real cosine,
    amrex::Real sine,
    const FireVec2& translation_m)
{
    return {
        cosine * point.x - sine * point.y + translation_m.x,
        sine * point.x + cosine * point.y + translation_m.y
    };
}

FirePerimeter
transform_perimeter (
    const FirePerimeter& perimeter,
    amrex::Real angle_rad,
    const FireVec2& translation_m)
{
    const amrex::Real cosine = std::cos(angle_rad);
    const amrex::Real sine = std::sin(angle_rad);
    std::vector<FireVec2> transformed;
    transformed.reserve(perimeter.size());

    for (const auto& vertex : perimeter.vertices_m()) {
        transformed.push_back(
            rotate_translate(vertex, cosine, sine, translation_m));
    }

    return FirePerimeter(std::move(transformed));
}

TEST(
    FireRemeshing,
    FrontRemeshPreservesRolesAreaAndAggregatesStatistics)
{
    const FireFront front(
        std::vector<FireFrontComponent>{
            {
                FireFrontRole::Outer,
                FirePerimeter({
                    {0.0, 0.0},
                    {4.0, 0.0},
                    {4.0, 4.0},
                    {0.0, 4.0}
                })
            },
            {
                FireFrontRole::Hole,
                FirePerimeter({
                    {1.0, 1.0},
                    {3.0, 1.0},
                    {3.0, 3.0},
                    {1.0, 3.0}
                })
            }
        });

    const amrex::Real burned_area_before =
        front.burned_area_m2();

    const auto result =
        ERFFire::remesh_front(
            front,
            FirePerimeterRemeshOptions{
                0.25,
                1.0,
                0.0
            });

    ASSERT_EQ(
        result.front.components().size(),
        2U);
    EXPECT_EQ(
        result.front.components()[0].role,
        FireFrontRole::Outer);
    EXPECT_EQ(
        result.front.components()[1].role,
        FireFrontRole::Hole);

    EXPECT_EQ(
        result.front.components()[0]
            .perimeter.size(),
        16U);
    EXPECT_EQ(
        result.front.components()[1]
            .perimeter.size(),
        8U);

    EXPECT_EQ(
        result.stats.vertices_removed,
        0U);
    EXPECT_EQ(
        result.stats.vertices_added,
        16U);

    EXPECT_DOUBLE_EQ(
        result.front.burned_area_m2(),
        burned_area_before);

    EXPECT_LE(
        maximum_edge_length_m(
            result.front.components()[0]
                .perimeter),
        1.0);
    EXPECT_LE(
        maximum_edge_length_m(
            result.front.components()[1]
                .perimeter),
        1.0);
}

TEST(FireRemeshing, StraightEdgeSubdivisionPreservesPolygonExactly)
{
    const FirePerimeter rectangle({
        {0.0, 0.0},
        {4.0, 0.0},
        {4.0, 2.0},
        {0.0, 2.0}
    });

    const auto result = ERFFire::remesh_perimeter(
        rectangle,
        FirePerimeterRemeshOptions{0.25, 0.75, 0.0});

    EXPECT_EQ(result.stats.vertices_removed, 0U);
    EXPECT_EQ(result.stats.vertices_added, 14U);
    EXPECT_EQ(result.perimeter.size(), 18U);
    EXPECT_DOUBLE_EQ(
        static_cast<double>(result.perimeter.area_m2()),
        static_cast<double>(rectangle.area_m2()));
    EXPECT_DOUBLE_EQ(
        static_cast<double>(result.perimeter.perimeter_length_m()),
        static_cast<double>(rectangle.perimeter_length_m()));
    EXPECT_LE(static_cast<double>(maximum_edge_length_m(result.perimeter)), 0.75);
}

TEST(FireRemeshing, RedundantCollinearVerticesCollapseWithoutGeometryChange)
{
    const FirePerimeter oversampled_rectangle({
        {0.0, 0.0},
        {0.5, 0.0},
        {1.0, 0.0},
        {1.5, 0.0},
        {2.0, 0.0},
        {2.0, 2.0},
        {0.0, 2.0}
    });

    const auto result = ERFFire::remesh_perimeter(
        oversampled_rectangle,
        FirePerimeterRemeshOptions{0.6, 2.0, 0.0});

    EXPECT_EQ(result.stats.vertices_removed, 3U);
    EXPECT_EQ(result.stats.vertices_added, 0U);
    EXPECT_EQ(result.perimeter.size(), 4U);
    EXPECT_DOUBLE_EQ(
        static_cast<double>(result.perimeter.area_m2()),
        static_cast<double>(oversampled_rectangle.area_m2()));
    EXPECT_DOUBLE_EQ(
        static_cast<double>(result.perimeter.perimeter_length_m()),
        static_cast<double>(oversampled_rectangle.perimeter_length_m()));
}

TEST(FireRemeshing, RigidTransformCommutesWithLocalRemeshing)
{
    const FirePerimeter original({
        {0.0, 0.0},
        {0.4, 0.0},
        {1.0, 0.0},
        {2.2, 0.3},
        {2.0, 1.8},
        {0.8, 2.1},
        {0.0, 1.6}
    });

    constexpr amrex::Real angle_rad = 0.713;
    const FireVec2 translation_m{7.5, -3.25};
    const FirePerimeterRemeshOptions options{0.5, 1.2, 0.03};

    const auto base = ERFFire::remesh_perimeter(original, options);
    const auto transformed = ERFFire::remesh_perimeter(
        transform_perimeter(original, angle_rad, translation_m),
        options);
    const FirePerimeter expected = transform_perimeter(
        base.perimeter, angle_rad, translation_m);

    ASSERT_EQ(transformed.perimeter.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(
            static_cast<double>(transformed.perimeter.vertices_m()[i].x),
            static_cast<double>(expected.vertices_m()[i].x),
            2.0e-14);
        EXPECT_NEAR(
            static_cast<double>(transformed.perimeter.vertices_m()[i].y),
            static_cast<double>(expected.vertices_m()[i].y),
            2.0e-14);
    }
}

TEST(FireRemeshing, ExpandingCircleMaintainsPhysicalSpacingAndAnalyticRadius)
{
    const FirePerimeterRemeshOptions options{0.6, 1.2, 0.03};
    FirePerimeter perimeter = ERFFire::remesh_perimeter(
        ERFFireTest::make_circle(256, 10.0), options).perimeter;

    const auto speed = [] (
        const FireVec2&,
        const FireVec2&,
        amrex::Real) -> amrex::Real
    {
        return 0.1;
    };

    ERFFireTest::write_snapshot("remesh_circle_t000.csv", perimeter);

    amrex::Real time_s = 0.0;
    for (int step = 1; step <= 100; ++step) {
        perimeter = ERFFire::advance_perimeter_rk2(
            perimeter, time_s, 1.0, speed);
        time_s += 1.0;
        perimeter = ERFFire::remesh_perimeter(perimeter, options).perimeter;

        if (step % 25 == 0) {
            std::ostringstream filename;
            filename << "remesh_circle_t"
                     << std::setw(3) << std::setfill('0') << step
                     << ".csv";
            ERFFireTest::write_snapshot(filename.str(), perimeter);
        }
    }

    EXPECT_GE(
        static_cast<double>(minimum_edge_length_m(perimeter)),
        static_cast<double>(options.min_edge_length_m) - 1.0e-12);
    EXPECT_LE(
        static_cast<double>(maximum_edge_length_m(perimeter)),
        static_cast<double>(options.max_edge_length_m) + 1.0e-12);
    EXPECT_NEAR(static_cast<double>(mean_radius_m(perimeter)), 20.0, 0.01);
    EXPECT_LT(static_cast<double>(maximum_radial_error_m(perimeter, 20.0)), 0.02);
}

TEST(FireRemeshing, SyntheticAnisotropyTracksHighResolutionReference)
{
    const FirePerimeterRemeshOptions options{0.6, 1.2, 0.03};
    FirePerimeter remeshed = ERFFire::remesh_perimeter(
        ERFFireTest::make_circle(256, 10.0), options).perimeter;
    FirePerimeter reference = ERFFireTest::make_circle(1024, 10.0);

    const auto speed = [] (
        const FireVec2&,
        const FireVec2& outward_normal,
        amrex::Real) -> amrex::Real
    {
        return amrex::Real(0.10)
            * (amrex::Real(1.0)
               + amrex::Real(0.25) * outward_normal.x
               + amrex::Real(0.10) * outward_normal.y);
    };

    amrex::Real time_s = 0.0;
    constexpr amrex::Real dt_s = 0.5;
    for (int step = 0; step < 200; ++step) {
        remeshed = ERFFire::advance_perimeter_rk2(
            remeshed, time_s, dt_s, speed);
        remeshed = ERFFire::remesh_perimeter(remeshed, options).perimeter;

        reference = ERFFire::advance_perimeter_rk2(
            reference, time_s, dt_s, speed);
        time_s += dt_s;
    }

    ERFFireTest::write_snapshot("remesh_nonuniform_t100.csv", remeshed);
    ERFFireTest::write_snapshot("reference_nonuniform_t100.csv", reference);

    const amrex::Real geometric_distance_m =
        symmetric_vertex_to_segments_distance_m(remeshed, reference);
    const amrex::Real relative_area_error =
        std::abs(remeshed.area_m2() - reference.area_m2())
        / reference.area_m2();
    const amrex::Real relative_perimeter_error =
        std::abs(
            remeshed.perimeter_length_m()
            - reference.perimeter_length_m())
        / reference.perimeter_length_m();

    EXPECT_LT(static_cast<double>(geometric_distance_m), 0.03);
    EXPECT_LT(static_cast<double>(relative_area_error), 0.002);
    EXPECT_LT(static_cast<double>(relative_perimeter_error), 0.001);
}

TEST(FireRemeshing, NarrowBacktrackingSpikeCollapsesWithoutShortEdges)
{
    const FirePerimeter hairpin({
        {0.0, 0.0},
        {1.0, 0.0},
        {2.0, 0.01},
        {1.5, 0.02},
        {2.0, 1.0},
        {0.0, 1.0}
    });

    const FirePerimeterRemeshOptions options{
        0.25,
        2.0,
        0.02
    };

    const amrex::Real incoming_edge_m =
        ERFFire::norm(
            hairpin.vertices_m()[2]
            - hairpin.vertices_m()[1]);
    const amrex::Real outgoing_edge_m =
        ERFFire::norm(
            hairpin.vertices_m()[3]
            - hairpin.vertices_m()[2]);

    ASSERT_GT(
        static_cast<double>(incoming_edge_m),
        static_cast<double>(options.min_edge_length_m));
    ASSERT_GT(
        static_cast<double>(outgoing_edge_m),
        static_cast<double>(options.min_edge_length_m));

    const auto result =
        ERFFire::remesh_perimeter(hairpin, options);

    EXPECT_EQ(result.stats.vertices_removed, 1U);
    EXPECT_EQ(result.stats.vertices_added, 0U);
    ASSERT_EQ(result.perimeter.size(), 5U);

    bool retained_spike_tip = false;
    for (const FireVec2& vertex : result.perimeter.vertices_m()) {
        if (std::abs(vertex.x - amrex::Real(2.0)) < amrex::Real(1.0e-14)
            && std::abs(vertex.y - amrex::Real(0.01)) < amrex::Real(1.0e-14)) {
            retained_spike_tip = true;
        }
    }

    EXPECT_FALSE(retained_spike_tip);
    EXPECT_GT(
        static_cast<double>(result.perimeter.area_m2()),
        0.0);
}

TEST(FireRemeshing, SharpCornerIsNotCollapsedPastChordTolerance)
{
    const FirePerimeter sharp_corner({
        {0.0, 0.0},
        {0.1, 0.0},
        {0.2, 1.0},
        {2.0, 0.0},
        {2.0, 2.0},
        {0.0, 2.0}
    });

    const auto result = ERFFire::remesh_perimeter(
        sharp_corner,
        FirePerimeterRemeshOptions{0.15, 2.0, 0.01});

    EXPECT_EQ(result.stats.vertices_removed, 0U);

    bool found_sharp_vertex = false;
    for (const auto& vertex : result.perimeter.vertices_m()) {
        if (std::abs(vertex.x - 0.1) < 1.0e-14
            && std::abs(vertex.y) < 1.0e-14) {
            found_sharp_vertex = true;
        }
    }
    EXPECT_TRUE(found_sharp_vertex);
}

TEST(FireRemeshing, RejectsInvalidPhysicalSpacingControls)
{
    const FirePerimeter perimeter = ERFFireTest::make_circle(32, 10.0);

    EXPECT_THROW(
        (void) ERFFire::remesh_perimeter(
            perimeter, FirePerimeterRemeshOptions{0.0, 1.0, 0.01}),
        std::invalid_argument);
    EXPECT_THROW(
        (void) ERFFire::remesh_perimeter(
            perimeter, FirePerimeterRemeshOptions{0.6, 1.0, 0.01}),
        std::invalid_argument);
    EXPECT_THROW(
        (void) ERFFire::remesh_perimeter(
            perimeter, FirePerimeterRemeshOptions{0.4, 1.0, -0.01}),
        std::invalid_argument);
}

} // namespace
