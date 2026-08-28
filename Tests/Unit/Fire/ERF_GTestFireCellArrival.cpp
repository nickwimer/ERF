#include <ERF_FireCellArrival.H>
#include <ERF_FireFrontTopology.H>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{

using ERFFire::FireCartesianCell2D;
using ERFFire::FireFront;
using ERFFire::FirePerimeter;
using ERFFire::FireVec2;

FirePerimeter
make_rectangle (
    amrex::Real xlo_m,
    amrex::Real xhi_m,
    amrex::Real ylo_m,
    amrex::Real yhi_m)
{
    return FirePerimeter(std::vector<FireVec2>{
        {xlo_m, ylo_m},
        {xhi_m, ylo_m},
        {xhi_m, yhi_m},
        {xlo_m, yhi_m}
    });
}

FirePerimeter
make_topology_event_start()
{
    return FirePerimeter(
        std::vector<FireVec2>{
            {0.0, 0.0},
            {10.0, 0.0},
            {10.0, 2.0},
            {3.0, 2.0},
            {3.0, 8.0},
            {7.0, 8.0},
            {7.0, 2.2},
            {8.0, 2.2},
            {9.0, 2.2},
            {9.0, 8.0},
            {10.0, 8.0},
            {10.0, 10.0},
            {0.0, 10.0}
        });
}

std::vector<FireVec2>
make_topology_event_vertices(
    const FirePerimeter& start)
{
    std::vector<FireVec2> vertices =
        start.vertices_m();

    // Move the complete left outer edge outward so the arrival regression
    // has a broad, linearly growing cell intersection. This avoids making
    // the bisection-tolerance test depend on near-degenerate triangular-area
    // resolution at the pinch itself.
    vertices[0].x = amrex::Real(-1.0);
    vertices[12].x = amrex::Real(-1.0);

    vertices[7] = {
        amrex::Real(8.0),
        amrex::Real(2.0)
    };
    return vertices;
}

FireFront
make_topology_event_front(
    const std::vector<FireVec2>& event_vertices)
{
    return ERFFire::split_perimeter_at_pinch(
        event_vertices,
        ERFFire::FirePerimeterPinch{
            2U,
            amrex::Real(2.0 / 7.0),
            7U,
            amrex::Real(0.0)
        });
}

TEST(FireCellArrival, AlreadyCoveredCellArrivesAtSweepStartExactly)
{
    const auto start = make_rectangle(-1.0, 0.5, -1.0, 2.0);
    const auto end = make_rectangle(-1.0, 2.0, -1.0, 2.0);
    const FireCartesianCell2D cell{0.0, 1.0, 0.0, 1.0};

    const auto result =
        ERFFire::fire_cell_first_arrival_time_linear_sweep(
            start, end, cell, 10.0, 20.0, 1.0e-9);

    ASSERT_TRUE(result.arrived);
    EXPECT_DOUBLE_EQ(
        static_cast<double>(result.arrival_time_s),
        10.0);
}

TEST(FireCellArrival, NoEndCoverageMeansNoArrivalInMonotoneSweep)
{
    const auto start = make_rectangle(-3.0, -2.0, -1.0, 2.0);
    const auto end = make_rectangle(-3.0, -0.5, -1.0, 2.0);
    const FireCartesianCell2D cell{0.0, 1.0, 0.0, 1.0};

    const auto result =
        ERFFire::fire_cell_first_arrival_time_linear_sweep(
            start, end, cell, 10.0, 20.0, 1.0e-9);

    EXPECT_FALSE(result.arrived);
}

TEST(FireCellArrival, FinalBoundaryTouchAloneIsNotArrival)
{
    const auto start = make_rectangle(-3.0, -2.0, -1.0, 2.0);
    const auto end = make_rectangle(-3.0, 0.0, -1.0, 2.0);
    const FireCartesianCell2D cell{0.0, 1.0, 0.0, 1.0};

    const auto result =
        ERFFire::fire_cell_first_arrival_time_linear_sweep(
            start, end, cell, 0.0, 4.0, 1.0e-10);

    EXPECT_FALSE(result.arrived);
}

TEST(FireCellArrival, PlanarLinearSweepMatchesAnalyticCrossingTime)
{
    const auto start = make_rectangle(-3.0, -1.0, -1.0, 2.0);
    const auto end = make_rectangle(-3.0, 2.0, -1.0, 2.0);
    const FireCartesianCell2D cell{0.0, 1.0, 0.0, 1.0};

    const auto result =
        ERFFire::fire_cell_first_arrival_time_linear_sweep(
            start, end, cell, 10.0, 16.0, 1.0e-9);

    ASSERT_TRUE(result.arrived);
    EXPECT_GE(
        static_cast<double>(result.arrival_time_s),
        12.0);
    EXPECT_LT(
        static_cast<double>(result.arrival_time_s - 12.0),
        1.0e-9);
}

TEST(FireCellArrival, SplittingPlanarSweepPreservesArrivalTime)
{
    const auto start = make_rectangle(-3.0, -1.0, -1.0, 2.0);
    const auto middle = make_rectangle(-3.0, 0.5, -1.0, 2.0);
    const auto end = make_rectangle(-3.0, 2.0, -1.0, 2.0);
    const FireCartesianCell2D cell{0.0, 1.0, 0.0, 1.0};

    const auto whole =
        ERFFire::fire_cell_first_arrival_time_linear_sweep(
            start, end, cell, 10.0, 16.0, 1.0e-9);

    const auto first_half =
        ERFFire::fire_cell_first_arrival_time_linear_sweep(
            start, middle, cell, 10.0, 13.0, 1.0e-9);

    ASSERT_TRUE(whole.arrived);
    ASSERT_TRUE(first_half.arrived);
    EXPECT_NEAR(
        static_cast<double>(whole.arrival_time_s),
        static_cast<double>(first_half.arrival_time_s),
        1.0e-9);
}

TEST(FireCellArrival, NearStartCrossingHonorsToleranceBeyond128Bisections)
{
    const amrex::Real tiny =
        std::ldexp(amrex::Real(1.0), -200);
    const amrex::Real tolerance_s =
        std::ldexp(amrex::Real(1.0), -220);

    const auto start = make_rectangle(
        -1.0, -tiny, -1.0, 2.0);
    const auto end = make_rectangle(
        -1.0, 1.0, -1.0, 2.0);
    const FireCartesianCell2D cell{
        0.0, 1.0, 0.0, 1.0};

    // The right face is x(alpha) = -2^-200 + alpha because
    // 1 + 2^-200 rounds to 1 in double precision. The zero-area
    // contact therefore occurs exactly at alpha = t = 2^-200.
    const auto result =
        ERFFire::fire_cell_first_arrival_time_linear_sweep(
            start,
            end,
            cell,
            0.0,
            1.0,
            tolerance_s);

    ASSERT_TRUE(result.arrived);
    EXPECT_GE(
        static_cast<double>(result.arrival_time_s),
        static_cast<double>(tiny));
    EXPECT_LE(
        static_cast<double>(
            result.arrival_time_s - tiny),
        static_cast<double>(tolerance_s));
}

TEST(FireCellArrival, TranslationAndClockShiftPreserveRelativeArrival)
{
    const FireVec2 offset{1.0e6, -2.0e6};

    const auto base_start = make_rectangle(-3.0, -1.0, -1.0, 2.0);
    const auto base_end = make_rectangle(-3.0, 2.0, -1.0, 2.0);

    auto translate = [&] (const FirePerimeter& perimeter)
    {
        std::vector<FireVec2> vertices;
        vertices.reserve(perimeter.size());
        for (const auto& vertex : perimeter.vertices_m()) {
            vertices.push_back(vertex + offset);
        }
        return FirePerimeter(std::move(vertices));
    };

    const FireCartesianCell2D base_cell{0.0, 1.0, 0.0, 1.0};
    const FireCartesianCell2D shifted_cell{
        base_cell.xlo_m + offset.x,
        base_cell.xhi_m + offset.x,
        base_cell.ylo_m + offset.y,
        base_cell.yhi_m + offset.y};

    const auto base =
        ERFFire::fire_cell_first_arrival_time_linear_sweep(
            base_start,
            base_end,
            base_cell,
            10.0,
            16.0,
            1.0e-8);

    const auto shifted =
        ERFFire::fire_cell_first_arrival_time_linear_sweep(
            translate(base_start),
            translate(base_end),
            shifted_cell,
            1010.0,
            1016.0,
            1.0e-8);

    ASSERT_TRUE(base.arrived);
    ASSERT_TRUE(shifted.arrived);
    EXPECT_NEAR(
        static_cast<double>(shifted.arrival_time_s - 1000.0),
        static_cast<double>(base.arrival_time_s),
        2.0e-8);
}

TEST(FireCellArrival, TighterToleranceTightensAnalyticUpperBound)
{
    const auto start = make_rectangle(-3.0, -1.0, -1.0, 2.0);
    const auto end = make_rectangle(-3.0, 2.0, -1.0, 2.0);
    const FireCartesianCell2D cell{0.0, 1.0, 0.0, 1.0};

    const auto coarse =
        ERFFire::fire_cell_first_arrival_time_linear_sweep(
            start, end, cell, 10.0, 16.0, 1.0e-3);
    const auto fine =
        ERFFire::fire_cell_first_arrival_time_linear_sweep(
            start, end, cell, 10.0, 16.0, 1.0e-9);

    ASSERT_TRUE(coarse.arrived);
    ASSERT_TRUE(fine.arrived);

    const amrex::Real coarse_error =
        coarse.arrival_time_s - amrex::Real(12.0);
    const amrex::Real fine_error =
        fine.arrival_time_s - amrex::Real(12.0);

    EXPECT_GE(static_cast<double>(coarse_error), 0.0);
    EXPECT_LT(static_cast<double>(coarse_error), 1.0e-3);
    EXPECT_GE(static_cast<double>(fine_error), 0.0);
    EXPECT_LT(static_cast<double>(fine_error), 1.0e-9);
    EXPECT_LE(
        static_cast<double>(fine_error),
        static_cast<double>(coarse_error));
}

TEST(
    FireCellArrival,
    TopologyEventSweepFindsArrivalBeforePinch)
{
    const FirePerimeter start =
        make_topology_event_start();
    const std::vector<FireVec2> event_vertices =
        make_topology_event_vertices(start);
    const FireFront event_front =
        make_topology_event_front(event_vertices);

    const FireCartesianCell2D cell{
        -0.75,
        -0.5,
        9.0,
        9.5
    };

    const auto result =
        ERFFire::
            fire_cell_first_arrival_time_topology_event_sweep(
                start,
                event_vertices,
                event_front,
                cell,
                amrex::Real(10.0),
                amrex::Real(12.0),
                amrex::Real(1.0e-9));

    ASSERT_TRUE(result.arrived);

    EXPECT_GE(
        static_cast<double>(
            result.arrival_time_s),
        11.0);
    EXPECT_LT(
        static_cast<double>(
            result.arrival_time_s
            - amrex::Real(11.0)),
        1.0e-9);
}

TEST(
    FireCellArrival,
    TopologyEventSweepDoesNotIgniteUnburnedPocket)
{
    const FirePerimeter start =
        make_topology_event_start();
    const std::vector<FireVec2> event_vertices =
        make_topology_event_vertices(start);
    const FireFront event_front =
        make_topology_event_front(event_vertices);

    const FireCartesianCell2D pocket_cell{
        4.0,
        4.5,
        4.0,
        4.5
    };

    const auto result =
        ERFFire::
            fire_cell_first_arrival_time_topology_event_sweep(
                start,
                event_vertices,
                event_front,
                pocket_cell,
                amrex::Real(10.0),
                amrex::Real(12.0),
                amrex::Real(1.0e-9));

    EXPECT_FALSE(result.arrived);
}

TEST(FireCellArrival, RejectsInvalidSweepInputs)
{
    const auto four_vertices = make_rectangle(-3.0, -1.0, -1.0, 2.0);
    const FirePerimeter three_vertices(
        std::vector<FireVec2>{
            {-3.0, -1.0},
            {-1.0,  0.0},
            {-3.0,  2.0}});
    const FireCartesianCell2D cell{0.0, 1.0, 0.0, 1.0};

    EXPECT_THROW(
        (void)ERFFire::fire_cell_first_arrival_time_linear_sweep(
            four_vertices,
            three_vertices,
            cell,
            0.0,
            1.0,
            1.0e-6),
        std::invalid_argument);

    const amrex::Real nan =
        std::numeric_limits<amrex::Real>::quiet_NaN();
    const amrex::Real max_value =
        std::numeric_limits<amrex::Real>::max();

    EXPECT_THROW(
        (void)ERFFire::fire_cell_first_arrival_time_linear_sweep(
            four_vertices,
            four_vertices,
            cell,
            -max_value,
            max_value,
            max_value),
        std::overflow_error);

    for (const auto& times : {
            std::vector<amrex::Real>{nan, 1.0, 1.0e-6},
            std::vector<amrex::Real>{0.0, nan, 1.0e-6},
            std::vector<amrex::Real>{0.0, 1.0, nan},
            std::vector<amrex::Real>{1.0, 1.0, 1.0e-6},
            std::vector<amrex::Real>{2.0, 1.0, 1.0e-6},
            std::vector<amrex::Real>{0.0, 1.0, 0.0},
            std::vector<amrex::Real>{0.0, 1.0, -1.0e-6},
            std::vector<amrex::Real>{0.0, 1.0, 2.0}}) {
        EXPECT_THROW(
            (void)ERFFire::fire_cell_first_arrival_time_linear_sweep(
                four_vertices,
                four_vertices,
                cell,
                times[0],
                times[1],
                times[2]),
            std::invalid_argument);
    }
}

} // namespace
