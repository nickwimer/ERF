#include <ERF_FireFirstArrivalRaster.H>
#include <ERF_FireFront.H>
#include <ERF_FireFrontTopology.H>
#include <ERF_FirePerimeter.H>
#include <ERF_FireTypes.H>

#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{

using ERFFire::FireFirstArrivalRaster;
using ERFFire::FireFront;
using ERFFire::FireFrontComponent;
using ERFFire::FireFrontRole;
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

TEST(FireFirstArrivalRaster, ConstructsUnsetHistoryOnSharedGeometry)
{
    FireFirstArrivalRaster raster({
        3,
        2,
        0.1,
        -0.2,
        0.5,
        1.0
    });

    EXPECT_EQ(raster.cell_count(), 6U);
    EXPECT_EQ(raster.arrived_cell_count(), 0U);

    for (std::size_t j = 0; j < 2; ++j) {
        for (std::size_t i = 0; i < 3; ++i) {
            EXPECT_FALSE(raster.has_arrived(i, j));
            EXPECT_THROW(
                (void)raster.first_arrival_time_s(i, j),
                std::logic_error);
        }
    }

    const auto cell = raster.cell_bounds(2, 1);
    EXPECT_DOUBLE_EQ(
        static_cast<double>(cell.xlo_m),
        1.1);
    EXPECT_DOUBLE_EQ(
        static_cast<double>(cell.xhi_m),
        1.6);
    EXPECT_DOUBLE_EQ(
        static_cast<double>(cell.ylo_m),
        0.8);
    EXPECT_DOUBLE_EQ(
        static_cast<double>(cell.yhi_m),
        1.8);
}

TEST(FireFirstArrivalRaster, InitialConditionSeedsIgnitionAndPinsFirstSweepStart)
{
    FireFirstArrivalRaster raster({
        3,
        1,
        0.0,
        0.0,
        1.0,
        1.0
    });

    const auto ignition =
        make_rectangle(-2.0, 1.0, -1.0, 2.0);

    const auto seeded =
        raster.initialize_from_perimeter(
            ignition, 5.0);

    EXPECT_EQ(seeded.newly_arrived_cell_count, 1U);
    EXPECT_EQ(seeded.arrived_cell_count, 1U);
    ASSERT_TRUE(raster.has_arrived(0, 0));
    EXPECT_FALSE(raster.has_arrived(1, 0));
    EXPECT_FALSE(raster.has_arrived(2, 0));
    EXPECT_DOUBLE_EQ(
        static_cast<double>(
            raster.first_arrival_time_s(0, 0)),
        5.0);

    EXPECT_THROW(
        (void)raster.initialize_from_perimeter(
            ignition, 5.0),
        std::logic_error);

    const auto end =
        make_rectangle(-2.0, 2.5, -1.0, 2.0);

    EXPECT_THROW(
        (void)raster.update_from_sweep(
            ignition,
            end,
            5.5,
            6.5,
            1.0e-9),
        std::invalid_argument);

    EXPECT_EQ(raster.arrived_cell_count(), 1U);
    EXPECT_DOUBLE_EQ(
        static_cast<double>(
            raster.first_arrival_time_s(0, 0)),
        5.0);

    const auto advanced =
        raster.update_from_sweep(
            ignition,
            end,
            5.0,
            6.0,
            1.0e-9);

    EXPECT_GT(advanced.newly_arrived_cell_count, 0U);
    EXPECT_GT(advanced.arrived_cell_count, 1U);
    EXPECT_DOUBLE_EQ(
        static_cast<double>(
            raster.first_arrival_time_s(0, 0)),
        5.0);
}

TEST(FireFirstArrivalRaster, AlreadyCoveredCellsRecordSweepStartExactly)
{
    FireFirstArrivalRaster raster({
        2,
        1,
        0.0,
        0.0,
        1.0,
        1.0
    });

    const auto start =
        make_rectangle(-2.0, 1.5, -1.0, 2.0);
    const auto end =
        make_rectangle(-2.0, 2.0, -1.0, 2.0);

    const auto update =
        raster.update_from_sweep(
            start,
            end,
            10.0,
            11.0,
            1.0e-9);

    EXPECT_EQ(update.newly_arrived_cell_count, 2U);
    EXPECT_EQ(update.arrived_cell_count, 2U);

    EXPECT_DOUBLE_EQ(
        static_cast<double>(
            raster.first_arrival_time_s(0, 0)),
        10.0);
    EXPECT_DOUBLE_EQ(
        static_cast<double>(
            raster.first_arrival_time_s(1, 0)),
        10.0);
}

TEST(FireFirstArrivalRaster, PlanarSweepStoresAnalyticArrivalOrdering)
{
    FireFirstArrivalRaster raster({
        4,
        1,
        0.0,
        0.0,
        1.0,
        1.0
    });

    const auto start =
        make_rectangle(-3.0, -1.0, -1.0, 2.0);
    const auto end =
        make_rectangle(-3.0, 4.0, -1.0, 2.0);

    const auto update =
        raster.update_from_sweep(
            start,
            end,
            10.0,
            15.0,
            1.0e-9);

    EXPECT_EQ(update.newly_arrived_cell_count, 4U);
    EXPECT_EQ(update.arrived_cell_count, 4U);

    for (std::size_t i = 0; i < 4; ++i) {
        ASSERT_TRUE(raster.has_arrived(i, 0));

        const amrex::Real exact_time_s =
            amrex::Real(11.0)
            + static_cast<amrex::Real>(i);
        const amrex::Real stored_time_s =
            raster.first_arrival_time_s(i, 0);

        EXPECT_GE(
            static_cast<double>(stored_time_s),
            static_cast<double>(exact_time_s));
        EXPECT_LT(
            static_cast<double>(
                stored_time_s - exact_time_s),
            1.0e-9);

        if (i > 0) {
            EXPECT_GT(
                static_cast<double>(stored_time_s),
                static_cast<double>(
                    raster.first_arrival_time_s(i - 1, 0)));
        }
    }
}

TEST(FireFirstArrivalRaster, FirstArrivalIsImmutableAcrossLaterSweeps)
{
    FireFirstArrivalRaster raster({
        2,
        1,
        0.0,
        0.0,
        1.0,
        1.0
    });

    const auto first_start =
        make_rectangle(-3.0, -1.0, -1.0, 2.0);
    const auto first_end =
        make_rectangle(-3.0, 0.5, -1.0, 2.0);

    const auto first =
        raster.update_from_sweep(
            first_start,
            first_end,
            0.0,
            3.0,
            1.0e-9);

    ASSERT_EQ(first.newly_arrived_cell_count, 1U);
    ASSERT_TRUE(raster.has_arrived(0, 0));
    ASSERT_FALSE(raster.has_arrived(1, 0));

    const amrex::Real original_time_s =
        raster.first_arrival_time_s(0, 0);

    const auto second_start =
        make_rectangle(-3.0, 0.5, -1.0, 2.0);
    const auto second_end =
        make_rectangle(-3.0, 2.0, -1.0, 2.0);

    const auto second =
        raster.update_from_sweep(
            second_start,
            second_end,
            3.0,
            6.0,
            1.0e-9);

    EXPECT_EQ(second.newly_arrived_cell_count, 1U);
    EXPECT_EQ(second.arrived_cell_count, 2U);

    EXPECT_DOUBLE_EQ(
        static_cast<double>(
            raster.first_arrival_time_s(0, 0)),
        static_cast<double>(original_time_s));

    EXPECT_GE(
        static_cast<double>(
            raster.first_arrival_time_s(1, 0)),
        4.0);
}

TEST(FireFirstArrivalRaster, SplitSweepMatchesSingleSweepHistory)
{
    FireFirstArrivalRaster whole({
        4,
        1,
        0.0,
        0.0,
        1.0,
        1.0
    });
    FireFirstArrivalRaster split(
        whole.geometry());

    const auto start =
        make_rectangle(-3.0, -1.0, -1.0, 2.0);
    const auto middle =
        make_rectangle(-3.0, 1.5, -1.0, 2.0);
    const auto end =
        make_rectangle(-3.0, 4.0, -1.0, 2.0);

    (void)whole.update_from_sweep(
        start,
        end,
        10.0,
        15.0,
        1.0e-9);

    (void)split.update_from_sweep(
        start,
        middle,
        10.0,
        12.5,
        1.0e-9);
    (void)split.update_from_sweep(
        middle,
        end,
        12.5,
        15.0,
        1.0e-9);

    EXPECT_EQ(
        split.arrived_cell_count(),
        whole.arrived_cell_count());

    for (std::size_t i = 0; i < 4; ++i) {
        ASSERT_TRUE(whole.has_arrived(i, 0));
        ASSERT_TRUE(split.has_arrived(i, 0));
        EXPECT_NEAR(
            static_cast<double>(
                split.first_arrival_time_s(i, 0)),
            static_cast<double>(
                whole.first_arrival_time_s(i, 0)),
            1.0e-9);
    }
}

TEST(FireFirstArrivalRaster, RejectsNoncontiguousSweepsWithoutChangingHistory)
{
    FireFirstArrivalRaster raster({
        3,
        1,
        0.0,
        0.0,
        1.0,
        1.0
    });

    const auto first_start =
        make_rectangle(-3.0, -1.0, -1.0, 2.0);
    const auto first_end =
        make_rectangle(-3.0, 0.5, -1.0, 2.0);

    (void)raster.update_from_sweep(
        first_start,
        first_end,
        0.0,
        1.0,
        1.0e-9);

    ASSERT_TRUE(raster.has_arrived(0, 0));
    ASSERT_FALSE(raster.has_arrived(1, 0));
    ASSERT_FALSE(raster.has_arrived(2, 0));

    const amrex::Real stored_time_s =
        raster.first_arrival_time_s(0, 0);
    const std::size_t stored_count =
        raster.arrived_cell_count();

    const auto later_end =
        make_rectangle(-3.0, 2.5, -1.0, 2.0);

    EXPECT_THROW(
        (void)raster.update_from_sweep(
            first_end,
            later_end,
            2.0,
            3.0,
            1.0e-9),
        std::invalid_argument);

    EXPECT_THROW(
        (void)raster.update_from_sweep(
            first_end,
            later_end,
            0.5,
            1.5,
            1.0e-9),
        std::invalid_argument);

    EXPECT_EQ(
        raster.arrived_cell_count(),
        stored_count);
    EXPECT_TRUE(raster.has_arrived(0, 0));
    EXPECT_FALSE(raster.has_arrived(1, 0));
    EXPECT_FALSE(raster.has_arrived(2, 0));
    EXPECT_DOUBLE_EQ(
        static_cast<double>(
            raster.first_arrival_time_s(0, 0)),
        static_cast<double>(stored_time_s));

    const auto contiguous =
        raster.update_from_sweep(
            first_end,
            later_end,
            1.0,
            2.0,
            1.0e-9);

    EXPECT_GT(contiguous.newly_arrived_cell_count, 0U);
    EXPECT_GT(
        contiguous.arrived_cell_count,
        stored_count);
}

TEST(FireFirstArrivalRaster, StationarySweepCreatesNoAdditionalHistory)
{
    FireFirstArrivalRaster raster({
        3,
        1,
        0.0,
        0.0,
        1.0,
        1.0
    });

    const auto perimeter =
        make_rectangle(-2.0, 1.25, -1.0, 2.0);

    const auto first =
        raster.update_from_sweep(
            perimeter,
            perimeter,
            5.0,
            6.0,
            1.0e-9);

    ASSERT_EQ(first.newly_arrived_cell_count, 2U);
    ASSERT_EQ(first.arrived_cell_count, 2U);

    const amrex::Real first_time_0 =
        raster.first_arrival_time_s(0, 0);
    const amrex::Real first_time_1 =
        raster.first_arrival_time_s(1, 0);

    const auto second =
        raster.update_from_sweep(
            perimeter,
            perimeter,
            6.0,
            7.0,
            1.0e-9);

    EXPECT_EQ(second.newly_arrived_cell_count, 0U);
    EXPECT_EQ(second.arrived_cell_count, 2U);
    EXPECT_FALSE(raster.has_arrived(2, 0));

    EXPECT_DOUBLE_EQ(
        static_cast<double>(
            raster.first_arrival_time_s(0, 0)),
        static_cast<double>(first_time_0));
    EXPECT_DOUBLE_EQ(
        static_cast<double>(
            raster.first_arrival_time_s(1, 0)),
        static_cast<double>(first_time_1));
}

TEST(FireFirstArrivalRaster, FailedSweepUpdateIsTransactional)
{
    FireFirstArrivalRaster raster({
        2,
        1,
        0.0,
        0.0,
        1.0,
        1.0
    });

    const auto initial_start =
        make_rectangle(-3.0, -1.0, -1.0, 2.0);
    const auto initial_end =
        make_rectangle(-3.0, 0.5, -1.0, 2.0);

    (void)raster.update_from_sweep(
        initial_start,
        initial_end,
        0.0,
        1.0,
        1.0e-9);

    ASSERT_TRUE(raster.has_arrived(0, 0));
    ASSERT_FALSE(raster.has_arrived(1, 0));

    const amrex::Real stored_time_s =
        raster.first_arrival_time_s(0, 0);
    const std::size_t stored_count =
        raster.arrived_cell_count();

    const FirePerimeter mismatched_end(
        std::vector<FireVec2>{
            {-3.0, -1.0},
            { 2.0,  0.0},
            {-3.0,  2.0}});

    EXPECT_THROW(
        (void)raster.update_from_sweep(
            initial_end,
            mismatched_end,
            1.0,
            2.0,
            1.0e-9),
        std::invalid_argument);

    EXPECT_EQ(
        raster.arrived_cell_count(),
        stored_count);
    EXPECT_TRUE(raster.has_arrived(0, 0));
    EXPECT_FALSE(raster.has_arrived(1, 0));
    EXPECT_DOUBLE_EQ(
        static_cast<double>(
            raster.first_arrival_time_s(0, 0)),
        static_cast<double>(stored_time_s));
}

TEST(FireFirstArrivalRaster, RejectsInvalidIndicesAndUnarrivedTimeQuery)
{
    FireFirstArrivalRaster raster({
        2,
        3,
        0.0,
        0.0,
        1.0,
        1.0
    });

    EXPECT_THROW(
        (void)raster.cell_bounds(2, 0),
        std::out_of_range);
    EXPECT_THROW(
        (void)raster.has_arrived(0, 3),
        std::out_of_range);
    EXPECT_THROW(
        (void)raster.first_arrival_time_s(1, 2),
        std::logic_error);
}

TEST(
    FireFirstArrivalRaster,
    FrontInitializationExcludesUnburnedHole)
{
    FireFirstArrivalRaster raster({
        4,
        4,
        0.0,
        0.0,
        1.0,
        1.0
    });

    const FireFront front(
        std::vector<FireFrontComponent>{
            {
                FireFrontRole::Outer,
                make_rectangle(
                    0.0, 4.0,
                    0.0, 4.0)
            },
            {
                FireFrontRole::Hole,
                make_rectangle(
                    1.0, 3.0,
                    1.0, 3.0)
            }
        });

    const auto initialized =
        raster.initialize_from_front(
            front,
            amrex::Real(7.25));

    EXPECT_EQ(
        initialized.newly_arrived_cell_count,
        12U);
    EXPECT_EQ(
        initialized.arrived_cell_count,
        12U);
    EXPECT_EQ(
        raster.arrived_cell_count(),
        12U);

    EXPECT_TRUE(
        raster.has_initial_condition());
    EXPECT_DOUBLE_EQ(
        static_cast<double>(
            raster.initial_condition_time_s()),
        7.25);

    for (std::size_t j = 0; j < 4; ++j) {
        for (std::size_t i = 0; i < 4; ++i) {
            const bool inside_hole =
                i >= 1 && i <= 2
                && j >= 1 && j <= 2;

            EXPECT_EQ(
                raster.has_arrived(i, j),
                !inside_hole);

            if (inside_hole) {
                EXPECT_THROW(
                    (void)raster.first_arrival_time_s(i, j),
                    std::logic_error);
            } else {
                EXPECT_DOUBLE_EQ(
                    static_cast<double>(
                        raster.first_arrival_time_s(i, j)),
                    7.25);
            }
        }
    }
}

TEST(
    FireFirstArrivalRaster,
    FrontLinearSweepRecordsSampledTransientArrival)
{
    FireFirstArrivalRaster raster({
        4,
        1,
        0.0,
        0.0,
        1.0,
        1.0
    });

    const FireFront start(
        std::vector<FireFrontComponent>{
            {
                FireFrontRole::Outer,
                make_rectangle(
                    -2.0,
                    -1.0,
                    0.0,
                    1.0)
            },
            {
                FireFrontRole::Hole,
                make_rectangle(
                    -1.75,
                    -1.25,
                    0.0,
                    1.0)
            }
        });

    const FireFront end(
        std::vector<FireFrontComponent>{
            {
                FireFrontRole::Outer,
                make_rectangle(
                    2.0,
                    3.0,
                    0.0,
                    1.0)
            },
            {
                FireFrontRole::Hole,
                make_rectangle(
                    2.25,
                    2.75,
                    0.0,
                    1.0)
            }
        });

    const auto initialized =
        raster.initialize_from_front(
            start,
            amrex::Real(10.0));

    EXPECT_EQ(
        initialized.arrived_cell_count,
        0U);

    const auto update =
        raster.update_from_front_linear_sweep(
            start,
            end,
            amrex::Real(10.0),
            amrex::Real(12.0),
            amrex::Real(1.0e-9),
            2U);

    EXPECT_EQ(
        update.newly_arrived_cell_count,
        2U);
    EXPECT_EQ(
        update.arrived_cell_count,
        2U);

    ASSERT_TRUE(
        raster.has_arrived(0, 0));
    const amrex::Real first_time_s =
        raster.first_arrival_time_s(0, 0);
    EXPECT_GE(
        static_cast<double>(first_time_s),
        10.5);
    EXPECT_LT(
        static_cast<double>(
            first_time_s
            - amrex::Real(10.5)),
        1.0e-9);

    ASSERT_TRUE(
        raster.has_arrived(2, 0));
    const amrex::Real second_time_s =
        raster.first_arrival_time_s(2, 0);
    EXPECT_GE(
        static_cast<double>(second_time_s),
        11.5);
    EXPECT_LT(
        static_cast<double>(
            second_time_s
            - amrex::Real(11.5)),
        1.0e-9);

    EXPECT_FALSE(
        raster.has_arrived(1, 0));
    EXPECT_FALSE(
        raster.has_arrived(3, 0));

    EXPECT_TRUE(
        raster.has_committed_sweep());
    EXPECT_DOUBLE_EQ(
        static_cast<double>(
            raster.last_sweep_end_time_s()),
        12.0);
}

TEST(
    FireFirstArrivalRaster,
    TopologyEventSweepCommitsArrivalAndPreservesPocket)
{
    FireFirstArrivalRaster raster({
        44,
        20,
        -1.0,
        0.0,
        0.25,
        0.5
    });

    const FirePerimeter start =
        make_topology_event_start();
    const std::vector<FireVec2> event_vertices =
        make_topology_event_vertices(start);
    const FireFront event_front =
        make_topology_event_front(event_vertices);

    const auto initialized =
        raster.initialize_from_perimeter(
            start,
            amrex::Real(10.0));

    EXPECT_GT(
        initialized.arrived_cell_count,
        0U);

    constexpr std::size_t arriving_i = 1U;
    constexpr std::size_t arriving_j = 18U;
    constexpr std::size_t pocket_i = 20U;
    constexpr std::size_t pocket_j = 8U;
    constexpr std::size_t existing_i = 4U;
    constexpr std::size_t existing_j = 18U;

    ASSERT_FALSE(
        raster.has_arrived(
            arriving_i,
            arriving_j));
    ASSERT_FALSE(
        raster.has_arrived(
            pocket_i,
            pocket_j));
    ASSERT_TRUE(
        raster.has_arrived(
            existing_i,
            existing_j));
    EXPECT_DOUBLE_EQ(
        static_cast<double>(
            raster.first_arrival_time_s(
                existing_i,
                existing_j)),
        10.0);

    const auto update =
        raster.update_from_topology_event_sweep(
            start,
            event_vertices,
            event_front,
            amrex::Real(10.0),
            amrex::Real(12.0),
            amrex::Real(1.0e-9));

    EXPECT_GT(
        update.newly_arrived_cell_count,
        0U);

    ASSERT_TRUE(
        raster.has_arrived(
            arriving_i,
            arriving_j));
    const amrex::Real arrival_time_s =
        raster.first_arrival_time_s(
            arriving_i,
            arriving_j);
    EXPECT_GE(
        static_cast<double>(arrival_time_s),
        11.0);
    EXPECT_LT(
        static_cast<double>(
            arrival_time_s
            - amrex::Real(11.0)),
        1.0e-9);

    EXPECT_FALSE(
        raster.has_arrived(
            pocket_i,
            pocket_j));

    EXPECT_DOUBLE_EQ(
        static_cast<double>(
            raster.first_arrival_time_s(
                existing_i,
                existing_j)),
        10.0);

    EXPECT_TRUE(
        raster.has_committed_sweep());
    EXPECT_DOUBLE_EQ(
        static_cast<double>(
            raster.last_sweep_end_time_s()),
        12.0);
}

TEST(
    FireFirstArrivalRaster,
    HoleExtinctionTopologyEventCommitsAndPreservesArrival)
{
    FireFirstArrivalRaster raster({
        1,
        1,
        1.0,
        1.0,
        1.0,
        1.0
    });

    const FirePerimeter outer =
        make_rectangle(
            0.0,
            4.0,
            0.0,
            4.0);

    const FirePerimeter hole(
        std::vector<FireVec2>{
            {1.0, 1.0},
            {3.0, 1.0},
            {1.0, 3.0}
        });

    const FireFront start_front(
        std::vector<FireFrontComponent>{
            {
                FireFrontRole::Outer,
                outer
            },
            {
                FireFrontRole::Hole,
                hole
            }
        });

    const auto initialized =
        raster.initialize_from_front(
            start_front,
            amrex::Real(10.0));

    EXPECT_EQ(
        initialized.arrived_cell_count,
        0U);
    EXPECT_FALSE(
        raster.has_arrived(0, 0));

    std::vector<std::vector<FireVec2>>
        event_vertices_m{
            outer.vertices_m(),
            hole.vertices_m()
        };

    event_vertices_m[1][0] = {
        amrex::Real(2.0),
        amrex::Real(2.0)
    };

    const FireFront event_front(
        std::vector<FireFrontComponent>{
            {
                FireFrontRole::Outer,
                outer
            }
        });

    const auto update =
        raster.update_from_front_topology_event_sweep(
            start_front,
            event_vertices_m,
            event_front,
            amrex::Real(10.0),
            amrex::Real(12.0),
            amrex::Real(1.0e-9));

    EXPECT_EQ(
        update.newly_arrived_cell_count,
        1U);
    EXPECT_EQ(
        update.arrived_cell_count,
        1U);
    ASSERT_TRUE(
        raster.has_arrived(0, 0));

    const amrex::Real arrival_time_s =
        raster.first_arrival_time_s(0, 0);

    EXPECT_GE(
        static_cast<double>(
            arrival_time_s),
        10.0);
    EXPECT_LT(
        static_cast<double>(
            arrival_time_s
            - amrex::Real(10.0)),
        1.0e-9);

    EXPECT_TRUE(
        raster.has_committed_sweep());
    EXPECT_DOUBLE_EQ(
        static_cast<double>(
            raster.last_sweep_end_time_s()),
        12.0);

    (void)raster.update_from_front_linear_sweep(
        event_front,
        event_front,
        amrex::Real(12.0),
        amrex::Real(13.0),
        amrex::Real(1.0e-9),
        1U);

    EXPECT_DOUBLE_EQ(
        static_cast<double>(
            raster.first_arrival_time_s(0, 0)),
        static_cast<double>(
            arrival_time_s));
}

} // namespace
