#include <ERF_FireFirstArrivalRaster.H>
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

} // namespace
