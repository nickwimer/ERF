#include <ERF_FireBurnedFractionRaster.H>
#include <ERF_FirePerimeter.H>
#include <ERF_FireTypes.H>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{

using ERFFire::FireBurnedFractionRaster;
using ERFFire::FireCartesianRasterGeometry2D;
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
make_non_grid_aligned_concave_fixture ()
{
    return FirePerimeter(std::vector<FireVec2>{
        {0.23, 0.31},
        {4.23, 0.31},
        {4.23, 4.31},
        {3.23, 4.31},
        {3.23, 1.31},
        {1.23, 1.31},
        {1.23, 4.31},
        {0.23, 4.31}
    });
}

TEST(FireBurnedFractionRaster, ConstructsZeroHistoryAndMapsPhysicalCells)
{
    FireBurnedFractionRaster raster({
        3,
        2,
        -1.0,
        4.0,
        0.5,
        2.0
    });

    EXPECT_EQ(raster.cell_count(), 6U);
    EXPECT_DOUBLE_EQ(
        static_cast<double>(raster.burned_area_m2()),
        0.0);

    for (std::size_t j = 0; j < 2; ++j) {
        for (std::size_t i = 0; i < 3; ++i) {
            EXPECT_DOUBLE_EQ(
                static_cast<double>(
                    raster.burned_fraction(i, j)),
                0.0);
        }
    }

    const auto cell = raster.cell_bounds(2, 1);
    EXPECT_DOUBLE_EQ(static_cast<double>(cell.xlo_m), 0.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(cell.xhi_m), 0.5);
    EXPECT_DOUBLE_EQ(static_cast<double>(cell.ylo_m), 6.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(cell.yhi_m), 8.0);
}

TEST(FireBurnedFractionRaster, AdjacentCellsShareIdenticalRepresentedFaces)
{
    FireBurnedFractionRaster raster({
        8,
        8,
        0.1,
        -0.2,
        0.1,
        0.1
    });

    for (std::size_t i = 0; i + 1 < 8; ++i) {
        const auto left = raster.cell_bounds(i, 0);
        const auto right = raster.cell_bounds(i + 1, 0);
        EXPECT_DOUBLE_EQ(
            static_cast<double>(left.xhi_m),
            static_cast<double>(right.xlo_m));
    }

    for (std::size_t j = 0; j + 1 < 8; ++j) {
        const auto lower = raster.cell_bounds(0, j);
        const auto upper = raster.cell_bounds(0, j + 1);
        EXPECT_DOUBLE_EQ(
            static_cast<double>(lower.yhi_m),
            static_cast<double>(upper.ylo_m));
    }
}

TEST(FireBurnedFractionRaster, TranslatedNonBinaryGridConservesRepresentedArea)
{
    FireBurnedFractionRaster raster({
        10,
        10,
        0.1,
        -0.2,
        0.1,
        0.1
    });

    const auto perimeter = make_rectangle(
        0.23, 0.87, -0.07, 0.53);

    const auto update =
        raster.update_from_perimeter(perimeter);

    EXPECT_NEAR(
        static_cast<double>(update.burned_area_m2),
        static_cast<double>(perimeter.area_m2()),
        3.0e-15);
    EXPECT_NEAR(
        static_cast<double>(update.newly_burned_area_m2),
        static_cast<double>(perimeter.area_m2()),
        3.0e-15);
    EXPECT_DOUBLE_EQ(
        static_cast<double>(update.burned_area_m2),
        static_cast<double>(raster.burned_area_m2()));
}

TEST(FireBurnedFractionRaster, OneUpdatePopulatesExactPartialFractionsAndArea)
{
    FireBurnedFractionRaster raster({
        2,
        2,
        0.0,
        0.0,
        1.0,
        1.0
    });

    const auto perimeter = make_rectangle(
        0.5, 1.5, 0.25, 1.25);

    const auto update = raster.update_from_perimeter(perimeter);

    EXPECT_DOUBLE_EQ(
        static_cast<double>(raster.burned_fraction(0, 0)),
        0.375);
    EXPECT_DOUBLE_EQ(
        static_cast<double>(raster.burned_fraction(1, 0)),
        0.375);
    EXPECT_DOUBLE_EQ(
        static_cast<double>(raster.burned_fraction(0, 1)),
        0.125);
    EXPECT_DOUBLE_EQ(
        static_cast<double>(raster.burned_fraction(1, 1)),
        0.125);

    EXPECT_DOUBLE_EQ(
        static_cast<double>(update.burned_area_m2),
        1.0);
    EXPECT_DOUBLE_EQ(
        static_cast<double>(update.newly_burned_area_m2),
        1.0);
    EXPECT_DOUBLE_EQ(
        static_cast<double>(raster.burned_area_m2()),
        1.0);
}

TEST(FireBurnedFractionRaster, RepeatingSamePerimeterIsIdempotent)
{
    FireBurnedFractionRaster raster({
        4,
        4,
        0.0,
        0.0,
        1.0,
        1.0
    });

    const auto perimeter = make_rectangle(
        0.25, 2.75, 0.50, 2.25);

    const auto first = raster.update_from_perimeter(perimeter);
    const auto second = raster.update_from_perimeter(perimeter);

    EXPECT_NEAR(
        static_cast<double>(first.burned_area_m2),
        4.375,
        2.0e-15);
    EXPECT_NEAR(
        static_cast<double>(first.newly_burned_area_m2),
        4.375,
        2.0e-15);

    EXPECT_DOUBLE_EQ(
        static_cast<double>(second.burned_area_m2),
        static_cast<double>(first.burned_area_m2));
    EXPECT_DOUBLE_EQ(
        static_cast<double>(second.newly_burned_area_m2),
        0.0);
}

TEST(FireBurnedFractionRaster, NestedExpansionTelescopesNewlyBurnedArea)
{
    FireBurnedFractionRaster raster({
        8,
        8,
        0.0,
        0.0,
        0.5,
        0.5
    });

    const auto first_perimeter = make_rectangle(
        0.25, 1.75, 0.50, 1.50);
    const auto second_perimeter = make_rectangle(
        0.25, 2.75, 0.25, 2.25);

    const auto first =
        raster.update_from_perimeter(first_perimeter);
    const auto second =
        raster.update_from_perimeter(second_perimeter);

    EXPECT_NEAR(
        static_cast<double>(first.burned_area_m2),
        1.5,
        3.0e-15);
    EXPECT_NEAR(
        static_cast<double>(first.newly_burned_area_m2),
        1.5,
        3.0e-15);

    EXPECT_NEAR(
        static_cast<double>(second.burned_area_m2),
        5.0,
        6.0e-15);
    EXPECT_NEAR(
        static_cast<double>(second.newly_burned_area_m2),
        3.5,
        6.0e-15);

    EXPECT_NEAR(
        static_cast<double>(
            first.burned_area_m2
            + second.newly_burned_area_m2),
        static_cast<double>(second.burned_area_m2),
        6.0e-15);
}

TEST(FireBurnedFractionRaster, RetreatCannotEraseBurnHistory)
{
    FireBurnedFractionRaster raster({
        8,
        8,
        0.0,
        0.0,
        0.5,
        0.5
    });

    const auto large = make_rectangle(
        0.25, 2.75, 0.25, 2.25);
    const auto smaller = make_rectangle(
        1.0, 2.0, 1.0, 2.0);

    const auto initial = raster.update_from_perimeter(large);

    std::vector<amrex::Real> before;
    before.reserve(raster.cell_count());
    for (std::size_t j = 0; j < 8; ++j) {
        for (std::size_t i = 0; i < 8; ++i) {
            before.push_back(raster.burned_fraction(i, j));
        }
    }

    const auto retreated =
        raster.update_from_perimeter(smaller);

    EXPECT_NEAR(
        static_cast<double>(initial.burned_area_m2),
        5.0,
        6.0e-15);
    EXPECT_DOUBLE_EQ(
        static_cast<double>(retreated.burned_area_m2),
        static_cast<double>(initial.burned_area_m2));
    EXPECT_DOUBLE_EQ(
        static_cast<double>(retreated.newly_burned_area_m2),
        0.0);

    std::size_t index = 0;
    for (std::size_t j = 0; j < 8; ++j) {
        for (std::size_t i = 0; i < 8; ++i) {
            EXPECT_DOUBLE_EQ(
                static_cast<double>(
                    raster.burned_fraction(i, j)),
                static_cast<double>(before[index++]));
        }
    }
}

TEST(FireBurnedFractionRaster, ExactCoverageConservesConcaveAreaAcrossResolutions)
{
    const auto perimeter =
        make_non_grid_aligned_concave_fixture();

    for (const std::size_t cells_per_side : {
            std::size_t(5),
            std::size_t(10),
            std::size_t(20)}) {
        const amrex::Real spacing =
            amrex::Real(5.0)
            / static_cast<amrex::Real>(cells_per_side);

        FireBurnedFractionRaster raster({
            cells_per_side,
            cells_per_side,
            0.0,
            0.0,
            spacing,
            spacing
        });

        const auto update =
            raster.update_from_perimeter(perimeter);

        EXPECT_NEAR(
            static_cast<double>(update.burned_area_m2),
            10.0,
            2.0e-12);
        EXPECT_NEAR(
            static_cast<double>(update.newly_burned_area_m2),
            10.0,
            2.0e-12);
        EXPECT_NEAR(
            static_cast<double>(raster.burned_area_m2()),
            10.0,
            2.0e-12);

        for (std::size_t j = 0; j < cells_per_side; ++j) {
            for (std::size_t i = 0; i < cells_per_side; ++i) {
                const amrex::Real fraction =
                    raster.burned_fraction(i, j);
                EXPECT_GE(static_cast<double>(fraction), 0.0);
                EXPECT_LE(static_cast<double>(fraction), 1.0);
            }
        }
    }
}

TEST(FireBurnedFractionRaster, RejectsInvalidGeometryAndIndices)
{
    const amrex::Real nan =
        std::numeric_limits<amrex::Real>::quiet_NaN();
    const amrex::Real inf =
        std::numeric_limits<amrex::Real>::infinity();

    for (const FireCartesianRasterGeometry2D geometry : {
            FireCartesianRasterGeometry2D{
                0, 1, 0.0, 0.0, 1.0, 1.0},
            FireCartesianRasterGeometry2D{
                1, 0, 0.0, 0.0, 1.0, 1.0},
            FireCartesianRasterGeometry2D{
                1, 1, nan, 0.0, 1.0, 1.0},
            FireCartesianRasterGeometry2D{
                1, 1, 0.0, inf, 1.0, 1.0},
            FireCartesianRasterGeometry2D{
                1, 1, 0.0, 0.0, 0.0, 1.0},
            FireCartesianRasterGeometry2D{
                1, 1, 0.0, 0.0, 1.0, -1.0}}) {
        EXPECT_THROW(
            (void)FireBurnedFractionRaster(geometry),
            std::invalid_argument);
    }

    EXPECT_THROW(
        (void)FireBurnedFractionRaster({
            std::numeric_limits<std::size_t>::max(),
            2,
            0.0,
            0.0,
            1.0,
            1.0
        }),
        std::overflow_error);

    EXPECT_THROW(
        (void)FireBurnedFractionRaster({
            2,
            1,
            0.0,
            0.0,
            std::numeric_limits<amrex::Real>::max(),
            1.0
        }),
        std::overflow_error);

    EXPECT_THROW(
        (void)FireBurnedFractionRaster({
            1,
            1,
            amrex::Real(1.0e20),
            0.0,
            1.0,
            1.0
        }),
        std::invalid_argument);

    EXPECT_THROW(
        (void)FireBurnedFractionRaster({
            2,
            2,
            0.0,
            0.0,
            amrex::Real(1.0e154),
            amrex::Real(1.0e154)
        }),
        std::overflow_error);

    FireBurnedFractionRaster raster({
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
        (void)raster.cell_bounds(0, 3),
        std::out_of_range);
    EXPECT_THROW(
        (void)raster.burned_fraction(2, 0),
        std::out_of_range);
    EXPECT_THROW(
        (void)raster.burned_fraction(0, 3),
        std::out_of_range);
}

} // namespace
