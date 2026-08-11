#include <ERF_FireCellCoverage.H>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{

using ERFFire::FireCartesianCell2D;
using ERFFire::FirePerimeter;
using ERFFire::FireVec2;

FirePerimeter
make_perimeter (std::initializer_list<FireVec2> vertices)
{
    return FirePerimeter(std::vector<FireVec2>(vertices));
}

FirePerimeter
translate_perimeter (
    const FirePerimeter& perimeter,
    const FireVec2& offset)
{
    std::vector<FireVec2> translated;
    translated.reserve(perimeter.size());

    for (const FireVec2& vertex : perimeter.vertices_m()) {
        translated.push_back(vertex + offset);
    }

    return FirePerimeter(std::move(translated));
}

FirePerimeter
scale_perimeter (
    const FirePerimeter& perimeter,
    amrex::Real scale)
{
    std::vector<FireVec2> scaled;
    scaled.reserve(perimeter.size());

    for (const FireVec2& vertex : perimeter.vertices_m()) {
        scaled.push_back(vertex * scale);
    }

    return FirePerimeter(std::move(scaled));
}

FireCartesianCell2D
translate_cell (
    const FireCartesianCell2D& cell,
    const FireVec2& offset)
{
    return {
        cell.xlo_m + offset.x,
        cell.xhi_m + offset.x,
        cell.ylo_m + offset.y,
        cell.yhi_m + offset.y
    };
}

FireCartesianCell2D
scale_cell (
    const FireCartesianCell2D& cell,
    amrex::Real scale)
{
    return {
        cell.xlo_m * scale,
        cell.xhi_m * scale,
        cell.ylo_m * scale,
        cell.yhi_m * scale
    };
}

TEST(FireCellCoverage, CellFullyInsidePerimeterHasUnitCoverage)
{
    const auto perimeter = make_perimeter({
        {-2.0, -2.0},
        { 2.0, -2.0},
        { 2.0,  2.0},
        {-2.0,  2.0}
    });
    const FireCartesianCell2D cell{-1.0, 1.0, -1.0, 1.0};

    EXPECT_DOUBLE_EQ(
        static_cast<double>(
            ERFFire::fire_perimeter_cell_intersection_area_m2(
                perimeter, cell)),
        4.0);
    EXPECT_DOUBLE_EQ(
        static_cast<double>(
            ERFFire::fire_perimeter_cell_coverage_fraction(
                perimeter, cell)),
        1.0);
}

TEST(FireCellCoverage, PerimeterFullyInsideCellReturnsPolygonFraction)
{
    const auto perimeter = make_perimeter({
        {0.25, 0.25},
        {1.25, 0.25},
        {0.25, 1.25}
    });
    const FireCartesianCell2D cell{0.0, 2.0, 0.0, 2.0};

    // Triangle area = 1*1/2 = 0.5 m^2; cell area = 4 m^2.
    EXPECT_NEAR(
        static_cast<double>(
            ERFFire::fire_perimeter_cell_intersection_area_m2(
                perimeter, cell)),
        0.5,
        2.0e-15);
    EXPECT_NEAR(
        static_cast<double>(
            ERFFire::fire_perimeter_cell_coverage_fraction(
                perimeter, cell)),
        0.125,
        2.0e-15);
}

TEST(FireCellCoverage, DisjointAndBoundaryOnlyContactHaveZeroCoverage)
{
    const auto perimeter = make_perimeter({
        {0.0, 0.0},
        {1.0, 0.0},
        {1.0, 1.0},
        {0.0, 1.0}
    });

    for (const FireCartesianCell2D cell : {
            FireCartesianCell2D{2.0, 3.0, 0.0, 1.0},
            FireCartesianCell2D{1.0, 2.0, 0.0, 1.0},
            FireCartesianCell2D{0.0, 1.0, 1.0, 2.0},
            FireCartesianCell2D{1.0, 2.0, 1.0, 2.0}}) {
        EXPECT_DOUBLE_EQ(
            static_cast<double>(
                ERFFire::fire_perimeter_cell_intersection_area_m2(
                    perimeter, cell)),
            0.0);
        EXPECT_DOUBLE_EQ(
            static_cast<double>(
                ERFFire::fire_perimeter_cell_coverage_fraction(
                    perimeter, cell)),
            0.0);
    }
}

TEST(FireCellCoverage, RectangleHalfCellFixtureIsExact)
{
    const auto perimeter = make_perimeter({
        {-1.0, -1.0},
        { 0.5, -1.0},
        { 0.5,  2.0},
        {-1.0,  2.0}
    });
    const FireCartesianCell2D cell{0.0, 1.0, 0.0, 1.0};

    EXPECT_DOUBLE_EQ(
        static_cast<double>(
            ERFFire::fire_perimeter_cell_intersection_area_m2(
                perimeter, cell)),
        0.5);
    EXPECT_DOUBLE_EQ(
        static_cast<double>(
            ERFFire::fire_perimeter_cell_coverage_fraction(
                perimeter, cell)),
        0.5);
}

TEST(FireCellCoverage, TrianglePartialCellFixtureMatchesAnalyticArea)
{
    const auto perimeter = make_perimeter({
        {0.0, 0.0},
        {2.0, 0.0},
        {0.0, 1.0}
    });
    const FireCartesianCell2D cell{0.0, 1.0, 0.0, 1.0};

    // y = 1 - x/2 over x in [0,1]:
    // integral_0^1 (1 - x/2) dx = 3/4.
    EXPECT_NEAR(
        static_cast<double>(
            ERFFire::fire_perimeter_cell_intersection_area_m2(
                perimeter, cell)),
        0.75,
        2.0e-15);
    EXPECT_NEAR(
        static_cast<double>(
            ERFFire::fire_perimeter_cell_coverage_fraction(
                perimeter, cell)),
        0.75,
        2.0e-15);
}

TEST(FireCellCoverage, DiamondCutsFourAnalyticCornerTriangles)
{
    const auto perimeter = make_perimeter({
        { 0.50, -0.25},
        { 1.25,  0.50},
        { 0.50,  1.25},
        {-0.25,  0.50}
    });
    const FireCartesianCell2D cell{0.0, 1.0, 0.0, 1.0};

    // The unit square loses four right triangles with legs 0.25:
    // 1 - 4 * (0.25*0.25/2) = 0.875.
    EXPECT_NEAR(
        static_cast<double>(
            ERFFire::fire_perimeter_cell_intersection_area_m2(
                perimeter, cell)),
        0.875,
        2.0e-15);
    EXPECT_NEAR(
        static_cast<double>(
            ERFFire::fire_perimeter_cell_coverage_fraction(
                perimeter, cell)),
        0.875,
        2.0e-15);
}

TEST(FireCellCoverage, ConcavePerimeterCanIntersectCellInTwoComponents)
{
    const auto perimeter = make_perimeter({
        {0.0, 0.0},
        {4.0, 0.0},
        {4.0, 4.0},
        {3.0, 4.0},
        {3.0, 1.0},
        {1.0, 1.0},
        {1.0, 4.0},
        {0.0, 4.0}
    });
    const FireCartesianCell2D cell{0.5, 3.5, 2.0, 3.0};

    // The U-shaped polygon intersects this 3x1 cell in two disjoint
    // 0.5x1 strips. Total area = 1, cell area = 3.
    EXPECT_NEAR(
        static_cast<double>(
            ERFFire::fire_perimeter_cell_intersection_area_m2(
                perimeter, cell)),
        1.0,
        4.0e-15);
    EXPECT_NEAR(
        static_cast<double>(
            ERFFire::fire_perimeter_cell_coverage_fraction(
                perimeter, cell)),
        1.0 / 3.0,
        2.0e-15);
}

TEST(FireCellCoverage, CoverageIsTranslationAndScaleInvariant)
{
    const auto perimeter = make_perimeter({
        {0.0, 0.0},
        {2.0, 0.0},
        {0.0, 1.0}
    });
    const FireCartesianCell2D cell{0.0, 1.0, 0.0, 1.0};
    const amrex::Real reference =
        ERFFire::fire_perimeter_cell_coverage_fraction(
            perimeter, cell);

    const FireVec2 offset{1.0e6, -2.0e6};
    const auto translated_perimeter =
        translate_perimeter(perimeter, offset);
    const auto translated_cell =
        translate_cell(cell, offset);

    EXPECT_NEAR(
        static_cast<double>(
            ERFFire::fire_perimeter_cell_coverage_fraction(
                translated_perimeter, translated_cell)),
        static_cast<double>(reference),
        2.0e-10);

    constexpr amrex::Real scale = 37.0;
    EXPECT_NEAR(
        static_cast<double>(
            ERFFire::fire_perimeter_cell_coverage_fraction(
                scale_perimeter(perimeter, scale),
                scale_cell(cell, scale))),
        static_cast<double>(reference),
        2.0e-15);
}

TEST(FireCellCoverage, CellAreaSumRecoversConcavePolygonArea)
{
    const auto perimeter = make_perimeter({
        {0.2, 0.3},
        {4.2, 0.3},
        {4.2, 4.3},
        {3.2, 4.3},
        {3.2, 1.3},
        {1.2, 1.3},
        {1.2, 4.3},
        {0.2, 4.3}
    });

    amrex::Real covered_area_m2 = 0.0;

    for (int j = 0; j < 5; ++j) {
        for (int i = 0; i < 5; ++i) {
            const FireCartesianCell2D cell{
                static_cast<amrex::Real>(i),
                static_cast<amrex::Real>(i + 1),
                static_cast<amrex::Real>(j),
                static_cast<amrex::Real>(j + 1)
            };

            const amrex::Real fraction =
                ERFFire::fire_perimeter_cell_coverage_fraction(
                    perimeter, cell);

            EXPECT_GE(static_cast<double>(fraction), 0.0);
            EXPECT_LE(static_cast<double>(fraction), 1.0);

            covered_area_m2 += fraction;
        }
    }

    EXPECT_NEAR(
        static_cast<double>(covered_area_m2),
        static_cast<double>(perimeter.area_m2()),
        3.0e-14);
    EXPECT_NEAR(
        static_cast<double>(covered_area_m2),
        10.0,
        3.0e-14);
}

TEST(FireCellCoverage, RejectsInvalidCellBounds)
{
    const auto perimeter = make_perimeter({
        {0.0, 0.0},
        {1.0, 0.0},
        {0.0, 1.0}
    });

    const amrex::Real nan =
        std::numeric_limits<amrex::Real>::quiet_NaN();
    const amrex::Real inf =
        std::numeric_limits<amrex::Real>::infinity();

    for (const FireCartesianCell2D cell : {
            FireCartesianCell2D{0.0, 0.0, 0.0, 1.0},
            FireCartesianCell2D{1.0, 0.0, 0.0, 1.0},
            FireCartesianCell2D{0.0, 1.0, 2.0, 2.0},
            FireCartesianCell2D{0.0, 1.0, 2.0, 1.0},
            FireCartesianCell2D{nan, 1.0, 0.0, 1.0},
            FireCartesianCell2D{0.0, inf, 0.0, 1.0}}) {
        EXPECT_THROW(
            (void)ERFFire::fire_perimeter_cell_coverage_fraction(
                perimeter, cell),
            std::invalid_argument);
    }
}

} // namespace
