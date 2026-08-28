#include <ERF_FirePerimeter.H>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace
{

using ERFFire::FirePerimeter;
using ERFFire::FireVec2;

TEST(FirePerimeter, NormalizesClockwiseInput)
{
    const std::vector<FireVec2> counter_clockwise{
        {-1.0, -1.0},
        { 1.0, -1.0},
        { 1.0,  1.0},
        {-1.0,  1.0}
    };

    auto clockwise = counter_clockwise;
    std::reverse(clockwise.begin(), clockwise.end());

    const FirePerimeter ccw(counter_clockwise);
    const FirePerimeter cw(clockwise);

    EXPECT_TRUE(ccw.is_counter_clockwise());
    EXPECT_TRUE(cw.is_counter_clockwise());
    EXPECT_DOUBLE_EQ(static_cast<double>(ccw.signed_area_m2()), 4.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(cw.signed_area_m2()), 4.0);
}

TEST(FirePerimeter, MeasuresSquareGeometry)
{
    const FirePerimeter square({
        {-1.0, -1.0},
        { 1.0, -1.0},
        { 1.0,  1.0},
        {-1.0,  1.0}
    });

    EXPECT_DOUBLE_EQ(static_cast<double>(square.area_m2()), 4.0);
    EXPECT_DOUBLE_EQ(
        static_cast<double>(square.perimeter_length_m()), 8.0);
}

TEST(FirePerimeter, OutwardNormalsPointAwayFromInterior)
{
    const FirePerimeter square({
        {-1.0, -1.0},
        { 1.0, -1.0},
        { 1.0,  1.0},
        {-1.0,  1.0}
    });

    for (std::size_t i = 0; i < square.size(); ++i) {
        const FireVec2 normal = square.outward_normal_unit(i);
        const FireVec2 radial = square.vertices_m()[i];

        EXPECT_NEAR(
            static_cast<double>(ERFFire::norm(normal)), 1.0, 1.0e-14);
        EXPECT_GT(
            static_cast<double>(ERFFire::dot(normal, radial)), 0.0);
    }
}

TEST(FirePerimeter, RejectsInvalidLoops)
{
    EXPECT_THROW(
        (FirePerimeter(std::vector<FireVec2>{
            {0.0, 0.0},
            {1.0, 0.0}
        })),
        std::invalid_argument);

    EXPECT_THROW(
        (FirePerimeter(std::vector<FireVec2>{
            {0.0, 0.0},
            {1.0, 0.0},
            {1.0, 0.0},
            {0.0, 1.0}
        })),
        std::invalid_argument);

    EXPECT_THROW(
        (FirePerimeter(std::vector<FireVec2>{
            {0.0, 0.0},
            {1.0, 0.0},
            {2.0, 0.0}
        })),
        std::invalid_argument);
}

TEST(FirePerimeter, RejectsSelfIntersectingLoop)
{
    EXPECT_THROW(
        (FirePerimeter(std::vector<FireVec2>{
            {0.0, 0.0},
            {4.0, 4.0},
            {0.0, 4.0},
            {3.0, 0.0},
            {0.0, 2.0}
        })),
        std::invalid_argument);
}

} // namespace
