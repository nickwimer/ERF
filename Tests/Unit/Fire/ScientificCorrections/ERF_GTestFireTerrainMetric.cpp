#include <ERF_FireTerrainMetric.H>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace
{
using amrex::Real;
using namespace ERFFire;
Real tol(Real x = Real(1))
{
    return Real(512) * std::numeric_limits<Real>::epsilon()
        * std::max(Real(1), std::abs(x));
}
FireVec2 unit(Real a) { return {std::cos(a), std::sin(a)}; }
FireVec2 rotate(FireVec2 v, Real a)
{
    return {std::cos(a)*v.x - std::sin(a)*v.y,
            std::sin(a)*v.x + std::cos(a)*v.y};
}
}

TEST(FireScientificTerrain, FlatIsExactlyTheExistingSupport)
{
    const auto ellipse = make_richards_ellipse(unit(Real(0.3)), Real(2), Real(3));
    const FireTerrainMetric metric({Real(0), Real(0)});
    for (int i = 0; i < 20; ++i) {
        const auto n = unit(Real(i) / Real(3));
        EXPECT_EQ(metric.normal_speed_mps(ellipse, n),
                  richards_normal_speed_mps(ellipse, n));
        EXPECT_EQ(metric.surface_unit_direction(n).x, n.x);
        EXPECT_EQ(metric.surface_unit_direction(n).y, n.y);
    }
}

TEST(FireScientificTerrain, IsotropicGroundSpeedUsesInverseMetric)
{
    const auto circle = make_richards_ellipse({Real(1), Real(0)}, Real(2), Real(0));
    for (const Real slope : {Real(0.3), Real(0.6), Real(1)}) {
        for (const Real angle : {Real(0), Real(0.4), Real(1.1)}) {
            const FireVec2 g = slope * unit(angle);
            const FireTerrainMetric metric(g);
            for (int i = 0; i < 25; ++i) {
                const FireVec2 n = unit(Real(i) / Real(4));
                // Independent Sherman-Morrison inverse of I + g g^T.
                const Real gn = dot(g, n);
                const Real expected = Real(2) * std::sqrt(
                    Real(1) - gn*gn / (Real(1) + slope*slope));
                EXPECT_NEAR(metric.normal_speed_mps(circle, n), expected, tol(expected));
            }
            EXPECT_NEAR(metric.normal_speed_mps(circle, unit(angle)),
                        Real(2) / std::sqrt(Real(1) + slope*slope), tol());
            EXPECT_NEAR(metric.normal_speed_mps(circle, unit(angle + Real(1.5707963267948966))),
                        Real(2), tol());
        }
    }
}

TEST(FireScientificTerrain, ObliqueEllipseMatchesThreeDimensionalProjection)
{
    const FireVec2 g{Real(0.3), Real(0.4)};
    const FireTerrainMetric metric(g);
    const auto ellipse = make_richards_ellipse(unit(Real(0.8)), Real(2), Real(3));
    const Real slope = Real(0.5);
    const FireVec2 uphill{Real(0.6), Real(0.8)};
    const FireVec2 contour{Real(-0.8), Real(0.6)};
    // Horizontal projections of the orthonormal 3-D tangent-plane basis.
    const FireVec2 projected_uphill = uphill / std::sqrt(Real(1) + slope*slope);
    const auto project = [&](FireVec2 v) {
        return projected_uphill * dot(uphill, v) + contour * dot(contour, v);
    };
    const FireVec2 heading = project(ellipse.heading_unit);
    const FireVec2 flank = project({-ellipse.heading_unit.y, ellipse.heading_unit.x});
    for (int i = 0; i < 32; ++i) {
        const auto n = unit(Real(i) / Real(5));
        const Real u = dot(n, heading), v = dot(n, flank);
        const Real expected = ellipse.center_translation_rate_mps*u
            + std::hypot(ellipse.semi_major_rate_mps*u,
                         ellipse.semi_minor_rate_mps*v);
        EXPECT_NEAR(metric.normal_speed_mps(ellipse, n), expected, tol(expected));
    }
}

TEST(FireScientificTerrain, WindAzimuthAndRotationAreConsistent)
{
    const FireVec2 g{Real(0.6), Real(0)};
    const FireVec2 wind = unit(Real(0.7));
    const FireTerrainMetric metric(g);
    const FireVec2 actual = metric.surface_unit_direction(wind);
    const Real expected_angle = std::atan2(wind.y,
        std::sqrt(Real(1.36))*wind.x);
    EXPECT_NEAR(actual.x, std::cos(expected_angle), tol());
    EXPECT_NEAR(actual.y, std::sin(expected_angle), tol());
    const auto ellipse = make_richards_ellipse(actual, Real(2), Real(3));
    const FireVec2 n = unit(Real(1.2));
    for (const Real angle : {Real(0.4), Real(1.3), Real(2.5)}) {
        const FireTerrainMetric rotated(rotate(g, angle));
        const auto direction = rotated.surface_unit_direction(rotate(wind, angle));
        const auto rotated_ellipse = make_richards_ellipse(direction, Real(2), Real(3));
        EXPECT_NEAR(rotated.normal_speed_mps(rotated_ellipse, rotate(n, angle)),
                    metric.normal_speed_mps(ellipse, n), tol());
    }
}

TEST(FireScientificTerrain, InvalidDirectionsDoNotSilentlyNormalize)
{
    EXPECT_THROW((void)FireTerrainMetric({std::numeric_limits<Real>::infinity(), Real(0)}),
                 std::invalid_argument);
    const FireTerrainMetric metric({Real(0.6), Real(0)});
    EXPECT_THROW((void)metric.surface_unit_direction({Real(0), Real(0)}), std::invalid_argument);
    const auto ellipse = make_richards_ellipse({Real(1), Real(0)}, Real(2), Real(0));
    EXPECT_THROW((void)metric.normal_speed_mps(ellipse, {Real(2), Real(0)}), std::invalid_argument);
}
