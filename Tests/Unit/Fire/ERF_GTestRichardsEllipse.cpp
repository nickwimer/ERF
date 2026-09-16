#include <ERF_RichardsEllipse.H>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace
{

using ERFFire::FireVec2;
using ERFFire::RichardsEllipse;

constexpr amrex::Real inv_sqrt_two =
    amrex::Real(0.707106781186547524400844362104849039L);

amrex::Real
normal_component_from_richards_equations_heading_x (
    const RichardsEllipse& ellipse,
    const FireVec2& outward_normal)
{
    // Finney (1998, revised 2004), equations 1-2, specialized to heading
    // along +x (azimuth theta = pi/2 in Finney's east/north convention).
    // For a CCW perimeter tangent t=(-n_y,n_x), x_s=-n_y, y_s=n_x.
    const amrex::Real xs = -outward_normal.y;
    const amrex::Real ys = outward_normal.x;

    const amrex::Real a = ellipse.semi_minor_rate_mps;
    const amrex::Real b = ellipse.semi_major_rate_mps;
    const amrex::Real c = ellipse.center_translation_rate_mps;
    const amrex::Real denominator = std::sqrt(
        b * b * ys * ys + a * a * xs * xs);

    const FireVec2 richards_velocity{
        b * b * ys / denominator + c,
        -a * a * xs / denominator
    };

    return ERFFire::dot(richards_velocity, outward_normal);
}

TEST(FireRichards, ZeroEffectiveWindIsExactlyIsotropic)
{
    const auto ellipse = ERFFire::make_richards_ellipse(
        {1.0, 0.0}, 0.2, 0.0);

    EXPECT_DOUBLE_EQ(static_cast<double>(ellipse.unclamped_length_to_breadth), 1.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(ellipse.length_to_breadth), 1.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(ellipse.head_to_back_ratio), 1.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(ellipse.eccentricity), 0.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(ellipse.semi_minor_rate_mps), 0.2);
    EXPECT_DOUBLE_EQ(static_cast<double>(ellipse.semi_major_rate_mps), 0.2);
    EXPECT_DOUBLE_EQ(static_cast<double>(ellipse.center_translation_rate_mps), 0.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(ellipse.backing_ros_mps()), 0.2);
    EXPECT_DOUBLE_EQ(static_cast<double>(ellipse.flank_ros_mps()), 0.2);

    for (const FireVec2 normal : {
            FireVec2{1.0, 0.0},
            FireVec2{-1.0, 0.0},
            FireVec2{0.0, 1.0},
            FireVec2{inv_sqrt_two, inv_sqrt_two}}) {
        EXPECT_NEAR(
            static_cast<double>(ERFFire::richards_normal_speed_mps(ellipse, normal)),
            0.2,
            2.0e-15);
    }
}

TEST(FireRichards, PublishedFarsiteEllipseDimensionsMatchReferenceFixture)
{
    // Independent numerical fixture generated from Finney equations 13-17:
    // heading R=0.5 m/s, effective midflame U=2.0 m/s, rear-focus origin.
    const auto ellipse = ERFFire::make_richards_ellipse(
        {1.0, 0.0}, 0.5, 2.0);

    EXPECT_NEAR(static_cast<double>(ellipse.unclamped_length_to_breadth),
                1.5049627492754702, 2.0e-15);
    EXPECT_NEAR(static_cast<double>(ellipse.length_to_breadth),
                1.5049627492754702, 2.0e-15);
    EXPECT_NEAR(static_cast<double>(ellipse.head_to_back_ratio),
                6.915039166104357, 2.0e-14);
    EXPECT_NEAR(static_cast<double>(ellipse.eccentricity),
                0.7473164746215192, 2.0e-15);
    EXPECT_NEAR(static_cast<double>(ellipse.backing_ros_mps()),
                0.07230617036138626, 2.0e-15);
    EXPECT_NEAR(static_cast<double>(ellipse.semi_minor_rate_mps),
                0.19013964652510834, 2.0e-15);
    EXPECT_NEAR(static_cast<double>(ellipse.semi_major_rate_mps),
                0.28615308518069316, 2.0e-15);
    EXPECT_NEAR(static_cast<double>(ellipse.center_translation_rate_mps),
                0.2138469148193069, 2.0e-15);

    EXPECT_NEAR(
        static_cast<double>(ellipse.center_translation_rate_mps
            * ellipse.center_translation_rate_mps),
        static_cast<double>(ellipse.semi_major_rate_mps
            * ellipse.semi_major_rate_mps
            - ellipse.semi_minor_rate_mps * ellipse.semi_minor_rate_mps),
        3.0e-17);
}

TEST(FireRichards, LengthToBreadthCapMatchesFarsite)
{
    const amrex::Real raw = ERFFire::farsite_unclamped_length_to_breadth(10.0);
    ASSERT_GT(static_cast<double>(raw), 8.0);

    const auto ellipse = ERFFire::make_richards_ellipse(
        {1.0, 0.0}, 0.5, 10.0);

    EXPECT_DOUBLE_EQ(static_cast<double>(ellipse.length_to_breadth), 8.0);
    EXPECT_GT(static_cast<double>(ellipse.unclamped_length_to_breadth), 8.0);
}

TEST(FireRichards, HeadFlankBackNormalsRecoverEllipseRates)
{
    const auto ellipse = ERFFire::make_richards_ellipse(
        {1.0, 0.0}, 0.5, 2.0);

    EXPECT_NEAR(
        static_cast<double>(ERFFire::richards_normal_speed_mps(ellipse, {1.0, 0.0})),
        static_cast<double>(ellipse.heading_ros_mps),
        2.0e-15);
    EXPECT_NEAR(
        static_cast<double>(ERFFire::richards_normal_speed_mps(ellipse, {-1.0, 0.0})),
        static_cast<double>(ellipse.backing_ros_mps()),
        2.0e-15);
    EXPECT_NEAR(
        static_cast<double>(ERFFire::richards_normal_speed_mps(ellipse, {0.0, 1.0})),
        static_cast<double>(ellipse.flank_ros_mps()),
        2.0e-15);
    EXPECT_NEAR(
        static_cast<double>(ERFFire::richards_normal_speed_mps(ellipse, {0.0, -1.0})),
        static_cast<double>(ellipse.flank_ros_mps()),
        2.0e-15);
}

TEST(FireRichards, NormalSupportMatchesRichardsDifferentialNormalComponent)
{
    const auto ellipse = ERFFire::make_richards_ellipse(
        {1.0, 0.0}, 0.5, 2.0);

    for (const FireVec2 normal : {
            FireVec2{1.0, 0.0},
            FireVec2{-1.0, 0.0},
            FireVec2{0.0, 1.0},
            FireVec2{inv_sqrt_two, inv_sqrt_two},
            FireVec2{-inv_sqrt_two, inv_sqrt_two}}) {
        const amrex::Real support =
            ERFFire::richards_normal_speed_mps(ellipse, normal);
        const amrex::Real richards_component =
            normal_component_from_richards_equations_heading_x(ellipse, normal);
        EXPECT_NEAR(
            static_cast<double>(support),
            static_cast<double>(richards_component),
            3.0e-15);
    }
}

TEST(FireRichards, ArbitraryNormalsArePositiveAndNoFasterThanHead)
{
    const auto ellipse = ERFFire::make_richards_ellipse(
        {1.0, 0.0}, 0.5, 5.0);

    constexpr int samples = 720;
    constexpr amrex::Real pi =
        amrex::Real(3.141592653589793238462643383279502884L);
    for (int i = 0; i < samples; ++i) {
        const amrex::Real angle =
            amrex::Real(2.0) * pi * static_cast<amrex::Real>(i)
            / static_cast<amrex::Real>(samples);
        const FireVec2 normal{std::cos(angle), std::sin(angle)};
        const amrex::Real speed =
            ERFFire::richards_normal_speed_mps(ellipse, normal);

        EXPECT_GT(static_cast<double>(speed), 0.0);
        EXPECT_LE(
            static_cast<double>(speed),
            static_cast<double>(ellipse.heading_ros_mps) + 2.0e-15);
    }
}

TEST(FireRichards, ReversingHeadingMirrorsSolutionAndSmallWindIsContinuous)
{
    const auto east = ERFFire::make_richards_ellipse(
        {1.0, 0.0}, 0.5, 2.0);
    const auto west = ERFFire::make_richards_ellipse(
        {-1.0, 0.0}, 0.5, 2.0);

    for (const FireVec2 normal : {
            FireVec2{1.0, 0.0},
            FireVec2{0.0, 1.0},
            FireVec2{inv_sqrt_two, inv_sqrt_two}}) {
        const FireVec2 mirrored{-normal.x, normal.y};
        EXPECT_NEAR(
            static_cast<double>(ERFFire::richards_normal_speed_mps(east, normal)),
            static_cast<double>(ERFFire::richards_normal_speed_mps(west, mirrored)),
            3.0e-15);
    }

    const auto almost_calm = ERFFire::make_richards_ellipse(
        {1.0, 0.0}, 0.5, 1.0e-9);
    EXPECT_NEAR(
        static_cast<double>(almost_calm.length_to_breadth),
        1.0,
        2.0e-10);
    EXPECT_NEAR(
        static_cast<double>(ERFFire::richards_normal_speed_mps(
            almost_calm, {0.0, 1.0})),
        0.5,
        1.0e-5);
}

TEST(FireRichards, RejectsInvalidInputs)
{
    EXPECT_THROW(
        (void)ERFFire::make_richards_ellipse({2.0, 0.0}, 0.5, 2.0),
        std::invalid_argument);
    EXPECT_THROW(
        (void)ERFFire::make_richards_ellipse({1.0, 0.0}, -0.1, 2.0),
        std::invalid_argument);
    EXPECT_THROW(
        (void)ERFFire::make_richards_ellipse({1.0, 0.0}, 0.5, -1.0),
        std::invalid_argument);
    EXPECT_THROW(
        (void)ERFFire::farsite_unclamped_length_to_breadth(
            std::numeric_limits<amrex::Real>::infinity()),
        std::invalid_argument);
    EXPECT_THROW(
        (void)ERFFire::farsite_unclamped_length_to_breadth(1.0e6),
        std::overflow_error);

    const auto ellipse = ERFFire::make_richards_ellipse(
        {1.0, 0.0}, 0.5, 2.0);
    EXPECT_THROW(
        (void)ERFFire::richards_normal_speed_mps(ellipse, {0.5, 0.0}),
        std::invalid_argument);
}

} // namespace
