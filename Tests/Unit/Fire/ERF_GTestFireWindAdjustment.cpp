#include <ERF_FireWindAdjustment.H>
#include <ERF_RichardsDirectionalSpread.H>
#include <ERF_RothermelFuel.H>
#include <ERF_RothermelModel.H>

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace
{

using ERFFire::FireVec2;
using ERFFire::RothermelInputs;

bool same_real_bits (const amrex::Real& lhs, const amrex::Real& rhs) noexcept
{
    return std::memcmp(&lhs, &rhs, sizeof(amrex::Real)) == 0;
}

TEST(FireWindAdjustment, UnityWafPreservesReferenceWindBits)
{
    const FireVec2 reference{-amrex::Real(0.0), amrex::Real(3.25)};
    const FireVec2 adjusted =
        ERFFire::fire_midflame_wind_from_20ft_reference(reference, 1.0);

    EXPECT_TRUE(same_real_bits(adjusted.x, reference.x));
    EXPECT_TRUE(same_real_bits(adjusted.y, reference.y));
    EXPECT_TRUE(std::signbit(static_cast<double>(adjusted.x)));
}

TEST(FireWindAdjustment, ZeroWafReturnsCanonicalZeroVector)
{
    const FireVec2 adjusted =
        ERFFire::fire_midflame_wind_from_20ft_reference(
            {-amrex::Real(0.0), amrex::Real(-4.5)}, 0.0);

    EXPECT_DOUBLE_EQ(static_cast<double>(adjusted.x), 0.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(adjusted.y), 0.0);
    EXPECT_FALSE(std::signbit(static_cast<double>(adjusted.x)));
    EXPECT_FALSE(std::signbit(static_cast<double>(adjusted.y)));
}

TEST(FireWindAdjustment, ScalesComponentsAndMagnitude)
{
    const FireVec2 reference{3.0, -4.0};
    const FireVec2 adjusted =
        ERFFire::fire_midflame_wind_from_20ft_reference(reference, 0.25);

    EXPECT_DOUBLE_EQ(static_cast<double>(adjusted.x), 0.75);
    EXPECT_DOUBLE_EQ(static_cast<double>(adjusted.y), -1.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(ERFFire::norm(reference)), 5.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(ERFFire::norm(adjusted)), 1.25);
}

TEST(FireWindAdjustment, PreservesNonAxisAlignedPushDirection)
{
    const FireVec2 reference{2.75, -1.25};
    const FireVec2 adjusted =
        ERFFire::fire_midflame_wind_from_20ft_reference(reference, 0.37);

    const FireVec2 reference_unit = reference / ERFFire::norm(reference);
    const FireVec2 adjusted_unit = adjusted / ERFFire::norm(adjusted);
    const double tolerance =
        64.0 * static_cast<double>(std::numeric_limits<amrex::Real>::epsilon());

    EXPECT_NEAR(
        static_cast<double>(adjusted_unit.x),
        static_cast<double>(reference_unit.x),
        tolerance);
    EXPECT_NEAR(
        static_cast<double>(adjusted_unit.y),
        static_cast<double>(reference_unit.y),
        tolerance);
    EXPECT_GT(static_cast<double>(ERFFire::dot(reference, adjusted)), 0.0);
}

TEST(FireWindAdjustment, RejectsNonfiniteAndOutOfRangeWaf)
{
    const FireVec2 reference{2.0, -1.0};

    for (const amrex::Real invalid_waf : {
            amrex::Real(-0.01),
            amrex::Real(1.01),
            std::numeric_limits<amrex::Real>::quiet_NaN(),
            std::numeric_limits<amrex::Real>::infinity(),
            -std::numeric_limits<amrex::Real>::infinity()}) {
        EXPECT_THROW(
            (void)ERFFire::fire_midflame_wind_from_20ft_reference(
                reference, invalid_waf),
            std::invalid_argument);
    }
}

TEST(FireWindAdjustment, ZeroWafLeavesSlopeOnlyBehaviorActive)
{
    const FireVec2 adjusted =
        ERFFire::fire_midflame_wind_from_20ft_reference({3.0, 4.0}, 0.0);
    const auto fuel = ERFFire::make_fm1_fuel_parameters();

    const auto baseline = ERFFire::evaluate_rothermel(
        fuel, RothermelInputs{0.08, 0.0, 0.20});
    const auto with_waf = ERFFire::evaluate_rothermel(
        fuel, RothermelInputs{0.08, ERFFire::norm(adjusted), 0.20});

    EXPECT_DOUBLE_EQ(static_cast<double>(with_waf.wind_factor), 0.0);
    EXPECT_DOUBLE_EQ(
        static_cast<double>(with_waf.slope_factor),
        static_cast<double>(baseline.slope_factor));
    EXPECT_DOUBLE_EQ(
        static_cast<double>(with_waf.aligned_heading_ros_mps),
        static_cast<double>(baseline.aligned_heading_ros_mps));
    EXPECT_GT(static_cast<double>(with_waf.slope_factor), 0.0);

    const FireVec2 upslope{-0.6, 0.8};
    const auto spread = ERFFire::make_richards_directional_spread(
        with_waf, {0.0, 0.0}, upslope);

    EXPECT_TRUE(spread.forcing.has_preferred_heading());
    EXPECT_NEAR(
        static_cast<double>(spread.forcing.heading_unit().x), -0.6, 1.0e-15);
    EXPECT_NEAR(
        static_cast<double>(spread.forcing.heading_unit().y), 0.8, 1.0e-15);
    EXPECT_GT(
        static_cast<double>(spread.ellipse.heading_ros_mps),
        static_cast<double>(with_waf.no_wind_no_slope_ros_mps));
}

TEST(FireWindAdjustment, UnityWafMatchesUnadjustedRothermelRichardsPath)
{
    const FireVec2 reference{0.6, 0.8};
    const FireVec2 adjusted =
        ERFFire::fire_midflame_wind_from_20ft_reference(reference, 1.0);
    const amrex::Real direct_speed = ERFFire::norm(reference);
    const amrex::Real adjusted_speed = ERFFire::norm(adjusted);
    const FireVec2 direct_push = reference / direct_speed;
    const FireVec2 adjusted_push = adjusted / adjusted_speed;
    const FireVec2 upslope{0.0, 1.0};
    const auto fuel = ERFFire::make_fm1_fuel_parameters();

    const auto direct_behavior = ERFFire::evaluate_rothermel(
        fuel, RothermelInputs{0.08, direct_speed, 0.20});
    const auto adjusted_behavior = ERFFire::evaluate_rothermel(
        fuel, RothermelInputs{0.08, adjusted_speed, 0.20});

    EXPECT_DOUBLE_EQ(
        static_cast<double>(adjusted_behavior.wind_factor),
        static_cast<double>(direct_behavior.wind_factor));
    EXPECT_DOUBLE_EQ(
        static_cast<double>(adjusted_behavior.slope_factor),
        static_cast<double>(direct_behavior.slope_factor));
    EXPECT_DOUBLE_EQ(
        static_cast<double>(adjusted_behavior.aligned_heading_ros_mps),
        static_cast<double>(direct_behavior.aligned_heading_ros_mps));

    const auto direct_spread = ERFFire::make_richards_directional_spread(
        direct_behavior, direct_push, upslope);
    const auto adjusted_spread = ERFFire::make_richards_directional_spread(
        adjusted_behavior, adjusted_push, upslope);

    EXPECT_DOUBLE_EQ(
        static_cast<double>(adjusted_spread.forcing.resultant_vector.x),
        static_cast<double>(direct_spread.forcing.resultant_vector.x));
    EXPECT_DOUBLE_EQ(
        static_cast<double>(adjusted_spread.forcing.resultant_vector.y),
        static_cast<double>(direct_spread.forcing.resultant_vector.y));
    EXPECT_DOUBLE_EQ(
        static_cast<double>(adjusted_spread.ellipse.heading_unit.x),
        static_cast<double>(direct_spread.ellipse.heading_unit.x));
    EXPECT_DOUBLE_EQ(
        static_cast<double>(adjusted_spread.ellipse.heading_unit.y),
        static_cast<double>(direct_spread.ellipse.heading_unit.y));
    EXPECT_DOUBLE_EQ(
        static_cast<double>(adjusted_spread.ellipse.heading_ros_mps),
        static_cast<double>(direct_spread.ellipse.heading_ros_mps));
    EXPECT_DOUBLE_EQ(
        static_cast<double>(adjusted_spread.ellipse.effective_midflame_wind_mps),
        static_cast<double>(direct_spread.ellipse.effective_midflame_wind_mps));
    EXPECT_DOUBLE_EQ(
        static_cast<double>(adjusted_spread.ellipse.semi_minor_rate_mps),
        static_cast<double>(direct_spread.ellipse.semi_minor_rate_mps));
    EXPECT_DOUBLE_EQ(
        static_cast<double>(adjusted_spread.ellipse.semi_major_rate_mps),
        static_cast<double>(direct_spread.ellipse.semi_major_rate_mps));
    EXPECT_DOUBLE_EQ(
        static_cast<double>(adjusted_spread.ellipse.center_translation_rate_mps),
        static_cast<double>(direct_spread.ellipse.center_translation_rate_mps));
}

} // namespace
