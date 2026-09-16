#include <ERF_RichardsDirectionalSpread.H>
#include <ERF_RothermelFuel.H>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace
{

using ERFFire::FireVec2;
using ERFFire::RothermelInputs;

TEST(FireRichardsDirectional, OrthogonalFactorsUseVectorMagnitude)
{
    const auto result = ERFFire::combine_rothermel_wind_slope_factors(
        3.0, {1.0, 0.0}, 4.0, {0.0, 1.0});

    EXPECT_TRUE(result.has_preferred_heading());
    EXPECT_DOUBLE_EQ(static_cast<double>(result.resultant_vector.x), 3.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(result.resultant_vector.y), 4.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(result.resultant_factor()), 5.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(result.heading_unit().x), 0.6);
    EXPECT_DOUBLE_EQ(static_cast<double>(result.heading_unit().y), 0.8);
}

TEST(FireRichardsDirectional, OpposingEqualFactorsCancelToIsotropy)
{
    const auto result = ERFFire::combine_rothermel_wind_slope_factors(
        2.0, {1.0, 0.0}, 2.0, {-1.0, 0.0});

    EXPECT_FALSE(result.has_preferred_heading());
    EXPECT_DOUBLE_EQ(static_cast<double>(result.resultant_vector.x), 0.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(result.resultant_vector.y), 0.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(result.resultant_factor()), 0.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(result.heading_unit().x), 1.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(result.heading_unit().y), 0.0);
}

TEST(FireRichardsDirectional, RothermelWindFactorInverseRoundTripsFM1)
{
    const auto fuel = ERFFire::make_fm1_fuel_parameters();

    const auto reference = ERFFire::evaluate_rothermel(
        fuel, RothermelInputs{0.08, 1.0, 0.0});

    // Independent FM1 fixtures for the SI power-law representation
    // Phi_w = K U_mps^B, obtained from the published Rothermel/Albini
    // coefficients and exact SI conversion.
    EXPECT_NEAR(
        static_cast<double>(reference.wind_factor_coefficient_si),
        4.049771770825199,
        1.0e-13);
    EXPECT_NEAR(
        static_cast<double>(reference.wind_factor_exponent),
        2.071238404800941,
        1.0e-14);

    for (const amrex::Real wind_mps : {
            amrex::Real(0.0),
            amrex::Real(0.1),
            amrex::Real(0.5),
            amrex::Real(1.0),
            amrex::Real(2.0)}) {
        const auto behavior = ERFFire::evaluate_rothermel(
            fuel, RothermelInputs{0.08, wind_mps, 0.0});
        const amrex::Real recovered =
            ERFFire::rothermel_model_wind_speed_for_factor_mps(
                behavior, behavior.wind_factor);

        EXPECT_NEAR(
            static_cast<double>(recovered),
            static_cast<double>(wind_mps),
            2.0e-14);
    }
}

TEST(FireRichardsDirectional, WindOnlyRecoversWindHeadingAndOriginalWind)
{
    const auto behavior = ERFFire::evaluate_rothermel(
        ERFFire::make_fm1_fuel_parameters(),
        RothermelInputs{0.08, 1.0, 0.0});

    const auto spread = ERFFire::make_richards_directional_spread(
        behavior, {0.6, 0.8}, {0.0, 0.0});

    EXPECT_TRUE(spread.forcing.has_preferred_heading());
    EXPECT_NEAR(static_cast<double>(spread.forcing.heading_unit().x), 0.6, 1.0e-15);
    EXPECT_NEAR(static_cast<double>(spread.forcing.heading_unit().y), 0.8, 1.0e-15);
    EXPECT_NEAR(
        static_cast<double>(spread.ellipse.effective_midflame_wind_mps),
        1.0,
        2.0e-14);
    EXPECT_NEAR(
        static_cast<double>(spread.ellipse.heading_ros_mps),
        static_cast<double>(behavior.aligned_heading_ros_mps),
        2.0e-14);
}

TEST(FireRichardsDirectional, SlopeOnlyPointsUpslopeWithEquivalentWind)
{
    const auto behavior = ERFFire::evaluate_rothermel(
        ERFFire::make_fm1_fuel_parameters(),
        RothermelInputs{0.08, 0.0, 0.20});

    const auto spread = ERFFire::make_richards_directional_spread(
        behavior, {0.0, 0.0}, {-0.6, 0.8});

    EXPECT_TRUE(spread.forcing.has_preferred_heading());
    EXPECT_NEAR(static_cast<double>(spread.forcing.heading_unit().x), -0.6, 1.0e-15);
    EXPECT_NEAR(static_cast<double>(spread.forcing.heading_unit().y), 0.8, 1.0e-15);
    EXPECT_NEAR(
        static_cast<double>(spread.ellipse.effective_midflame_wind_mps),
        0.647442889557713,
        2.0e-14);
    EXPECT_NEAR(
        static_cast<double>(spread.ellipse.heading_ros_mps),
        static_cast<double>(behavior.aligned_heading_ros_mps),
        2.0e-14);
}

TEST(FireRichardsDirectional, AlignedWindAndSlopeRecoverScalarRothermelRate)
{
    const auto behavior = ERFFire::evaluate_rothermel(
        ERFFire::make_fm1_fuel_parameters(),
        RothermelInputs{0.08, 1.0, 0.20});

    const auto spread = ERFFire::make_richards_directional_spread(
        behavior, {0.0, 1.0}, {0.0, 1.0});

    EXPECT_NEAR(
        static_cast<double>(spread.forcing.resultant_factor()),
        static_cast<double>(behavior.wind_factor + behavior.slope_factor),
        1.0e-14);
    EXPECT_NEAR(static_cast<double>(spread.forcing.heading_unit().x), 0.0, 1.0e-15);
    EXPECT_NEAR(static_cast<double>(spread.forcing.heading_unit().y), 1.0, 1.0e-15);
    EXPECT_NEAR(
        static_cast<double>(spread.ellipse.effective_midflame_wind_mps),
        1.1789824120921781,
        3.0e-14);
    EXPECT_NEAR(
        static_cast<double>(spread.ellipse.heading_ros_mps),
        static_cast<double>(behavior.aligned_heading_ros_mps),
        2.0e-14);
}

TEST(FireRichardsDirectional, ObliqueWindSlopeMatchesIndependentFM1FixtureAndMirrors)
{
    const auto behavior = ERFFire::evaluate_rothermel(
        ERFFire::make_fm1_fuel_parameters(),
        RothermelInputs{0.08, 1.0, 0.20});

    const auto north_slope = ERFFire::make_richards_directional_spread(
        behavior, {1.0, 0.0}, {0.0, 1.0});
    const auto south_slope = ERFFire::make_richards_directional_spread(
        behavior, {1.0, 0.0}, {0.0, -1.0});

    // Independent numerical fixture: vector the published dimensionless
    // coefficients, then invert the published wind-factor power law.
    EXPECT_NEAR(
        static_cast<double>(north_slope.forcing.resultant_factor()),
        4.371429149314283,
        2.0e-14);
    EXPECT_NEAR(
        static_cast<double>(north_slope.forcing.heading_unit().x),
        0.9264182564780811,
        2.0e-15);
    EXPECT_NEAR(
        static_cast<double>(north_slope.forcing.heading_unit().y),
        0.37649596819104475,
        2.0e-15);
    EXPECT_NEAR(
        static_cast<double>(north_slope.ellipse.effective_midflame_wind_mps),
        1.0375896416122365,
        3.0e-14);
    EXPECT_NEAR(
        static_cast<double>(north_slope.ellipse.heading_ros_mps),
        0.10852086861083189,
        3.0e-14);

    EXPECT_NEAR(
        static_cast<double>(south_slope.forcing.heading_unit().x),
        static_cast<double>(north_slope.forcing.heading_unit().x),
        2.0e-15);
    EXPECT_NEAR(
        static_cast<double>(south_slope.forcing.heading_unit().y),
        static_cast<double>(-north_slope.forcing.heading_unit().y),
        2.0e-15);
    EXPECT_NEAR(
        static_cast<double>(south_slope.ellipse.effective_midflame_wind_mps),
        static_cast<double>(north_slope.ellipse.effective_midflame_wind_mps),
        2.0e-14);
    EXPECT_NEAR(
        static_cast<double>(south_slope.ellipse.heading_ros_mps),
        static_cast<double>(north_slope.ellipse.heading_ros_mps),
        2.0e-14);
}

TEST(FireRichardsDirectional, ComposedObliqueWaveletNormalsMatchFixedFixture)
{
    const auto behavior = ERFFire::evaluate_rothermel(
        ERFFire::make_fm1_fuel_parameters(),
        RothermelInputs{0.08, 1.0, 0.20});

    const auto spread = ERFFire::make_richards_directional_spread(
        behavior, {1.0, 0.0}, {0.0, 1.0});

    const FireVec2 heading = spread.forcing.heading_unit();
    const FireVec2 backing{-heading.x, -heading.y};
    const FireVec2 flank{-heading.y, heading.x};

    EXPECT_NEAR(
        static_cast<double>(
            ERFFire::richards_normal_speed_mps(spread.ellipse, heading)),
        0.10852086861083189,
        3.0e-14);
    EXPECT_NEAR(
        static_cast<double>(
            ERFFire::richards_normal_speed_mps(spread.ellipse, backing)),
        0.029717476468763306,
        3.0e-14);
    EXPECT_NEAR(
        static_cast<double>(
            ERFFire::richards_normal_speed_mps(spread.ellipse, flank)),
        0.056788787267489274,
        3.0e-14);
}

TEST(FireRichardsDirectional, ExtinguishedBehaviorProducesStationaryWavelet)
{
    const auto behavior = ERFFire::evaluate_rothermel(
        ERFFire::make_fm1_fuel_parameters(),
        RothermelInputs{0.12, 1.0, 0.20});

    const auto spread = ERFFire::make_richards_directional_spread(
        behavior, {1.0, 0.0}, {0.0, 1.0});

    EXPECT_DOUBLE_EQ(static_cast<double>(spread.ellipse.heading_ros_mps), 0.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(spread.ellipse.semi_minor_rate_mps), 0.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(spread.ellipse.semi_major_rate_mps), 0.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(spread.ellipse.center_translation_rate_mps), 0.0);

    for (const FireVec2 normal : {
            FireVec2{1.0, 0.0},
            FireVec2{-1.0, 0.0},
            FireVec2{0.0, 1.0}}) {
        EXPECT_DOUBLE_EQ(
            static_cast<double>(ERFFire::richards_normal_speed_mps(
                spread.ellipse, normal)),
            0.0);
    }
}

TEST(FireRichardsDirectional, RejectsInvalidDirectionalInputs)
{
    EXPECT_THROW(
        (void)ERFFire::combine_rothermel_wind_slope_factors(
            -1.0, {1.0, 0.0}, 0.0, {0.0, 0.0}),
        std::invalid_argument);
    EXPECT_THROW(
        (void)ERFFire::combine_rothermel_wind_slope_factors(
            1.0, {2.0, 0.0}, 0.0, {0.0, 0.0}),
        std::invalid_argument);
    EXPECT_THROW(
        (void)ERFFire::combine_rothermel_wind_slope_factors(
            0.0,
            {std::numeric_limits<amrex::Real>::infinity(), 0.0},
            0.0,
            {0.0, 0.0}),
        std::invalid_argument);

    auto behavior = ERFFire::evaluate_rothermel(
        ERFFire::make_fm1_fuel_parameters(),
        RothermelInputs{0.08, 1.0, 0.0});
    EXPECT_THROW(
        (void)ERFFire::rothermel_model_wind_speed_for_factor_mps(
            behavior, -1.0),
        std::invalid_argument);

    behavior.wind_factor_coefficient_si = 0.0;
    EXPECT_THROW(
        (void)ERFFire::rothermel_model_wind_speed_for_factor_mps(
            behavior, 1.0),
        std::invalid_argument);
}

} // namespace
