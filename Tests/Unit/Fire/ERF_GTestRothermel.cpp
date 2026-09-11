#include <ERF_RothermelFuel.H>
#include <ERF_RothermelModel.H>

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace
{

using ERFFire::RothermelFuelParameters;
using ERFFire::RothermelInputs;

constexpr double mph_to_mps = 0.44704;
constexpr double mps_to_chains_per_hour = 3600.0 / 20.1168;

TEST(FireRothermelFuel, FM1MatchesPublishedOriginalFuelModelParameters)
{
    const RothermelFuelParameters fuel =
        ERFFire::make_fm1_fuel_parameters();

    // Independent SI conversions of Albini (1976), Appendix III, table 7,
    // plus the common original-model particle properties documented there.
    EXPECT_NEAR(static_cast<double>(fuel.dead_1h_load_kg_m2),
                0.16600253963702372, 1.0e-14);
    EXPECT_NEAR(static_cast<double>(fuel.dead_1h_sav_m_inv),
                11482.939632545931, 1.0e-10);
    EXPECT_DOUBLE_EQ(static_cast<double>(fuel.fuel_bed_depth_m), 0.3048);
    EXPECT_DOUBLE_EQ(static_cast<double>(fuel.dead_heat_content_j_kg),
                     18608000.0);
    EXPECT_NEAR(static_cast<double>(fuel.particle_density_kg_m3),
                512.5908279667244, 1.0e-12);
    EXPECT_DOUBLE_EQ(static_cast<double>(fuel.total_mineral_fraction), 0.0555);
    EXPECT_DOUBLE_EQ(static_cast<double>(fuel.effective_mineral_fraction), 0.01);
    EXPECT_DOUBLE_EQ(static_cast<double>(fuel.dead_moisture_of_extinction), 0.12);
}

TEST(FireRothermel, FM1EightPercentZeroWindSlopeMatchesIndependentFixture)
{
    const auto result = ERFFire::evaluate_rothermel(
        ERFFire::make_fm1_fuel_parameters(),
        RothermelInputs{0.08, 0.0, 0.0});

    // Numerical fixture independently evaluated from the published
    // Rothermel (1972) equations with the Albini (1976) computer-form
    // corrections. The fixture contains values only; no production equation
    // is reused by this test.
    EXPECT_NEAR(static_cast<double>(result.net_fuel_loading_kg_m2),
                0.15678939868716891, 1.0e-13);
    EXPECT_NEAR(static_cast<double>(result.bulk_density_kg_m3),
                0.5446277547146448, 1.0e-13);
    EXPECT_NEAR(static_cast<double>(result.packing_ratio),
                0.0010625, 1.0e-15);
    EXPECT_NEAR(static_cast<double>(result.optimum_packing_ratio),
                0.004193224627380653, 1.0e-15);
    EXPECT_NEAR(static_cast<double>(result.reaction_velocity_exponent),
                0.2086558654295252, 1.0e-14);
    EXPECT_NEAR(static_cast<double>(result.max_reaction_velocity_s_inv),
                0.269728280402528, 1.0e-13);
    EXPECT_NEAR(static_cast<double>(result.reaction_velocity_s_inv),
                0.23668933177455445, 1.0e-13);
    EXPECT_NEAR(static_cast<double>(result.moisture_damping),
                0.5014814814814816, 1.0e-14);
    EXPECT_NEAR(static_cast<double>(result.mineral_damping),
                0.4173969279093913, 1.0e-14);
    EXPECT_NEAR(static_cast<double>(result.reaction_intensity_w_m2),
                144543.71878014246, 1.0e-7);
    EXPECT_NEAR(static_cast<double>(result.propagating_flux_ratio),
                0.0577521709187699, 1.0e-14);
    EXPECT_NEAR(static_cast<double>(result.effective_heating_number),
                0.9613386185824466, 1.0e-14);
    EXPECT_NEAR(static_cast<double>(result.heat_of_preignition_j_kg),
                789165.28, 1.0e-7);
    EXPECT_NEAR(static_cast<double>(result.heat_sink_j_m3),
                413184.60198975785, 1.0e-7);
    EXPECT_NEAR(static_cast<double>(result.no_wind_no_slope_ros_mps),
                0.02020335102524543, 1.0e-14);
    EXPECT_DOUBLE_EQ(static_cast<double>(result.wind_factor), 0.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(result.slope_factor), 0.0);
    EXPECT_NEAR(static_cast<double>(result.aligned_heading_ros_mps),
                0.02020335102524543, 1.0e-14);
}

TEST(FireRothermel, MoistureAtOrAboveExtinctionStopsSpreadExactly)
{
    const auto fuel = ERFFire::make_fm1_fuel_parameters();

    for (const amrex::Real moisture : {amrex::Real(0.12), amrex::Real(0.20)}) {
        const auto result = ERFFire::evaluate_rothermel(
            fuel,
            RothermelInputs{moisture, 2.0, 0.30});

        EXPECT_DOUBLE_EQ(static_cast<double>(result.moisture_damping), 0.0);
        EXPECT_DOUBLE_EQ(static_cast<double>(result.reaction_intensity_w_m2), 0.0);
        EXPECT_DOUBLE_EQ(static_cast<double>(result.no_wind_no_slope_ros_mps), 0.0);
        EXPECT_DOUBLE_EQ(static_cast<double>(result.aligned_heading_ros_mps), 0.0);
    }
}

TEST(FireRothermel, WindOnlyMatchesIndependentFixture)
{
    const auto result = ERFFire::evaluate_rothermel(
        ERFFire::make_fm1_fuel_parameters(),
        RothermelInputs{0.08, 1.0, 0.0});

    EXPECT_NEAR(static_cast<double>(result.wind_factor),
                4.049771770825199, 1.0e-13);
    EXPECT_DOUBLE_EQ(static_cast<double>(result.slope_factor), 0.0);
    EXPECT_NEAR(static_cast<double>(result.aligned_heading_ros_mps),
                0.10202231168335671, 1.0e-13);
}

TEST(FireRothermel, SlopeOnlyMatchesIndependentFixture)
{
    const auto result = ERFFire::evaluate_rothermel(
        ERFFire::make_fm1_fuel_parameters(),
        RothermelInputs{0.08, 0.0, 0.20});

    EXPECT_DOUBLE_EQ(static_cast<double>(result.wind_factor), 0.0);
    EXPECT_NEAR(static_cast<double>(result.slope_factor),
                1.645825449949636, 1.0e-13);
    EXPECT_NEAR(static_cast<double>(result.aligned_heading_ros_mps),
                0.053454540316860436, 1.0e-13);
}

TEST(FireRothermel, WindAndSlopeAmplificationsAddInPublishedScalarEquation)
{
    const auto result = ERFFire::evaluate_rothermel(
        ERFFire::make_fm1_fuel_parameters(),
        RothermelInputs{0.08, 1.0, 0.20});

    EXPECT_NEAR(static_cast<double>(result.wind_factor),
                4.049771770825199, 1.0e-13);
    EXPECT_NEAR(static_cast<double>(result.slope_factor),
                1.645825449949636, 1.0e-13);
    EXPECT_NEAR(static_cast<double>(result.aligned_heading_ros_mps),
                0.1352735009749717, 1.0e-13);

    EXPECT_NEAR(
        static_cast<double>(result.aligned_heading_ros_mps),
        static_cast<double>(
            result.no_wind_no_slope_ros_mps
            * (1.0 + result.wind_factor + result.slope_factor)),
        1.0e-13);
}

TEST(FireRothermel, AndersonRepresentativeFM1CaseIsRecoveredWithinPublishedPrecision)
{
    const auto result = ERFFire::evaluate_rothermel(
        ERFFire::make_fm1_fuel_parameters(),
        RothermelInputs{0.08, 5.0 * mph_to_mps, 0.0});

    const double chains_per_hour =
        static_cast<double>(result.aligned_heading_ros_mps)
        * mps_to_chains_per_hour;

    // Anderson (1982) reports a representative FM1 rate of 78 chains/hour
    // at 8% moisture and 5 mi/h wind. The publication presents rounded
    // operational values, so this is intentionally a coarse independent
    // reference rather than the tight equation-level fixture above.
    EXPECT_NEAR(chains_per_hour, 78.0, 5.0);
}

TEST(FireRothermel, RejectsInvalidPhysicalInputs)
{
    const auto fuel = ERFFire::make_fm1_fuel_parameters();

    EXPECT_THROW(
        (void)ERFFire::evaluate_rothermel(
            fuel, RothermelInputs{-0.01, 0.0, 0.0}),
        std::invalid_argument);
    EXPECT_THROW(
        (void)ERFFire::evaluate_rothermel(
            fuel, RothermelInputs{0.08, -1.0, 0.0}),
        std::invalid_argument);
    EXPECT_THROW(
        (void)ERFFire::evaluate_rothermel(
            fuel, RothermelInputs{0.08, 0.0, -0.1}),
        std::invalid_argument);
    EXPECT_THROW(
        (void)ERFFire::evaluate_rothermel(
            fuel,
            RothermelInputs{
                std::numeric_limits<amrex::Real>::infinity(), 0.0, 0.0}),
        std::invalid_argument);

    auto bad_fuel = fuel;
    bad_fuel.fuel_bed_depth_m = 0.0;
    EXPECT_THROW(
        (void)ERFFire::evaluate_rothermel(
            bad_fuel, RothermelInputs{0.08, 0.0, 0.0}),
        std::invalid_argument);
}

} // namespace
