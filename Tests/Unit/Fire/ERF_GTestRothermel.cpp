#include <ERF_RothermelFuel.H>
#include <ERF_RothermelModel.H>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace
{

using ERFFire::RothermelFuelParameters;
using ERFFire::RothermelInputs;
using ERFFire::RothermelMulticlassInputs;

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


TEST(FireRothermelFuel, Anderson13CatalogMatchesAlbiniTable7)
{
    struct NativeExpected
    {
        double sav_1h{};
        double load_1h{};
        double load_10h{};
        double load_100h{};
        double sav_live{};
        double load_live{};
        double depth_ft{};
        double extinction{};
    };

    // Albini (1976), Appendix III, table 7. Loading values are lb/ft^2,
    // SAV values are 1/ft, and depth is ft. Dead 10-h/100-h SAV are the
    // published 109 and 30 1/ft whenever those classes are present.
    constexpr std::array<NativeExpected, 13> expected{{
        {3500.0, 0.034, 0.000, 0.000,    0.0, 0.000, 1.0, 0.12},
        {3000.0, 0.092, 0.046, 0.023, 1500.0, 0.023, 1.0, 0.15},
        {1500.0, 0.138, 0.000, 0.000,    0.0, 0.000, 2.5, 0.25},
        {2000.0, 0.230, 0.184, 0.092, 1500.0, 0.230, 6.0, 0.20},
        {2000.0, 0.046, 0.023, 0.000, 1500.0, 0.092, 2.0, 0.20},
        {1750.0, 0.069, 0.115, 0.092,    0.0, 0.000, 2.5, 0.25},
        {1750.0, 0.052, 0.086, 0.069, 1550.0, 0.017, 2.5, 0.40},
        {2000.0, 0.069, 0.046, 0.115,    0.0, 0.000, 0.2, 0.30},
        {2500.0, 0.134, 0.019, 0.007,    0.0, 0.000, 0.2, 0.25},
        {2000.0, 0.138, 0.092, 0.230, 1500.0, 0.092, 1.0, 0.25},
        {1500.0, 0.069, 0.207, 0.253,    0.0, 0.000, 1.0, 0.15},
        {1500.0, 0.184, 0.644, 0.759,    0.0, 0.000, 2.3, 0.20},
        {1500.0, 0.322, 1.058, 1.288,    0.0, 0.000, 3.0, 0.25}
    }};

    constexpr double foot_m = 0.3048;
    constexpr double pound_kg = 0.45359237;
    constexpr double btu_j = 1055.05585262;

    const auto load_lb_ft2 = [=](amrex::Real value) {
        return static_cast<double>(value)
            * foot_m * foot_m / pound_kg;
    };
    const auto sav_ft_inv = [=](amrex::Real value) {
        return static_cast<double>(value) * foot_m;
    };

    for (std::size_t index = 0; index < expected.size(); ++index) {
        const auto fuel =
            ERFFire::make_anderson13_fuel_parameters(
                static_cast<int>(index + 1));
        const auto& e = expected[index];

        EXPECT_NEAR(load_lb_ft2(fuel.dead_1h.dry_load_kg_m2),
                    e.load_1h, 1.0e-14);
        EXPECT_NEAR(sav_ft_inv(fuel.dead_1h.sav_m_inv),
                    e.sav_1h, 1.0e-11);

        EXPECT_NEAR(load_lb_ft2(fuel.dead_10h.dry_load_kg_m2),
                    e.load_10h, 1.0e-14);
        EXPECT_NEAR(load_lb_ft2(fuel.dead_100h.dry_load_kg_m2),
                    e.load_100h, 1.0e-14);
        EXPECT_NEAR(load_lb_ft2(fuel.live_foliage.dry_load_kg_m2),
                    e.load_live, 1.0e-14);

        if (e.load_10h > 0.0) {
            EXPECT_NEAR(sav_ft_inv(fuel.dead_10h.sav_m_inv),
                        109.0, 1.0e-12);
        } else {
            EXPECT_DOUBLE_EQ(
                static_cast<double>(fuel.dead_10h.sav_m_inv), 0.0);
        }

        if (e.load_100h > 0.0) {
            EXPECT_NEAR(sav_ft_inv(fuel.dead_100h.sav_m_inv),
                        30.0, 1.0e-12);
        } else {
            EXPECT_DOUBLE_EQ(
                static_cast<double>(fuel.dead_100h.sav_m_inv), 0.0);
        }

        if (e.load_live > 0.0) {
            EXPECT_NEAR(sav_ft_inv(fuel.live_foliage.sav_m_inv),
                        e.sav_live, 1.0e-11);
        } else {
            EXPECT_DOUBLE_EQ(
                static_cast<double>(fuel.live_foliage.sav_m_inv), 0.0);
        }

        EXPECT_NEAR(
            static_cast<double>(fuel.fuel_bed_depth_m) / foot_m,
            e.depth_ft, 1.0e-14);
        EXPECT_DOUBLE_EQ(
            static_cast<double>(fuel.dead_moisture_of_extinction),
            e.extinction);

        EXPECT_NEAR(
            static_cast<double>(fuel.heat_content_j_kg)
                * pound_kg / btu_j,
            8000.0, 1.0e-11);
        EXPECT_NEAR(
            static_cast<double>(fuel.particle_density_kg_m3)
                * foot_m * foot_m * foot_m / pound_kg,
            32.0, 1.0e-13);
        EXPECT_DOUBLE_EQ(
            static_cast<double>(fuel.total_mineral_fraction),
            0.0555);
        EXPECT_DOUBLE_EQ(
            static_cast<double>(fuel.effective_mineral_fraction),
            0.01);
    }
}

TEST(FireRothermelFuel, FM1LegacyParametersAreExactCatalogProjection)
{
    const auto legacy = ERFFire::make_fm1_fuel_parameters();
    const auto catalog = ERFFire::make_anderson13_fuel_parameters(1);

    EXPECT_EQ(legacy.dead_1h_load_kg_m2,
              catalog.dead_1h.dry_load_kg_m2);
    EXPECT_EQ(legacy.dead_1h_sav_m_inv,
              catalog.dead_1h.sav_m_inv);
    EXPECT_EQ(legacy.fuel_bed_depth_m,
              catalog.fuel_bed_depth_m);
    EXPECT_EQ(legacy.dead_heat_content_j_kg,
              catalog.heat_content_j_kg);
    EXPECT_EQ(legacy.particle_density_kg_m3,
              catalog.particle_density_kg_m3);
    EXPECT_EQ(legacy.total_mineral_fraction,
              catalog.total_mineral_fraction);
    EXPECT_EQ(legacy.effective_mineral_fraction,
              catalog.effective_mineral_fraction);
    EXPECT_EQ(legacy.dead_moisture_of_extinction,
              catalog.dead_moisture_of_extinction);

    EXPECT_EQ(catalog.dead_10h.dry_load_kg_m2, amrex::Real(0));
    EXPECT_EQ(catalog.dead_100h.dry_load_kg_m2, amrex::Real(0));
    EXPECT_EQ(catalog.live_foliage.dry_load_kg_m2, amrex::Real(0));
}

TEST(FireRothermelFuel, Anderson13CatalogRejectsInvalidModelNumber)
{
    EXPECT_THROW(
        (void)ERFFire::make_anderson13_fuel_parameters(0),
        std::invalid_argument);
    EXPECT_THROW(
        (void)ERFFire::make_anderson13_fuel_parameters(14),
        std::invalid_argument);
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

TEST(FireRothermelMulticlass, FM1ReducesToLegacySingleClassPhysics)
{
    const auto legacy = ERFFire::evaluate_rothermel(
        ERFFire::make_fm1_fuel_parameters(),
        RothermelInputs{0.08, 1.0, 0.20});

    // Moistures for absent classes are intentionally different; they must
    // have no influence on the projected FM1 result.
    const auto multi = ERFFire::evaluate_rothermel_multiclass(
        ERFFire::make_anderson13_fuel_parameters(1),
        RothermelMulticlassInputs{
            0.08, 0.31, 0.47, 1.70, 1.0, 0.20});

    EXPECT_NEAR(
        static_cast<double>(multi.dead_net_fuel_loading_kg_m2),
        static_cast<double>(legacy.net_fuel_loading_kg_m2),
        1.0e-14);
    EXPECT_DOUBLE_EQ(
        static_cast<double>(multi.live_net_fuel_loading_kg_m2), 0.0);
    EXPECT_NEAR(
        static_cast<double>(multi.bulk_density_kg_m3),
        static_cast<double>(legacy.bulk_density_kg_m3),
        1.0e-14);
    EXPECT_NEAR(
        static_cast<double>(multi.packing_ratio),
        static_cast<double>(legacy.packing_ratio),
        1.0e-15);
    EXPECT_NEAR(
        static_cast<double>(multi.optimum_packing_ratio),
        static_cast<double>(legacy.optimum_packing_ratio),
        1.0e-15);
    EXPECT_NEAR(
        static_cast<double>(multi.reaction_velocity_exponent),
        static_cast<double>(legacy.reaction_velocity_exponent),
        1.0e-14);
    EXPECT_NEAR(
        static_cast<double>(multi.max_reaction_velocity_s_inv),
        static_cast<double>(legacy.max_reaction_velocity_s_inv),
        1.0e-13);
    EXPECT_NEAR(
        static_cast<double>(multi.reaction_velocity_s_inv),
        static_cast<double>(legacy.reaction_velocity_s_inv),
        1.0e-13);
    EXPECT_NEAR(
        static_cast<double>(multi.dead_moisture_damping),
        static_cast<double>(legacy.moisture_damping),
        1.0e-14);
    EXPECT_DOUBLE_EQ(
        static_cast<double>(multi.live_moisture_damping), 0.0);
    EXPECT_NEAR(
        static_cast<double>(multi.mineral_damping),
        static_cast<double>(legacy.mineral_damping),
        1.0e-14);
    EXPECT_NEAR(
        static_cast<double>(multi.dead_reaction_intensity_w_m2),
        static_cast<double>(legacy.reaction_intensity_w_m2),
        1.0e-7);
    EXPECT_DOUBLE_EQ(
        static_cast<double>(multi.live_reaction_intensity_w_m2), 0.0);
    EXPECT_NEAR(
        static_cast<double>(multi.reaction_intensity_w_m2),
        static_cast<double>(legacy.reaction_intensity_w_m2),
        1.0e-7);
    EXPECT_NEAR(
        static_cast<double>(multi.propagating_flux_ratio),
        static_cast<double>(legacy.propagating_flux_ratio),
        1.0e-14);
    EXPECT_NEAR(
        static_cast<double>(multi.heat_sink_j_m3),
        static_cast<double>(legacy.heat_sink_j_m3),
        1.0e-7);
    EXPECT_NEAR(
        static_cast<double>(multi.no_wind_no_slope_ros_mps),
        static_cast<double>(legacy.no_wind_no_slope_ros_mps),
        1.0e-14);
    EXPECT_NEAR(
        static_cast<double>(multi.wind_factor),
        static_cast<double>(legacy.wind_factor),
        1.0e-13);
    EXPECT_NEAR(
        static_cast<double>(multi.slope_factor),
        static_cast<double>(legacy.slope_factor),
        1.0e-13);
    EXPECT_NEAR(
        static_cast<double>(multi.aligned_heading_ros_mps),
        static_cast<double>(legacy.aligned_heading_ros_mps),
        1.0e-13);
}

TEST(FireRothermelMulticlass, Model2LiveDeadFixtureMatchesIndependentCalculation)
{
    const auto result = ERFFire::evaluate_rothermel_multiclass(
        ERFFire::make_anderson13_fuel_parameters(2),
        RothermelMulticlassInputs{
            0.08, 0.08, 0.08, 1.00,
            5.0 * mph_to_mps, 0.0});

    // Independently evaluated from the Rothermel surface-area weighting and
    // Albini computer-form corrections using the table-7 model-2 inputs.
    EXPECT_NEAR(static_cast<double>(result.dead_net_fuel_loading_kg_m2),
                0.4196986901241104, 1.0e-13);
    EXPECT_NEAR(static_cast<double>(result.live_net_fuel_loading_kg_m2),
                0.10606341675896719, 1.0e-13);
    EXPECT_NEAR(static_cast<double>(result.characteristic_dead_moisture_fraction),
                0.08, 1.0e-15);
    EXPECT_DOUBLE_EQ(
        static_cast<double>(result.characteristic_live_moisture_fraction),
        1.0);
    EXPECT_NEAR(static_cast<double>(result.live_moisture_of_extinction),
                8.073269029040183, 1.0e-12);
    EXPECT_NEAR(static_cast<double>(result.characteristic_sav_m_inv),
                9133.913155203867, 1.0e-9);
    EXPECT_NEAR(static_cast<double>(result.dead_reaction_intensity_w_m2),
                464814.36502015276, 1.0e-6);
    EXPECT_NEAR(static_cast<double>(result.live_reaction_intensity_w_m2),
                163893.35963937454, 1.0e-6);
    EXPECT_NEAR(static_cast<double>(result.heat_sink_j_m3),
                2881375.3018895136, 1.0e-6);
    EXPECT_NEAR(static_cast<double>(result.no_wind_no_slope_ros_mps),
                0.011595890573710788, 1.0e-14);
    EXPECT_NEAR(static_cast<double>(result.wind_factor),
                14.663076039651468, 1.0e-12);
    EXPECT_NEAR(static_cast<double>(result.aligned_heading_ros_mps),
                0.18162731580350977, 1.0e-13);
}

TEST(FireRothermelMulticlass, Model11AreaWeightsDeadMoistureClasses)
{
    const auto result = ERFFire::evaluate_rothermel_multiclass(
        ERFFire::make_anderson13_fuel_parameters(11),
        RothermelMulticlassInputs{
            0.04, 0.08, 0.12, 0.0,
            0.8, 0.20});

    EXPECT_NEAR(static_cast<double>(result.characteristic_dead_moisture_fraction),
                0.05129581827568404, 1.0e-14);
    EXPECT_NEAR(static_cast<double>(result.characteristic_sav_m_inv),
                3876.9517355761604, 1.0e-10);
    EXPECT_NEAR(static_cast<double>(result.bulk_density_kg_m3),
                8.473767124824914, 1.0e-12);
    EXPECT_NEAR(static_cast<double>(result.dead_moisture_damping),
                0.5711092964059014, 1.0e-13);
    EXPECT_DOUBLE_EQ(
        static_cast<double>(result.live_reaction_intensity_w_m2), 0.0);
    EXPECT_NEAR(static_cast<double>(result.dead_reaction_intensity_w_m2),
                457227.3297030352, 1.0e-6);
    EXPECT_NEAR(static_cast<double>(result.heat_sink_j_m3),
                4424503.227893632, 1.0e-6);
    EXPECT_NEAR(static_cast<double>(result.no_wind_no_slope_ros_mps),
                0.0034777923915500185, 1.0e-15);
    EXPECT_NEAR(static_cast<double>(result.wind_factor),
                3.0132182036836346, 1.0e-13);
    EXPECT_NEAR(static_cast<double>(result.slope_factor),
                0.7224216407916382, 1.0e-13);
    EXPECT_NEAR(static_cast<double>(result.aligned_heading_ros_mps),
                0.016469572220237216, 1.0e-14);
}

TEST(FireRothermelMulticlass, DeadExtinctionDoesNotSuppressLiveReaction)
{
    const auto result = ERFFire::evaluate_rothermel_multiclass(
        ERFFire::make_anderson13_fuel_parameters(4),
        RothermelMulticlassInputs{
            0.20, 0.20, 0.20, 0.05,
            4.0, 0.30});

    // The characteristic dead moisture is mathematically the 0.20 input for
    // all three dead classes. Roundoff in surface-area weighting must not
    // leave a residual dead reaction at the exact dead extinction boundary.
    EXPECT_NEAR(
        static_cast<double>(result.characteristic_dead_moisture_fraction),
        0.20, 1.0e-15);
    EXPECT_DOUBLE_EQ(
        static_cast<double>(result.dead_moisture_damping), 0.0);
    EXPECT_DOUBLE_EQ(
        static_cast<double>(result.dead_reaction_intensity_w_m2), 0.0);

    // Rothermel/Albini damp live and dead reaction intensity separately.
    // With live moisture 0.05 and live extinction limited to at least the
    // dead extinction value 0.20, the live term remains active.
    EXPECT_NEAR(
        static_cast<double>(result.live_moisture_of_extinction),
        0.20, 1.0e-15);
    EXPECT_NEAR(
        static_cast<double>(result.live_moisture_damping),
        0.616875, 1.0e-14);
    EXPECT_NEAR(
        static_cast<double>(result.live_reaction_intensity_w_m2),
        1198914.2533346876, 1.0e-6);
    EXPECT_EQ(
        result.reaction_intensity_w_m2,
        result.live_reaction_intensity_w_m2);
    EXPECT_NEAR(
        static_cast<double>(result.no_wind_no_slope_ros_mps),
        0.02315414913343321, 1.0e-14);
    EXPECT_GT(
        static_cast<double>(result.aligned_heading_ros_mps), 0.0);
    EXPECT_GT(
        static_cast<double>(result.heat_sink_j_m3), 0.0);
}

TEST(FireRothermelMulticlass, AndersonRepresentativeRatesHavePublishedScale)
{
    constexpr std::array<double, 13> published_chains_per_hour{{
        78.0, 35.0, 104.0,
        75.0, 18.0, 32.0, 20.0,
        1.6, 7.5, 7.9,
        6.0, 13.0, 13.5
    }};

    for (std::size_t index = 0;
         index < published_chains_per_hour.size();
         ++index) {
        const auto result = ERFFire::evaluate_rothermel_multiclass(
            ERFFire::make_anderson13_fuel_parameters(
                static_cast<int>(index + 1)),
            RothermelMulticlassInputs{
                0.08, 0.08, 0.08, 1.00,
                5.0 * mph_to_mps, 0.0});

        const double chains_per_hour =
            static_cast<double>(result.aligned_heading_ros_mps)
            * mps_to_chains_per_hour;

        // Anderson (1982) gives representative rounded rates from the
        // operational nomographs. The core intentionally omits the historical
        // wind-speed cap, so this is a scale-level independent reference, not
        // an equation-level equality test.
        EXPECT_NEAR(
            chains_per_hour,
            published_chains_per_hour[index],
            0.30 * published_chains_per_hour[index])
            << "fuel model " << (index + 1);
    }
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
