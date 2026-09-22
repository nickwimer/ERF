#include <ERF_FireCombustion.H>
#include <ERF_FireFuelBurnTime.H>
#include <ERF_FireFuelCombustion.H>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{
using amrex::Real;
using namespace ERFFire;

FireFuelRasterCell material(int model)
{
    FireFuelRasterCell cell;
    cell.model_id = static_cast<FireFuelModelId>(model);
    const auto contract = fire_fuel_model_moisture_contract(cell.model_id);
    for (int n = 0; n < FireFuelMoisture::component_count; ++n) {
        const auto c = static_cast<FireFuelMoistureClass>(n);
        if ((contract.required_mask & fire_fuel_moisture_component_bit(c)) != 0u) {
            cell.moisture.set(c, n < 3 ? Real(0.08) : Real(0.8));
        }
    }
    return cell;
}

Real tolerance(Real value)
{
    return Real(512) * std::numeric_limits<Real>::epsilon()
        * std::max(Real(1), std::abs(value));
}
}

TEST(FireScientificBurnTime, ProductionUsesFuelDependentTimes)
{
    // Independent table from SFIRE's DATA weight, not an expectation obtained
    // by invoking the implementation's table accessor.
    const int weights[] = {7, 7, 7, 180, 100, 100, 100,
                           900, 900, 900, 900, 900, 900};
    const auto base = make_fm1_combustion_parameters(Real(0.08));
    for (int model = 1; model <= 13; ++model) {
        SCOPED_TRACE(model);
        const auto cell = material(model);
        const auto actual = make_spatial_fire_combustion_accounting(base, cell);
        const Real expected_tau = Real(weights[model - 1]) / Real(0.85);
        EXPECT_NEAR(actual.parameters.burn_time_constant_s,
                    expected_tau, tolerance(expected_tau));

        FireCombustionState initial;
        initial = add_fire_combustion_ignition(initial, actual.parameters, Real(1));
        const auto result = advance_fire_combustion(initial, actual.parameters, Real(60));
        const Real expected_remaining = actual.parameters.dry_fuel_load_kg_m2
            * std::exp(-Real(60) / expected_tau);
        EXPECT_NEAR(result.state.remaining_dry_fuel_kg_m2,
                    expected_remaining, tolerance(expected_remaining));
        EXPECT_NEAR(result.state.remaining_dry_fuel_kg_m2
                        + result.state.consumed_dry_fuel_kg_m2,
                    actual.parameters.dry_fuel_load_kg_m2,
                    tolerance(actual.parameters.dry_fuel_load_kg_m2));
    }
}

TEST(FireScientificBurnTime, SlashDoesNotBurnLikeGrass)
{
    const auto base = make_fm1_combustion_parameters(Real(0.08));
    const auto grass = make_spatial_fire_combustion_accounting(base, material(1));
    const auto slash = make_spatial_fire_combustion_accounting(base, material(13));
    // At 60 s the selected FM13 law consumes about 5.51%, not 99.93%.
    const Real slash_consumed_fraction =
        -std::expm1(-Real(60) / slash.parameters.burn_time_constant_s);
    EXPECT_GT(slash_consumed_fraction, Real(0.055));
    EXPECT_LT(slash_consumed_fraction, Real(0.056));
    EXPECT_GT(-std::expm1(-Real(60) / grass.parameters.burn_time_constant_s), Real(0.999));
    EXPECT_EQ(anderson13_sfire_burn_time_s(0), Real(0));
    EXPECT_EQ(anderson13_sfire_burn_time_s(14), Real(0));
}

TEST(FireScientificBurnTime, AccountingOnlyApiStillHonorsExplicitBase)
{
    auto base = make_fm1_combustion_parameters(Real(0.08));
    base.burn_time_constant_s = Real(123);
    // The accounting-only API is not the production default policy.
    EXPECT_EQ(make_anderson13_fire_combustion_accounting(base, material(13))
                  .parameters.burn_time_constant_s, Real(123));
    EXPECT_EQ(make_spatial_fire_combustion_accounting(base, material(1))
                  .parameters.burn_time_constant_s, Real(123));
}
