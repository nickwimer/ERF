#include <ERF_FireCombustion.H>
#include <ERF_RothermelFuel.H>

#include <AMReX_Gpu.H>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace
{

using amrex::Real;
using ERFFire::FireCombustionParameters;
using ERFFire::FireCombustionState;

Real scaled_tolerance(Real expected)
{
    return Real(1024)
        * std::numeric_limits<Real>::epsilon()
        * std::max(Real(1), std::abs(expected));
}

FireCombustionParameters synthetic_parameters()
{
    return {
        Real(2.0),
        Real(10.0),
        Real(0.25),
        Real(4.0),
        Real(0.5)
    };
}

} // namespace

TEST(FireCombustion, BurnTimeWeightMatchesSfirePointEightFiveRule)
{
    const Real weight = Real(7.0);
    const Real expected = weight / Real(0.85);

    EXPECT_NEAR(
        ERFFire::fire_burn_time_constant_from_weight(weight),
        expected,
        scaled_tolerance(expected));
    EXPECT_NEAR(
        ERFFire::fire_burn_time_constant_from_weight(weight),
        Real(8.235294117647058),
        Real(2.0e-14));
}

TEST(FireCombustion, FM1CombustionUsesValidatedDryLoadAndSfireHeatRelease)
{
    const Real moisture = Real(0.08);
    const auto combustion =
        ERFFire::make_fm1_combustion_parameters(moisture);
    const auto fuel = ERFFire::make_fm1_fuel_parameters();

    EXPECT_EQ(
        combustion.dry_fuel_load_kg_m2,
        fuel.dead_1h_load_kg_m2);
    EXPECT_EQ(
        combustion.sensible_heat_release_j_kg_dry,
        Real(17.433e6));
    EXPECT_NE(
        combustion.sensible_heat_release_j_kg_dry,
        fuel.dead_heat_content_j_kg);
    EXPECT_EQ(
        combustion.fuel_moisture_fraction,
        moisture);
    EXPECT_EQ(
        combustion.combustion_water_yield_kg_per_kg_dry,
        Real(0.56));

    const Real expected_burn_time =
        Real(7.0) / Real(0.85);
    EXPECT_NEAR(
        combustion.burn_time_constant_s,
        expected_burn_time,
        scaled_tolerance(expected_burn_time));
}

TEST(FireCombustion, IgnitionAddsDryFuelWithoutInstantConsumption)
{
    const auto p = synthetic_parameters();

    const auto state =
        ERFFire::add_fire_combustion_ignition(
            FireCombustionState{}, p, Real(0.3));

    EXPECT_EQ(state.ignited_area_fraction, Real(0.3));
    EXPECT_EQ(state.remaining_dry_fuel_kg_m2, Real(0.6));
    EXPECT_EQ(state.consumed_dry_fuel_kg_m2, Real(0.0));
    EXPECT_EQ(state.sensible_energy_j_m2, Real(0.0));
    EXPECT_EQ(state.water_released_kg_m2, Real(0.0));
}

TEST(FireCombustion, ExponentialDecayMatchesIndependentHalfLifeFixture)
{
    const auto p = synthetic_parameters();
    const auto state =
        ERFFire::add_fire_combustion_ignition(
            FireCombustionState{}, p, Real(1.0));

    const Real dt = p.burn_time_constant_s * std::log(Real(2.0));
    const auto update =
        ERFFire::advance_fire_combustion(state, p, dt);

    EXPECT_NEAR(
        update.state.remaining_dry_fuel_kg_m2,
        Real(1.0),
        scaled_tolerance(Real(1.0)));
    EXPECT_NEAR(
        update.newly_consumed_dry_fuel_kg_m2,
        Real(1.0),
        scaled_tolerance(Real(1.0)));
    EXPECT_NEAR(
        update.sensible_energy_increment_j_m2,
        Real(10.0),
        scaled_tolerance(Real(10.0)));
    EXPECT_NEAR(
        update.water_released_increment_kg_m2,
        Real(0.75),
        scaled_tolerance(Real(0.75)));
}

TEST(FireCombustion, SplitAdvanceHasExactExponentialSemigroupProperty)
{
    const auto p = synthetic_parameters();
    const auto state =
        ERFFire::add_fire_combustion_ignition(
            FireCombustionState{}, p, Real(0.8));

    const Real dt1 = Real(1.25);
    const Real dt2 = Real(2.75);

    const auto once =
        ERFFire::advance_fire_combustion(
            state, p, dt1 + dt2);
    const auto first =
        ERFFire::advance_fire_combustion(
            state, p, dt1);
    const auto split =
        ERFFire::advance_fire_combustion(
            first.state, p, dt2);

    EXPECT_NEAR(
        split.state.remaining_dry_fuel_kg_m2,
        once.state.remaining_dry_fuel_kg_m2,
        scaled_tolerance(once.state.remaining_dry_fuel_kg_m2));
    EXPECT_NEAR(
        split.state.consumed_dry_fuel_kg_m2,
        once.state.consumed_dry_fuel_kg_m2,
        scaled_tolerance(once.state.consumed_dry_fuel_kg_m2));
    EXPECT_NEAR(
        split.state.sensible_energy_j_m2,
        once.state.sensible_energy_j_m2,
        scaled_tolerance(once.state.sensible_energy_j_m2));
    EXPECT_NEAR(
        split.state.water_released_kg_m2,
        once.state.water_released_kg_m2,
        scaled_tolerance(once.state.water_released_kg_m2));
}

TEST(FireCombustion, StaggeredIgnitionCohortsSuperposeInOneReservoir)
{
    const auto p = synthetic_parameters();

    auto state =
        ERFFire::add_fire_combustion_ignition(
            FireCombustionState{}, p, Real(0.4));
    state =
        ERFFire::advance_fire_combustion(
            state, p, Real(2.0)).state;
    state =
        ERFFire::add_fire_combustion_ignition(
            state, p, Real(0.3));
    state =
        ERFFire::advance_fire_combustion(
            state, p, Real(3.0)).state;

    const Real first_initial =
        Real(0.4) * p.dry_fuel_load_kg_m2;
    const Real second_initial =
        Real(0.3) * p.dry_fuel_load_kg_m2;
    const Real expected_remaining =
        first_initial * std::exp(-Real(5.0) / p.burn_time_constant_s)
        + second_initial * std::exp(-Real(3.0) / p.burn_time_constant_s);
    const Real expected_ignited =
        Real(0.7) * p.dry_fuel_load_kg_m2;
    const Real expected_consumed =
        expected_ignited - expected_remaining;

    EXPECT_NEAR(
        state.remaining_dry_fuel_kg_m2,
        expected_remaining,
        scaled_tolerance(expected_remaining));
    EXPECT_NEAR(
        state.consumed_dry_fuel_kg_m2,
        expected_consumed,
        scaled_tolerance(expected_consumed));
    EXPECT_NEAR(
        state.sensible_energy_j_m2,
        expected_consumed * p.sensible_heat_release_j_kg_dry,
        scaled_tolerance(expected_consumed * p.sensible_heat_release_j_kg_dry));
    EXPECT_NEAR(
        state.water_released_kg_m2,
        expected_consumed
            * (p.fuel_moisture_fraction
               + p.combustion_water_yield_kg_per_kg_dry),
        scaled_tolerance(
            expected_consumed
            * (p.fuel_moisture_fraction
               + p.combustion_water_yield_kg_per_kg_dry)));
}

TEST(FireCombustion, ValidLongRunStateAdvancesWithoutDerivedEnergyDrift)
{
    const auto p =
        ERFFire::make_fm1_combustion_parameters(
            Real(0.08));

    // Regression fixture captured from the 1024^2 x 80
    // four-H100 terrain/two-way capability run immediately
    // before its long-time combustion accounting failure.
    const FireCombustionState state{
        Real(1.0),
        Real(2.01719275819238694e-02),
        Real(1.45830612055105335e-01),
        Real(2.54226505995636247e+06),
        Real(9.33315917152680308e-02)};

    // This state is valid before the failing positive-time
    // half-step.
    ERFFire::FireCombustionAdvance zero_update{};
    ASSERT_EQ(
        ERFFire::try_advance_fire_combustion(
            state,
            p,
            Real(0.0),
            zero_update),
        ERFFire::FireCombustionStatus::success);

    // Reconstruct the production half-substep from the
    // diagnosed remaining-fuel transition.
    const Real target_remaining =
        Real(2.01718481189893974e-02);
    const Real half_substep_dt =
        -p.burn_time_constant_s
        * std::log(
            target_remaining
            / state.remaining_dry_fuel_kg_m2);

    ERFFire::FireCombustionAdvance update{};
    ASSERT_EQ(
        ERFFire::try_advance_fire_combustion(
            state,
            p,
            half_substep_dt,
            update),
        ERFFire::FireCombustionStatus::success);

    EXPECT_NEAR(
        update.state.remaining_dry_fuel_kg_m2,
        target_remaining,
        scaled_tolerance(target_remaining));

    EXPECT_EQ(
        update.state.sensible_energy_j_m2,
        update.state.consumed_dry_fuel_kg_m2
            * p.sensible_heat_release_j_kg_dry);

    EXPECT_EQ(
        update.state.water_released_kg_m2,
        update.state.consumed_dry_fuel_kg_m2
            * (p.fuel_moisture_fraction
               + p.combustion_water_yield_kg_per_kg_dry));
}

TEST(FireCombustion, ZeroDtIsExactlyIdempotent)
{
    const auto p = synthetic_parameters();
    const auto state =
        ERFFire::add_fire_combustion_ignition(
            FireCombustionState{}, p, Real(0.65));

    const auto update =
        ERFFire::advance_fire_combustion(
            state, p, Real(0.0));

    EXPECT_EQ(update.state.ignited_area_fraction, state.ignited_area_fraction);
    EXPECT_EQ(
        update.state.remaining_dry_fuel_kg_m2,
        state.remaining_dry_fuel_kg_m2);
    EXPECT_EQ(
        update.state.consumed_dry_fuel_kg_m2,
        state.consumed_dry_fuel_kg_m2);
    EXPECT_EQ(update.state.sensible_energy_j_m2, state.sensible_energy_j_m2);
    EXPECT_EQ(update.state.water_released_kg_m2, state.water_released_kg_m2);
    EXPECT_EQ(update.newly_consumed_dry_fuel_kg_m2, Real(0.0));
    EXPECT_EQ(update.sensible_energy_increment_j_m2, Real(0.0));
    EXPECT_EQ(update.water_released_increment_kg_m2, Real(0.0));
}

#ifdef AMREX_USE_GPU
namespace
{

struct DeviceCombustionProbe
{
    ERFFire::FireCombustionAdvance update{};
    int status{};
    int invalid_status{};
};

DeviceCombustionProbe
run_device_combustion_probe(
    const FireCombustionParameters& p,
    Real ignition_fraction,
    Real dt)
{
    amrex::Gpu::DeviceScalar<ERFFire::FireCombustionAdvance>
        device_update;
    amrex::Gpu::DeviceScalar<int> device_status;
    amrex::Gpu::DeviceScalar<int> device_invalid_status;

    auto* update_ptr = device_update.dataPtr();
    auto* status_ptr = device_status.dataPtr();
    auto* invalid_status_ptr =
        device_invalid_status.dataPtr();

    amrex::ParallelFor(
        1,
        [=] AMREX_GPU_DEVICE (int) noexcept
        {
            FireCombustionState ignited{};
            ERFFire::FireCombustionAdvance update{};
            auto status =
                ERFFire::try_add_fire_combustion_ignition(
                    FireCombustionState{},
                    p,
                    ignition_fraction,
                    ignited);
            if (status
                == ERFFire::FireCombustionStatus::success) {
                status =
                    ERFFire::try_advance_fire_combustion(
                        ignited,
                        p,
                        dt,
                        update);
            }

            ERFFire::FireCombustionAdvance rejected{};
            const auto invalid_status =
                ERFFire::try_advance_fire_combustion(
                    ignited,
                    p,
                    Real(-1.0),
                    rejected);

            *update_ptr = update;
            *status_ptr = static_cast<int>(status);
            *invalid_status_ptr =
                static_cast<int>(invalid_status);
        });

    return {
        device_update.dataValue(),
        device_status.dataValue(),
        device_invalid_status.dataValue()};
}

} // namespace

TEST(FireCombustion, DeviceSafeScalarApiMatchesHost)
{
    const auto p = synthetic_parameters();
    const Real ignition_fraction = Real(0.8);
    const Real dt = Real(4.0);

    const FireCombustionState host_ignited =
        ERFFire::add_fire_combustion_ignition(
            FireCombustionState{},
            p,
            ignition_fraction);
    const auto host_update =
        ERFFire::advance_fire_combustion(
            host_ignited,
            p,
            dt);

    const DeviceCombustionProbe actual =
        run_device_combustion_probe(
            p,
            ignition_fraction,
            dt);

    EXPECT_EQ(
        actual.status,
        static_cast<int>(
            ERFFire::FireCombustionStatus::success));
    EXPECT_EQ(
        actual.invalid_status,
        static_cast<int>(
            ERFFire::FireCombustionStatus::invalid_argument));

    EXPECT_NEAR(
        actual.update.state.ignited_area_fraction,
        host_update.state.ignited_area_fraction,
        scaled_tolerance(
            host_update.state.ignited_area_fraction));
    EXPECT_NEAR(
        actual.update.state.remaining_dry_fuel_kg_m2,
        host_update.state.remaining_dry_fuel_kg_m2,
        scaled_tolerance(
            host_update.state.remaining_dry_fuel_kg_m2));
    EXPECT_NEAR(
        actual.update.state.consumed_dry_fuel_kg_m2,
        host_update.state.consumed_dry_fuel_kg_m2,
        scaled_tolerance(
            host_update.state.consumed_dry_fuel_kg_m2));
    EXPECT_NEAR(
        actual.update.state.sensible_energy_j_m2,
        host_update.state.sensible_energy_j_m2,
        scaled_tolerance(
            host_update.state.sensible_energy_j_m2));
    EXPECT_NEAR(
        actual.update.state.water_released_kg_m2,
        host_update.state.water_released_kg_m2,
        scaled_tolerance(
            host_update.state.water_released_kg_m2));
}
#endif

TEST(FireCombustion, RejectsInvalidParametersStateAndTransitions)
{
    const auto p = synthetic_parameters();

    EXPECT_THROW(
        (void)ERFFire::fire_burn_time_constant_from_weight(Real(0.0)),
        std::invalid_argument);

    auto invalid_p = p;
    invalid_p.burn_time_constant_s = Real(0.0);
    EXPECT_THROW(
        (void)ERFFire::advance_fire_combustion(
            FireCombustionState{}, invalid_p, Real(1.0)),
        std::invalid_argument);

    EXPECT_THROW(
        (void)ERFFire::add_fire_combustion_ignition(
            FireCombustionState{}, p, Real(-0.01)),
        std::invalid_argument);

    auto state =
        ERFFire::add_fire_combustion_ignition(
            FireCombustionState{}, p, Real(0.8));

    EXPECT_THROW(
        (void)ERFFire::add_fire_combustion_ignition(
            state, p, Real(0.21)),
        std::invalid_argument);
    EXPECT_THROW(
        (void)ERFFire::advance_fire_combustion(
            state, p, Real(-1.0)),
        std::invalid_argument);

    state.sensible_energy_j_m2 = Real(1.0);
    EXPECT_THROW(
        (void)ERFFire::advance_fire_combustion(
            state, p, Real(1.0)),
        std::invalid_argument);
}
