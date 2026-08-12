#include <ERF_FireBurnedFractionRaster.H>
#include <ERF_FireCombustionRaster.H>
#include <ERF_FirePerimeter.H>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{

using amrex::Real;
using ERFFire::FireBurnedFractionRaster;
using ERFFire::FireCartesianRasterGeometry2D;
using ERFFire::FireCombustionParameters;
using ERFFire::FireCombustionRaster;
using ERFFire::FireCombustionRasterOptions;
using ERFFire::FirePerimeter;
using ERFFire::FireVec2;

FireCartesianRasterGeometry2D
unit_geometry()
{
    return {
        1,
        1,
        Real(0.0),
        Real(0.0),
        Real(1.0),
        Real(1.0)
    };
}

FireCombustionParameters
synthetic_parameters()
{
    return {
        Real(2.0),
        Real(10.0),
        Real(0.25),
        Real(4.0),
        Real(0.5)
    };
}

FirePerimeter
sweep_rectangle(
    Real right_x,
    Real yhi = Real(1.0),
    Real left_x = Real(-1.0))
{
    std::vector<FireVec2> vertices{
        {left_x, Real(0.0)},
        {right_x, Real(0.0)},
        {right_x, yhi},
        {left_x, yhi}
    };
    return FirePerimeter(std::move(vertices));
}

FireBurnedFractionRaster
burned_from(
    const FireCartesianRasterGeometry2D& geometry,
    const FirePerimeter& perimeter)
{
    FireBurnedFractionRaster burned(geometry);
    (void)burned.update_from_perimeter(perimeter);
    return burned;
}

Real
remaining_after_planar_sweep(std::size_t temporal_substeps)
{
    const auto geometry = unit_geometry();
    const auto parameters = synthetic_parameters();
    const FirePerimeter start = sweep_rectangle(Real(0.0));
    const FirePerimeter end = sweep_rectangle(Real(1.0));

    const auto before = burned_from(geometry, start);
    const auto after = burned_from(geometry, end);

    FireCombustionRaster combustion(
        geometry,
        parameters,
        FireCombustionRasterOptions{temporal_substeps});
    (void)combustion.initialize_from_burned_fraction(before);
    (void)combustion.advance_from_linear_sweep(
        start,
        end,
        before,
        after,
        Real(4.0));

    return combustion.totals().remaining_dry_fuel_kg;
}

} // namespace

TEST(FireCombustionRaster, InitializesFractionalHistoryAsFuelMass)
{
    const auto geometry = unit_geometry();
    const auto parameters = synthetic_parameters();
    const auto initial =
        burned_from(
            geometry,
            sweep_rectangle(Real(0.25)));

    FireCombustionRaster combustion(
        geometry,
        parameters);

    const auto totals =
        combustion.initialize_from_burned_fraction(initial);

    EXPECT_TRUE(combustion.initialized());
    EXPECT_DOUBLE_EQ(
        combustion.state(0, 0).ignited_area_fraction,
        Real(0.25));
    EXPECT_DOUBLE_EQ(
        combustion.state(0, 0).remaining_dry_fuel_kg_m2,
        Real(0.5));
    EXPECT_DOUBLE_EQ(
        totals.remaining_dry_fuel_kg,
        Real(0.5));
    EXPECT_DOUBLE_EQ(totals.consumed_dry_fuel_kg, Real(0.0));
    EXPECT_DOUBLE_EQ(totals.sensible_energy_j, Real(0.0));
    EXPECT_DOUBLE_EQ(totals.water_released_kg, Real(0.0));
}

TEST(FireCombustionRaster, StationarySweepBurnsExistingFuelExactly)
{
    const auto geometry = unit_geometry();
    const auto parameters = synthetic_parameters();
    const FirePerimeter stationary =
        sweep_rectangle(Real(0.5));
    const auto burned =
        burned_from(geometry, stationary);

    FireCombustionRaster combustion(
        geometry,
        parameters,
        FireCombustionRasterOptions{8});
    (void)combustion.initialize_from_burned_fraction(burned);

    const auto update =
        combustion.advance_from_linear_sweep(
            stationary,
            stationary,
            burned,
            burned,
            Real(4.0));

    const Real initial_dry_fuel_kg = Real(1.0);
    const Real expected_remaining =
        initial_dry_fuel_kg * std::exp(Real(-1.0));
    const Real expected_consumed =
        initial_dry_fuel_kg - expected_remaining;

    EXPECT_NEAR(
        update.totals.remaining_dry_fuel_kg,
        expected_remaining,
        Real(2.0e-14));
    EXPECT_NEAR(
        update.newly_consumed_dry_fuel_kg,
        expected_consumed,
        Real(2.0e-14));
    EXPECT_NEAR(
        update.totals.consumed_dry_fuel_kg,
        expected_consumed,
        Real(2.0e-14));
}

TEST(FireCombustionRaster, PlanarSweepConvergesToContinuousIgnitionOracle)
{
    const Real parameters_tau_s =
        synthetic_parameters().burn_time_constant_s;
    const Real duration_s = Real(4.0);
    const Real load_kg_m2 =
        synthetic_parameters().dry_fuel_load_kg_m2;

    // Exact continuous-ignition oracle for B(t)=t/T:
    //
    // remaining(T) =
    //   load * integral_0^T exp(-(T-t)/tau) d(t/T)
    // = load * (tau/T) * (1-exp(-T/tau)).
    const Real expected_remaining =
        load_kg_m2
        * (parameters_tau_s / duration_s)
        * (Real(1.0)
           - std::exp(-duration_s / parameters_tau_s));

    const Real error4 =
        std::abs(
            remaining_after_planar_sweep(4)
            - expected_remaining);
    const Real error8 =
        std::abs(
            remaining_after_planar_sweep(8)
            - expected_remaining);
    const Real error16 =
        std::abs(
            remaining_after_planar_sweep(16)
            - expected_remaining);

    EXPECT_LT(error8, error4);
    EXPECT_LT(error16, error8);
    EXPECT_LT(error16, Real(2.1e-4));
}

TEST(FireCombustionRaster, SplitSweepMatchesAlignedTemporalPartition)
{
    const auto geometry = unit_geometry();
    const auto parameters = synthetic_parameters();

    const FirePerimeter start =
        sweep_rectangle(Real(0.0));
    const FirePerimeter middle =
        sweep_rectangle(Real(0.5));
    const FirePerimeter end =
        sweep_rectangle(Real(1.0));

    const auto burned_start =
        burned_from(geometry, start);
    const auto burned_middle =
        burned_from(geometry, middle);
    const auto burned_end =
        burned_from(geometry, end);

    FireCombustionRaster whole(
        geometry,
        parameters,
        FireCombustionRasterOptions{16});
    (void)whole.initialize_from_burned_fraction(
        burned_start);
    (void)whole.advance_from_linear_sweep(
        start,
        end,
        burned_start,
        burned_end,
        Real(4.0));

    FireCombustionRaster split(
        geometry,
        parameters,
        FireCombustionRasterOptions{8});
    (void)split.initialize_from_burned_fraction(
        burned_start);
    (void)split.advance_from_linear_sweep(
        start,
        middle,
        burned_start,
        burned_middle,
        Real(2.0));
    (void)split.advance_from_linear_sweep(
        middle,
        end,
        burned_middle,
        burned_end,
        Real(2.0));

    const auto whole_totals = whole.totals();
    const auto split_totals = split.totals();

    EXPECT_NEAR(
        split_totals.remaining_dry_fuel_kg,
        whole_totals.remaining_dry_fuel_kg,
        Real(5.0e-14));
    EXPECT_NEAR(
        split_totals.consumed_dry_fuel_kg,
        whole_totals.consumed_dry_fuel_kg,
        Real(5.0e-14));
    EXPECT_NEAR(
        split_totals.sensible_energy_j,
        whole_totals.sensible_energy_j,
        Real(5.0e-13));
    EXPECT_NEAR(
        split_totals.water_released_kg,
        whole_totals.water_released_kg,
        Real(5.0e-14));
}

TEST(FireCombustionRaster, PartialAreaIgnitionIsNotCollapsedToSweepStart)
{
    const auto geometry = unit_geometry();
    const auto parameters = synthetic_parameters();
    const FirePerimeter start =
        sweep_rectangle(Real(0.0));
    const FirePerimeter end =
        sweep_rectangle(Real(1.0));

    const auto before = burned_from(geometry, start);
    const auto after = burned_from(geometry, end);

    FireCombustionRaster combustion(
        geometry,
        parameters,
        FireCombustionRasterOptions{16});
    (void)combustion.initialize_from_burned_fraction(before);
    (void)combustion.advance_from_linear_sweep(
        start,
        end,
        before,
        after,
        Real(4.0));

    const Real if_entire_cell_ignited_at_sweep_start =
        parameters.dry_fuel_load_kg_m2 * std::exp(Real(-1.0));

    EXPECT_GT(
        combustion.totals().remaining_dry_fuel_kg,
        if_entire_cell_ignited_at_sweep_start);
    EXPECT_DOUBLE_EQ(
        combustion.state(0, 0).ignited_area_fraction,
        Real(1.0));
}

TEST(FireCombustionRaster, ExtensiveTotalsScaleWithPhysicalCellArea)
{
    const FireCartesianRasterGeometry2D geometry{
        1,
        1,
        Real(0.0),
        Real(0.0),
        Real(2.0),
        Real(3.0)
    };
    const auto parameters = synthetic_parameters();

    const FirePerimeter half_cell =
        sweep_rectangle(
            Real(1.0),
            Real(3.0),
            Real(-2.0));
    const auto burned =
        burned_from(geometry, half_cell);

    FireCombustionRaster combustion(
        geometry,
        parameters,
        FireCombustionRasterOptions{4});
    const auto initial =
        combustion.initialize_from_burned_fraction(burned);

    // 0.5 burned fraction * 6 m^2 * 2 kg/m^2 = 6 kg.
    EXPECT_NEAR(
        initial.remaining_dry_fuel_kg,
        Real(6.0),
        Real(2.0e-14));

    const auto update =
        combustion.advance_from_linear_sweep(
            half_cell,
            half_cell,
            burned,
            burned,
            Real(4.0));

    const Real expected_remaining =
        Real(6.0) * std::exp(Real(-1.0));
    const Real expected_consumed =
        Real(6.0) - expected_remaining;

    EXPECT_NEAR(
        update.totals.remaining_dry_fuel_kg,
        expected_remaining,
        Real(1.0e-13));
    EXPECT_NEAR(
        update.totals.consumed_dry_fuel_kg,
        expected_consumed,
        Real(1.0e-13));
    EXPECT_NEAR(
        update.totals.sensible_energy_j,
        expected_consumed * Real(10.0),
        Real(1.0e-12));
    EXPECT_NEAR(
        update.totals.water_released_kg,
        expected_consumed * Real(0.75),
        Real(1.0e-13));
}

TEST(FireCombustionRaster, FailedBurnHistoryConsistencyCheckIsTransactional)
{
    const auto geometry = unit_geometry();
    const auto parameters = synthetic_parameters();
    const FirePerimeter start =
        sweep_rectangle(Real(0.0));
    const FirePerimeter end =
        sweep_rectangle(Real(1.0));

    const auto burned_start =
        burned_from(geometry, start);

    FireCombustionRaster combustion(
        geometry,
        parameters,
        FireCombustionRasterOptions{8});
    (void)combustion.initialize_from_burned_fraction(
        burned_start);

    const auto before_totals = combustion.totals();
    const auto before_state = combustion.state(0, 0);

    EXPECT_THROW(
        (void)combustion.advance_from_linear_sweep(
            start,
            end,
            burned_start,
            burned_start,
            Real(4.0)),
        std::invalid_argument);

    EXPECT_DOUBLE_EQ(
        combustion.totals().remaining_dry_fuel_kg,
        before_totals.remaining_dry_fuel_kg);
    EXPECT_DOUBLE_EQ(
        combustion.totals().consumed_dry_fuel_kg,
        before_totals.consumed_dry_fuel_kg);
    EXPECT_DOUBLE_EQ(
        combustion.state(0, 0).ignited_area_fraction,
        before_state.ignited_area_fraction);
    EXPECT_DOUBLE_EQ(
        combustion.state(0, 0).remaining_dry_fuel_kg_m2,
        before_state.remaining_dry_fuel_kg_m2);
}

TEST(FireCombustionRaster, RejectsInvalidOptionsGeometryAndTopology)
{
    const auto geometry = unit_geometry();
    const auto parameters = synthetic_parameters();

    EXPECT_THROW(
        (void)FireCombustionRaster(
            geometry,
            parameters,
            FireCombustionRasterOptions{0}),
        std::invalid_argument);

    FireCombustionRaster combustion(
        geometry,
        parameters,
        FireCombustionRasterOptions{4});

    const FireCartesianRasterGeometry2D other_geometry{
        2,
        1,
        Real(0.0),
        Real(0.0),
        Real(0.5),
        Real(1.0)
    };
    FireBurnedFractionRaster wrong_burned(other_geometry);

    EXPECT_THROW(
        (void)combustion.initialize_from_burned_fraction(
            wrong_burned),
        std::invalid_argument);

    const FirePerimeter start =
        sweep_rectangle(Real(0.0));
    const auto burned_start =
        burned_from(geometry, start);
    (void)combustion.initialize_from_burned_fraction(
        burned_start);

    std::vector<FireVec2> triangle_vertices{
        {Real(-1.0), Real(0.0)},
        {Real(1.0), Real(0.0)},
        {Real(0.0), Real(1.0)}
    };
    const FirePerimeter triangle(
        std::move(triangle_vertices));

    EXPECT_THROW(
        (void)combustion.advance_from_linear_sweep(
            start,
            triangle,
            burned_start,
            burned_start,
            Real(1.0)),
        std::invalid_argument);
}
