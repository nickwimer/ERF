#include <ERF_FireBurnedFractionRaster.H>
#include <ERF_FireCombustionRaster.H>
#include <ERF_FireSurfaceFeedback.H>

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
using ERFFire::FireSurfaceFeedbackRaster;
using ERFFire::FireVec2;

FireCombustionParameters
parameters()
{
    return {
        Real(2.0),
        Real(10.0),
        Real(0.25),
        Real(4.0),
        Real(0.5)
    };
}

FireCartesianRasterGeometry2D
geometry(
    std::size_t nx = 1,
    std::size_t ny = 1,
    Real dx = Real(1.0),
    Real dy = Real(1.0))
{
    return {
        nx,
        ny,
        Real(0.0),
        Real(0.0),
        dx,
        dy
    };
}

FirePerimeter
rectangle(
    Real xlo,
    Real xhi,
    Real ylo,
    Real yhi)
{
    std::vector<FireVec2> vertices{
        {xlo, ylo},
        {xhi, ylo},
        {xhi, yhi},
        {xlo, yhi}
    };
    return FirePerimeter(std::move(vertices));
}

FireBurnedFractionRaster
burned_from(
    const FireCartesianRasterGeometry2D& raster_geometry,
    const FirePerimeter& perimeter)
{
    FireBurnedFractionRaster burned(raster_geometry);
    (void)burned.update_from_perimeter(perimeter);
    return burned;
}

FireCombustionRaster
initialized_combustion(
    const FireCartesianRasterGeometry2D& raster_geometry,
    const FireCombustionParameters& combustion_parameters,
    const FirePerimeter& perimeter,
    std::size_t temporal_substeps = 8)
{
    FireCombustionRaster combustion(
        raster_geometry,
        combustion_parameters,
        FireCombustionRasterOptions{temporal_substeps});
    const auto burned =
        burned_from(raster_geometry, perimeter);
    (void)combustion.initialize_from_burned_fraction(
        burned);
    return combustion;
}

FireSurfaceFeedbackRaster
stationary_feedback(
    const FireCartesianRasterGeometry2D& raster_geometry,
    const FirePerimeter& perimeter,
    Real dt_s = Real(4.0))
{
    const auto p = parameters();
    const auto burned =
        burned_from(raster_geometry, perimeter);

    FireCombustionRaster before(
        raster_geometry,
        p,
        FireCombustionRasterOptions{8});
    (void)before.initialize_from_burned_fraction(
        burned);

    FireCombustionRaster after = before;
    (void)after.advance_from_linear_sweep(
        perimeter,
        perimeter,
        burned,
        burned,
        dt_s);

    return ERFFire::make_fire_surface_feedback_increment(
        before,
        after);
}

Real
feedback_tolerance(Real expected)
{
    return Real(2.0e-12)
        * std::max(
            Real(1.0),
            std::abs(expected));
}

} // namespace

TEST(FireSurfaceFeedback, IdenticalStatesProduceExactlyZeroFeedback)
{
    const auto geom = geometry();
    const auto p = parameters();
    const auto full =
        rectangle(
            Real(-1.0), Real(1.0),
            Real(0.0), Real(1.0));

    const auto combustion =
        initialized_combustion(geom, p, full);

    const FireSurfaceFeedbackRaster feedback =
        ERFFire::make_fire_surface_feedback_increment(
            combustion, combustion);

    const auto cell = feedback.cell(0, 0);
    EXPECT_EQ(cell.consumed_dry_fuel_kg, Real(0.0));
    EXPECT_EQ(cell.sensible_energy_j, Real(0.0));
    EXPECT_EQ(cell.water_released_kg, Real(0.0));

    const auto totals = feedback.totals();
    EXPECT_EQ(totals.consumed_dry_fuel_kg, Real(0.0));
    EXPECT_EQ(totals.sensible_energy_j, Real(0.0));
    EXPECT_EQ(totals.water_released_kg, Real(0.0));
}

TEST(FireSurfaceFeedback, StationaryBurnMatchesIndependentExponentialOracle)
{
    const auto geom = geometry();
    const auto p = parameters();
    const auto full =
        rectangle(
            Real(-1.0), Real(1.0),
            Real(0.0), Real(1.0));

    const auto burned = burned_from(geom, full);
    FireCombustionRaster before(
        geom, p, FireCombustionRasterOptions{4});
    (void)before.initialize_from_burned_fraction(burned);

    FireCombustionRaster after = before;
    (void)after.advance_from_linear_sweep(
        full, full, burned, burned, Real(4.0));

    const auto feedback =
        ERFFire::make_fire_surface_feedback_increment(
            before, after);

    const Real initial_dry_fuel_kg =
        p.dry_fuel_load_kg_m2;
    const Real expected_consumed =
        initial_dry_fuel_kg
        * (Real(1.0)
           - std::exp(
               -Real(4.0)
               / p.burn_time_constant_s));
    const Real expected_energy =
        expected_consumed
        * p.sensible_heat_release_j_kg_dry;
    const Real expected_water =
        expected_consumed
        * (p.fuel_moisture_fraction
           + p.combustion_water_yield_kg_per_kg_dry);

    const auto cell = feedback.cell(0, 0);
    EXPECT_NEAR(
        cell.consumed_dry_fuel_kg,
        expected_consumed,
        Real(2.0e-14));
    EXPECT_NEAR(
        cell.sensible_energy_j,
        expected_energy,
        Real(2.0e-13));
    EXPECT_NEAR(
        cell.water_released_kg,
        expected_water,
        Real(2.0e-14));
}

TEST(FireSurfaceFeedback, ExtensiveFeedbackScalesWithPhysicalCellArea)
{
    const auto geom =
        geometry(1, 1, Real(2.0), Real(3.0));
    const auto p = parameters();
    const auto full =
        rectangle(
            Real(-1.0), Real(2.0),
            Real(0.0), Real(3.0));

    const auto burned = burned_from(geom, full);
    FireCombustionRaster before(
        geom, p, FireCombustionRasterOptions{4});
    (void)before.initialize_from_burned_fraction(burned);

    FireCombustionRaster after = before;
    (void)after.advance_from_linear_sweep(
        full, full, burned, burned, Real(4.0));

    const auto feedback =
        ERFFire::make_fire_surface_feedback_increment(
            before, after);

    const Real represented_area_m2 = Real(6.0);
    const Real expected_consumed =
        represented_area_m2
        * p.dry_fuel_load_kg_m2
        * (Real(1.0) - std::exp(Real(-1.0)));

    EXPECT_NEAR(
        feedback.cell(0, 0).consumed_dry_fuel_kg,
        expected_consumed,
        Real(1.0e-13));
}

TEST(FireSurfaceFeedback, TotalsEqualIndependentCumulativeStateDifferences)
{
    const auto geom =
        geometry(2, 1, Real(1.0), Real(1.0));
    const auto p = parameters();

    const auto initial_perimeter =
        rectangle(
            Real(-1.0), Real(1.5),
            Real(0.0), Real(1.0));
    const auto end_perimeter =
        rectangle(
            Real(-1.0), Real(2.0),
            Real(0.0), Real(1.0));

    const auto burned_before =
        burned_from(geom, initial_perimeter);
    const auto burned_after =
        burned_from(geom, end_perimeter);

    FireCombustionRaster before(
        geom, p, FireCombustionRasterOptions{8});
    (void)before.initialize_from_burned_fraction(
        burned_before);

    FireCombustionRaster after = before;
    (void)after.advance_from_linear_sweep(
        initial_perimeter,
        end_perimeter,
        burned_before,
        burned_after,
        Real(2.0));

    const auto before_totals = before.totals();
    const auto after_totals = after.totals();

    const auto feedback =
        ERFFire::make_fire_surface_feedback_increment(
            before, after);
    const auto feedback_totals = feedback.totals();

    EXPECT_NEAR(
        feedback_totals.consumed_dry_fuel_kg,
        after_totals.consumed_dry_fuel_kg
            - before_totals.consumed_dry_fuel_kg,
        Real(2.0e-14));
    EXPECT_NEAR(
        feedback_totals.sensible_energy_j,
        after_totals.sensible_energy_j
            - before_totals.sensible_energy_j,
        Real(2.0e-13));
    EXPECT_NEAR(
        feedback_totals.water_released_kg,
        after_totals.water_released_kg
            - before_totals.water_released_kg,
        Real(2.0e-14));
}

TEST(FireSurfaceFeedback, NewIgnitionAndExistingBurnProducePositiveRelease)
{
    const auto geom = geometry();
    const auto p = parameters();

    const auto start =
        rectangle(
            Real(-1.0), Real(0.25),
            Real(0.0), Real(1.0));
    const auto end =
        rectangle(
            Real(-1.0), Real(1.0),
            Real(0.0), Real(1.0));

    const auto burned_before = burned_from(geom, start);
    const auto burned_after = burned_from(geom, end);

    FireCombustionRaster before(
        geom, p, FireCombustionRasterOptions{16});
    (void)before.initialize_from_burned_fraction(
        burned_before);

    FireCombustionRaster after = before;
    (void)after.advance_from_linear_sweep(
        start,
        end,
        burned_before,
        burned_after,
        Real(4.0));

    const auto feedback =
        ERFFire::make_fire_surface_feedback_increment(
            before, after);
    const auto cell = feedback.cell(0, 0);

    EXPECT_GT(cell.consumed_dry_fuel_kg, Real(0.0));
    EXPECT_GT(cell.sensible_energy_j, Real(0.0));
    EXPECT_GT(cell.water_released_kg, Real(0.0));

    EXPECT_NEAR(
        cell.sensible_energy_j,
        cell.consumed_dry_fuel_kg
            * p.sensible_heat_release_j_kg_dry,
        Real(2.0e-13));
    EXPECT_NEAR(
        cell.water_released_kg,
        cell.consumed_dry_fuel_kg
            * (p.fuel_moisture_fraction
               + p.combustion_water_yield_kg_per_kg_dry),
        Real(2.0e-14));
}

TEST(FireSurfaceFeedback, ConservativeRegridSplitsCoarseCellsToFine)
{
    const auto source_geometry =
        geometry(
            2,
            1,
            Real(2.0),
            Real(2.0));

    const auto left_half =
        rectangle(
            Real(0.0),
            Real(2.0),
            Real(0.0),
            Real(2.0));

    const auto source =
        stationary_feedback(
            source_geometry,
            left_half);

    const auto target =
        ERFFire::conservatively_regrid_fire_surface_feedback(
            source,
            geometry(
                4,
                2,
                Real(1.0),
                Real(1.0)));

    const auto source_left =
        source.cell(0, 0);
    const auto source_right =
        source.cell(1, 0);

    EXPECT_GT(
        source_left.sensible_energy_j,
        Real(0.0));
    EXPECT_EQ(
        source_right.sensible_energy_j,
        Real(0.0));

    for (std::size_t j = 0; j < 2; ++j) {
        for (std::size_t i = 0; i < 4; ++i) {
            const auto cell =
                target.cell(i, j);

            if (i < 2) {
                EXPECT_NEAR(
                    cell.consumed_dry_fuel_kg,
                    source_left.consumed_dry_fuel_kg
                        / Real(4.0),
                    feedback_tolerance(
                        source_left.consumed_dry_fuel_kg));
                EXPECT_NEAR(
                    cell.sensible_energy_j,
                    source_left.sensible_energy_j
                        / Real(4.0),
                    feedback_tolerance(
                        source_left.sensible_energy_j));
                EXPECT_NEAR(
                    cell.water_released_kg,
                    source_left.water_released_kg
                        / Real(4.0),
                    feedback_tolerance(
                        source_left.water_released_kg));
            } else {
                EXPECT_EQ(
                    cell.consumed_dry_fuel_kg,
                    Real(0.0));
                EXPECT_EQ(
                    cell.sensible_energy_j,
                    Real(0.0));
                EXPECT_EQ(
                    cell.water_released_kg,
                    Real(0.0));
            }
        }
    }

    EXPECT_EQ(
        target.totals().consumed_dry_fuel_kg,
        source.totals().consumed_dry_fuel_kg);
    EXPECT_EQ(
        target.totals().sensible_energy_j,
        source.totals().sensible_energy_j);
    EXPECT_EQ(
        target.totals().water_released_kg,
        source.totals().water_released_kg);
}

TEST(FireSurfaceFeedback, ConservativeRegridSumsFineCellsToCoarse)
{
    const auto source_geometry =
        geometry(
            4,
            2,
            Real(1.0),
            Real(1.0));

    const auto left_half =
        rectangle(
            Real(0.0),
            Real(2.0),
            Real(0.0),
            Real(2.0));

    const auto source =
        stationary_feedback(
            source_geometry,
            left_half);

    const auto target =
        ERFFire::conservatively_regrid_fire_surface_feedback(
            source,
            geometry(
                2,
                1,
                Real(2.0),
                Real(2.0)));

    const auto left =
        target.cell(0, 0);
    const auto right =
        target.cell(1, 0);
    const auto totals =
        source.totals();

    EXPECT_NEAR(
        left.consumed_dry_fuel_kg,
        totals.consumed_dry_fuel_kg,
        feedback_tolerance(
            totals.consumed_dry_fuel_kg));
    EXPECT_NEAR(
        left.sensible_energy_j,
        totals.sensible_energy_j,
        feedback_tolerance(
            totals.sensible_energy_j));
    EXPECT_NEAR(
        left.water_released_kg,
        totals.water_released_kg,
        feedback_tolerance(
            totals.water_released_kg));

    EXPECT_EQ(
        right.consumed_dry_fuel_kg,
        Real(0.0));
    EXPECT_EQ(
        right.sensible_energy_j,
        Real(0.0));
    EXPECT_EQ(
        right.water_released_kg,
        Real(0.0));

    EXPECT_EQ(
        target.totals().consumed_dry_fuel_kg,
        totals.consumed_dry_fuel_kg);
    EXPECT_EQ(
        target.totals().sensible_energy_j,
        totals.sensible_energy_j);
    EXPECT_EQ(
        target.totals().water_released_kg,
        totals.water_released_kg);
}

TEST(FireSurfaceFeedback, ConservativeRegridRejectsMixedAxisDirection)
{
    const auto source =
        stationary_feedback(
            geometry(
                4,
                1,
                Real(1.0),
                Real(2.0)),
            rectangle(
                Real(0.0),
                Real(4.0),
                Real(0.0),
                Real(2.0)));

    EXPECT_THROW(
        (void)ERFFire::conservatively_regrid_fire_surface_feedback(
            source,
            geometry(
                2,
                2,
                Real(2.0),
                Real(1.0))),
        std::invalid_argument);
}

TEST(FireSurfaceFeedback, ConservativeRegridRejectsDifferentDomain)
{
    const auto source =
        stationary_feedback(
            geometry(
                2,
                1,
                Real(1.0),
                Real(1.0)),
            rectangle(
                Real(0.0),
                Real(2.0),
                Real(0.0),
                Real(1.0)));

    EXPECT_THROW(
        (void)ERFFire::conservatively_regrid_fire_surface_feedback(
            source,
            geometry(
                1,
                1,
                Real(3.0),
                Real(1.0))),
        std::invalid_argument);
}

TEST(FireSurfaceFeedback, RejectsGeometryMismatch)
{
    const auto p = parameters();
    const auto full =
        rectangle(
            Real(-1.0), Real(2.0),
            Real(0.0), Real(1.0));

    const auto a =
        initialized_combustion(
            geometry(1, 1), p, full);
    const auto b =
        initialized_combustion(
            geometry(2, 1, Real(0.5), Real(1.0)),
            p,
            full);

    EXPECT_THROW(
        (void)ERFFire::make_fire_surface_feedback_increment(
            a, b),
        std::invalid_argument);
}

TEST(FireSurfaceFeedback, RejectsCombustionParameterMismatch)
{
    const auto geom = geometry();
    auto p1 = parameters();
    auto p2 = p1;
    p2.sensible_heat_release_j_kg_dry += Real(1.0);

    const auto full =
        rectangle(
            Real(-1.0), Real(1.0),
            Real(0.0), Real(1.0));

    const auto a =
        initialized_combustion(geom, p1, full);
    const auto b =
        initialized_combustion(geom, p2, full);

    EXPECT_THROW(
        (void)ERFFire::make_fire_surface_feedback_increment(
            a, b),
        std::invalid_argument);
}

TEST(FireSurfaceFeedback, RejectsReversedCombustionHistory)
{
    const auto geom = geometry();
    const auto p = parameters();
    const auto full =
        rectangle(
            Real(-1.0), Real(1.0),
            Real(0.0), Real(1.0));

    const auto burned = burned_from(geom, full);
    FireCombustionRaster before(
        geom, p, FireCombustionRasterOptions{4});
    (void)before.initialize_from_burned_fraction(burned);

    FireCombustionRaster after = before;
    (void)after.advance_from_linear_sweep(
        full, full, burned, burned, Real(4.0));

    EXPECT_THROW(
        (void)ERFFire::make_fire_surface_feedback_increment(
            after, before),
        std::invalid_argument);
}
