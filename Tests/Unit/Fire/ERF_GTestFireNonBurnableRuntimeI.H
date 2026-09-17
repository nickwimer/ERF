#ifndef ERF_GTEST_FIRE_NONBURNABLE_RUNTIME_I_H_
#define ERF_GTEST_FIRE_NONBURNABLE_RUNTIME_I_H_

#include <ERF_FireBurnedFractionRaster.H>
#include <ERF_FireCombustionRaster.H>
#include <ERF_FireFuelRaster.H>
#include <ERF_FireSpreadRuntime.H>

#include <AMReX_ParallelDescriptor.H>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace
{

using NonBurnableReal = amrex::Real;
using ERFFire::ERFFireSpreadConfig;
using ERFFire::ERFFireSpreadRuntime;
using ERFFire::FireBurnedFractionRaster;
using ERFFire::FireCartesianRasterGeometry2D;
using ERFFire::FireCombustionRaster;
using ERFFire::FireCombustionRasterOptions;
using ERFFire::FireFuelModelId;
using ERFFire::FireFuelMoistureClass;
using ERFFire::FireFuelRaster;
using ERFFire::FireFuelRasterState;
using ERFFire::FirePerimeter;
using ERFFire::FirePerimeterRemeshOptions;
using ERFFire::FireVec2;

FireCartesianRasterGeometry2D
nonburnable_combustion_geometry()
{
    return {
        2,
        1,
        NonBurnableReal(0),
        NonBurnableReal(0),
        NonBurnableReal(1),
        NonBurnableReal(1)};
}

FireCartesianRasterGeometry2D
nonburnable_runtime_geometry()
{
    return {
        8,
        8,
        NonBurnableReal(0),
        NonBurnableReal(0),
        NonBurnableReal(1),
        NonBurnableReal(1)};
}

FireFuelRaster
make_nonburnable_test_raster(
    const FireCartesianRasterGeometry2D& geometry,
    const std::vector<std::pair<std::size_t, std::size_t>>& blocked)
{
    FireFuelRasterState state;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        state.cells.resize(geometry.nx * geometry.ny);
        for (std::size_t j = 0; j < geometry.ny; ++j) {
            for (std::size_t i = 0; i < geometry.nx; ++i) {
                auto& cell = state.cells[j * geometry.nx + i];
                const bool is_blocked =
                    std::find(
                        blocked.begin(),
                        blocked.end(),
                        std::make_pair(i, j))
                    != blocked.end();
                if (is_blocked) {
                    cell.model_id = FireFuelModelId::NonBurnable;
                    continue;
                }
                cell.model_id = FireFuelModelId::FM1;
                cell.moisture.set(
                    FireFuelMoistureClass::Dead1h,
                    NonBurnableReal(0.08));
            }
        }
    }

    return FireFuelRaster::collective_from_io_rank_state(
        geometry,
        state);
}

FireFuelRaster
make_nonburnable_combustion_raster()
{
    return make_nonburnable_test_raster(
        nonburnable_combustion_geometry(),
        {{1U, 0U}});
}

FirePerimeter
first_combustion_cell_perimeter()
{
    return FirePerimeter(
        std::vector<FireVec2>{
            {NonBurnableReal(0), NonBurnableReal(0)},
            {NonBurnableReal(1), NonBurnableReal(0)},
            {NonBurnableReal(1), NonBurnableReal(1)},
            {NonBurnableReal(0), NonBurnableReal(1)}});
}

FirePerimeter
full_combustion_domain_perimeter()
{
    return FirePerimeter(
        std::vector<FireVec2>{
            {NonBurnableReal(0), NonBurnableReal(0)},
            {NonBurnableReal(2), NonBurnableReal(0)},
            {NonBurnableReal(2), NonBurnableReal(1)},
            {NonBurnableReal(0), NonBurnableReal(1)}});
}

bool
combustion_state_is_zero(
    const ERFFire::FireCombustionState& state)
{
    return state.ignited_area_fraction == NonBurnableReal(0)
        && state.remaining_dry_fuel_kg_m2 == NonBurnableReal(0)
        && state.consumed_dry_fuel_kg_m2 == NonBurnableReal(0)
        && state.sensible_energy_j_m2 == NonBurnableReal(0)
        && state.water_released_kg_m2 == NonBurnableReal(0);
}

ERFFireSpreadConfig
nonburnable_runtime_config()
{
    const auto geometry = nonburnable_runtime_geometry();
    return {
        ERFFire::make_fm1_fuel_parameters(),
        NonBurnableReal(0.08),
        ERFFire::make_fm1_combustion_parameters(
            NonBurnableReal(0.08)),
        FireCombustionRasterOptions{8},
        FirePerimeterRemeshOptions{
            NonBurnableReal(1.0e-6),
            NonBurnableReal(10),
            NonBurnableReal(0)},
        geometry,
        NonBurnableReal(1.0e-7)};
}

ERFFire::FireEnvironmentBatchFunction
nonburnable_east_wind()
{
    return [](
        const std::vector<FireVec2>& positions) {
        return std::vector<ERFFire::FireEnvironmentSample>(
            positions.size(),
            ERFFire::FireEnvironmentSample{
                FireVec2{
                    NonBurnableReal(10),
                    NonBurnableReal(0)},
                FireVec2{}});
    };
}

std::vector<std::pair<std::size_t, std::size_t>>
nonburnable_full_height_wall()
{
    std::vector<std::pair<std::size_t, std::size_t>> result;
    for (std::size_t j = 0; j < 8U; ++j) {
        result.push_back({4U, j});
    }
    return result;
}

TEST(
    FireFuelCombustionRaster,
    NonBurnableCellRemainsCanonicalZero)
{
    const auto geometry = nonburnable_combustion_geometry();
    const auto fuel = make_nonburnable_combustion_raster();
    const auto perimeter = first_combustion_cell_perimeter();

    FireBurnedFractionRaster burned(geometry);
    (void)burned.update_from_perimeter(perimeter);

    FireCombustionRaster combustion(
        geometry,
        ERFFire::make_fm1_combustion_parameters(
            NonBurnableReal(0.08)),
        FireCombustionRasterOptions{4});

    (void)combustion.initialize_from_burned_fraction(
        burned,
        fuel);
    (void)combustion.advance_from_linear_sweep(
        perimeter,
        perimeter,
        burned,
        burned,
        fuel,
        NonBurnableReal(2));

    const auto state =
        combustion.collective_snapshot_state_to_io_rank();
    if (amrex::ParallelDescriptor::IOProcessor()) {
        ASSERT_EQ(state.cells.size(), 2U);
        EXPECT_GT(
            state.cells[0].consumed_dry_fuel_kg_m2,
            NonBurnableReal(0));
        EXPECT_TRUE(combustion_state_is_zero(state.cells[1]));
    }
}

TEST(
    FireFuelCombustionRaster,
    BurnedAreaInNonBurnableCellIsRejectedTransactionally)
{
    const auto geometry = nonburnable_combustion_geometry();
    const auto fuel = make_nonburnable_combustion_raster();

    FireBurnedFractionRaster burned(geometry);
    (void)burned.update_from_perimeter(
        full_combustion_domain_perimeter());

    FireCombustionRaster combustion(
        geometry,
        ERFFire::make_fm1_combustion_parameters(
            NonBurnableReal(0.08)),
        FireCombustionRasterOptions{4});

    EXPECT_THROW(
        (void)combustion.initialize_from_burned_fraction(
            burned,
            fuel),
        std::invalid_argument);
    EXPECT_FALSE(combustion.initialized());

    const auto totals = combustion.totals();
    EXPECT_EQ(totals.remaining_dry_fuel_kg, NonBurnableReal(0));
    EXPECT_EQ(totals.consumed_dry_fuel_kg, NonBurnableReal(0));
    EXPECT_EQ(totals.sensible_energy_j, NonBurnableReal(0));
    EXPECT_EQ(totals.water_released_kg, NonBurnableReal(0));
}

TEST(
    FireFuelRuntime,
    NonBurnableWallStopsProductionSpatialRuntime)
{
    const auto config = nonburnable_runtime_config();
    const auto wall = nonburnable_full_height_wall();
    const auto fuel =
        make_nonburnable_test_raster(
            config.raster_geometry,
            wall);

    ERFFireSpreadRuntime runtime(
        FirePerimeter(
            std::vector<FireVec2>{
                {NonBurnableReal(1.5), NonBurnableReal(2.5)},
                {NonBurnableReal(3.9), NonBurnableReal(2.5)},
                {NonBurnableReal(3.9), NonBurnableReal(5.5)},
                {NonBurnableReal(1.5), NonBurnableReal(5.5)}}),
        NonBurnableReal(0),
        config,
        fuel);

    const auto environment = nonburnable_east_wind();
    (void)runtime.advance_direct_reference_wind_batched(
        environment,
        NonBurnableReal(2));
    (void)runtime.advance_direct_reference_wind_batched(
        environment,
        NonBurnableReal(1));

    NonBurnableReal maximum_x =
        -std::numeric_limits<NonBurnableReal>::infinity();
    for (const auto& component : runtime.front().components()) {
        for (const FireVec2& vertex : component.perimeter.vertices_m()) {
            maximum_x = std::max(maximum_x, vertex.x);
        }
    }
    EXPECT_EQ(maximum_x, NonBurnableReal(4));

    const auto burned =
        runtime.burned_fraction_raster()
            .collective_snapshot_state_to_io_rank();
    const auto arrival =
        runtime.first_arrival_raster()
            .collective_snapshot_state_to_io_rank();
    const auto combustion =
        runtime.combustion_raster()
            .collective_snapshot_state_to_io_rank();

    if (amrex::ParallelDescriptor::IOProcessor()) {
        const auto& geometry = config.raster_geometry;
        ASSERT_EQ(
            burned.burned_fraction.size(),
            geometry.nx * geometry.ny);
        ASSERT_EQ(arrival.arrived.size(), geometry.nx * geometry.ny);
        ASSERT_EQ(combustion.cells.size(), geometry.nx * geometry.ny);

        bool saw_consumption = false;
        for (std::size_t j = 0; j < geometry.ny; ++j) {
            const std::size_t barrier_index = j * geometry.nx + 4U;
            EXPECT_EQ(
                burned.burned_fraction[barrier_index],
                NonBurnableReal(0));
            EXPECT_EQ(arrival.arrived[barrier_index], std::uint8_t(0));
            EXPECT_TRUE(
                combustion_state_is_zero(
                    combustion.cells[barrier_index]));

            for (std::size_t i = 0; i < 4U; ++i) {
                if (combustion.cells[j * geometry.nx + i]
                        .consumed_dry_fuel_kg_m2
                    > NonBurnableReal(0)) {
                    saw_consumption = true;
                }
            }
        }
        EXPECT_TRUE(saw_consumption);
    }
}

TEST(
    FireFuelRuntime,
    SpatialIgnitionCannotEncloseNonBurnableIsland)
{
    const auto config = nonburnable_runtime_config();
    const auto fuel =
        make_nonburnable_test_raster(
            config.raster_geometry,
            {{4U, 4U}});

    EXPECT_THROW(
        (void)ERFFireSpreadRuntime(
            FirePerimeter(
                std::vector<FireVec2>{
                    {NonBurnableReal(3), NonBurnableReal(3)},
                    {NonBurnableReal(6), NonBurnableReal(3)},
                    {NonBurnableReal(6), NonBurnableReal(6)},
                    {NonBurnableReal(3), NonBurnableReal(6)}}),
            NonBurnableReal(0),
            config,
            fuel),
        std::invalid_argument);
}

} // namespace

#endif // ERF_GTEST_FIRE_NONBURNABLE_RUNTIME_I_H_
