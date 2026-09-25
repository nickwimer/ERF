#include <ERF_FireBurnedFractionRaster.H>
#include <ERF_FireCombustionRaster.H>
#include <ERF_FireFuelCombustion.H>
#include <ERF_FireFuelRaster.H>
#include <ERF_FireFuelSource.H>
#include <ERF_FireFuelSpread.H>
#include <ERF_FirePerimeter.H>

#include <AMReX_Gpu.H>
#include <AMReX_ParallelDescriptor.H>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
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
using ERFFire::FireCombustionState;
using ERFFire::FireFuelCombustionAccounting;
using ERFFire::FireFuelCombustionAccountingStatus;
using ERFFire::FireFuelCombustionParameterStatus;
using ERFFire::FireFuelModelId;
using ERFFire::FireFuelModelMoistureContract;
using ERFFire::FireFuelMoisture;
using ERFFire::FireFuelRaster;
using ERFFire::FireFuelRasterCell;
using ERFFire::FireFuelRasterState;
using ERFFire::FirePerimeter;
using ERFFire::FireVec2;
using MoistureClass = ERFFire::FireFuelMoistureClass;

FireCartesianRasterGeometry2D
fuel_raster_geometry()
{
    return {
        4,
        3,
        Real(100),
        Real(-50),
        Real(2),
        Real(4)};
}

FireCartesianRasterGeometry2D
combustion_geometry()
{
    return {
        2,
        1,
        Real(0),
        Real(0),
        Real(1),
        Real(1)};
}

FireCombustionParameters
fuel_combustion_base_parameters()
{
    return {
        Real(2.0),
        Real(10.0),
        Real(0.25),
        Real(4.0),
        Real(0.5)};
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool
expected_nonburnable(int i, int j) noexcept
{
    return ((i + j) % 4) == 0;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real
expected_dead_1h_moisture(int i, int j) noexcept
{
    return Real(0.05)
        + Real(0.01) * static_cast<Real>(i)
        + Real(0.001) * static_cast<Real>(j);
}

FireFuelRasterCell
make_expected_cell(int i, int j)
{
    FireFuelRasterCell cell;
    if (expected_nonburnable(i, j)) {
        cell.model_id = FireFuelModelId::NonBurnable;
        return cell;
    }

    cell.model_id = FireFuelModelId::FM1;
    cell.moisture.set(
        MoistureClass::Dead1h,
        expected_dead_1h_moisture(i, j));

    if ((i % 2) == 0) {
        cell.moisture.set(
            MoistureClass::Dead10h,
            Real(0.10));
    }
    if ((j % 2) == 1) {
        cell.moisture.set(
            MoistureClass::LiveHerbaceous,
            Real(1.20) + Real(0.01) * static_cast<Real>(i));
    }
    return cell;
}

FireFuelRasterState
make_io_rank_state(const FireCartesianRasterGeometry2D& geometry)
{
    FireFuelRasterState state;
    if (!amrex::ParallelDescriptor::IOProcessor()) {
        return state;
    }

    state.cells.reserve(geometry.nx * geometry.ny);
    for (std::size_t j = 0; j < geometry.ny; ++j) {
        for (std::size_t i = 0; i < geometry.nx; ++i) {
            state.cells.push_back(
                make_expected_cell(
                    static_cast<int>(i),
                    static_cast<int>(j)));
        }
    }
    return state;
}

FireFuelRaster
make_combustion_fuel_raster(
    Real first_moisture,
    Real second_moisture)
{
    const auto geometry = combustion_geometry();
    FireFuelRasterState state;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        state.cells.resize(2);
        state.cells[0].model_id = FireFuelModelId::FM1;
        state.cells[0].moisture.set(
            MoistureClass::Dead1h,
            first_moisture);
        state.cells[1].model_id = FireFuelModelId::FM1;
        state.cells[1].moisture.set(
            MoistureClass::Dead1h,
            second_moisture);
    }
    return FireFuelRaster::collective_from_io_rank_state(
        geometry,
        state);
}

FireFuelRaster
make_anderson_combustion_fuel_raster()
{
    const auto geometry = combustion_geometry();
    FireFuelRasterState state;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        state.cells.resize(2);

        state.cells[0].model_id = FireFuelModelId::FM1;
        state.cells[0].moisture.set(
            MoistureClass::Dead1h,
            Real(0.08));

        state.cells[1].model_id = FireFuelModelId::FM2;
        state.cells[1].moisture.set(
            MoistureClass::Dead1h,
            Real(0.08));
        state.cells[1].moisture.set(
            MoistureClass::Dead10h,
            Real(0.09));
        state.cells[1].moisture.set(
            MoistureClass::Dead100h,
            Real(0.10));
        state.cells[1].moisture.set(
            MoistureClass::LiveHerbaceous,
            Real(0.80));
    }
    return FireFuelRaster::collective_from_io_rank_state(
        geometry,
        state);
}

FireFuelRaster
make_invalid_combustion_fuel_raster(bool nonburnable)
{
    const auto geometry = combustion_geometry();
    FireFuelRasterState state;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        state.cells.resize(2);
        for (auto& cell : state.cells) {
            cell.model_id = FireFuelModelId::FM1;
            cell.moisture.set(
                MoistureClass::Dead1h,
                Real(0.08));
        }
        if (nonburnable) {
            state.cells[1].model_id = FireFuelModelId::NonBurnable;
            state.cells[1].moisture = {};
        } else {
            state.cells[1].moisture = {};
        }
    }
    return FireFuelRaster::collective_from_io_rank_state(
        geometry,
        state);
}

FirePerimeter
full_combustion_perimeter()
{
    return FirePerimeter(
        std::vector<FireVec2>{
            {Real(0), Real(0)},
            {Real(2), Real(0)},
            {Real(2), Real(1)},
            {Real(0), Real(1)}});
}

FireBurnedFractionRaster
fully_burned_combustion_raster()
{
    FireBurnedFractionRaster burned(combustion_geometry());
    (void)burned.update_from_perimeter(
        full_combustion_perimeter());
    return burned;
}

void
expect_same_moisture(
    const FireFuelMoisture& actual,
    const FireFuelMoisture& expected)
{
    for (int component = 0;
         component < FireFuelMoisture::component_count;
         ++component) {
        const auto moisture_class =
            static_cast<MoistureClass>(component);
        EXPECT_EQ(
            actual.has(moisture_class),
            expected.has(moisture_class));

        Real actual_value = Real(-1);
        Real expected_value = Real(-2);
        const bool actual_present =
            actual.try_get(moisture_class, actual_value);
        const bool expected_present =
            expected.try_get(moisture_class, expected_value);
        EXPECT_EQ(actual_present, expected_present);
        if (actual_present && expected_present) {
            EXPECT_EQ(actual_value, expected_value);
        }
    }
}

void
expect_same_combustion_state(
    const FireCombustionState& actual,
    const FireCombustionState& expected)
{
    EXPECT_EQ(
        actual.ignited_area_fraction,
        expected.ignited_area_fraction);
    EXPECT_EQ(
        actual.remaining_dry_fuel_kg_m2,
        expected.remaining_dry_fuel_kg_m2);
    EXPECT_EQ(
        actual.consumed_dry_fuel_kg_m2,
        expected.consumed_dry_fuel_kg_m2);
    EXPECT_EQ(
        actual.sensible_energy_j_m2,
        expected.sensible_energy_j_m2);
    EXPECT_EQ(
        actual.water_released_kg_m2,
        expected.water_released_kg_m2);
}

void
expect_same_totals(
    const ERFFire::FireCombustionRasterTotals& actual,
    const ERFFire::FireCombustionRasterTotals& expected)
{
    EXPECT_EQ(
        actual.remaining_dry_fuel_kg,
        expected.remaining_dry_fuel_kg);
    EXPECT_EQ(
        actual.consumed_dry_fuel_kg,
        expected.consumed_dry_fuel_kg);
    EXPECT_EQ(
        actual.sensible_energy_j,
        expected.sensible_energy_j);
    EXPECT_EQ(
        actual.water_released_kg,
        expected.water_released_kg);
}

#ifdef AMREX_USE_GPU
int
device_decode_mismatch_count(const FireFuelRaster& raster)
{
    const auto arrays =
        raster.distributed_values().const_arrays();
    const FireCombustionParameters base =
        fuel_combustion_base_parameters();

    const auto reduced =
        amrex::ParReduce(
            amrex::TypeList<amrex::ReduceOpSum>{},
            amrex::TypeList<int>{},
            raster.distributed_values(),
            [=] AMREX_GPU_DEVICE (
                int box_no,
                int i,
                int j,
                int) noexcept -> amrex::GpuTuple<int>
            {
                FireFuelRasterCell cell;
                if (!ERFFire::detail::try_decode_fire_fuel_raster_cell(
                        arrays[box_no], i, j, cell)) {
                    return {1};
                }

                FireCombustionParameters local{};
                const auto combustion_status =
                    ERFFire::try_make_fire_combustion_parameters_for_fuel_cell(
                        base,
                        cell,
                        local);

                if (expected_nonburnable(i, j)) {
                    if (cell.model_id
                            != FireFuelModelId::NonBurnable
                        || cell.moisture.has(MoistureClass::Dead1h)
                        || cell.moisture.has(MoistureClass::Dead10h)
                        || cell.moisture.has(MoistureClass::LiveHerbaceous)
                        || combustion_status
                            != FireFuelCombustionParameterStatus::nonburnable) {
                        return {1};
                    }
                    return {0};
                }

                if (cell.model_id != FireFuelModelId::FM1
                    || combustion_status
                        != FireFuelCombustionParameterStatus::success) {
                    return {1};
                }

                Real dead_1h = Real(-1);
                if (!cell.moisture.try_get(
                        MoistureClass::Dead1h,
                        dead_1h)
                    || dead_1h
                        != expected_dead_1h_moisture(i, j)
                    || local.fuel_moisture_fraction != dead_1h
                    || local.dry_fuel_load_kg_m2
                        != base.dry_fuel_load_kg_m2
                    || local.sensible_heat_release_j_kg_dry
                        != base.sensible_heat_release_j_kg_dry
                    || local.burn_time_constant_s
                        != base.burn_time_constant_s
                    || local.combustion_water_yield_kg_per_kg_dry
                        != base.combustion_water_yield_kg_per_kg_dry) {
                    return {1};
                }

                Real dead_10h = Real(-1);
                const bool has_dead_10h =
                    cell.moisture.try_get(
                        MoistureClass::Dead10h,
                        dead_10h);
                if (has_dead_10h != ((i % 2) == 0)) {
                    return {1};
                }
                if (has_dead_10h && dead_10h != Real(0.10)) {
                    return {1};
                }

                Real live_herbaceous = Real(-1);
                const bool has_live_herbaceous =
                    cell.moisture.try_get(
                        MoistureClass::LiveHerbaceous,
                        live_herbaceous);
                if (has_live_herbaceous != ((j % 2) == 1)) {
                    return {1};
                }
                if (has_live_herbaceous
                    && live_herbaceous
                        != Real(1.20)
                            + Real(0.01)
                                * static_cast<Real>(i)) {
                    return {1};
                }

                return {0};
            });

    return reduced;
}
#endif

void
write_fuel_source_text(
    const std::string& filename,
    const std::string& contents)
{
    int failed = 0;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream stream(
            filename,
            std::ios::out | std::ios::trunc);
        if (!stream.good()) {
            failed = 1;
        } else {
            stream << contents;
            stream.close();
            if (stream.fail()) {
                failed = 1;
            }
        }
    }
    amrex::ParallelDescriptor::Bcast(
        &failed,
        1,
        amrex::ParallelDescriptor::IOProcessorNumber());
    ASSERT_EQ(failed, 0);
    amrex::ParallelDescriptor::Barrier();
}

void
remove_fuel_source_text(const std::string& filename)
{
    amrex::ParallelDescriptor::Barrier();
    if (amrex::ParallelDescriptor::IOProcessor()) {
        (void)std::remove(filename.c_str());
    }
    amrex::ParallelDescriptor::Barrier();
}

Real
fuel_accounting_tolerance(Real expected)
{
    return Real(1024)
        * std::numeric_limits<Real>::epsilon()
        * std::max(
            Real(1),
            std::abs(expected));
}

FireFuelMoisture
complete_model_moisture(FireFuelModelId id)
{
    FireFuelMoisture moisture;
    const auto contract =
        ERFFire::fire_fuel_model_moisture_contract(id);

    for (int component = 0;
         component < FireFuelMoisture::component_count;
         ++component) {
        const auto moisture_class =
            static_cast<MoistureClass>(component);
        const unsigned int bit =
            ERFFire::fire_fuel_moisture_component_bit(
                moisture_class);
        if ((contract.required_mask & bit) != 0u) {
            moisture.set(
                moisture_class,
                Real(0.05)
                    + Real(0.10)
                        * static_cast<Real>(component));
        }
    }
    return moisture;
}

#ifdef AMREX_USE_GPU
struct DeviceFuelCombustionAccountingProbe
{
    FireFuelCombustionAccounting accounting{};
    int status{};
};

DeviceFuelCombustionAccountingProbe
run_device_fuel_combustion_accounting_probe(
    const FireCombustionParameters& base,
    const FireFuelRasterCell& cell)
{
    amrex::Gpu::DeviceScalar<FireFuelCombustionAccounting>
        device_accounting;
    amrex::Gpu::DeviceScalar<int> device_status;

    auto* accounting_ptr =
        device_accounting.dataPtr();
    auto* status_ptr =
        device_status.dataPtr();

    amrex::ParallelFor(
        1,
        [=] AMREX_GPU_DEVICE (int) noexcept
        {
            FireFuelCombustionAccounting result{};
            const auto status =
                ERFFire::try_make_anderson13_fire_combustion_accounting(
                    base,
                    cell,
                    result);
            *accounting_ptr = result;
            *status_ptr = static_cast<int>(status);
        });

    return {
        device_accounting.dataValue(),
        device_status.dataValue()};
}

DeviceFuelCombustionAccountingProbe
run_device_spatial_fuel_combustion_accounting_probe(
    const FireCombustionParameters& base,
    const FireFuelRasterCell& cell)
{
    amrex::Gpu::DeviceScalar<FireFuelCombustionAccounting>
        device_accounting;
    amrex::Gpu::DeviceScalar<int> device_status;

    auto* accounting_ptr =
        device_accounting.dataPtr();
    auto* status_ptr =
        device_status.dataPtr();

    amrex::ParallelFor(
        1,
        [=] AMREX_GPU_DEVICE (int) noexcept
        {
            FireFuelCombustionAccounting result{};
            const auto status =
                ERFFire::try_make_spatial_fire_combustion_accounting(
                    base,
                    cell,
                    result);
            *accounting_ptr = result;
            *status_ptr = static_cast<int>(status);
        });

    return {
        device_accounting.dataValue(),
        device_status.dataValue()};
}

#endif

} // namespace

TEST(FireFuelModel, Anderson13MoistureContractsMatchPublishedClasses)
{
    const auto bit =
        [](MoistureClass component) {
            return ERFFire::fire_fuel_moisture_component_bit(
                component);
        };

    const unsigned int d1 = bit(MoistureClass::Dead1h);
    const unsigned int d10 = bit(MoistureClass::Dead10h);
    const unsigned int d100 = bit(MoistureClass::Dead100h);
    const unsigned int lh = bit(MoistureClass::LiveHerbaceous);
    const unsigned int lw = bit(MoistureClass::LiveWoody);

    const unsigned int expected[] = {
        0u,
        d1,
        d1 | d10 | d100 | lh,
        d1,
        d1 | d10 | d100 | lw,
        d1 | d10 | lw,
        d1 | d10 | d100,
        d1 | d10 | d100 | lw,
        d1 | d10 | d100,
        d1 | d10 | d100,
        d1 | d10 | d100 | lw,
        d1 | d10 | d100,
        d1 | d10 | d100,
        d1 | d10 | d100
    };

    for (int raw = 0; raw <= 13; ++raw) {
        const auto id =
            static_cast<FireFuelModelId>(raw);
        const FireFuelModelMoistureContract contract =
            ERFFire::fire_fuel_model_moisture_contract(id);

        EXPECT_TRUE(contract.valid_model);
        EXPECT_EQ(contract.burnable, raw != 0);
        EXPECT_EQ(contract.required_mask, expected[raw]);
        EXPECT_EQ(
            ERFFire::fire_fuel_model_number(id),
            raw == 0 ? 0 : raw);
        EXPECT_EQ(
            ERFFire::fire_fuel_model_is_anderson13(id),
            raw != 0);
    }

    const auto invalid =
        ERFFire::fire_fuel_model_moisture_contract(
            static_cast<FireFuelModelId>(99));
    EXPECT_FALSE(invalid.valid_model);
    EXPECT_FALSE(invalid.burnable);
    EXPECT_EQ(invalid.required_mask, 0u);
    EXPECT_FALSE(
        ERFFire::fire_fuel_model_id_valid(
            static_cast<FireFuelModelId>(99)));
}

TEST(FireFuelModel, MissingRequiredMoistureIsExplicit)
{
    FireFuelMoisture moisture;
    moisture.set(MoistureClass::Dead1h, Real(0.05));
    moisture.set(MoistureClass::Dead10h, Real(0.07));
    moisture.set(MoistureClass::Dead100h, Real(0.09));

    const unsigned int lh =
        ERFFire::fire_fuel_moisture_component_bit(
            MoistureClass::LiveHerbaceous);
    const unsigned int lw =
        ERFFire::fire_fuel_moisture_component_bit(
            MoistureClass::LiveWoody);

    EXPECT_EQ(
        ERFFire::fire_fuel_missing_required_moisture_mask(
            FireFuelModelId::FM2,
            moisture),
        lh);
    EXPECT_FALSE(
        ERFFire::fire_fuel_model_moisture_complete(
            FireFuelModelId::FM2,
            moisture));

    moisture.set(
        MoistureClass::LiveHerbaceous,
        Real(0.80));
    EXPECT_TRUE(
        ERFFire::fire_fuel_model_moisture_complete(
            FireFuelModelId::FM2,
            moisture));

    EXPECT_EQ(
        ERFFire::fire_fuel_missing_required_moisture_mask(
            FireFuelModelId::FM4,
            moisture),
        lw);

    moisture.set(
        MoistureClass::LiveWoody,
        Real(0.65));
    EXPECT_TRUE(
        ERFFire::fire_fuel_model_moisture_complete(
            FireFuelModelId::FM4,
            moisture));

    EXPECT_TRUE(
        ERFFire::fire_fuel_model_moisture_complete(
            FireFuelModelId::NonBurnable,
            FireFuelMoisture{}));
    EXPECT_FALSE(
        ERFFire::fire_fuel_model_moisture_complete(
            static_cast<FireFuelModelId>(99),
            moisture));
}

TEST(FireFuelSpread, Anderson13MaterialsMapRequiredMoistureExactly)
{
    for (int raw = 1; raw <= 13; ++raw) {
        const auto id =
            static_cast<FireFuelModelId>(raw);
        const FireFuelMoisture moisture =
            complete_model_moisture(id);

        ERFFire::FireFuelSpreadInputs inputs{};
        EXPECT_EQ(
            ERFFire::try_make_fire_fuel_spread_inputs(
                id,
                moisture,
                Real(2.5),
                Real(0.30),
                inputs),
            ERFFire::FireFuelSpreadInputStatus::success);

        EXPECT_EQ(
            inputs.anderson13_model_number,
            raw);
        EXPECT_EQ(
            inputs.rothermel.model_wind_speed_mps,
            Real(2.5));
        EXPECT_EQ(
            inputs.rothermel.slope_tangent_magnitude,
            Real(0.30));

        const auto required_value =
            [&moisture, id](MoistureClass component) {
                const auto contract =
                    ERFFire::fire_fuel_model_moisture_contract(
                        id);
                const unsigned int bit =
                    ERFFire::fire_fuel_moisture_component_bit(
                        component);
                return (contract.required_mask & bit) != 0u
                    ? moisture.get(component)
                    : Real(0);
            };

        EXPECT_EQ(
            inputs.rothermel.dead_1h_moisture_fraction,
            required_value(MoistureClass::Dead1h));
        EXPECT_EQ(
            inputs.rothermel.dead_10h_moisture_fraction,
            required_value(MoistureClass::Dead10h));
        EXPECT_EQ(
            inputs.rothermel.dead_100h_moisture_fraction,
            required_value(MoistureClass::Dead100h));
        EXPECT_EQ(
            inputs.rothermel.live_foliage_moisture_fraction,
            required_value(MoistureClass::LiveHerbaceous)
                + required_value(MoistureClass::LiveWoody));

        const auto behavior =
            ERFFire::evaluate_rothermel_multiclass(
                ERFFire::make_anderson13_fuel_parameters(
                    inputs.anderson13_model_number),
                inputs.rothermel);
        EXPECT_GE(
            behavior.aligned_heading_ros_mps,
            Real(0));
    }
}

TEST(FireFuelSpread, MissingMoistureAndNonburnableRemainDistinct)
{
    ERFFire::FireFuelSpreadInputs inputs{};

    FireFuelMoisture incomplete;
    incomplete.set(
        MoistureClass::Dead1h,
        Real(0.08));

    EXPECT_EQ(
        ERFFire::try_make_fire_fuel_spread_inputs(
            FireFuelModelId::FM2,
            incomplete,
            Real(1),
            Real(0),
            inputs),
        ERFFire::FireFuelSpreadInputStatus::
            missing_required_moisture);
    EXPECT_EQ(inputs.anderson13_model_number, 0);

    EXPECT_EQ(
        ERFFire::try_make_fire_fuel_spread_inputs(
            FireFuelModelId::NonBurnable,
            FireFuelMoisture{},
            Real(1),
            Real(0),
            inputs),
        ERFFire::FireFuelSpreadInputStatus::nonburnable);

    EXPECT_EQ(
        ERFFire::try_make_fire_fuel_spread_inputs(
            static_cast<FireFuelModelId>(99),
            FireFuelMoisture{},
            Real(1),
            Real(0),
            inputs),
        ERFFire::FireFuelSpreadInputStatus::invalid_model);
}

TEST(FireFuelSource, Anderson13TextInputPreservesResolvedMaterial)
{
    const std::string filename =
        "fire_fuel_source_anderson13_test.txt";

    write_fuel_source_text(
        filename,
        "ERF_FIRE_FUEL_RASTER 1\n"
        "nx 2\n"
        "ny 1\n"
        "xlo_m 0\n"
        "ylo_m 0\n"
        "dx_m 1\n"
        "dy_m 1\n"
        "components 7\n"
        "cells\n"
        "2 15 0.08 0.09 0.10 0.80 0\n"
        "10 23 0.06 0.07 0.08 0 0.65\n"
        "END_ERF_FIRE_FUEL_RASTER\n");

    const FireCartesianRasterGeometry2D geometry{
        2, 1, Real(0), Real(0), Real(1), Real(1)};

    const FireFuelRaster raster =
        ERFFire::read_erf_fire_aligned_fuel_raster_text_file(
            filename,
            geometry);
    const auto state =
        raster.collective_snapshot_state_to_io_rank();

    if (amrex::ParallelDescriptor::IOProcessor()) {
        ASSERT_EQ(state.cells.size(), 2U);
        EXPECT_EQ(
            state.cells[0].model_id,
            FireFuelModelId::FM2);
        EXPECT_EQ(
            state.cells[1].model_id,
            FireFuelModelId::FM10);
        EXPECT_TRUE(
            ERFFire::fire_fuel_model_moisture_complete(
                state.cells[0].model_id,
                state.cells[0].moisture));
        EXPECT_TRUE(
            ERFFire::fire_fuel_model_moisture_complete(
                state.cells[1].model_id,
                state.cells[1].moisture));
        EXPECT_EQ(
            state.cells[0].moisture.get(
                MoistureClass::LiveHerbaceous),
            Real(0.80));
        EXPECT_EQ(
            state.cells[1].moisture.get(
                MoistureClass::LiveWoody),
            Real(0.65));
    }

    remove_fuel_source_text(filename);
}

TEST(FireFuelSource, RejectsMissingRequiredModelMoisture)
{
    const std::string filename =
        "fire_fuel_source_missing_moisture_test.txt";

    write_fuel_source_text(
        filename,
        "ERF_FIRE_FUEL_RASTER 1\n"
        "nx 1\n"
        "ny 1\n"
        "xlo_m 0\n"
        "ylo_m 0\n"
        "dx_m 1\n"
        "dy_m 1\n"
        "components 7\n"
        "cells\n"
        "2 7 0.08 0.09 0.10 0 0\n"
        "END_ERF_FIRE_FUEL_RASTER\n");

    const FireCartesianRasterGeometry2D geometry{
        1, 1, Real(0), Real(0), Real(1), Real(1)};

    EXPECT_THROW(
        (void)ERFFire::read_erf_fire_aligned_fuel_raster_text_file(
            filename,
            geometry),
        std::runtime_error);

    remove_fuel_source_text(filename);
}

TEST(FireFuelRaster, Anderson13ModelIdsRoundTripCollectively)
{
    const FireCartesianRasterGeometry2D geometry{
        14,
        1,
        Real(0),
        Real(0),
        Real(1),
        Real(1)};

    FireFuelRasterState source;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        source.cells.resize(14);

        for (int raw = 0; raw <= 13; ++raw) {
            auto& cell =
                source.cells[
                    static_cast<std::size_t>(raw)];
            cell.model_id =
                static_cast<FireFuelModelId>(raw);

            const auto contract =
                ERFFire::fire_fuel_model_moisture_contract(
                    cell.model_id);
            for (int component = 0;
                 component
                     < FireFuelMoisture::component_count;
                 ++component) {
                const auto moisture_class =
                    static_cast<MoistureClass>(
                        component);
                const unsigned int bit =
                    ERFFire::fire_fuel_moisture_component_bit(
                        moisture_class);
                if ((contract.required_mask & bit) != 0u) {
                    cell.moisture.set(
                        moisture_class,
                        Real(0.05)
                            + Real(0.01)
                                * static_cast<Real>(
                                    component));
                }
            }
        }
    }

    const FireFuelRaster raster =
        FireFuelRaster::collective_from_io_rank_state(
            geometry,
            source);
    const auto restored =
        raster.collective_snapshot_state_to_io_rank();

    if (!amrex::ParallelDescriptor::IOProcessor()) {
        EXPECT_TRUE(restored.cells.empty());
        return;
    }

    ASSERT_EQ(restored.cells.size(), 14U);
    for (int raw = 0; raw <= 13; ++raw) {
        const auto& cell =
            restored.cells[
                static_cast<std::size_t>(raw)];
        EXPECT_EQ(
            cell.model_id,
            static_cast<FireFuelModelId>(raw));
        EXPECT_TRUE(
            ERFFire::fire_fuel_model_moisture_complete(
                cell.model_id,
                cell.moisture));
    }
}

TEST(FireFuelCombustionAccounting, AllModelsConserveDryFuelAndPrescribedWater)
{
    const FireCombustionParameters base =
        ERFFire::make_fm1_combustion_parameters(
            Real(0.08));

    for (int raw = 1; raw <= 13; ++raw) {
        FireFuelRasterCell cell;
        cell.model_id =
            static_cast<FireFuelModelId>(raw);
        cell.moisture =
            complete_model_moisture(cell.model_id);

        const FireFuelCombustionAccounting accounting =
            ERFFire::make_anderson13_fire_combustion_accounting(
                base,
                cell);
        const auto fuel =
            ERFFire::make_anderson13_fuel_parameters(raw);

        const Real expected_dry =
            fuel.dead_1h.dry_load_kg_m2
            + fuel.dead_10h.dry_load_kg_m2
            + fuel.dead_100h.dry_load_kg_m2
            + fuel.live_foliage.dry_load_kg_m2;

        const auto required_moisture =
            [&cell](MoistureClass component) {
                const auto contract =
                    ERFFire::fire_fuel_model_moisture_contract(
                        cell.model_id);
                const unsigned int bit =
                    ERFFire::fire_fuel_moisture_component_bit(
                        component);
                if ((contract.required_mask & bit) == 0u) {
                    return Real(0);
                }
                return cell.moisture.get(component);
            };

        const Real expected_live_moisture =
            required_moisture(
                MoistureClass::LiveHerbaceous)
            + required_moisture(
                MoistureClass::LiveWoody);
        const Real expected_water =
            fuel.dead_1h.dry_load_kg_m2
                * required_moisture(
                    MoistureClass::Dead1h)
            + fuel.dead_10h.dry_load_kg_m2
                * required_moisture(
                    MoistureClass::Dead10h)
            + fuel.dead_100h.dry_load_kg_m2
                * required_moisture(
                    MoistureClass::Dead100h)
            + fuel.live_foliage.dry_load_kg_m2
                * expected_live_moisture;

        EXPECT_NEAR(
            accounting.parameters.dry_fuel_load_kg_m2,
            expected_dry,
            fuel_accounting_tolerance(expected_dry))
            << "fuel model " << raw;
        EXPECT_NEAR(
            accounting.prescribed_water_load_kg_m2,
            expected_water,
            fuel_accounting_tolerance(expected_water))
            << "fuel model " << raw;
        const Real expected_moisture =
            expected_water / expected_dry;
        EXPECT_NEAR(
            accounting.parameters.fuel_moisture_fraction,
            expected_moisture,
            fuel_accounting_tolerance(expected_moisture))
            << "fuel model " << raw;

        EXPECT_EQ(
            accounting.parameters
                .sensible_heat_release_j_kg_dry,
            base.sensible_heat_release_j_kg_dry);
        EXPECT_EQ(
            accounting.parameters.burn_time_constant_s,
            base.burn_time_constant_s);
        EXPECT_EQ(
            accounting.parameters
                .combustion_water_yield_kg_per_kg_dry,
            base.combustion_water_yield_kg_per_kg_dry);

        const Real reconstructed_water =
            accounting.parameters.dry_fuel_load_kg_m2
            * accounting.parameters
                .fuel_moisture_fraction;
        EXPECT_NEAR(
            reconstructed_water,
            accounting.prescribed_water_load_kg_m2,
            fuel_accounting_tolerance(
                accounting.prescribed_water_load_kg_m2));
    }
}

TEST(FireFuelCombustionAccounting, FM1IsExactLegacyLocalParameterProjection)
{
    const FireCombustionParameters base =
        ERFFire::make_fm1_combustion_parameters(
            Real(0.08));

    FireFuelRasterCell cell;
    cell.model_id = FireFuelModelId::FM1;
    cell.moisture.set(
        MoistureClass::Dead1h,
        Real(0.11));

    const FireCombustionParameters legacy =
        ERFFire::make_fire_combustion_parameters_for_fuel_cell(
            base,
            cell);
    const FireFuelCombustionAccounting accounting =
        ERFFire::make_anderson13_fire_combustion_accounting(
            base,
            cell);

    EXPECT_EQ(
        accounting.parameters.dry_fuel_load_kg_m2,
        legacy.dry_fuel_load_kg_m2);
    EXPECT_EQ(
        accounting.parameters.sensible_heat_release_j_kg_dry,
        legacy.sensible_heat_release_j_kg_dry);
    EXPECT_EQ(
        accounting.parameters.fuel_moisture_fraction,
        legacy.fuel_moisture_fraction);
    EXPECT_EQ(
        accounting.parameters.burn_time_constant_s,
        legacy.burn_time_constant_s);
    EXPECT_EQ(
        accounting.parameters.combustion_water_yield_kg_per_kg_dry,
        legacy.combustion_water_yield_kg_per_kg_dry);
    EXPECT_EQ(
        accounting.prescribed_water_load_kg_m2,
        legacy.dry_fuel_load_kg_m2
            * legacy.fuel_moisture_fraction);
}

TEST(FireFuelCombustionAccounting, MaterialFailuresRemainDistinct)
{
    const FireCombustionParameters base =
        ERFFire::make_fm1_combustion_parameters(
            Real(0.08));
    FireFuelCombustionAccounting accounting{};

    FireFuelRasterCell incomplete;
    incomplete.model_id = FireFuelModelId::FM2;
    incomplete.moisture.set(
        MoistureClass::Dead1h,
        Real(0.08));

    EXPECT_EQ(
        ERFFire::try_make_anderson13_fire_combustion_accounting(
            base,
            incomplete,
            accounting),
        FireFuelCombustionAccountingStatus::
            missing_required_moisture);

    FireFuelRasterCell nonburnable;
    nonburnable.model_id =
        FireFuelModelId::NonBurnable;
    EXPECT_EQ(
        ERFFire::try_make_anderson13_fire_combustion_accounting(
            base,
            nonburnable,
            accounting),
        FireFuelCombustionAccountingStatus::nonburnable);

    FireFuelRasterCell invalid;
    invalid.model_id =
        static_cast<FireFuelModelId>(99);
    EXPECT_EQ(
        ERFFire::try_make_anderson13_fire_combustion_accounting(
            base,
            invalid,
            accounting),
        FireFuelCombustionAccountingStatus::invalid_model);

    FireCombustionParameters invalid_base = base;
    invalid_base.burn_time_constant_s = Real(0);
    FireFuelRasterCell fm1;
    fm1.model_id = FireFuelModelId::FM1;
    fm1.moisture.set(
        MoistureClass::Dead1h,
        Real(0.08));
    EXPECT_EQ(
        ERFFire::try_make_anderson13_fire_combustion_accounting(
            invalid_base,
            fm1,
            accounting),
        FireFuelCombustionAccountingStatus::
            invalid_base_parameters);
}

TEST(FireFuelCombustionAccounting, ProductionResolverRemainsFailClosedForFM2)
{
    const FireCombustionParameters base =
        ERFFire::make_fm1_combustion_parameters(
            Real(0.08));

    FireFuelRasterCell cell;
    cell.model_id = FireFuelModelId::FM2;
    cell.moisture =
        complete_model_moisture(cell.model_id);

    FireFuelCombustionAccounting accounting{};
    ASSERT_EQ(
        ERFFire::try_make_anderson13_fire_combustion_accounting(
            base,
            cell,
            accounting),
        FireFuelCombustionAccountingStatus::success);

    FireCombustionParameters production{};
    EXPECT_EQ(
        ERFFire::try_make_fire_combustion_parameters_for_fuel_cell(
            base,
            cell,
            production),
        FireFuelCombustionParameterStatus::unsupported_model);
    EXPECT_EQ(production.dry_fuel_load_kg_m2, Real(0));
}

#ifdef AMREX_USE_GPU
TEST(FireFuelCombustionAccounting, DeviceSafeResolutionMatchesHost)
{
    const FireCombustionParameters base =
        ERFFire::make_fm1_combustion_parameters(
            Real(0.08));

    FireFuelRasterCell cell;
    cell.model_id = FireFuelModelId::FM10;
    cell.moisture =
        complete_model_moisture(cell.model_id);

    FireFuelCombustionAccounting host{};
    const auto host_status =
        ERFFire::try_make_anderson13_fire_combustion_accounting(
            base,
            cell,
            host);
    ASSERT_EQ(
        host_status,
        FireFuelCombustionAccountingStatus::success);

    const DeviceFuelCombustionAccountingProbe actual =
        run_device_fuel_combustion_accounting_probe(
            base,
            cell);

    EXPECT_EQ(
        actual.status,
        static_cast<int>(
            FireFuelCombustionAccountingStatus::success));
    EXPECT_NEAR(
        actual.accounting.parameters.dry_fuel_load_kg_m2,
        host.parameters.dry_fuel_load_kg_m2,
        fuel_accounting_tolerance(
            host.parameters.dry_fuel_load_kg_m2));
    EXPECT_NEAR(
        actual.accounting.parameters.fuel_moisture_fraction,
        host.parameters.fuel_moisture_fraction,
        fuel_accounting_tolerance(
            host.parameters.fuel_moisture_fraction));
    EXPECT_NEAR(
        actual.accounting.prescribed_water_load_kg_m2,
        host.prescribed_water_load_kg_m2,
        fuel_accounting_tolerance(
            host.prescribed_water_load_kg_m2));
    EXPECT_EQ(
        actual.accounting.parameters.sensible_heat_release_j_kg_dry,
        host.parameters.sensible_heat_release_j_kg_dry);
    EXPECT_EQ(
        actual.accounting.parameters.burn_time_constant_s,
        host.parameters.burn_time_constant_s);
    EXPECT_EQ(
        actual.accounting.parameters.combustion_water_yield_kg_per_kg_dry,
        host.parameters.combustion_water_yield_kg_per_kg_dry);
}
#endif

#ifdef AMREX_USE_GPU
TEST(FireFuelCombustionAccounting, DeviceSafeSpatialFm2ResolutionMatchesHost)
{
    const FireCombustionParameters base =
        ERFFire::make_fm1_combustion_parameters(
            Real(0.08));

    FireFuelRasterCell cell;
    cell.model_id = FireFuelModelId::FM2;
    cell.moisture.set(MoistureClass::Dead1h, Real(0.08));
    cell.moisture.set(MoistureClass::Dead10h, Real(0.09));
    cell.moisture.set(MoistureClass::Dead100h, Real(0.10));
    cell.moisture.set(MoistureClass::LiveHerbaceous, Real(0.80));

    const auto host =
        ERFFire::make_spatial_fire_combustion_accounting(
            base,
            cell);
    const auto actual =
        run_device_spatial_fuel_combustion_accounting_probe(
            base,
            cell);

    ASSERT_EQ(
        actual.status,
        static_cast<int>(
            FireFuelCombustionAccountingStatus::success));
    EXPECT_NEAR(
        actual.accounting.parameters.dry_fuel_load_kg_m2,
        host.parameters.dry_fuel_load_kg_m2,
        fuel_accounting_tolerance(
            host.parameters.dry_fuel_load_kg_m2));
    EXPECT_NEAR(
        actual.accounting.parameters.fuel_moisture_fraction,
        host.parameters.fuel_moisture_fraction,
        fuel_accounting_tolerance(
            host.parameters.fuel_moisture_fraction));
    EXPECT_NEAR(
        actual.accounting.prescribed_water_load_kg_m2,
        host.prescribed_water_load_kg_m2,
        fuel_accounting_tolerance(
            host.prescribed_water_load_kg_m2));
    EXPECT_EQ(
        actual.accounting.parameters.burn_time_constant_s,
        host.parameters.burn_time_constant_s);
    EXPECT_EQ(
        actual.accounting.parameters.combustion_water_yield_kg_per_kg_dry,
        host.parameters.combustion_water_yield_kg_per_kg_dry);
}
#endif

TEST(FireFuelCombustion, AndersonModelsRemainUnsupportedUntilCombustionPolicy)
{
    const auto base = fuel_combustion_base_parameters();

    FireFuelRasterCell cell;
    cell.model_id = FireFuelModelId::FM2;
    cell.moisture.set(
        MoistureClass::Dead1h,
        Real(0.08));
    cell.moisture.set(
        MoistureClass::Dead10h,
        Real(0.09));
    cell.moisture.set(
        MoistureClass::Dead100h,
        Real(0.10));
    cell.moisture.set(
        MoistureClass::LiveHerbaceous,
        Real(0.80));

    ASSERT_TRUE(
        ERFFire::fire_fuel_model_moisture_complete(
            cell.model_id,
            cell.moisture));

    FireCombustionParameters local = base;
    EXPECT_EQ(
        ERFFire::try_make_fire_combustion_parameters_for_fuel_cell(
            base,
            cell,
            local),
        FireFuelCombustionParameterStatus::unsupported_model);
    EXPECT_EQ(local.dry_fuel_load_kg_m2, Real(0));
    EXPECT_EQ(local.fuel_moisture_fraction, Real(0));
}

TEST(FireFuelRaster, CanonicalRoundTripPreservesCategoricalCells)
{
    const auto geometry = fuel_raster_geometry();
    const auto source = make_io_rank_state(geometry);
    const FireFuelRaster raster =
        FireFuelRaster::collective_from_io_rank_state(
            geometry,
            source);

    const auto restored =
        raster.collective_snapshot_state_to_io_rank();

    if (!amrex::ParallelDescriptor::IOProcessor()) {
        EXPECT_TRUE(restored.cells.empty());
        return;
    }

    ASSERT_EQ(restored.cells.size(), source.cells.size());
    for (std::size_t index = 0;
         index < source.cells.size();
         ++index) {
        EXPECT_EQ(
            restored.cells[index].model_id,
            source.cells[index].model_id);
        expect_same_moisture(
            restored.cells[index].moisture,
            source.cells[index].moisture);
    }
}

TEST(FireFuelRaster, MissingAndLiveMoistureRemainDistinct)
{
    const auto geometry = fuel_raster_geometry();
    const FireFuelRaster raster =
        FireFuelRaster::collective_from_io_rank_state(
            geometry,
            make_io_rank_state(geometry));

    if (amrex::ParallelDescriptor::NProcs() != 1) {
        GTEST_SKIP() << "scalar FireFuelRaster access is intentionally single-rank";
    }

    const auto nonburnable = raster.state(0, 0);
    EXPECT_EQ(
        nonburnable.model_id,
        FireFuelModelId::NonBurnable);
    EXPECT_FALSE(
        nonburnable.moisture.has(MoistureClass::Dead1h));

    const auto live = raster.state(0, 1);
    EXPECT_EQ(live.model_id, FireFuelModelId::FM1);
    EXPECT_TRUE(
        live.moisture.has(MoistureClass::LiveHerbaceous));
    EXPECT_GT(
        live.moisture.get(MoistureClass::LiveHerbaceous),
        Real(1));
    EXPECT_FALSE(
        live.moisture.has(MoistureClass::LiveWoody));
}

TEST(FireFuelRaster, RejectsWrongCellCountAndUnsupportedModel)
{
    const auto geometry = fuel_raster_geometry();

    FireFuelRasterState wrong_count;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        wrong_count.cells.resize(1);
    }
    EXPECT_THROW(
        (void)FireFuelRaster::collective_from_io_rank_state(
            geometry,
            wrong_count),
        std::invalid_argument);

    FireFuelRasterState invalid_model =
        make_io_rank_state(geometry);
    if (amrex::ParallelDescriptor::IOProcessor()) {
        invalid_model.cells[1].model_id =
            static_cast<FireFuelModelId>(99);
    }
    EXPECT_THROW(
        (void)FireFuelRaster::collective_from_io_rank_state(
            geometry,
            invalid_model),
        std::invalid_argument);
}

TEST(FireFuelRaster, CopiesShareImmutableDistributedStorage)
{
    const auto geometry = fuel_raster_geometry();
    const FireFuelRaster original =
        FireFuelRaster::collective_from_io_rank_state(
            geometry,
            make_io_rank_state(geometry));
    const FireFuelRaster copy = original;

    EXPECT_EQ(
        &original.distributed_values(),
        &copy.distributed_values());
    EXPECT_EQ(original.geometry().nx, copy.geometry().nx);
    EXPECT_EQ(original.geometry().ny, copy.geometry().ny);
}

#ifdef AMREX_USE_GPU
TEST(FireFuelRaster, DeviceDecoderMatchesCanonicalMaterial)
{
    const auto geometry = fuel_raster_geometry();
    const FireFuelRaster raster =
        FireFuelRaster::collective_from_io_rank_state(
            geometry,
            make_io_rank_state(geometry));

    EXPECT_EQ(device_decode_mismatch_count(raster), 0);
}
#endif

TEST(FireFuelRaster, CollectivePointSamplerPreservesOrderAndFaceOwnership)
{
    const auto geometry = fuel_raster_geometry();
    const FireFuelRaster raster =
        FireFuelRaster::collective_from_io_rank_state(
            geometry,
            make_io_rank_state(geometry));

    const std::vector<FireVec2> positions{
        {Real(101), Real(-48)},
        {Real(104), Real(-48)},
        {Real(101), Real(-46)},
        {Real(108), Real(-38)},
        {Real(104), Real(-48)}};
    const int expected_i[]{0, 2, 0, 3, 2};
    const int expected_j[]{0, 0, 1, 2, 0};

    const auto sampled =
        raster.collective_sample_points(positions);

    ASSERT_EQ(sampled.size(), positions.size());
    for (std::size_t index = 0;
         index < sampled.size();
         ++index) {
        const auto expected =
            make_expected_cell(
                expected_i[index],
                expected_j[index]);
        EXPECT_EQ(
            sampled[index].model_id,
            expected.model_id);
        expect_same_moisture(
            sampled[index].moisture,
            expected.moisture);
    }
}

TEST(FireFuelRaster, CollectivePointSamplerEmptyBatchIsEmpty)
{
    const auto geometry = fuel_raster_geometry();
    const FireFuelRaster raster =
        FireFuelRaster::collective_from_io_rank_state(
            geometry,
            make_io_rank_state(geometry));

    EXPECT_TRUE(
        raster.collective_sample_points({}).empty());
}

TEST(FireFuelRaster, CollectivePointSamplerRejectsInvalidPoints)
{
    const auto geometry = fuel_raster_geometry();
    const FireFuelRaster raster =
        FireFuelRaster::collective_from_io_rank_state(
            geometry,
            make_io_rank_state(geometry));
    const Real inf = std::numeric_limits<Real>::infinity();

    EXPECT_THROW(
        (void)raster.collective_sample_points(
            std::vector<FireVec2>{
                {
                    std::nextafter(geometry.xlo_m, -inf),
                    geometry.ylo_m}}),
        std::out_of_range);

    EXPECT_THROW(
        (void)raster.collective_sample_points(
            std::vector<FireVec2>{
                {
                    std::numeric_limits<Real>::quiet_NaN(),
                    geometry.ylo_m}}),
        std::invalid_argument);
}

TEST(FireFuelCombustionRaster, SpatialRuntimePreservesFm1AndUsesAndersonAccounting)
{
    const auto geometry = combustion_geometry();
    const auto fuel_raster =
        make_anderson_combustion_fuel_raster();
    const auto burned =
        fully_burned_combustion_raster();
    const auto base =
        ERFFire::make_fm1_combustion_parameters(
            Real(0.08));

    FireCombustionRaster combustion(
        geometry,
        base,
        FireCombustionRasterOptions{4});
    (void)combustion.initialize_from_burned_fraction(
        burned,
        fuel_raster);

    auto snapshot =
        combustion.collective_snapshot_state_to_io_rank();
    if (amrex::ParallelDescriptor::IOProcessor()) {
        ASSERT_EQ(snapshot.cells.size(), 2U);

        FireFuelRasterCell fm1_cell;
        fm1_cell.model_id = FireFuelModelId::FM1;
        fm1_cell.moisture.set(
            MoistureClass::Dead1h,
            Real(0.08));
        const auto fm1 =
            ERFFire::make_spatial_fire_combustion_accounting(
                base,
                fm1_cell);

        FireFuelRasterCell fm2_cell;
        fm2_cell.model_id = FireFuelModelId::FM2;
        fm2_cell.moisture.set(MoistureClass::Dead1h, Real(0.08));
        fm2_cell.moisture.set(MoistureClass::Dead10h, Real(0.09));
        fm2_cell.moisture.set(MoistureClass::Dead100h, Real(0.10));
        fm2_cell.moisture.set(MoistureClass::LiveHerbaceous, Real(0.80));
        const auto fm2 =
            ERFFire::make_spatial_fire_combustion_accounting(
                base,
                fm2_cell);
        const auto fm2_anderson =
            ERFFire::make_anderson13_fire_combustion_accounting(
                base,
                fm2_cell);

        EXPECT_EQ(
            fm1.parameters.dry_fuel_load_kg_m2,
            base.dry_fuel_load_kg_m2);
        EXPECT_EQ(
            fm2.parameters.dry_fuel_load_kg_m2,
            fm2_anderson.parameters.dry_fuel_load_kg_m2);

        EXPECT_NEAR(
            snapshot.cells[0].remaining_dry_fuel_kg_m2,
            fm1.parameters.dry_fuel_load_kg_m2,
            fuel_accounting_tolerance(
                fm1.parameters.dry_fuel_load_kg_m2));
        EXPECT_NEAR(
            snapshot.cells[1].remaining_dry_fuel_kg_m2,
            fm2.parameters.dry_fuel_load_kg_m2,
            fuel_accounting_tolerance(
                fm2.parameters.dry_fuel_load_kg_m2));
        EXPECT_GT(
            snapshot.cells[1].remaining_dry_fuel_kg_m2,
            snapshot.cells[0].remaining_dry_fuel_kg_m2);
    }

    const FirePerimeter perimeter =
        full_combustion_perimeter();
    const auto update =
        combustion.advance_from_linear_sweep(
            perimeter,
            perimeter,
            burned,
            burned,
            fuel_raster,
            Real(1.0));

    EXPECT_GT(update.newly_consumed_dry_fuel_kg, Real(0));
    EXPECT_GT(update.sensible_energy_increment_j, Real(0));
    EXPECT_GT(update.water_released_increment_kg, Real(0));

    snapshot =
        combustion.collective_snapshot_state_to_io_rank();
    if (amrex::ParallelDescriptor::IOProcessor()) {
        ASSERT_EQ(snapshot.cells.size(), 2U);

        FireFuelRasterCell cells[2];
        cells[0].model_id = FireFuelModelId::FM1;
        cells[0].moisture.set(MoistureClass::Dead1h, Real(0.08));
        cells[1].model_id = FireFuelModelId::FM2;
        cells[1].moisture.set(MoistureClass::Dead1h, Real(0.08));
        cells[1].moisture.set(MoistureClass::Dead10h, Real(0.09));
        cells[1].moisture.set(MoistureClass::Dead100h, Real(0.10));
        cells[1].moisture.set(MoistureClass::LiveHerbaceous, Real(0.80));

        for (int index = 0; index < 2; ++index) {
            const auto local =
                ERFFire::make_spatial_fire_combustion_accounting(
                    base,
                    cells[index]);
            const auto& state =
                snapshot.cells[static_cast<std::size_t>(index)];

            EXPECT_NEAR(
                state.remaining_dry_fuel_kg_m2
                    + state.consumed_dry_fuel_kg_m2,
                local.parameters.dry_fuel_load_kg_m2,
                fuel_accounting_tolerance(
                    local.parameters.dry_fuel_load_kg_m2));
            EXPECT_NEAR(
                state.sensible_energy_j_m2,
                state.consumed_dry_fuel_kg_m2
                    * local.parameters.sensible_heat_release_j_kg_dry,
                fuel_accounting_tolerance(
                    state.sensible_energy_j_m2));
            EXPECT_NEAR(
                state.water_released_kg_m2,
                state.consumed_dry_fuel_kg_m2
                    * (local.parameters.fuel_moisture_fraction
                       + local.parameters
                           .combustion_water_yield_kg_per_kg_dry),
                fuel_accounting_tolerance(
                    state.water_released_kg_m2));
        }
    }
}

TEST(FireFuelCombustion, Fm1OverridesOnlyDead1hMoisture)
{
    const auto base = fuel_combustion_base_parameters();
    FireFuelRasterCell cell;
    cell.model_id = FireFuelModelId::FM1;
    cell.moisture.set(MoistureClass::Dead1h, Real(0.08));
    cell.moisture.set(MoistureClass::Dead10h, Real(0.12));
    cell.moisture.set(MoistureClass::Dead100h, Real(0.17));

    FireCombustionParameters local{};
    EXPECT_EQ(
        ERFFire::try_make_fire_combustion_parameters_for_fuel_cell(
            base,
            cell,
            local),
        FireFuelCombustionParameterStatus::success);

    EXPECT_EQ(local.dry_fuel_load_kg_m2, base.dry_fuel_load_kg_m2);
    EXPECT_EQ(
        local.sensible_heat_release_j_kg_dry,
        base.sensible_heat_release_j_kg_dry);
    EXPECT_EQ(local.fuel_moisture_fraction, Real(0.08));
    EXPECT_EQ(local.burn_time_constant_s, base.burn_time_constant_s);
    EXPECT_EQ(
        local.combustion_water_yield_kg_per_kg_dry,
        base.combustion_water_yield_kg_per_kg_dry);
}

TEST(FireFuelCombustion, LiveMoistureDoesNotChangeFm1Policy)
{
    const auto base = fuel_combustion_base_parameters();
    FireFuelRasterCell without_live;
    without_live.model_id = FireFuelModelId::FM1;
    without_live.moisture.set(MoistureClass::Dead1h, Real(0.08));

    FireFuelRasterCell with_live = without_live;
    with_live.moisture.set(MoistureClass::LiveHerbaceous, Real(1.20));
    with_live.moisture.set(MoistureClass::LiveWoody, Real(2.10));

    const auto first =
        ERFFire::make_fire_combustion_parameters_for_fuel_cell(
            base,
            without_live);
    const auto second =
        ERFFire::make_fire_combustion_parameters_for_fuel_cell(
            base,
            with_live);

    EXPECT_EQ(first.dry_fuel_load_kg_m2, second.dry_fuel_load_kg_m2);
    EXPECT_EQ(
        first.sensible_heat_release_j_kg_dry,
        second.sensible_heat_release_j_kg_dry);
    EXPECT_EQ(first.fuel_moisture_fraction, second.fuel_moisture_fraction);
    EXPECT_EQ(first.burn_time_constant_s, second.burn_time_constant_s);
    EXPECT_EQ(
        first.combustion_water_yield_kg_per_kg_dry,
        second.combustion_water_yield_kg_per_kg_dry);
}

TEST(FireFuelCombustion, MoistureAboveOneRemainsValidMassRatio)
{
    const auto base = fuel_combustion_base_parameters();
    FireFuelRasterCell cell;
    cell.model_id = FireFuelModelId::FM1;
    cell.moisture.set(MoistureClass::Dead1h, Real(1.25));

    const auto local =
        ERFFire::make_fire_combustion_parameters_for_fuel_cell(
            base,
            cell);

    EXPECT_EQ(local.fuel_moisture_fraction, Real(1.25));
}

TEST(FireFuelCombustion, MissingDead1hMoistureIsNotDryFuel)
{
    const auto base = fuel_combustion_base_parameters();
    FireFuelRasterCell cell;
    cell.model_id = FireFuelModelId::FM1;
    cell.moisture.set(MoistureClass::Dead10h, Real(0.08));

    FireCombustionParameters local = base;
    EXPECT_EQ(
        ERFFire::try_make_fire_combustion_parameters_for_fuel_cell(
            base,
            cell,
            local),
        FireFuelCombustionParameterStatus::missing_dead_1h_moisture);
    EXPECT_EQ(local.dry_fuel_load_kg_m2, Real(0));
    EXPECT_EQ(local.fuel_moisture_fraction, Real(0));
    EXPECT_THROW(
        (void)ERFFire::make_fire_combustion_parameters_for_fuel_cell(
            base,
            cell),
        std::invalid_argument);
}

TEST(FireFuelCombustion, NonburnableRejectedUntilBarrierDynamicsExist)
{
    const auto base = fuel_combustion_base_parameters();
    FireFuelRasterCell cell;
    cell.model_id = FireFuelModelId::NonBurnable;
    cell.moisture.set(MoistureClass::Dead1h, Real(0.08));

    FireCombustionParameters local = base;
    EXPECT_EQ(
        ERFFire::try_make_fire_combustion_parameters_for_fuel_cell(
            base,
            cell,
            local),
        FireFuelCombustionParameterStatus::nonburnable);
    EXPECT_EQ(local.dry_fuel_load_kg_m2, Real(0));
    EXPECT_THROW(
        (void)ERFFire::make_fire_combustion_parameters_for_fuel_cell(
            base,
            cell),
        std::invalid_argument);
}

TEST(FireFuelCombustion, InvalidBaseAndUnknownModelAreExplicit)
{
    FireFuelRasterCell cell;
    cell.model_id = FireFuelModelId::FM1;
    cell.moisture.set(MoistureClass::Dead1h, Real(0.08));

    auto invalid_base = fuel_combustion_base_parameters();
    invalid_base.dry_fuel_load_kg_m2 = Real(0);
    FireCombustionParameters local{};
    EXPECT_EQ(
        ERFFire::try_make_fire_combustion_parameters_for_fuel_cell(
            invalid_base,
            cell,
            local),
        FireFuelCombustionParameterStatus::invalid_base_parameters);

    cell.model_id = static_cast<FireFuelModelId>(13);
    EXPECT_EQ(
        ERFFire::try_make_fire_combustion_parameters_for_fuel_cell(
            fuel_combustion_base_parameters(),
            cell,
            local),
        FireFuelCombustionParameterStatus::unsupported_model);
}

TEST(FireFuelCombustionRaster, UniformSpatialFm1MatchesLegacyAccounting)
{
    const auto geometry = combustion_geometry();
    const auto base = fuel_combustion_base_parameters();
    const auto fuel =
        make_combustion_fuel_raster(
            base.fuel_moisture_fraction,
            base.fuel_moisture_fraction);
    const auto perimeter = full_combustion_perimeter();
    const auto burned = fully_burned_combustion_raster();

    FireCombustionRaster legacy(
        geometry,
        base,
        FireCombustionRasterOptions{4});
    FireCombustionRaster spatial(
        geometry,
        base,
        FireCombustionRasterOptions{4});

    const auto legacy_initial =
        legacy.initialize_from_burned_fraction(burned);
    const auto spatial_initial =
        spatial.initialize_from_burned_fraction(
            burned,
            fuel);
    expect_same_totals(spatial_initial, legacy_initial);

    const auto legacy_update =
        legacy.advance_from_linear_sweep(
            perimeter,
            perimeter,
            burned,
            burned,
            Real(2));
    const auto spatial_update =
        spatial.advance_from_linear_sweep(
            perimeter,
            perimeter,
            burned,
            burned,
            fuel,
            Real(2));

    expect_same_totals(
        spatial_update.totals,
        legacy_update.totals);
    EXPECT_EQ(
        spatial_update.newly_consumed_dry_fuel_kg,
        legacy_update.newly_consumed_dry_fuel_kg);
    EXPECT_EQ(
        spatial_update.sensible_energy_increment_j,
        legacy_update.sensible_energy_increment_j);
    EXPECT_EQ(
        spatial_update.water_released_increment_kg,
        legacy_update.water_released_increment_kg);

    const auto legacy_state =
        legacy.collective_snapshot_state_to_io_rank();
    const auto spatial_state =
        spatial.collective_snapshot_state_to_io_rank();
    if (amrex::ParallelDescriptor::IOProcessor()) {
        ASSERT_EQ(
            spatial_state.cells.size(),
            legacy_state.cells.size());
        for (std::size_t index = 0;
             index < spatial_state.cells.size();
             ++index) {
            expect_same_combustion_state(
                spatial_state.cells[index],
                legacy_state.cells[index]);
        }
    }
}

TEST(FireFuelCombustionRaster, SpatialMoistureChangesOnlyWaterAccounting)
{
    const auto geometry = combustion_geometry();
    const auto base = fuel_combustion_base_parameters();
    const Real moisture[2]{Real(0.05), Real(0.45)};
    const auto fuel =
        make_combustion_fuel_raster(
            moisture[0],
            moisture[1]);
    const auto perimeter = full_combustion_perimeter();
    const auto burned = fully_burned_combustion_raster();

    FireCombustionRaster legacy(
        geometry,
        base,
        FireCombustionRasterOptions{4});
    FireCombustionRaster spatial(
        geometry,
        base,
        FireCombustionRasterOptions{4});

    (void)legacy.initialize_from_burned_fraction(burned);
    (void)spatial.initialize_from_burned_fraction(
        burned,
        fuel);

    const auto legacy_update =
        legacy.advance_from_linear_sweep(
            perimeter,
            perimeter,
            burned,
            burned,
            Real(2));
    const auto spatial_update =
        spatial.advance_from_linear_sweep(
            perimeter,
            perimeter,
            burned,
            burned,
            fuel,
            Real(2));

    EXPECT_EQ(
        spatial_update.newly_consumed_dry_fuel_kg,
        legacy_update.newly_consumed_dry_fuel_kg);
    EXPECT_EQ(
        spatial_update.sensible_energy_increment_j,
        legacy_update.sensible_energy_increment_j);

    const auto legacy_state =
        legacy.collective_snapshot_state_to_io_rank();
    const auto spatial_state =
        spatial.collective_snapshot_state_to_io_rank();

    if (amrex::ParallelDescriptor::IOProcessor()) {
        ASSERT_EQ(spatial_state.cells.size(), 2u);
        ASSERT_EQ(
            legacy_state.cells.size(),
            spatial_state.cells.size());

        for (std::size_t index = 0; index < 2; ++index) {
            const auto& actual = spatial_state.cells[index];
            const auto& reference = legacy_state.cells[index];

            EXPECT_EQ(
                actual.ignited_area_fraction,
                reference.ignited_area_fraction);
            EXPECT_EQ(
                actual.remaining_dry_fuel_kg_m2,
                reference.remaining_dry_fuel_kg_m2);
            EXPECT_EQ(
                actual.consumed_dry_fuel_kg_m2,
                reference.consumed_dry_fuel_kg_m2);
            EXPECT_EQ(
                actual.sensible_energy_j_m2,
                reference.sensible_energy_j_m2);

            const Real expected_water =
                actual.consumed_dry_fuel_kg_m2
                * (moisture[index]
                   + base.combustion_water_yield_kg_per_kg_dry);
            EXPECT_EQ(
                actual.water_released_kg_m2,
                expected_water);
        }

        EXPECT_EQ(
            spatial_state.cells[0].remaining_dry_fuel_kg_m2,
            spatial_state.cells[1].remaining_dry_fuel_kg_m2);
        EXPECT_EQ(
            spatial_state.cells[0].consumed_dry_fuel_kg_m2,
            spatial_state.cells[1].consumed_dry_fuel_kg_m2);
        EXPECT_EQ(
            spatial_state.cells[0].sensible_energy_j_m2,
            spatial_state.cells[1].sensible_energy_j_m2);
        EXPECT_NE(
            spatial_state.cells[0].water_released_kg_m2,
            spatial_state.cells[1].water_released_kg_m2);
    }

    EXPECT_GT(
        spatial_update.water_released_increment_kg,
        Real(0));
}

TEST(FireFuelCombustionRaster, RejectsNonburnableAndMissingMoistureBeforeMutation)
{
    const auto geometry = combustion_geometry();
    const auto base = fuel_combustion_base_parameters();
    const auto burned = fully_burned_combustion_raster();

    for (bool nonburnable : {true, false}) {
        const auto fuel =
            make_invalid_combustion_fuel_raster(nonburnable);
        FireCombustionRaster combustion(
            geometry,
            base,
            FireCombustionRasterOptions{4});

        EXPECT_THROW(
            (void)combustion.initialize_from_burned_fraction(
                burned,
                fuel),
            std::invalid_argument);
        EXPECT_FALSE(combustion.initialized());
        expect_same_totals(
            combustion.totals(),
            ERFFire::FireCombustionRasterTotals{});
    }
}

TEST(FireFuelCombustionRaster, ChangingSpatialFuelAfterConsumptionIsRejected)
{
    const auto geometry = combustion_geometry();
    const auto base = fuel_combustion_base_parameters();
    const auto first_fuel =
        make_combustion_fuel_raster(
            Real(0.05),
            Real(0.45));
    const auto changed_fuel =
        make_combustion_fuel_raster(
            Real(0.45),
            Real(0.05));
    const auto perimeter = full_combustion_perimeter();
    const auto burned = fully_burned_combustion_raster();

    FireCombustionRaster combustion(
        geometry,
        base,
        FireCombustionRasterOptions{4});
    (void)combustion.initialize_from_burned_fraction(
        burned,
        first_fuel);
    (void)combustion.advance_from_linear_sweep(
        perimeter,
        perimeter,
        burned,
        burned,
        first_fuel,
        Real(2));

    const auto before =
        combustion.collective_snapshot_state_to_io_rank();
    const auto before_totals = combustion.totals();

    EXPECT_THROW(
        (void)combustion.advance_from_linear_sweep(
            perimeter,
            perimeter,
            burned,
            burned,
            changed_fuel,
            Real(1)),
        std::invalid_argument);

    expect_same_totals(combustion.totals(), before_totals);
    const auto after =
        combustion.collective_snapshot_state_to_io_rank();
    if (amrex::ParallelDescriptor::IOProcessor()) {
        ASSERT_EQ(after.cells.size(), before.cells.size());
        for (std::size_t index = 0;
             index < after.cells.size();
             ++index) {
            expect_same_combustion_state(
                after.cells[index],
                before.cells[index]);
        }
    }
}
