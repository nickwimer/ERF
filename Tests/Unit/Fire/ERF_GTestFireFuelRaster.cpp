#include <ERF_FireFuelCombustion.H>
#include <ERF_FireFuelRaster.H>

#include <AMReX_ParallelDescriptor.H>
#include <gtest/gtest.h>

#include <cstddef>
#include <stdexcept>

namespace
{

using amrex::Real;
using ERFFire::FireCartesianRasterGeometry2D;
using ERFFire::FireCombustionParameters;
using ERFFire::FireFuelCombustionParameterStatus;
using ERFFire::FireFuelModelId;
using ERFFire::FireFuelMoisture;
using ERFFire::FireFuelRaster;
using ERFFire::FireFuelRasterCell;
using ERFFire::FireFuelRasterState;
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

} // namespace

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
