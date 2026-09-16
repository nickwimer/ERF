#include <ERF_FireFuelField.H>
#include <ERF_FireRuntimeInit.H>
#include <ERF_FireSpreadRuntime.H>

#include <AMReX_Gpu.H>
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace
{

using amrex::Real;
using ERFFire::FireCartesianRasterGeometry2D;
using ERFFire::FireFuelField;
using ERFFire::FireFuelMoisture;
using ERFFire::FireFuelProperties;
using Component = ERFFire::FireFuelMoistureClass;
using ERFFire::detail::fire_cartesian_raster_cell_bounds;
using ERFFire::detail::fire_cartesian_raster_locate_cell;

constexpr Component components[] = {
    Component::Dead1h,
    Component::Dead10h,
    Component::Dead100h,
    Component::LiveHerbaceous,
    Component::LiveWoody};

FireCartesianRasterGeometry2D
geometry()
{
    return {4, 3, Real(100), Real(-50), Real(2), Real(4)};
}

void
expect_fuel_bits_equal(
    const ERFFire::RothermelFuelParameters& a,
    const ERFFire::RothermelFuelParameters& b)
{
    using Fuel = ERFFire::RothermelFuelParameters;
    const Real Fuel::* members[] = {
        &Fuel::dead_1h_load_kg_m2,
        &Fuel::dead_1h_sav_m_inv,
        &Fuel::fuel_bed_depth_m,
        &Fuel::dead_heat_content_j_kg,
        &Fuel::particle_density_kg_m3,
        &Fuel::total_mineral_fraction,
        &Fuel::effective_mineral_fraction,
        &Fuel::dead_moisture_of_extinction};

    for (auto member : members) {
        EXPECT_EQ(
            std::memcmp(&(a.*member), &(b.*member), sizeof(Real)),
            0);
    }
}

void
expect_behavior_bits_equal(
    const ERFFire::RothermelResult& a,
    const ERFFire::RothermelResult& b)
{
    using Result = ERFFire::RothermelResult;
    const Real Result::* members[] = {
        &Result::net_fuel_loading_kg_m2,
        &Result::bulk_density_kg_m3,
        &Result::packing_ratio,
        &Result::optimum_packing_ratio,
        &Result::reaction_velocity_exponent,
        &Result::max_reaction_velocity_s_inv,
        &Result::reaction_velocity_s_inv,
        &Result::moisture_damping,
        &Result::mineral_damping,
        &Result::reaction_intensity_w_m2,
        &Result::propagating_flux_ratio,
        &Result::effective_heating_number,
        &Result::heat_of_preignition_j_kg,
        &Result::heat_sink_j_m3,
        &Result::no_wind_no_slope_ros_mps,
        &Result::wind_factor_coefficient_si,
        &Result::wind_factor_exponent,
        &Result::wind_factor,
        &Result::slope_factor,
        &Result::aligned_heading_ros_mps};

    for (auto member : members) {
        EXPECT_EQ(
            std::memcmp(&(a.*member), &(b.*member), sizeof(Real)),
            0);
    }
}

#ifdef AMREX_USE_GPU
struct MoistureDeviceProbe
{
    FireFuelMoisture value{};
    Real absent_output{};
    int flags{};
};

MoistureDeviceProbe
run_device_moisture_probe(FireFuelMoisture moisture)
{
    amrex::Gpu::DeviceScalar<MoistureDeviceProbe> device;
    auto* output = device.dataPtr();
    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    const Real inf = std::numeric_limits<Real>::infinity();

    amrex::ParallelFor(
        1,
        [=] AMREX_GPU_DEVICE (int) noexcept
        {
            MoistureDeviceProbe result;
            result.value = moisture;
            result.absent_output = Real(9);

            const bool missing =
                !result.value.try_get(
                    Component::LiveWoody,
                    result.absent_output);
            const bool added =
                result.value.try_set(
                    Component::LiveWoody,
                    Real(1.5));
            const bool negative =
                !result.value.try_set(
                    Component::Dead1h,
                    Real(-1));
            const bool invalid_id =
                !result.value.try_set(
                    static_cast<Component>(32),
                    Real(0));
            const bool invalid_nan =
                !result.value.try_set(
                    Component::Dead1h,
                    nan);
            const bool invalid_inf =
                !result.value.try_set(
                    Component::Dead1h,
                    inf);

            result.flags =
                missing
                && added
                && negative
                && invalid_id
                && invalid_nan
                && invalid_inf;

            *output = result;
        });

    return device.dataValue();
}

struct UniformMaterialDeviceProbe
{
    Real moisture{};
    Real load{};
    int found{};
};

UniformMaterialDeviceProbe
run_device_uniform_material_probe(FireFuelProperties material)
{
    amrex::Gpu::DeviceScalar<UniformMaterialDeviceProbe> device;
    auto* output = device.dataPtr();

    amrex::ParallelFor(
        1,
        [=] AMREX_GPU_DEVICE (int) noexcept
        {
            UniformMaterialDeviceProbe result;
            result.moisture = Real(-1);
            result.found =
                material.moisture.try_get(
                    Component::Dead1h,
                    result.moisture)
                    ? 1
                    : 0;
            result.load =
                material.single_dead_class.dead_1h_load_kg_m2;
            *output = result;
        });

    return device.dataValue();
}
#endif

} // namespace

TEST(FireFuelMoisture, MissingIsNotDry)
{
    FireFuelMoisture moisture;

    for (auto component : components) {
        Real value = Real(7);
        EXPECT_FALSE(moisture.has(component));
        EXPECT_FALSE(moisture.try_get(component, value));
        EXPECT_EQ(value, Real(7));
        EXPECT_THROW(
            (void)moisture.get(component),
            std::invalid_argument);
    }

    moisture.set(Component::Dead1h, Real(0));
    EXPECT_TRUE(moisture.has(Component::Dead1h));
    EXPECT_EQ(moisture.get(Component::Dead1h), Real(0));
}

TEST(FireFuelMoisture, DeadScalarDoesNotInventLiveMoisture)
{
    const auto moisture =
        FireFuelMoisture::uniform_dead(Real(0.08));

    for (auto component : {
             Component::Dead1h,
             Component::Dead10h,
             Component::Dead100h}) {
        EXPECT_TRUE(moisture.has(component));
        EXPECT_EQ(moisture.get(component), Real(0.08));
    }

    EXPECT_FALSE(moisture.has(Component::LiveHerbaceous));
    EXPECT_FALSE(moisture.has(Component::LiveWoody));
}

TEST(FireFuelMoisture, ComponentsAreIndependent)
{
    FireFuelMoisture moisture;

    for (int i = 0;
         i < FireFuelMoisture::component_count;
         ++i) {
        moisture.set(
            components[i],
            Real(i + 1) / Real(10));
    }

    moisture.set(Component::LiveWoody, Real(1.5));

    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(
            moisture.get(components[i]),
            Real(i + 1) / Real(10));
    }

    EXPECT_EQ(
        moisture.get(Component::LiveWoody),
        Real(1.5));
}

TEST(FireFuelMoisture, AcceptsLiveMoistureAboveOne)
{
    FireFuelMoisture moisture;
    moisture.set(Component::LiveHerbaceous, Real(1.2));
    moisture.set(Component::LiveWoody, Real(2.5));

    EXPECT_EQ(
        moisture.get(Component::LiveHerbaceous),
        Real(1.2));
    EXPECT_EQ(
        moisture.get(Component::LiveWoody),
        Real(2.5));
    EXPECT_FALSE(moisture.has(Component::Dead1h));
}

TEST(FireFuelMoisture, RejectsInvalidValuesWithoutMutation)
{
    const Real invalid[] = {
        Real(-0.01),
        std::numeric_limits<Real>::infinity(),
        -std::numeric_limits<Real>::infinity(),
        std::numeric_limits<Real>::quiet_NaN()};

    for (auto component : components) {
        FireFuelMoisture moisture;

        for (Real value : invalid) {
            EXPECT_FALSE(moisture.try_set(component, value));
            EXPECT_FALSE(moisture.has(component));
        }

        moisture.set(component, Real(0.2));

        for (Real value : invalid) {
            EXPECT_FALSE(moisture.try_set(component, value));
            EXPECT_THROW(
                moisture.set(component, value),
                std::invalid_argument);
            EXPECT_EQ(
                moisture.get(component),
                Real(0.2));
        }
    }

    EXPECT_THROW(
        (void)FireFuelMoisture::uniform_dead(Real(-1)),
        std::invalid_argument);
}

TEST(FireFuelMoisture, RejectsInvalidComponentsWithoutMutation)
{
    auto moisture =
        FireFuelMoisture::uniform_dead(Real(0.08));

    for (int index : {
             -1,
             5,
             32,
             std::numeric_limits<int>::max()}) {
        const auto component =
            static_cast<Component>(index);
        Real output = Real(9);

        EXPECT_FALSE(moisture.has(component));
        EXPECT_FALSE(moisture.try_get(component, output));
        EXPECT_EQ(output, Real(9));
        EXPECT_FALSE(
            moisture.try_set(
                component,
                Real(0.2)));
        EXPECT_THROW(
            (void)moisture.get(component),
            std::invalid_argument);
    }

    EXPECT_EQ(
        moisture.get(Component::Dead1h),
        Real(0.08));
    EXPECT_FALSE(moisture.has(Component::LiveWoody));
}

TEST(FireFuelMoisture, PreservesSignedZeroAndCopy)
{
    const auto original =
        FireFuelMoisture::uniform_dead(-Real(0));
    auto copy = original;
    copy.set(Component::LiveWoody, Real(1.4));

    EXPECT_TRUE(
        std::signbit(
            original.get(Component::Dead1h)));
    EXPECT_TRUE(
        std::signbit(
            copy.get(Component::Dead100h)));
    EXPECT_FALSE(
        original.has(Component::LiveWoody));
    EXPECT_EQ(
        copy.get(Component::LiveWoody),
        Real(1.4));

    static_assert(
        std::is_trivially_copyable_v<FireFuelMoisture>);
}

TEST(FireRasterLookup, TranslatedRectangularCellsAndCorners)
{
    const auto g = geometry();

    for (std::size_t j = 0; j < g.ny; ++j) {
        for (std::size_t i = 0; i < g.nx; ++i) {
            const auto bounds =
                fire_cartesian_raster_cell_bounds(
                    g,
                    i,
                    j);
            const auto corner =
                fire_cartesian_raster_locate_cell(
                    g,
                    bounds.xlo_m,
                    bounds.ylo_m);
            const auto center =
                fire_cartesian_raster_locate_cell(
                    g,
                    (bounds.xlo_m + bounds.xhi_m)
                        / Real(2),
                    (bounds.ylo_m + bounds.yhi_m)
                        / Real(2));

            EXPECT_EQ(corner.i, i);
            EXPECT_EQ(corner.j, j);
            EXPECT_EQ(center.i, i);
            EXPECT_EQ(center.j, j);
        }
    }
}

TEST(FireRasterLookup, NextafterInternalRepresentedFaces)
{
    const FireCartesianRasterGeometry2D g{
        31,
        19,
        Real(10.3),
        Real(-4.7),
        Real(0.1),
        Real(0.3)};

    (void)ERFFire::detail::
        validate_fire_cartesian_raster_geometry(g);

    const Real inf =
        std::numeric_limits<Real>::infinity();

    for (std::size_t i = 1; i < g.nx; ++i) {
        const Real face =
            fire_cartesian_raster_cell_bounds(
                g,
                i,
                0).xlo_m;

        EXPECT_EQ(
            fire_cartesian_raster_locate_cell(
                g,
                std::nextafter(face, -inf),
                g.ylo_m).i,
            i - 1);
        EXPECT_EQ(
            fire_cartesian_raster_locate_cell(
                g,
                face,
                g.ylo_m).i,
            i);
        EXPECT_EQ(
            fire_cartesian_raster_locate_cell(
                g,
                std::nextafter(face, inf),
                g.ylo_m).i,
            i);
    }

    for (std::size_t j = 1; j < g.ny; ++j) {
        const Real face =
            fire_cartesian_raster_cell_bounds(
                g,
                0,
                j).ylo_m;

        EXPECT_EQ(
            fire_cartesian_raster_locate_cell(
                g,
                g.xlo_m,
                std::nextafter(face, -inf)).j,
            j - 1);
        EXPECT_EQ(
            fire_cartesian_raster_locate_cell(
                g,
                g.xlo_m,
                face).j,
            j);
        EXPECT_EQ(
            fire_cartesian_raster_locate_cell(
                g,
                g.xlo_m,
                std::nextafter(face, inf)).j,
            j);
    }
}

TEST(FireRasterLookup, UpperBoundaryIsLastCellWithoutExteriorClamping)
{
    const auto g = geometry();
    const auto last =
        fire_cartesian_raster_cell_bounds(
            g,
            g.nx - 1,
            g.ny - 1);
    const auto location =
        fire_cartesian_raster_locate_cell(
            g,
            last.xhi_m,
            last.yhi_m);

    EXPECT_EQ(location.i, g.nx - 1);
    EXPECT_EQ(location.j, g.ny - 1);

    const Real inf =
        std::numeric_limits<Real>::infinity();

    EXPECT_THROW(
        (void)fire_cartesian_raster_locate_cell(
            g,
            std::nextafter(g.xlo_m, -inf),
            g.ylo_m),
        std::out_of_range);
    EXPECT_THROW(
        (void)fire_cartesian_raster_locate_cell(
            g,
            g.xlo_m,
            std::nextafter(g.ylo_m, -inf)),
        std::out_of_range);
    EXPECT_THROW(
        (void)fire_cartesian_raster_locate_cell(
            g,
            std::nextafter(last.xhi_m, inf),
            g.ylo_m),
        std::out_of_range);
    EXPECT_THROW(
        (void)fire_cartesian_raster_locate_cell(
            g,
            g.xlo_m,
            std::nextafter(last.yhi_m, inf)),
        std::out_of_range);
}

TEST(FireRasterLookup, RejectsNonfinitePoints)
{
    const auto g = geometry();

    for (Real invalid : {
             std::numeric_limits<Real>::infinity(),
             -std::numeric_limits<Real>::infinity(),
             std::numeric_limits<Real>::quiet_NaN()}) {
        EXPECT_THROW(
            (void)fire_cartesian_raster_locate_cell(
                g,
                invalid,
                g.ylo_m),
            std::invalid_argument);
        EXPECT_THROW(
            (void)fire_cartesian_raster_locate_cell(
                g,
                g.xlo_m,
                invalid),
            std::invalid_argument);
    }

    EXPECT_THROW(
        (void)fire_cartesian_raster_locate_cell(
            {},
            Real(0),
            Real(0)),
        std::invalid_argument);
}

TEST(FireRasterLookup, SingletonGrid)
{
    const FireCartesianRasterGeometry2D g{
        1,
        1,
        Real(-1),
        Real(2),
        Real(3),
        Real(5)};

    for (Real x : {Real(-1), Real(0), Real(2)}) {
        for (Real y : {Real(2), Real(4), Real(7)}) {
            const auto location =
                fire_cartesian_raster_locate_cell(
                    g,
                    x,
                    y);
            EXPECT_EQ(location.i, 0u);
            EXPECT_EQ(location.j, 0u);
        }
    }
}

TEST(FireFuelField, AllCellsShareExactParameters)
{
    const auto g = geometry();
    const auto fuel =
        ERFFire::make_fm1_fuel_parameters();
    const FireFuelField field(
        g,
        fuel,
        Real(0.08));

    for (std::size_t j = 0; j < g.ny; ++j) {
        for (std::size_t i = 0; i < g.nx; ++i) {
            const auto bounds =
                fire_cartesian_raster_cell_bounds(
                    g,
                    i,
                    j);
            const auto& value =
                field.sample(
                    bounds.xlo_m,
                    bounds.ylo_m);

            EXPECT_EQ(&value, &field.cell(i, j));
            EXPECT_EQ(
                &value,
                &field.uniform_properties());
            expect_fuel_bits_equal(
                value.single_dead_class,
                fuel);
            EXPECT_EQ(
                value.moisture.get(Component::Dead1h),
                Real(0.08));
            EXPECT_FALSE(
                value.moisture.has(
                    Component::LiveHerbaceous));
        }
    }
}

TEST(FireFuelField, PreservesSuppliedNonFM1Parameters)
{
    auto fuel =
        ERFFire::make_fm1_fuel_parameters();
    fuel.dead_1h_load_kg_m2 *= Real(2);
    fuel.dead_1h_sav_m_inv *= Real(0.75);
    fuel.fuel_bed_depth_m *= Real(1.5);

    const FireFuelField field(
        geometry(),
        fuel,
        Real(0.09));

    expect_fuel_bits_equal(
        field.uniform_properties().single_dead_class,
        fuel);
}

TEST(FireFuelField, OwnsCopiesAndHasNoPerCellStorage)
{
    auto fuel =
        ERFFire::make_fm1_fuel_parameters();
    auto g = geometry();

    const FireFuelField field(
        g,
        fuel,
        Real(0.08));
    const auto copy = field;

    fuel.dead_1h_load_kg_m2 *= Real(2);
    g.nx = 9;

    EXPECT_EQ(field.geometry().nx, 4u);
    expect_fuel_bits_equal(
        field.uniform_properties().single_dead_class,
        ERFFire::make_fm1_fuel_parameters());
    expect_fuel_bits_equal(
        copy.uniform_properties().single_dead_class,
        field.uniform_properties().single_dead_class);

    static_assert(
        std::is_trivially_copyable_v<FireFuelField>);

    const FireFuelField large(
        {
            1000000,
            1000000,
            Real(0),
            Real(0),
            Real(1),
            Real(1)},
        ERFFire::make_fm1_fuel_parameters(),
        Real(0.08));

    EXPECT_EQ(
        &large.cell(999999, 999999),
        &large.cell(0, 0));
}

TEST(FireFuelField, RejectsInvalidGeometryAndMaterial)
{
    const auto fuel =
        ERFFire::make_fm1_fuel_parameters();

    EXPECT_THROW(
        (void)FireFuelField(
            {},
            fuel,
            Real(0.08)),
        std::invalid_argument);

    auto g = geometry();
    g.dx_m = Real(0);
    EXPECT_THROW(
        (void)FireFuelField(
            g,
            fuel,
            Real(0.08)),
        std::invalid_argument);

    auto invalid_fuel = fuel;
    invalid_fuel.dead_1h_load_kg_m2 = Real(0);
    EXPECT_THROW(
        (void)FireFuelField(
            geometry(),
            invalid_fuel,
            Real(0.08)),
        std::invalid_argument);

    EXPECT_THROW(
        (void)FireFuelField(
            geometry(),
            fuel,
            Real(-0.01)),
        std::invalid_argument);
}

TEST(FireFuelField, RejectsOutsideIndicesAndPoints)
{
    const FireFuelField field(
        geometry(),
        ERFFire::make_fm1_fuel_parameters(),
        Real(0.08));

    EXPECT_THROW(
        (void)field.cell(4, 0),
        std::out_of_range);
    EXPECT_THROW(
        (void)field.cell(0, 3),
        std::out_of_range);
    EXPECT_THROW(
        (void)field.sample(Real(99), Real(-50)),
        std::out_of_range);
}

TEST(FireFuelField, LegacyRothermelResultsAreBitwiseEqual)
{
    const auto fuel =
        ERFFire::make_fm1_fuel_parameters();

    for (Real moisture : {
             Real(0),
             Real(0.08),
             Real(0.12),
             Real(0.2)}) {
        const FireFuelField field(
            geometry(),
            fuel,
            moisture);
        const auto& material =
            field.uniform_properties();

        for (Real wind : {
                 Real(0),
                 Real(1),
                 Real(3)}) {
            for (Real slope : {
                     Real(0),
                     Real(0.2)}) {
                const auto legacy =
                    ERFFire::evaluate_rothermel(
                        fuel,
                        {moisture, wind, slope});
                const auto via_field =
                    ERFFire::evaluate_rothermel(
                        material.single_dead_class,
                        {
                            material.moisture.get(
                                Component::Dead1h),
                            wind,
                            slope});

                expect_behavior_bits_equal(
                    legacy,
                    via_field);
            }
        }
    }
}

TEST(FireFuelField, ExtinguishedSpreadDoesNotEraseFuel)
{
    const FireFuelField field(
        geometry(),
        ERFFire::make_fm1_fuel_parameters(),
        Real(0.2));
    const auto& material =
        field.uniform_properties();

    const auto behavior =
        ERFFire::evaluate_rothermel(
            material.single_dead_class,
            {
                material.moisture.get(Component::Dead1h),
                Real(1),
                Real(0.2)});

    EXPECT_EQ(
        behavior.aligned_heading_ros_mps,
        Real(0));
    EXPECT_GT(
        material.single_dead_class.dead_1h_load_kg_m2,
        Real(0));
}

TEST(FireFuelField, LegacyConfigAdapterPreservesIndependentGeometryAndParameters)
{
    const FireCartesianRasterGeometry2D geometries[] = {
        geometry(),
        {8, 6, Real(100), Real(-50), Real(1), Real(2)},
        {2, 1, Real(100), Real(-50), Real(4), Real(12)}};

    for (const auto& g : geometries) {
        ERFFire::ERFFireSpreadConfig config{};
        config.raster_geometry = g;
        config.fuel =
            ERFFire::make_fm1_fuel_parameters();
        config.fuel.dead_1h_load_kg_m2 *= Real(1.25);
        config.dead_fuel_moisture_fraction = Real(0.07);

        const auto field =
            ERFFire::make_erf_fire_uniform_fuel_field(
                config);

        EXPECT_EQ(field.geometry().nx, g.nx);
        EXPECT_EQ(field.geometry().ny, g.ny);
        EXPECT_EQ(field.geometry().xlo_m, g.xlo_m);
        EXPECT_EQ(field.geometry().ylo_m, g.ylo_m);
        EXPECT_EQ(field.geometry().dx_m, g.dx_m);
        EXPECT_EQ(field.geometry().dy_m, g.dy_m);

        expect_fuel_bits_equal(
            field.uniform_properties().single_dead_class,
            config.fuel);
        EXPECT_EQ(
            field.uniform_properties().moisture.get(
                Component::Dead1h),
            config.dead_fuel_moisture_fraction);
        EXPECT_FALSE(
            field.uniform_properties().moisture.has(
                Component::LiveWoody));
    }
}

#ifdef AMREX_USE_GPU
TEST(FireFuelMoisture, DeviceSafeApiMatchesHost)
{
    auto moisture =
        FireFuelMoisture::uniform_dead(Real(0.08));
    moisture.set(
        Component::LiveHerbaceous,
        Real(1.2));

    const MoistureDeviceProbe actual =
        run_device_moisture_probe(moisture);

    EXPECT_EQ(actual.flags, 1);
    EXPECT_EQ(actual.absent_output, Real(9));
    EXPECT_EQ(
        actual.value.get(Component::Dead1h),
        Real(0.08));
    EXPECT_EQ(
        actual.value.get(Component::LiveHerbaceous),
        Real(1.2));
    EXPECT_EQ(
        actual.value.get(Component::LiveWoody),
        Real(1.5));
}

TEST(FireFuelField, UniformMaterialCanBeCapturedByValueOnDevice)
{
    const FireFuelField field(
        geometry(),
        ERFFire::make_fm1_fuel_parameters(),
        Real(0.08));
    const auto material =
        field.uniform_properties();

    const UniformMaterialDeviceProbe actual =
        run_device_uniform_material_probe(material);

    EXPECT_EQ(actual.found, 1);
    EXPECT_EQ(actual.moisture, Real(0.08));
    EXPECT_EQ(
        actual.load,
        material.single_dead_class.dead_1h_load_kg_m2);
}
#endif
