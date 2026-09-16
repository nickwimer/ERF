#include <gtest/gtest.h>

#include <ERF_FireRuntimeInit.H>
#include <ERF_FireRuntimeOptions.H>
#include <ERF_FireSpreadRuntime.H>
#include <ERF_FireSpreadOutput.H>
#include <ERF_FireFrontPropagator.H>
#include <ERF_FireWindAdjustment.H>
#include <ERF_RichardsDirectionalSpread.H>
#include <ERF_RothermelModel.H>
#include <ERF_VectorPerimeterPropagator.H>

#include <AMReX_Array.H>
#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_RealBox.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

using amrex::Array;
using amrex::Box;
using amrex::Geometry;
using amrex::IntVect;
using amrex::Real;
using amrex::RealBox;
using ERFFire::ERFFireCouplingMode;
using ERFFire::ERFFireRuntimeOptions;
using ERFFire::make_erf_fire_spread_config;

Geometry
make_geometry()
{
    const Box domain{
        IntVect(0, 0, 0),
        IntVect(3, 2, 1)};

    const RealBox real_box{
        Real(100.0),
        Real(50.0),
        Real(0.0),
        Real(108.0),
        Real(62.0),
        Real(20.0)};

    const Array<int, AMREX_SPACEDIM> periodic{{0, 0, 0}};

    return Geometry{
        domain,
        real_box,
        amrex::CoordSys::cartesian,
        periodic};
}

ERFFireRuntimeOptions
make_options()
{
    ERFFireRuntimeOptions options;
    options.enabled = true;
    options.dead_fuel_moisture_fraction = Real(0.08);
    options.remesh_min_edge_length_m = Real(0.05);
    options.remesh_max_edge_length_m = Real(0.25);
    options.remesh_max_chord_error_m = Real(0.002);
    return options;
}

TEST(FireRuntimeInit, DefaultFireGridMatchesLevel0)
{
    const Geometry geometry = make_geometry();
    const auto config =
        make_erf_fire_spread_config(
            make_options(),
            geometry);

    EXPECT_EQ(config.raster_geometry.nx, 4u);
    EXPECT_EQ(config.raster_geometry.ny, 3u);
    EXPECT_EQ(config.raster_geometry.xlo_m, Real(100.0));
    EXPECT_EQ(config.raster_geometry.ylo_m, Real(50.0));
    EXPECT_EQ(config.raster_geometry.dx_m, Real(2.0));
    EXPECT_EQ(config.raster_geometry.dy_m, Real(4.0));
}

TEST(FireRuntimeInit, FinerFireGridUsesSamePhysicalDomain)
{
    const Geometry geometry = make_geometry();
    auto options = make_options();
    options.n_cell_x = 8;
    options.n_cell_y = 6;

    const auto config =
        make_erf_fire_spread_config(
            options,
            geometry);

    EXPECT_EQ(config.raster_geometry.nx, 8u);
    EXPECT_EQ(config.raster_geometry.ny, 6u);
    EXPECT_EQ(config.raster_geometry.xlo_m, Real(100.0));
    EXPECT_EQ(config.raster_geometry.ylo_m, Real(50.0));
    EXPECT_EQ(config.raster_geometry.dx_m, Real(1.0));
    EXPECT_EQ(config.raster_geometry.dy_m, Real(2.0));

    EXPECT_EQ(
        config.raster_geometry.xlo_m
            + Real(config.raster_geometry.nx)
                * config.raster_geometry.dx_m,
        Real(108.0));
    EXPECT_EQ(
        config.raster_geometry.ylo_m
            + Real(config.raster_geometry.ny)
                * config.raster_geometry.dy_m,
        Real(62.0));
}

TEST(FireRuntimeInit, CoarserFireGridUsesSamePhysicalDomain)
{
    const Geometry geometry = make_geometry();
    auto options = make_options();
    options.n_cell_x = 2;
    options.n_cell_y = 1;

    const auto config =
        make_erf_fire_spread_config(
            options,
            geometry);

    EXPECT_EQ(config.raster_geometry.nx, 2u);
    EXPECT_EQ(config.raster_geometry.ny, 1u);
    EXPECT_EQ(config.raster_geometry.dx_m, Real(4.0));
    EXPECT_EQ(config.raster_geometry.dy_m, Real(12.0));
}

TEST(FireRuntimeInit, RejectsNoncommensurateFireGrid)
{
    const Geometry geometry = make_geometry();
    auto options = make_options();
    options.n_cell_x = 6;
    options.n_cell_y = 6;

    EXPECT_THROW(
        (void)make_erf_fire_spread_config(
            options,
            geometry),
        std::invalid_argument);
}

TEST(FireRuntimeInit, RejectsIncompleteFireGridCounts)
{
    const Geometry geometry = make_geometry();
    auto options = make_options();
    options.n_cell_x = 8;
    options.n_cell_y = 0;

    EXPECT_THROW(
        (void)make_erf_fire_spread_config(
            options,
            geometry),
        std::invalid_argument);
}

TEST(FireRuntimeInit, TwoWayIndependentGridAllowsUniformlyNestedResolutions)
{
    const Geometry geometry = make_geometry();
    auto options = make_options();
    options.coupling_mode =
        ERFFireCouplingMode::TwoWay;

    options.n_cell_x = 8;
    options.n_cell_y = 6;
    EXPECT_NO_THROW(
        (void)make_erf_fire_spread_config(
            options,
            geometry));

    options.n_cell_x = 2;
    options.n_cell_y = 1;
    EXPECT_NO_THROW(
        (void)make_erf_fire_spread_config(
            options,
            geometry));

    options.n_cell_x = 8;
    options.n_cell_y = 1;
    EXPECT_THROW(
        (void)make_erf_fire_spread_config(
            options,
            geometry),
        std::invalid_argument);
}

}

namespace
{

using Runtime = ERFFire::ERFFireSpreadRuntime;
using RuntimeState = ERFFire::ERFFireSpreadRuntimeState;
using SpreadConfig = ERFFire::ERFFireSpreadConfig;
using ERFFire::FireVec2;
using ERFFire::FirePerimeter;
using ERFFire::FireFront;
using ERFFire::FireFrontRole;
using ERFFire::FireFrontComponent;
using ERFFire::FireFlatEnvironmentSampler;
using ERFFire::FireTerrainSurface;
using ERFFire::FireCartesianRasterGeometry2D;

SpreadConfig
fuel_runtime_config(int resolution = 0, int material_case = 0)
{
    auto options = make_options();
    if (resolution < 0) {
        options.n_cell_x = 2;
        options.n_cell_y = 1;
    } else if (resolution > 0) {
        options.n_cell_x = 8;
        options.n_cell_y = 6;
    }
    options.dead_fuel_moisture_fraction =
        material_case == 2 ? Real(0.2) : Real(0.08);
    auto config = make_erf_fire_spread_config(options, make_geometry());
    if (material_case == 1) {
        // A supplied single-class bed must not be silently reconstructed as FM1.
        config.fuel.dead_1h_load_kg_m2 *= Real(1.25);
        config.fuel.dead_1h_sav_m_inv *= Real(0.75);
        config.fuel.fuel_bed_depth_m *= Real(1.5);
        config.dead_fuel_moisture_fraction = Real(0.07);
        config.combustion_parameters.dry_fuel_load_kg_m2 =
            config.fuel.dead_1h_load_kg_m2;
        config.combustion_parameters.fuel_moisture_fraction =
            config.dead_fuel_moisture_fraction;
    }
    return config;
}

FirePerimeter
fuel_runtime_ignition()
{
    constexpr int count = 32;
    const Real pi = Real(3.141592653589793238462643383279502884L);
    std::vector<FireVec2> vertices;
    for (int i = 0; i < count; ++i) {
        const Real angle = Real(2) * pi * Real(i) / Real(count);
        vertices.push_back({Real(104) + std::cos(angle),
                            Real(56) + std::sin(angle)});
    }
    return FirePerimeter(std::move(vertices));
}

FireFlatEnvironmentSampler
fuel_runtime_environment(const FireCartesianRasterGeometry2D& g, FireVec2 wind)
{
    ERFFire::FireFlatEnvironmentLayout2D layout(
        g.xlo_m, g.ylo_m, g.dx_m, g.dy_m, g.nx, g.ny);
    std::vector<Real> u(layout.u_storage_size(), wind.x);
    std::vector<Real> v(layout.v_storage_size(), wind.y);
    return FireFlatEnvironmentSampler(
        std::move(layout), Real(6.096), std::move(u), std::move(v));
}

FireTerrainSurface
fuel_runtime_terrain(const FireCartesianRasterGeometry2D& g)
{
    std::vector<Real> heights;
    for (std::size_t j = 0; j <= g.ny; ++j) {
        for (std::size_t i = 0; i <= g.nx; ++i) {
            heights.push_back(Real(0.125) * Real(i) * g.dx_m
                              + Real(0.0625) * Real(j) * g.dy_m);
        }
    }
    return FireTerrainSurface(g, std::move(heights));
}

ERFFire::FireEnvironmentBatchFunction
fuel_runtime_batch(const FireFlatEnvironmentSampler& environment)
{
    return [&environment](const std::vector<FireVec2>& positions) {
        std::vector<ERFFire::FireEnvironmentSample> samples;
        samples.reserve(positions.size());
        for (const auto& position : positions) {
            samples.push_back(environment.sample(position.x, position.y));
        }
        return samples;
    };
}

std::string
fuel_runtime_state_bytes(const RuntimeState& state)
{
    auto options = make_options();
    options.reference_height_agl_m = Real(0.5);
    std::ostringstream stream;
    // The existing writer uses max_digits10, including all persistent raster
    // values and supplied configuration. Compare complete payloads, not totals.
    ERFFire::write_erf_fire_checkpoint_state(state, options, stream);
    return stream.str();
}

void
expect_runtime_material(const Runtime& runtime, const SpreadConfig& config)
{
    const auto& field = runtime.fuel_field();
    const auto& material = field.uniform_properties();
    const auto& g = field.geometry();
    const auto& expected = config.raster_geometry;
    EXPECT_EQ(g.nx, expected.nx);
    EXPECT_EQ(g.ny, expected.ny);
    EXPECT_EQ(g.xlo_m, expected.xlo_m);
    EXPECT_EQ(g.ylo_m, expected.ylo_m);
    EXPECT_EQ(g.dx_m, expected.dx_m);
    EXPECT_EQ(g.dy_m, expected.dy_m);
    EXPECT_EQ(&field.cell(0, 0), &field.cell(g.nx - 1, g.ny - 1));

    using Fuel = ERFFire::RothermelFuelParameters;
    const Real Fuel::* members[] = {
        &Fuel::dead_1h_load_kg_m2, &Fuel::dead_1h_sav_m_inv,
        &Fuel::fuel_bed_depth_m, &Fuel::dead_heat_content_j_kg,
        &Fuel::particle_density_kg_m3, &Fuel::total_mineral_fraction,
        &Fuel::effective_mineral_fraction, &Fuel::dead_moisture_of_extinction};
    for (auto member : members) {
        EXPECT_EQ(std::memcmp(&(material.single_dead_class.*member),
                              &(config.fuel.*member), sizeof(Real)), 0);
    }
    using Component = ERFFire::FireFuelMoistureClass;
    for (auto component : {Component::Dead1h, Component::Dead10h, Component::Dead100h}) {
        const Real moisture = material.moisture.get(component);
        EXPECT_EQ(std::memcmp(&moisture, &config.dead_fuel_moisture_fraction,
                              sizeof(Real)), 0);
    }
    EXPECT_FALSE(material.moisture.has(Component::LiveHerbaceous));
    EXPECT_FALSE(material.moisture.has(Component::LiveWoody));
}

// Frozen legacy normal-speed arithmetic from the pre-field runtime. This
// reference deliberately reads the old config, never FireFuelField. It uses
// the unchanged low-level geometry/history algorithms but does not call a
// production Runtime::advance_* entry point, so constructor equivalence alone
// cannot make this comparison pass.
RuntimeState
legacy_uniform_step(
    const Runtime& before,
    const FireFlatEnvironmentSampler& environment,
    const FireTerrainSurface* terrain,
    bool use_waf,
    bool batched,
    Real dt)
{
    const auto& config = before.config();
    const Real start = before.current_time_s();
    const Real end = start + dt;
    const auto normal_speed = [&](const FireVec2& position,
                                  const FireVec2& normal, Real) {
        const auto reference = environment.sample(position.x, position.y).horizontal_wind_mps;
        const auto wind = use_waf
            ? ERFFire::fire_midflame_wind_from_20ft_reference(reference, Real(0.4))
            : reference;
        const auto gradient = terrain != nullptr
            ? terrain->terrain_gradient_m_per_m(position.x, position.y) : FireVec2{};
        const Real speed = ERFFire::norm(wind);
        const Real slope = ERFFire::norm(gradient);
        const Real wind_inactive = speed > Real(0) ? Real(0) : Real(1);
        const Real slope_inactive = slope > Real(0) ? Real(0) : Real(1);
        const auto wind_direction = FireVec2{wind.x + wind_inactive, wind.y}
            / (speed + wind_inactive);
        const auto upslope = FireVec2{gradient.x + slope_inactive, gradient.y}
            / (slope + slope_inactive);
        const auto behavior = ERFFire::evaluate_rothermel(config.fuel,
            {config.dead_fuel_moisture_fraction, speed, slope});
        const auto spread = ERFFire::make_richards_directional_spread(
            behavior, wind_direction, upslope);
        return ERFFire::richards_normal_speed_mps(spread.ellipse, normal);
    };
    const auto normal_speeds = [&](const std::vector<FireVec2>& positions,
                                   const std::vector<FireVec2>& normals, Real time) {
        std::vector<Real> speeds;
        for (std::size_t i = 0; i < positions.size(); ++i) {
            speeds.push_back(normal_speed(positions[i], normals[i], time));
        }
        return speeds;
    };

    FireFront advanced = [&]() {
        if (batched) {
            auto result = ERFFire::advance_front_rk2_batched_until_topology_event(
                before.front(), start, dt, normal_speeds);
            if (result.topology_event.has_value() || !result.completed_front.has_value()
                || result.advanced_dt_s != dt) {
                throw std::logic_error("uniform equivalence fixture unexpectedly changed topology");
            }
            return std::move(*result.completed_front);
        }
        return FireFront(std::vector<FireFrontComponent>{{FireFrontRole::Outer,
            ERFFire::advance_perimeter_rk2(before.perimeter(), start, dt, normal_speed)}});
    }();
    const auto& perimeter = advanced.components().front().perimeter;
    auto burned = before.burned_fraction_raster();
    auto arrival = before.first_arrival_raster();
    auto combustion = before.combustion_raster();
    if (batched) {
        (void)arrival.update_from_front_linear_sweep(before.front(), advanced,
            start, end, std::min(config.arrival_time_tolerance_s, dt),
            config.combustion_options.temporal_substeps);
        (void)burned.update_from_front_linear_sweep(before.front(), advanced,
            config.combustion_options.temporal_substeps);
        (void)combustion.advance_from_front_linear_sweep(before.front(), advanced,
            before.burned_fraction_raster(), burned, dt);
    } else {
        (void)arrival.update_from_sweep(before.perimeter(), perimeter,
            start, end, config.arrival_time_tolerance_s);
        (void)burned.update_from_linear_sweep(before.perimeter(), perimeter,
            config.combustion_options.temporal_substeps);
        (void)combustion.advance_from_linear_sweep(before.perimeter(), perimeter,
            before.burned_fraction_raster(), burned, dt);
    }
    const std::vector<FireVec2> vertices = batched
        ? ERFFire::remesh_front(advanced, config.remesh_options)
              .front.components().front().perimeter.vertices_m()
        : ERFFire::remesh_perimeter(perimeter, config.remesh_options).perimeter.vertices_m();
    RuntimeState expected;
    expected.config = config;
    expected.current_time_s = end;
    expected.perimeter_vertices_m = vertices;
    expected.front_components.push_back({FireFrontRole::Outer, vertices});
    expected.burned_fraction = burned.snapshot_state();
    expected.first_arrival = arrival.snapshot_state();
    expected.combustion = combustion.snapshot_state();
    return expected;
}

void
advance_uniform_fixture(Runtime& runtime, const FireFlatEnvironmentSampler& environment,
                        const FireTerrainSurface* terrain, bool use_waf, bool batched, Real dt)
{
    const auto batch = fuel_runtime_batch(environment);
    if (batched) {
        if (use_waf) {
            if (terrain != nullptr) {
                (void)runtime.advance_explicit_waf_20ft_batched(batch, *terrain, Real(0.4), dt);
            } else {
                (void)runtime.advance_explicit_waf_20ft_batched(batch, Real(0.4), dt);
            }
        } else if (terrain != nullptr) {
            (void)runtime.advance_direct_reference_wind_batched(batch, *terrain, dt);
        } else {
            (void)runtime.advance_direct_reference_wind_batched(batch, dt);
        }
    } else if (use_waf) {
        if (terrain != nullptr) {
            (void)runtime.advance_explicit_waf_20ft(environment, *terrain, Real(0.4), dt);
        } else {
            (void)runtime.advance_explicit_waf_20ft(environment, Real(0.4), dt);
        }
    } else if (terrain != nullptr) {
        (void)runtime.advance_direct_reference_wind(environment, *terrain, dt);
    } else {
        (void)runtime.advance_direct_reference_wind(environment, dt);
    }
}

void
check_uniform_runtime_equivalence(bool terrain_enabled, bool use_waf)
{
    if (amrex::ParallelDescriptor::NProcs() != 1) {
        GTEST_SKIP() << "Exact canonical-state oracle requires one MPI rank";
    }
    const auto atmosphere_geometry = fuel_runtime_config().raster_geometry;
    for (int resolution : {-1, 0, 1}) {
        for (int material_case : {0, 1, 2}) {
            for (bool batched : {false, true}) {
                SCOPED_TRACE(::testing::Message() << "resolution=" << resolution
                    << " material_case=" << material_case << " batched=" << batched);
                const auto config = fuel_runtime_config(resolution, material_case);
                Runtime runtime(fuel_runtime_ignition(), Real(0), config);
                const FireVec2 wind = material_case == 2 ? FireVec2{}
                    : (material_case == 1 ? FireVec2{Real(-0.5), Real(0.25)}
                                          : FireVec2{Real(1), Real(0)});
                // The legacy scalar terrain overload requires matching sampler
                // geometry. The production batched path also tests an independent
                // Fire grid against an atmosphere-resolution wind sampler.
                const auto environment = fuel_runtime_environment(
                    terrain_enabled && !batched ? config.raster_geometry : atmosphere_geometry,
                    wind);
                const auto surface = fuel_runtime_terrain(config.raster_geometry);
                const auto* terrain = terrain_enabled ? &surface : nullptr;
                for (int step = 0; step < 2; ++step) {
                    SCOPED_TRACE(step);
                    const auto expected = legacy_uniform_step(
                        runtime, environment, terrain, use_waf, batched, Real(0.25));
                    advance_uniform_fixture(runtime, environment, terrain, use_waf, batched, Real(0.25));
                    EXPECT_EQ(fuel_runtime_state_bytes(runtime.snapshot_state()),
                              fuel_runtime_state_bytes(expected));
                    expect_runtime_material(runtime, config);
                }
            }
        }
    }
}

TEST(FireFuelRuntime, FreshConstructorsPreserveSuppliedMaterial)
{
    const auto config = fuel_runtime_config(1, 1);
    const Runtime perimeter_runtime(fuel_runtime_ignition(), Real(0), config);
    const Runtime front_runtime(FireFront(std::vector<FireFrontComponent>{
        {FireFrontRole::Outer, fuel_runtime_ignition()}}), Real(0), config);
    expect_runtime_material(perimeter_runtime, config);
    expect_runtime_material(front_runtime, config);
}

TEST(FireFuelRuntime, CopiesAndMovesOwnMaterialValues)
{
    if (amrex::ParallelDescriptor::NProcs() != 1) {
        GTEST_SKIP() << "Canonical-state comparison requires one MPI rank";
    }
    const auto config = fuel_runtime_config(1, 1);
    Runtime source(fuel_runtime_ignition(), Real(0), config);
    const auto expected = fuel_runtime_state_bytes(source.snapshot_state());
    Runtime copy = source;
    Runtime moved = std::move(copy);
    Runtime assigned(fuel_runtime_ignition(), Real(0), fuel_runtime_config());
    assigned = moved;
    Runtime move_assigned(fuel_runtime_ignition(), Real(0), fuel_runtime_config());
    move_assigned = std::move(assigned);
    source = Runtime(fuel_runtime_ignition(), Real(0), fuel_runtime_config(-1, 2));
    for (const Runtime* runtime : {&moved, &move_assigned}) {
        expect_runtime_material(*runtime, config);
        EXPECT_EQ(fuel_runtime_state_bytes(runtime->snapshot_state()), expected);
        EXPECT_NE(&runtime->fuel_field().uniform_properties(),
                  &source.fuel_field().uniform_properties());
    }
}

TEST(FireFuelRuntime, FlatDirectMatchesLegacyCompleteState)
{
    check_uniform_runtime_equivalence(false, false);
}

TEST(FireFuelRuntime, FlatWafMatchesLegacyCompleteState)
{
    check_uniform_runtime_equivalence(false, true);
}

TEST(FireFuelRuntime, TerrainDirectMatchesLegacyCompleteState)
{
    check_uniform_runtime_equivalence(true, false);
}

TEST(FireFuelRuntime, TerrainWafMatchesLegacyCompleteState)
{
    check_uniform_runtime_equivalence(true, true);
}

TEST(FireFuelRuntime, LegacyRestorePathsReconstructMaterialAndContinue)
{
    if (amrex::ParallelDescriptor::NProcs() != 1) {
        GTEST_SKIP() << "Canonical-state round-trip oracle requires one MPI rank";
    }
    const auto config = fuel_runtime_config(1, 1);
    const auto environment = fuel_runtime_environment(
        config.raster_geometry, {Real(1), Real(0)});
    Runtime original(fuel_runtime_ignition(), Real(0), config);
    advance_uniform_fixture(original, environment, nullptr, false, true, Real(0.25));
    const auto before = fuel_runtime_state_bytes(original.snapshot_state());
    Runtime continued = original;
    advance_uniform_fixture(continued, environment, nullptr, false, true, Real(0.25));
    const auto after = fuel_runtime_state_bytes(continued.snapshot_state());
    const auto raster = ERFFire::make_erf_fire_checkpoint_v2_raster(original);
    auto options = make_options();
    options.reference_height_agl_m = Real(0.5);

    for (int version : {0, 1, 2, 3}) {
        SCOPED_TRACE(version);
        Runtime restored = [&]() {
            if (version == 0) {
                return Runtime::restore_from_state(original.snapshot_state());
            }
            std::stringstream stream;
            if (version == 1) {
                ERFFire::write_erf_fire_checkpoint_state(original, options, stream);
            } else if (version == 2) {
                ERFFire::write_erf_fire_checkpoint_v2_metadata(original, options, stream);
            } else {
                ERFFire::write_erf_fire_checkpoint_v3_metadata(original, options, stream);
            }
            EXPECT_EQ(ERFFire::read_erf_fire_checkpoint_version(stream), version);
            stream.clear();
            stream.seekg(0);
            if (version == 1) {
                auto checkpoint = ERFFire::read_erf_fire_checkpoint_state(stream);
                return Runtime::collective_restore_from_io_rank_state(
                    std::move(checkpoint.runtime_state));
            }
            auto state = version == 2
                ? ERFFire::read_erf_fire_checkpoint_v2_metadata(stream).checkpoint.runtime_state
                : ERFFire::read_erf_fire_checkpoint_v3_metadata(stream).checkpoint.runtime_state;
            return Runtime::collective_restore_from_checkpoint_raster(std::move(state), raster);
        }();
        expect_runtime_material(restored, config);
        EXPECT_EQ(fuel_runtime_state_bytes(restored.snapshot_state()), before);
        advance_uniform_fixture(restored, environment, nullptr, false, true, Real(0.25));
        EXPECT_EQ(fuel_runtime_state_bytes(restored.snapshot_state()), after);
    }
}

TEST(FireFuelRuntime, FailedMidpointSamplingPreservesMaterialAndHistory)
{
    if (amrex::ParallelDescriptor::NProcs() != 1) {
        GTEST_SKIP() << "Canonical-state comparison requires one MPI rank";
    }
    const auto config = fuel_runtime_config(0, 1);
    Runtime runtime(fuel_runtime_ignition(), Real(0), config);
    const auto before = fuel_runtime_state_bytes(runtime.snapshot_state());
    int calls = 0;
    const ERFFire::FireEnvironmentBatchFunction environment =
        [&calls](const std::vector<FireVec2>& positions) {
            if (++calls == 2) {
                throw std::runtime_error("intentional midpoint sampling failure");
            }
            return std::vector<ERFFire::FireEnvironmentSample>(positions.size(),
                ERFFire::FireEnvironmentSample{{Real(1), Real(0)}, {}});
        };
    EXPECT_THROW((void)runtime.advance_direct_reference_wind_batched(environment, Real(0.25)),
                 std::runtime_error);
    EXPECT_EQ(calls, 2);
    expect_runtime_material(runtime, config);
    EXPECT_EQ(fuel_runtime_state_bytes(runtime.snapshot_state()), before);
}

} // namespace
