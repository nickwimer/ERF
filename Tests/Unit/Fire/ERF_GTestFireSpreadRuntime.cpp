#include <ERF_FireSpreadRuntime.H>
#include <ERF_FireCombustion.H>

#include <ERF_RichardsDirectionalSpread.H>
#include <ERF_RothermelModel.H>
#include <ERF_VectorPerimeterPropagator.H>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{

using amrex::Real;
using ERFFire::ERFFireSpreadConfig;
using ERFFire::ERFFireSpreadRuntime;
using ERFFire::FireCartesianRasterGeometry2D;
using ERFFire::FireCombustionRasterOptions;
using ERFFire::FireFlatEnvironmentLayout2D;
using ERFFire::FireFlatEnvironmentSampler;
using ERFFire::FirePerimeter;
using ERFFire::FirePerimeterRemeshOptions;
using ERFFire::FireVec2;

constexpr Real pi =
    Real(3.141592653589793238462643383279502884L);

constexpr Real wind_only_flank_ros_mps =
    Real(0.05412799019404888);
constexpr Real wind_only_semi_major_rate_mps =
    Real(0.06536997242848683);
constexpr Real wind_only_center_translation_rate_mps =
    Real(0.03665233925486989);

FirePerimeter
make_circle(std::size_t count, FireVec2 center, Real radius_m)
{
    std::vector<FireVec2> vertices;
    vertices.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const Real angle =
            Real(2.0) * pi * static_cast<Real>(i)
            / static_cast<Real>(count);
        vertices.push_back({
            center.x + radius_m * std::cos(angle),
            center.y + radius_m * std::sin(angle)});
    }
    return FirePerimeter(std::move(vertices));
}

FirePerimeter
make_wind_only_exact_wavelet(
    std::size_t count,
    FireVec2 ignition_m,
    Real age_s)
{
    std::vector<FireVec2> vertices;
    vertices.reserve(count);
    const FireVec2 center{
        ignition_m.x
            + wind_only_center_translation_rate_mps * age_s,
        ignition_m.y};

    for (std::size_t i = 0; i < count; ++i) {
        const Real angle =
            Real(2.0) * pi * static_cast<Real>(i)
            / static_cast<Real>(count);
        vertices.push_back({
            center.x
                + wind_only_semi_major_rate_mps
                    * age_s * std::cos(angle),
            center.y
                + wind_only_flank_ros_mps
                    * age_s * std::sin(angle)});
    }
    return FirePerimeter(std::move(vertices));
}

Real
polygon_support(
    const FirePerimeter& perimeter,
    const FireVec2& direction)
{
    Real result = -std::numeric_limits<Real>::infinity();
    for (const FireVec2& vertex : perimeter.vertices_m()) {
        result = std::max(result, ERFFire::dot(vertex, direction));
    }
    return result;
}

Real
exact_wind_only_support(
    FireVec2 ignition_m,
    Real age_s,
    FireVec2 direction)
{
    const Real center =
        ERFFire::dot(ignition_m, direction)
        + wind_only_center_translation_rate_mps
            * age_s * direction.x;
    const Real major =
        wind_only_semi_major_rate_mps * age_s * direction.x;
    const Real minor =
        wind_only_flank_ros_mps * age_s * direction.y;
    return center + std::sqrt(major * major + minor * minor);
}

FireFlatEnvironmentSampler
make_uniform_sampler(
    Real xlo,
    Real ylo,
    Real dx,
    Real dy,
    std::size_t nx,
    std::size_t ny,
    FireVec2 wind_mps)
{
    FireFlatEnvironmentLayout2D layout(
        xlo, ylo, dx, dy, nx, ny);
    std::vector<Real> u(
        layout.u_storage_size(), wind_mps.x);
    std::vector<Real> v(
        layout.v_storage_size(), wind_mps.y);
    return FireFlatEnvironmentSampler(
        std::move(layout),
        Real(0.5),
        std::move(u),
        std::move(v));
}

Real
affine_u(Real x, Real y)
{
    return Real(0.8) + Real(0.02) * x - Real(0.01) * y;
}

Real
affine_v(Real x, Real y)
{
    return Real(0.15) + Real(0.005) * x + Real(0.01) * y;
}

FireFlatEnvironmentSampler
make_affine_sampler()
{
    FireFlatEnvironmentLayout2D layout(
        Real(0.0), Real(0.0), Real(1.0), Real(1.0), 24, 24);
    std::vector<Real> u(layout.u_storage_size());
    std::vector<Real> v(layout.v_storage_size());

    const int nx = static_cast<int>(layout.nx());
    const int ny = static_cast<int>(layout.ny());

    for (int j = -1; j <= ny; ++j) {
        for (int i = -1; i <= nx + 1; ++i) {
            const Real x =
                layout.xlo_m() + static_cast<Real>(i) * layout.dx_m();
            const Real y =
                layout.ylo_m()
                + (static_cast<Real>(j) + Real(0.5)) * layout.dy_m();
            u[layout.u_storage_index(i, j)] = affine_u(x, y);
        }
    }

    for (int j = -1; j <= ny + 1; ++j) {
        for (int i = -1; i <= nx; ++i) {
            const Real x =
                layout.xlo_m()
                + (static_cast<Real>(i) + Real(0.5)) * layout.dx_m();
            const Real y =
                layout.ylo_m() + static_cast<Real>(j) * layout.dy_m();
            v[layout.v_storage_index(i, j)] = affine_v(x, y);
        }
    }

    return FireFlatEnvironmentSampler(
        std::move(layout),
        Real(0.5),
        std::move(u),
        std::move(v));
}

ERFFireSpreadConfig
make_config(
    FireCartesianRasterGeometry2D raster,
    FirePerimeterRemeshOptions remesh =
        FirePerimeterRemeshOptions{
            Real(1.0e-6), Real(10.0), Real(0.0)})
{
    return {
        ERFFire::make_fm1_fuel_parameters(),
        Real(0.08),
        ERFFire::make_fm1_combustion_parameters(Real(0.08)),
        FireCombustionRasterOptions{16},
        remesh,
        raster,
        Real(1.0e-7)};
}

Real
direct_normal_speed(
    FireVec2 wind_mps,
    const FireVec2& outward_normal)
{
    const Real speed = ERFFire::norm(wind_mps);
    const FireVec2 direction =
        speed > Real(0.0)
        ? wind_mps / speed
        : FireVec2{Real(1.0), Real(0.0)};

    const auto behavior = ERFFire::evaluate_rothermel(
        ERFFire::make_fm1_fuel_parameters(),
        ERFFire::RothermelInputs{
            Real(0.08), speed, Real(0.0)});

    const auto spread =
        ERFFire::make_richards_directional_spread(
            behavior,
            direction,
            FireVec2{Real(1.0), Real(0.0)});

    return ERFFire::richards_normal_speed_mps(
        spread.ellipse, outward_normal);
}

} // namespace

TEST(FireSpreadRuntime, UniformDirectReferenceWindTracksIndependentWavelet)
{
    constexpr Real initial_age_s = Real(20.0);
    constexpr Real dt_s = Real(1.0);
    const FireVec2 ignition{Real(8.0), Real(8.0)};

    const FirePerimeter initial =
        make_wind_only_exact_wavelet(512, ignition, initial_age_s);

    ERFFireSpreadRuntime runtime(
        initial,
        Real(0.0),
        make_config(
            FireCartesianRasterGeometry2D{
                16, 16, Real(0.0), Real(0.0), Real(1.0), Real(1.0)}));

    const auto environment =
        make_uniform_sampler(
            Real(0.0), Real(0.0), Real(1.0), Real(1.0),
            16, 16, FireVec2{Real(1.0), Real(0.0)});

    const auto diagnostics =
        runtime.advance_direct_reference_wind(environment, dt_s);

    EXPECT_EQ(diagnostics.start_time_s, Real(0.0));
    EXPECT_EQ(diagnostics.end_time_s, dt_s);
    EXPECT_EQ(runtime.current_time_s(), dt_s);

    const Real final_age_s = initial_age_s + dt_s;
    for (const FireVec2 direction : std::vector<FireVec2>{
             {Real(1.0), Real(0.0)},
             {Real(-1.0), Real(0.0)},
             {Real(0.0), Real(1.0)},
             {Real(0.0), Real(-1.0)}}) {
        EXPECT_NEAR(
            polygon_support(runtime.perimeter(), direction),
            exact_wind_only_support(ignition, final_age_s, direction),
            2.0e-3);
    }
}

TEST(FireSpreadRuntime, AffineSnapshotIsResampledAtCurrentAndMidpointPositions)
{
    constexpr Real dt_s = Real(2.0);
    const FireVec2 center{Real(8.0), Real(8.0)};
    const FirePerimeter initial =
        make_circle(96, center, Real(1.5));

    ERFFireSpreadRuntime runtime(
        initial,
        Real(0.0),
        make_config(
            FireCartesianRasterGeometry2D{
                24, 24, Real(0.0), Real(0.0), Real(1.0), Real(1.0)}));

    const FireFlatEnvironmentSampler environment =
        make_affine_sampler();

    const auto analytic_speed = [](
        const FireVec2& position,
        const FireVec2& outward_normal,
        Real) -> Real
    {
        return direct_normal_speed(
            FireVec2{
                affine_u(position.x, position.y),
                affine_v(position.x, position.y)},
            outward_normal);
    };

    const FirePerimeter expected =
        ERFFire::advance_perimeter_rk2(
            initial, Real(0.0), dt_s, analytic_speed);

    (void)runtime.advance_direct_reference_wind(
        environment, dt_s);

    ASSERT_EQ(runtime.perimeter().size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(
            runtime.perimeter().vertices_m()[i].x,
            expected.vertices_m()[i].x,
            2.0e-12);
        EXPECT_NEAR(
            runtime.perimeter().vertices_m()[i].y,
            expected.vertices_m()[i].y,
            2.0e-12);
    }

    const FireVec2 one_global_wind{
        affine_u(center.x, center.y),
        affine_v(center.x, center.y)};
    const auto global_speed = [one_global_wind](
        const FireVec2&,
        const FireVec2& outward_normal,
        Real) -> Real
    {
        return direct_normal_speed(
            one_global_wind, outward_normal);
    };
    const FirePerimeter global_result =
        ERFFire::advance_perimeter_rk2(
            initial, Real(0.0), dt_s, global_speed);

    Real maximum_global_difference = Real(0.0);
    for (std::size_t i = 0; i < expected.size(); ++i) {
        maximum_global_difference = std::max(
            maximum_global_difference,
            ERFFire::norm(
                runtime.perimeter().vertices_m()[i]
                - global_result.vertices_m()[i]));
    }
    EXPECT_GT(maximum_global_difference, Real(1.0e-5));
}

TEST(FireSpreadRuntime, ArrivalBurnHistoryPrecedesRemeshingAndAdvancesMonotonically)
{
    const FirePerimeter initial =
        make_circle(
            24,
            FireVec2{Real(8.0), Real(8.0)},
            Real(1.2));

    ERFFireSpreadRuntime runtime(
        initial,
        Real(0.0),
        make_config(
            FireCartesianRasterGeometry2D{
                16, 16, Real(0.0), Real(0.0), Real(1.0), Real(1.0)},
            FirePerimeterRemeshOptions{
                Real(0.05), Real(0.25), Real(0.002)}));

    const Real initial_burned_area =
        runtime.burned_fraction_raster().burned_area_m2();
    const std::size_t initial_arrived_count =
        runtime.first_arrival_raster().arrived_cell_count();

    EXPECT_GT(initial_burned_area, Real(0.0));
    EXPECT_GT(initial_arrived_count, 0U);
    ASSERT_TRUE(
        runtime.first_arrival_raster().has_arrived(8, 8));
    EXPECT_EQ(
        runtime.first_arrival_raster().first_arrival_time_s(8, 8),
        Real(0.0));

    const auto environment =
        make_uniform_sampler(
            Real(0.0), Real(0.0), Real(1.0), Real(1.0),
            16, 16, FireVec2{Real(1.0), Real(0.0)});

    const auto diagnostics =
        runtime.advance_direct_reference_wind(
            environment, Real(1.0));

    EXPECT_GT(diagnostics.pre_remesh_vertex_count, 2U);
    EXPECT_GT(diagnostics.post_remesh_vertex_count, 2U);
    EXPECT_EQ(
        diagnostics.arrived_cell_count,
        runtime.first_arrival_raster().arrived_cell_count());
    EXPECT_EQ(
        diagnostics.burned_area_m2,
        runtime.burned_fraction_raster().burned_area_m2());

    EXPECT_GT(
        diagnostics.burned_area_m2,
        initial_burned_area);
    EXPECT_GT(
        diagnostics.newly_burned_area_m2,
        Real(0.0));
    EXPECT_GE(
        diagnostics.arrived_cell_count,
        initial_arrived_count);

    ASSERT_TRUE(
        runtime.first_arrival_raster().has_arrived(8, 8));
    EXPECT_EQ(
        runtime.first_arrival_raster().first_arrival_time_s(8, 8),
        Real(0.0));
}


TEST(FireSpreadRuntime, CombustionInitializesFromIgnitionBurnHistory)
{
    const FirePerimeter initial =
        make_circle(
            64,
            FireVec2{Real(8.0), Real(8.0)},
            Real(1.0));

    ERFFireSpreadRuntime runtime(
        initial,
        Real(0.0),
        make_config(
            FireCartesianRasterGeometry2D{
                16, 16, Real(0.0), Real(0.0), Real(1.0), Real(1.0)}));

    const auto& combustion = runtime.combustion_raster();
    ASSERT_TRUE(combustion.initialized());

    const auto totals = combustion.totals();
    const Real expected_ignited_dry_fuel_kg =
        runtime.burned_fraction_raster().burned_area_m2()
        * combustion.parameters().dry_fuel_load_kg_m2;
    const Real tolerance =
        Real(1.0e-12)
        * std::max(Real(1.0), std::abs(expected_ignited_dry_fuel_kg));

    EXPECT_GT(totals.remaining_dry_fuel_kg, Real(0.0));
    EXPECT_NEAR(
        totals.remaining_dry_fuel_kg,
        expected_ignited_dry_fuel_kg,
        tolerance);
    EXPECT_EQ(totals.consumed_dry_fuel_kg, Real(0.0));
    EXPECT_EQ(totals.sensible_energy_j, Real(0.0));
    EXPECT_EQ(totals.water_released_kg, Real(0.0));
}

TEST(FireSpreadRuntime, CombustionAdvanceProvidesConservativeExtensiveDiagnostics)
{
    const FirePerimeter initial =
        make_circle(
            64,
            FireVec2{Real(8.0), Real(8.0)},
            Real(1.0));

    ERFFireSpreadRuntime runtime(
        initial,
        Real(0.0),
        make_config(
            FireCartesianRasterGeometry2D{
                16, 16, Real(0.0), Real(0.0), Real(1.0), Real(1.0)}));

    const auto before = runtime.combustion_raster().totals();

    const auto environment =
        make_uniform_sampler(
            Real(0.0), Real(0.0), Real(1.0), Real(1.0),
            16, 16, FireVec2{Real(1.0), Real(0.0)});

    const auto diagnostics =
        runtime.advance_direct_reference_wind(
            environment, Real(1.0));

    const auto after = runtime.combustion_raster().totals();
    const auto& parameters =
        runtime.combustion_raster().parameters();

    const Real expected_ignited_dry_fuel_kg =
        runtime.burned_fraction_raster().burned_area_m2()
        * parameters.dry_fuel_load_kg_m2;
    const Real expected_energy_j =
        after.consumed_dry_fuel_kg
        * parameters.sensible_heat_release_j_kg_dry;
    const Real expected_water_kg =
        after.consumed_dry_fuel_kg
        * (parameters.fuel_moisture_fraction
           + parameters.combustion_water_yield_kg_per_kg_dry);

    const auto tolerance = [] (Real value) {
        return Real(1.0e-11)
            * std::max(Real(1.0), std::abs(value));
    };

    EXPECT_GT(
        diagnostics.newly_consumed_dry_fuel_kg,
        Real(0.0));
    EXPECT_GT(
        diagnostics.sensible_energy_increment_j,
        Real(0.0));
    EXPECT_GT(
        diagnostics.water_released_increment_kg,
        Real(0.0));

    EXPECT_NEAR(
        diagnostics.newly_consumed_dry_fuel_kg,
        after.consumed_dry_fuel_kg
            - before.consumed_dry_fuel_kg,
        tolerance(after.consumed_dry_fuel_kg));
    EXPECT_NEAR(
        diagnostics.sensible_energy_increment_j,
        after.sensible_energy_j
            - before.sensible_energy_j,
        tolerance(after.sensible_energy_j));
    EXPECT_NEAR(
        diagnostics.water_released_increment_kg,
        after.water_released_kg
            - before.water_released_kg,
        tolerance(after.water_released_kg));

    EXPECT_NEAR(
        diagnostics.remaining_dry_fuel_kg,
        after.remaining_dry_fuel_kg,
        tolerance(after.remaining_dry_fuel_kg));
    EXPECT_NEAR(
        diagnostics.consumed_dry_fuel_kg,
        after.consumed_dry_fuel_kg,
        tolerance(after.consumed_dry_fuel_kg));
    EXPECT_NEAR(
        diagnostics.sensible_energy_j,
        after.sensible_energy_j,
        tolerance(after.sensible_energy_j));
    EXPECT_NEAR(
        diagnostics.water_released_kg,
        after.water_released_kg,
        tolerance(after.water_released_kg));

    EXPECT_NEAR(
        after.remaining_dry_fuel_kg
            + after.consumed_dry_fuel_kg,
        expected_ignited_dry_fuel_kg,
        tolerance(expected_ignited_dry_fuel_kg));
    EXPECT_NEAR(
        after.sensible_energy_j,
        expected_energy_j,
        tolerance(expected_energy_j));
    EXPECT_NEAR(
        after.water_released_kg,
        expected_water_kg,
        tolerance(expected_water_kg));
}

TEST(FireSpreadRuntime, EndpointOutsideEnvironmentIsRejectedTransactionally)
{
    const FirePerimeter initial =
        make_circle(
            32,
            FireVec2{Real(3.87), Real(2.0)},
            Real(0.04));

    ERFFireSpreadRuntime runtime(
        initial,
        Real(0.0),
        make_config(
            FireCartesianRasterGeometry2D{
                4, 4, Real(0.0), Real(0.0), Real(1.0), Real(1.0)}));

    const auto environment =
        make_uniform_sampler(
            Real(0.0), Real(0.0), Real(1.0), Real(1.0),
            4, 4, FireVec2{Real(1.0), Real(0.0)});

    const std::vector<FireVec2> before_vertices =
        runtime.perimeter().vertices_m();
    const Real before_burned_area =
        runtime.burned_fraction_raster().burned_area_m2();
    const std::size_t before_arrived =
        runtime.first_arrival_raster().arrived_cell_count();
    const auto before_combustion =
        runtime.combustion_raster().totals();

    EXPECT_THROW(
        runtime.advance_direct_reference_wind(
            environment, Real(1.0)),
        std::out_of_range);

    EXPECT_EQ(runtime.current_time_s(), Real(0.0));
    EXPECT_EQ(
        runtime.first_arrival_raster().arrived_cell_count(),
        before_arrived);
    EXPECT_EQ(
        runtime.burned_fraction_raster().burned_area_m2(),
        before_burned_area);
    const auto after_combustion =
        runtime.combustion_raster().totals();
    EXPECT_EQ(
        after_combustion.remaining_dry_fuel_kg,
        before_combustion.remaining_dry_fuel_kg);
    EXPECT_EQ(
        after_combustion.consumed_dry_fuel_kg,
        before_combustion.consumed_dry_fuel_kg);
    EXPECT_EQ(
        after_combustion.sensible_energy_j,
        before_combustion.sensible_energy_j);
    EXPECT_EQ(
        after_combustion.water_released_kg,
        before_combustion.water_released_kg);
    ASSERT_EQ(
        runtime.perimeter().vertices_m().size(),
        before_vertices.size());

    for (std::size_t i = 0; i < before_vertices.size(); ++i) {
        EXPECT_EQ(
            runtime.perimeter().vertices_m()[i].x,
            before_vertices[i].x);
        EXPECT_EQ(
            runtime.perimeter().vertices_m()[i].y,
            before_vertices[i].y);
    }
}

TEST(FireSpreadRuntime, FailedMidpointSampleLeavesAllPersistentStateUnchanged)
{
    const FirePerimeter initial =
        make_circle(
            32,
            FireVec2{Real(3.94), Real(2.0)},
            Real(0.04));

    ERFFireSpreadRuntime runtime(
        initial,
        Real(0.0),
        make_config(
            FireCartesianRasterGeometry2D{
                4, 4, Real(0.0), Real(0.0), Real(1.0), Real(1.0)}));

    const auto environment =
        make_uniform_sampler(
            Real(0.0), Real(0.0), Real(1.0), Real(1.0),
            4, 4, FireVec2{Real(1.0), Real(0.0)});

    const std::vector<FireVec2> before_vertices =
        runtime.perimeter().vertices_m();
    const Real before_burned_area =
        runtime.burned_fraction_raster().burned_area_m2();
    const std::size_t before_arrived =
        runtime.first_arrival_raster().arrived_cell_count();
    const auto before_combustion =
        runtime.combustion_raster().totals();

    EXPECT_THROW(
        runtime.advance_direct_reference_wind(
            environment, Real(1.0)),
        std::out_of_range);

    EXPECT_EQ(runtime.current_time_s(), Real(0.0));
    EXPECT_EQ(
        runtime.first_arrival_raster().arrived_cell_count(),
        before_arrived);
    EXPECT_EQ(
        runtime.burned_fraction_raster().burned_area_m2(),
        before_burned_area);
    const auto after_combustion =
        runtime.combustion_raster().totals();
    EXPECT_EQ(
        after_combustion.remaining_dry_fuel_kg,
        before_combustion.remaining_dry_fuel_kg);
    EXPECT_EQ(
        after_combustion.consumed_dry_fuel_kg,
        before_combustion.consumed_dry_fuel_kg);
    EXPECT_EQ(
        after_combustion.sensible_energy_j,
        before_combustion.sensible_energy_j);
    EXPECT_EQ(
        after_combustion.water_released_kg,
        before_combustion.water_released_kg);
    ASSERT_EQ(
        runtime.perimeter().vertices_m().size(),
        before_vertices.size());

    for (std::size_t i = 0; i < before_vertices.size(); ++i) {
        EXPECT_EQ(
            runtime.perimeter().vertices_m()[i].x,
            before_vertices[i].x);
        EXPECT_EQ(
            runtime.perimeter().vertices_m()[i].y,
            before_vertices[i].y);
    }
}
