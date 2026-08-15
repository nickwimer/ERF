#include <ERF_FireCombustion.H>
#include <ERF_FireFlatEnvironmentSampler.H>
#include <ERF_FireRuntimeOptions.H>
#include <ERF_FireSpreadRuntime.H>
#include <ERF_FireTerrainSurface.H>
#include <ERF_RothermelFuel.H>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
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
using ERFFire::FireCombustionRasterTotals;
using ERFFire::FireFlatEnvironmentLayout2D;
using ERFFire::FireFlatEnvironmentSampler;
using ERFFire::FirePerimeter;
using ERFFire::FirePerimeterRemeshOptions;
using ERFFire::FireTerrainSurface;
using ERFFire::FireVec2;

constexpr Real pi =
    Real(3.141592653589793238462643383279502884L);
constexpr Real fuel_moisture = Real(0.08);
constexpr Real center_x_m = Real(12.0);
constexpr Real center_y_m = Real(12.0);
constexpr Real initial_radius_m = Real(2.0);
constexpr Real dt_s = Real(2.0);
constexpr int step_count = 6;
constexpr Real total_time_s = dt_s * Real(step_count);

FireCartesianRasterGeometry2D
science_geometry()
{
    return {
        48, 48,
        Real(0.0), Real(0.0),
        Real(0.5), Real(0.5)};
}

FirePerimeter
make_circle(
    std::size_t count,
    FireVec2 center,
    Real radius_m)
{
    std::vector<FireVec2> vertices;
    vertices.reserve(count);

    for (std::size_t i = 0; i < count; ++i) {
        const Real angle =
            Real(2.0) * pi
            * static_cast<Real>(i)
            / static_cast<Real>(count);
        vertices.push_back({
            center.x + radius_m * std::cos(angle),
            center.y + radius_m * std::sin(angle)});
    }

    return FirePerimeter(std::move(vertices));
}

ERFFireSpreadConfig
make_config(const FireCartesianRasterGeometry2D& geometry)
{
    return {
        ERFFire::make_fm1_fuel_parameters(),
        fuel_moisture,
        ERFFire::make_fm1_combustion_parameters(fuel_moisture),
        FireCombustionRasterOptions{16},
        FirePerimeterRemeshOptions{
            Real(1.0e-6),
            Real(10.0),
            Real(0.0)},
        geometry,
        Real(1.0e-7)};
}

FireFlatEnvironmentSampler
make_uniform_sampler(
    const FireCartesianRasterGeometry2D& geometry,
    FireVec2 reference_wind_mps)
{
    FireFlatEnvironmentLayout2D layout(
        geometry.xlo_m,
        geometry.ylo_m,
        geometry.dx_m,
        geometry.dy_m,
        geometry.nx,
        geometry.ny);

    std::vector<Real> u(
        layout.u_storage_size(),
        reference_wind_mps.x);
    std::vector<Real> v(
        layout.v_storage_size(),
        reference_wind_mps.y);

    return FireFlatEnvironmentSampler(
        std::move(layout),
        ERFFire::explicit_waf_20ft_reference_height_agl_m,
        std::move(u),
        std::move(v));
}

FireTerrainSurface
make_planar_terrain(
    const FireCartesianRasterGeometry2D& geometry,
    FireVec2 gradient)
{
    std::vector<Real> nodal;
    nodal.reserve(
        (geometry.nx + 1)
        * (geometry.ny + 1));

    for (std::size_t j = 0; j <= geometry.ny; ++j) {
        const Real y =
            geometry.ylo_m
            + static_cast<Real>(j) * geometry.dy_m;

        for (std::size_t i = 0; i <= geometry.nx; ++i) {
            const Real x =
                geometry.xlo_m
                + static_cast<Real>(i) * geometry.dx_m;
            nodal.push_back(
                gradient.x * x
                + gradient.y * y);
        }
    }

    return FireTerrainSurface(
        geometry,
        std::move(nodal));
}

Real
support(
    const FirePerimeter& perimeter,
    FireVec2 direction)
{
    Real result = -std::numeric_limits<Real>::infinity();
    for (const FireVec2& vertex : perimeter.vertices_m()) {
        result = std::max(
            result,
            ERFFire::dot(vertex, direction));
    }
    return result;
}

FireVec2
polygon_centroid(const FirePerimeter& perimeter)
{
    const auto& vertices = perimeter.vertices_m();
    Real twice_signed_area = Real(0.0);
    Real x_numerator = Real(0.0);
    Real y_numerator = Real(0.0);

    for (std::size_t i = 0; i < vertices.size(); ++i) {
        const FireVec2& a = vertices[i];
        const FireVec2& b = vertices[(i + 1) % vertices.size()];
        const Real cross = a.x * b.y - b.x * a.y;
        twice_signed_area += cross;
        x_numerator += (a.x + b.x) * cross;
        y_numerator += (a.y + b.y) * cross;
    }

    if (!std::isfinite(twice_signed_area)
        || !(std::abs(twice_signed_area) > Real(0.0))) {
        throw std::runtime_error(
            "polygon centroid requires finite nonzero area");
    }

    const Real denominator = Real(3.0) * twice_signed_area;
    const FireVec2 centroid{
        x_numerator / denominator,
        y_numerator / denominator};

    if (!std::isfinite(centroid.x)
        || !std::isfinite(centroid.y)) {
        throw std::runtime_error(
            "polygon centroid is not finite");
    }

    return centroid;
}

struct ScienceMetrics
{
    Real downwind_support_ros_mps{};
    Real upwind_support_ros_mps{};
    Real cross_positive_support_ros_mps{};
    Real cross_negative_support_ros_mps{};
    FireVec2 centroid_m{};
    Real polygon_area_m2{};
    Real burned_area_m2{};
    std::size_t arrived_cell_count{};
    FireCombustionRasterTotals combustion{};
};

ScienceMetrics
science_metrics(
    const ERFFireSpreadRuntime& runtime,
    const FirePerimeter& initial_perimeter,
    Real elapsed_s)
{
    const auto support_ros = [&] (FireVec2 direction) {
        return (
            support(runtime.perimeter(), direction)
            - support(initial_perimeter, direction))
            / elapsed_s;
    };

    return {
        support_ros({Real(1.0), Real(0.0)}),
        support_ros({Real(-1.0), Real(0.0)}),
        support_ros({Real(0.0), Real(1.0)}),
        support_ros({Real(0.0), Real(-1.0)}),
        polygon_centroid(runtime.perimeter()),
        runtime.perimeter().area_m2(),
        runtime.burned_fraction_raster().burned_area_m2(),
        runtime.first_arrival_raster().arrived_cell_count(),
        runtime.combustion_raster().totals()};
}

void
advance_direct(
    ERFFireSpreadRuntime& runtime,
    const FireFlatEnvironmentSampler& environment,
    const FireTerrainSurface* terrain = nullptr)
{
    for (int step = 0; step < step_count; ++step) {
        if (terrain == nullptr) {
            (void)runtime.advance_direct_reference_wind(
                environment,
                dt_s);
        } else {
            (void)runtime.advance_direct_reference_wind(
                environment,
                *terrain,
                dt_s);
        }
    }
}

void
advance_waf(
    ERFFireSpreadRuntime& runtime,
    const FireFlatEnvironmentSampler& environment,
    Real waf,
    const FireTerrainSurface* terrain = nullptr)
{
    for (int step = 0; step < step_count; ++step) {
        if (terrain == nullptr) {
            (void)runtime.advance_explicit_waf_20ft(
                environment,
                waf,
                dt_s);
        } else {
            (void)runtime.advance_explicit_waf_20ft(
                environment,
                *terrain,
                waf,
                dt_s);
        }
    }
}

void
expect_conservation_closure(
    const ERFFireSpreadRuntime& runtime)
{
    const Real burned_area_m2 =
        runtime.burned_fraction_raster().burned_area_m2();
    const Real perimeter_area_m2 =
        runtime.perimeter().area_m2();
    const auto totals =
        runtime.combustion_raster().totals();
    const auto& parameters =
        runtime.combustion_raster().parameters();

    const Real expected_ignited_dry_fuel_kg =
        burned_area_m2
        * parameters.dry_fuel_load_kg_m2;
    const Real represented_ignited_dry_fuel_kg =
        totals.remaining_dry_fuel_kg
        + totals.consumed_dry_fuel_kg;
    const Real expected_energy_j =
        totals.consumed_dry_fuel_kg
        * parameters.sensible_heat_release_j_kg_dry;
    const Real expected_water_kg =
        totals.consumed_dry_fuel_kg
        * (parameters.fuel_moisture_fraction
           + parameters.combustion_water_yield_kg_per_kg_dry);

    const auto tolerance = [] (Real value) {
        return Real(2.0e-10)
            * std::max(Real(1.0), std::abs(value));
    };

    EXPECT_NEAR(
        represented_ignited_dry_fuel_kg,
        expected_ignited_dry_fuel_kg,
        tolerance(expected_ignited_dry_fuel_kg));
    EXPECT_NEAR(
        totals.sensible_energy_j,
        expected_energy_j,
        tolerance(expected_energy_j));
    EXPECT_NEAR(
        totals.water_released_kg,
        expected_water_kg,
        tolerance(expected_water_kg));

    EXPECT_NEAR(
        burned_area_m2,
        perimeter_area_m2,
        Real(2.0e-8)
            * std::max(Real(1.0), std::abs(perimeter_area_m2)));
}

void
expect_runtime_exact(
    const ERFFireSpreadRuntime& lhs,
    const ERFFireSpreadRuntime& rhs)
{
    EXPECT_EQ(lhs.current_time_s(), rhs.current_time_s());
    ASSERT_EQ(lhs.perimeter().size(), rhs.perimeter().size());

    for (std::size_t i = 0; i < lhs.perimeter().size(); ++i) {
        EXPECT_EQ(
            lhs.perimeter().vertices_m()[i].x,
            rhs.perimeter().vertices_m()[i].x);
        EXPECT_EQ(
            lhs.perimeter().vertices_m()[i].y,
            rhs.perimeter().vertices_m()[i].y);
    }

    const auto& geometry = lhs.config().raster_geometry;
    ASSERT_EQ(geometry.nx, rhs.config().raster_geometry.nx);
    ASSERT_EQ(geometry.ny, rhs.config().raster_geometry.ny);

    for (std::size_t j = 0; j < geometry.ny; ++j) {
        for (std::size_t i = 0; i < geometry.nx; ++i) {
            EXPECT_EQ(
                lhs.burned_fraction_raster().burned_fraction(i, j),
                rhs.burned_fraction_raster().burned_fraction(i, j));

            const bool lhs_arrived =
                lhs.first_arrival_raster().has_arrived(i, j);
            const bool rhs_arrived =
                rhs.first_arrival_raster().has_arrived(i, j);
            EXPECT_EQ(lhs_arrived, rhs_arrived);
            if (lhs_arrived && rhs_arrived) {
                EXPECT_EQ(
                    lhs.first_arrival_raster().first_arrival_time_s(i, j),
                    rhs.first_arrival_raster().first_arrival_time_s(i, j));
            }

            const auto& lhs_state =
                lhs.combustion_raster().state(i, j);
            const auto& rhs_state =
                rhs.combustion_raster().state(i, j);
            EXPECT_EQ(
                lhs_state.ignited_area_fraction,
                rhs_state.ignited_area_fraction);
            EXPECT_EQ(
                lhs_state.remaining_dry_fuel_kg_m2,
                rhs_state.remaining_dry_fuel_kg_m2);
            EXPECT_EQ(
                lhs_state.consumed_dry_fuel_kg_m2,
                rhs_state.consumed_dry_fuel_kg_m2);
            EXPECT_EQ(
                lhs_state.sensible_energy_j_m2,
                rhs_state.sensible_energy_j_m2);
            EXPECT_EQ(
                lhs_state.water_released_kg_m2,
                rhs_state.water_released_kg_m2);
        }
    }
}

void
print_metrics(
    const char* geometry_name,
    Real waf,
    const ScienceMetrics& metrics)
{
    std::cout
        << std::setprecision(17)
        << "FIRE_WIND_SCIENCE_METRICS"
        << " geometry=" << geometry_name
        << " waf=" << waf
        << " downwind_support_ros_mps="
        << metrics.downwind_support_ros_mps
        << " upwind_support_ros_mps="
        << metrics.upwind_support_ros_mps
        << " cross_positive_support_ros_mps="
        << metrics.cross_positive_support_ros_mps
        << " cross_negative_support_ros_mps="
        << metrics.cross_negative_support_ros_mps
        << " centroid_x_m=" << metrics.centroid_m.x
        << " centroid_y_m=" << metrics.centroid_m.y
        << " polygon_area_m2=" << metrics.polygon_area_m2
        << " burned_area_m2=" << metrics.burned_area_m2
        << " arrived_cell_count=" << metrics.arrived_cell_count
        << " remaining_dry_fuel_kg="
        << metrics.combustion.remaining_dry_fuel_kg
        << " consumed_dry_fuel_kg="
        << metrics.combustion.consumed_dry_fuel_kg
        << " sensible_energy_j="
        << metrics.combustion.sensible_energy_j
        << " water_released_kg="
        << metrics.combustion.water_released_kg
        << "\n";
}

} // namespace

TEST(FireWindScience, FlatMatchedHeightUnityWafIsExactControl)
{
    const auto geometry = science_geometry();
    const FirePerimeter initial = make_circle(
        128,
        {center_x_m, center_y_m},
        initial_radius_m);
    const auto environment =
        make_uniform_sampler(geometry, {Real(1.0), Real(0.25)});

    ERFFireSpreadRuntime direct_runtime(
        initial,
        Real(0.0),
        make_config(geometry));
    ERFFireSpreadRuntime waf_runtime = direct_runtime;

    advance_direct(direct_runtime, environment);
    advance_waf(waf_runtime, environment, Real(1.0));

    expect_runtime_exact(direct_runtime, waf_runtime);
    expect_conservation_closure(direct_runtime);
    expect_conservation_closure(waf_runtime);
}

TEST(FireWindScience, FlatWafResponseHasOrderedRosCentroidAndConservation)
{
    const auto geometry = science_geometry();
    const FirePerimeter initial = make_circle(
        128,
        {center_x_m, center_y_m},
        initial_radius_m);
    const auto environment =
        make_uniform_sampler(geometry, {Real(1.0), Real(0.0)});

    ERFFireSpreadRuntime waf_one(
        initial,
        Real(0.0),
        make_config(geometry));
    ERFFireSpreadRuntime waf_half = waf_one;
    ERFFireSpreadRuntime waf_zero = waf_one;

    advance_waf(waf_one, environment, Real(1.0));
    advance_waf(waf_half, environment, Real(0.5));
    advance_waf(waf_zero, environment, Real(0.0));

    const ScienceMetrics one =
        science_metrics(waf_one, initial, total_time_s);
    const ScienceMetrics half =
        science_metrics(waf_half, initial, total_time_s);
    const ScienceMetrics zero =
        science_metrics(waf_zero, initial, total_time_s);

    EXPECT_GT(one.downwind_support_ros_mps, half.downwind_support_ros_mps);
    EXPECT_GT(half.downwind_support_ros_mps, zero.downwind_support_ros_mps);

    EXPECT_GT(one.centroid_m.x, half.centroid_m.x);
    EXPECT_GT(half.centroid_m.x, zero.centroid_m.x);
    EXPECT_NEAR(zero.centroid_m.x, center_x_m, Real(2.0e-10));
    EXPECT_NEAR(zero.centroid_m.y, center_y_m, Real(2.0e-10));

    EXPECT_NEAR(
        zero.downwind_support_ros_mps,
        zero.upwind_support_ros_mps,
        Real(2.0e-10));
    EXPECT_NEAR(
        zero.cross_positive_support_ros_mps,
        zero.cross_negative_support_ros_mps,
        Real(2.0e-10));
    EXPECT_NEAR(
        zero.downwind_support_ros_mps,
        zero.cross_positive_support_ros_mps,
        Real(2.0e-10));

    EXPECT_GT(
        std::abs(one.burned_area_m2 - half.burned_area_m2),
        Real(1.0e-4));
    EXPECT_GT(
        std::abs(half.burned_area_m2 - zero.burned_area_m2),
        Real(1.0e-4));

    expect_conservation_closure(waf_one);
    expect_conservation_closure(waf_half);
    expect_conservation_closure(waf_zero);

    print_metrics("flat", Real(1.0), one);
    print_metrics("flat", Real(0.5), half);
    print_metrics("flat", Real(0.0), zero);
}

TEST(FireWindScience, TerrainUnityAndZeroWafRespectIndependentSlope)
{
    const auto geometry = science_geometry();
    const FirePerimeter initial = make_circle(
        128,
        {center_x_m, center_y_m},
        initial_radius_m);
    const auto reference_environment =
        make_uniform_sampler(geometry, {Real(1.0), Real(0.0)});
    const auto zero_environment =
        make_uniform_sampler(geometry, {Real(0.0), Real(0.0)});
    const FireTerrainSurface terrain =
        make_planar_terrain(
            geometry,
            {Real(0.0), Real(0.20)});

    ERFFireSpreadRuntime direct_wind(
        initial,
        Real(0.0),
        make_config(geometry));
    ERFFireSpreadRuntime waf_one = direct_wind;
    ERFFireSpreadRuntime direct_zero = direct_wind;
    ERFFireSpreadRuntime waf_zero = direct_wind;

    advance_direct(direct_wind, reference_environment, &terrain);
    advance_waf(waf_one, reference_environment, Real(1.0), &terrain);
    advance_direct(direct_zero, zero_environment, &terrain);
    advance_waf(waf_zero, reference_environment, Real(0.0), &terrain);

    expect_runtime_exact(direct_wind, waf_one);
    expect_runtime_exact(direct_zero, waf_zero);

    const ScienceMetrics slope_only =
        science_metrics(waf_zero, initial, total_time_s);
    EXPECT_NEAR(slope_only.centroid_m.x, center_x_m, Real(2.0e-10));
    EXPECT_GT(slope_only.centroid_m.y, center_y_m);
    EXPECT_GT(
        slope_only.cross_positive_support_ros_mps,
        slope_only.cross_negative_support_ros_mps);

    expect_conservation_closure(waf_one);
    expect_conservation_closure(waf_zero);
}

TEST(FireWindScience, TerrainWafReductionChangesWindResponseWithoutBreakingConservation)
{
    const auto geometry = science_geometry();
    const FirePerimeter initial = make_circle(
        128,
        {center_x_m, center_y_m},
        initial_radius_m);
    const auto environment =
        make_uniform_sampler(geometry, {Real(1.0), Real(0.0)});
    const FireTerrainSurface terrain =
        make_planar_terrain(
            geometry,
            {Real(0.0), Real(0.20)});

    ERFFireSpreadRuntime waf_one(
        initial,
        Real(0.0),
        make_config(geometry));
    ERFFireSpreadRuntime waf_half = waf_one;
    ERFFireSpreadRuntime waf_zero = waf_one;

    advance_waf(waf_one, environment, Real(1.0), &terrain);
    advance_waf(waf_half, environment, Real(0.5), &terrain);
    advance_waf(waf_zero, environment, Real(0.0), &terrain);

    const ScienceMetrics one =
        science_metrics(waf_one, initial, total_time_s);
    const ScienceMetrics half =
        science_metrics(waf_half, initial, total_time_s);
    const ScienceMetrics zero =
        science_metrics(waf_zero, initial, total_time_s);

    EXPECT_GT(one.downwind_support_ros_mps, half.downwind_support_ros_mps);
    EXPECT_GT(half.downwind_support_ros_mps, zero.downwind_support_ros_mps);

    EXPECT_GT(one.centroid_m.x, half.centroid_m.x);
    EXPECT_GT(half.centroid_m.x, zero.centroid_m.x);
    EXPECT_NEAR(zero.centroid_m.x, center_x_m, Real(2.0e-10));

    EXPECT_GT(one.centroid_m.y, center_y_m);
    EXPECT_GT(half.centroid_m.y, center_y_m);
    EXPECT_GT(zero.centroid_m.y, center_y_m);

    EXPECT_GT(
        std::abs(one.burned_area_m2 - half.burned_area_m2),
        Real(1.0e-4));
    EXPECT_GT(
        std::abs(half.burned_area_m2 - zero.burned_area_m2),
        Real(1.0e-4));

    expect_conservation_closure(waf_one);
    expect_conservation_closure(waf_half);
    expect_conservation_closure(waf_zero);

    print_metrics("terrain", Real(1.0), one);
    print_metrics("terrain", Real(0.5), half);
    print_metrics("terrain", Real(0.0), zero);
}
