#include "ERF_FireSpreadRuntime.H"

#include <ERF_FireWindAdjustment.H>
#include <ERF_RichardsDirectionalSpread.H>
#include <ERF_RothermelModel.H>
#include <ERF_VectorPerimeterPropagator.H>

#include <cmath>
#include <stdexcept>
#include <utility>

namespace ERFFire
{
namespace
{

void
require(bool condition, const char* message)
{
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

FireVec2
wind_push_unit(const FireVec2& wind_mps, amrex::Real speed_mps)
{
    if (!(speed_mps > amrex::Real(0.0))) {
        return {amrex::Real(1.0), amrex::Real(0.0)};
    }
    return wind_mps / speed_mps;
}


FireVec2
terrain_upslope_unit(
    const FireVec2& gradient_m_per_m,
    amrex::Real slope_tangent)
{
    if (!(slope_tangent > amrex::Real(0.0))) {
        return {
            amrex::Real(1.0),
            amrex::Real(0.0)};
    }
    return gradient_m_per_m / slope_tangent;
}

bool
same_horizontal_geometry(
    const FireCartesianRasterGeometry2D& lhs,
    const FireCartesianRasterGeometry2D& rhs) noexcept
{
    return lhs.nx == rhs.nx
        && lhs.ny == rhs.ny
        && lhs.xlo_m == rhs.xlo_m
        && lhs.ylo_m == rhs.ylo_m
        && lhs.dx_m == rhs.dx_m
        && lhs.dy_m == rhs.dy_m;
}

bool
environment_matches_geometry(
    const FireFlatEnvironmentSampler& environment,
    const FireCartesianRasterGeometry2D& geometry) noexcept
{
    const auto& layout = environment.layout();

    return layout.nx() == geometry.nx
        && layout.ny() == geometry.ny
        && layout.xlo_m() == geometry.xlo_m
        && layout.ylo_m() == geometry.ylo_m
        && layout.dx_m() == geometry.dx_m
        && layout.dy_m() == geometry.dy_m;
}

void
require_terrain_runtime_geometry(
    const FireFlatEnvironmentSampler& environment,
    const FireTerrainSurface& terrain,
    const FireCartesianRasterGeometry2D& runtime_geometry)
{
    require(
        same_horizontal_geometry(
            terrain.geometry(),
            runtime_geometry),
        "fire terrain geometry must match the Fire runtime raster geometry");

    require(
        environment_matches_geometry(
            environment,
            runtime_geometry),
        "fire terrain reference-wind geometry must match the Fire runtime raster geometry");
}

void
require_perimeter_inside_environment(
    const FirePerimeter& perimeter,
    const FireFlatEnvironmentSampler& environment)
{
    for (const FireVec2& vertex : perimeter.vertices_m()) {
        if (!environment.layout().contains_physical_point(
                vertex.x, vertex.y)) {
            throw std::out_of_range(
                "fire spread propagated perimeter leaves the physical environment domain");
        }
    }
}

} // namespace

ERFFireSpreadRuntime::ERFFireSpreadRuntime(
    FirePerimeter initial_perimeter,
    amrex::Real initial_time_s,
    ERFFireSpreadConfig config)
    : config_(std::move(config)),
      perimeter_(std::move(initial_perimeter)),
      burned_fraction_(config_.raster_geometry),
      first_arrival_(config_.raster_geometry),
      combustion_(
          config_.raster_geometry,
          config_.combustion_parameters,
          config_.combustion_options),
      current_time_s_(initial_time_s)
{
    require(
        std::isfinite(current_time_s_) &&
            current_time_s_ >= amrex::Real(0.0),
        "fire spread initial time must be finite and nonnegative");

    require(
        std::isfinite(config_.arrival_time_tolerance_s) &&
            config_.arrival_time_tolerance_s > amrex::Real(0.0),
        "fire spread arrival tolerance must be finite and positive");

    (void)evaluate_rothermel(
        config_.fuel,
        RothermelInputs{
            config_.dead_fuel_moisture_fraction,
            amrex::Real(0.0),
            amrex::Real(0.0)});

    auto initial_remesh =
        remesh_perimeter(perimeter_, config_.remesh_options);
    perimeter_ = std::move(initial_remesh.perimeter);

    (void)first_arrival_.initialize_from_perimeter(
        perimeter_, current_time_s_);
    (void)burned_fraction_.update_from_perimeter(perimeter_);
    (void)combustion_.initialize_from_burned_fraction(
        burned_fraction_);
}

ERFFireStepDiagnostics
ERFFireSpreadRuntime::advance_direct_reference_wind(
    const FireFlatEnvironmentSampler& environment,
    amrex::Real dt_s)
{
    return advance_wind_impl(
        environment,
        nullptr,
        WindInputMode::DirectReference,
        amrex::Real(1.0),
        dt_s);
}

ERFFireStepDiagnostics
ERFFireSpreadRuntime::advance_direct_reference_wind(
    const FireFlatEnvironmentSampler& environment,
    const FireTerrainSurface& terrain,
    amrex::Real dt_s)
{
    require_terrain_runtime_geometry(
        environment,
        terrain,
        config_.raster_geometry);

    return advance_wind_impl(
        environment,
        &terrain,
        WindInputMode::DirectReference,
        amrex::Real(1.0),
        dt_s);
}

ERFFireStepDiagnostics
ERFFireSpreadRuntime::advance_explicit_waf_20ft(
    const FireFlatEnvironmentSampler& environment,
    amrex::Real wind_adjustment_factor,
    amrex::Real dt_s)
{
    return advance_wind_impl(
        environment,
        nullptr,
        WindInputMode::ExplicitWaf20ft,
        wind_adjustment_factor,
        dt_s);
}

ERFFireStepDiagnostics
ERFFireSpreadRuntime::advance_explicit_waf_20ft(
    const FireFlatEnvironmentSampler& environment,
    const FireTerrainSurface& terrain,
    amrex::Real wind_adjustment_factor,
    amrex::Real dt_s)
{
    require_terrain_runtime_geometry(
        environment,
        terrain,
        config_.raster_geometry);

    return advance_wind_impl(
        environment,
        &terrain,
        WindInputMode::ExplicitWaf20ft,
        wind_adjustment_factor,
        dt_s);
}

ERFFireStepDiagnostics
ERFFireSpreadRuntime::advance_wind_impl(
    const FireFlatEnvironmentSampler& environment,
    const FireTerrainSurface* terrain_surface,
    WindInputMode wind_input_mode,
    amrex::Real wind_adjustment_factor,
    amrex::Real dt_s)
{
    if (wind_input_mode == WindInputMode::ExplicitWaf20ft) {
        require(
            std::isfinite(wind_adjustment_factor)
                && wind_adjustment_factor >= amrex::Real(0.0)
                && wind_adjustment_factor <= amrex::Real(1.0),
            "fire explicit 20-ft wind adjustment factor must be finite and in [0,1]");
    }

    require(
        std::isfinite(dt_s) && dt_s > amrex::Real(0.0),
        "fire spread dt must be finite and positive");
    require(
        config_.arrival_time_tolerance_s <= dt_s,
        "fire spread arrival tolerance must not exceed the fire dt");

    const amrex::Real start_time_s = current_time_s_;
    const amrex::Real end_time_s = start_time_s + dt_s;
    if (!std::isfinite(end_time_s) || !(end_time_s > start_time_s)) {
        throw std::overflow_error(
            "fire spread end time must be finite and representably later");
    }

    const auto normal_speed =
        [this,
         &environment,
         terrain_surface,
         wind_input_mode,
         wind_adjustment_factor](
            const FireVec2& position_m,
            const FireVec2& outward_normal,
            amrex::Real) -> amrex::Real
    {
        const FireEnvironmentSample sample =
            environment.sample(position_m.x, position_m.y);
        const FireVec2 reference_wind_mps =
            sample.horizontal_wind_mps;

        if (!std::isfinite(reference_wind_mps.x)
            || !std::isfinite(reference_wind_mps.y)) {
            throw std::overflow_error(
                "fire spread sampled reference wind is not finite");
        }

        const FireVec2 wind_mps =
            wind_input_mode == WindInputMode::ExplicitWaf20ft
                ? fire_midflame_wind_from_20ft_reference(
                    reference_wind_mps,
                    wind_adjustment_factor)
                : reference_wind_mps;

        const amrex::Real speed_mps = norm(wind_mps);
        if (!std::isfinite(speed_mps)) {
            throw std::overflow_error(
                "fire spread model wind magnitude is not finite");
        }

        FireVec2 terrain_gradient_m_per_m{};
        if (terrain_surface != nullptr) {
            terrain_gradient_m_per_m =
                terrain_surface->terrain_gradient_m_per_m(
                    position_m.x,
                    position_m.y);
        }

        const amrex::Real slope_tangent =
            norm(terrain_gradient_m_per_m);
        if (!std::isfinite(slope_tangent)) {
            throw std::overflow_error(
                "fire spread sampled terrain slope magnitude is not finite");
        }

        const FireVec2 upslope_unit =
            terrain_upslope_unit(
                terrain_gradient_m_per_m,
                slope_tangent);

        const RothermelResult behavior =
            evaluate_rothermel(
                config_.fuel,
                RothermelInputs{
                    config_.dead_fuel_moisture_fraction,
                    speed_mps,
                    slope_tangent});

        const RichardsDirectionalSpread spread =
            make_richards_directional_spread(
                behavior,
                wind_push_unit(
                    wind_mps,
                    speed_mps),
                upslope_unit);

        return richards_normal_speed_mps(
            spread.ellipse, outward_normal);
    };

    FirePerimeter advanced =
        advance_perimeter_rk2(
            perimeter_, start_time_s, dt_s, normal_speed);

    // RK2 samples current and midpoint locations. Explicitly reject a final
    // perimeter outside the supported physical environment before committing
    // history, since the endpoint is not itself an RK sampling location.
    require_perimeter_inside_environment(advanced, environment);

    FireFirstArrivalRaster next_arrival = first_arrival_;
    const FireFirstArrivalRasterUpdate arrival_update =
        next_arrival.update_from_sweep(
            perimeter_,
            advanced,
            start_time_s,
            end_time_s,
            config_.arrival_time_tolerance_s);

    FireBurnedFractionRaster next_burned = burned_fraction_;
    const FireRasterBurnedAreaUpdate burned_update =
        next_burned.update_from_perimeter(advanced);

    FireCombustionRaster next_combustion = combustion_;
    const FireCombustionRasterAdvance combustion_update =
        next_combustion.advance_from_linear_sweep(
            perimeter_,
            advanced,
            burned_fraction_,
            next_burned,
            dt_s);

    const std::size_t pre_remesh_vertex_count = advanced.size();
    FirePerimeterRemeshResult remeshed =
        remesh_perimeter(advanced, config_.remesh_options);

    ERFFireStepDiagnostics diagnostics{
        start_time_s,
        end_time_s,
        pre_remesh_vertex_count,
        remeshed.perimeter.size(),
        remeshed.stats.vertices_removed,
        remeshed.stats.vertices_added,
        arrival_update.newly_arrived_cell_count,
        arrival_update.arrived_cell_count,
        burned_update.newly_burned_area_m2,
        burned_update.burned_area_m2,
        combustion_update.newly_consumed_dry_fuel_kg,
        combustion_update.totals.remaining_dry_fuel_kg,
        combustion_update.totals.consumed_dry_fuel_kg,
        combustion_update.sensible_energy_increment_j,
        combustion_update.totals.sensible_energy_j,
        combustion_update.water_released_increment_kg,
        combustion_update.totals.water_released_kg};

    perimeter_ = std::move(remeshed.perimeter);
    first_arrival_ = std::move(next_arrival);
    burned_fraction_ = std::move(next_burned);
    combustion_ = std::move(next_combustion);
    current_time_s_ = end_time_s;

    return diagnostics;
}

} // namespace ERFFire
