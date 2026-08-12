#include "ERF_FireSpreadRuntime.H"

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
}

ERFFireStepDiagnostics
ERFFireSpreadRuntime::advance_direct_reference_wind(
    const FireFlatEnvironmentSampler& environment,
    amrex::Real dt_s)
{
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

    const auto normal_speed = [this, &environment](
        const FireVec2& position_m,
        const FireVec2& outward_normal,
        amrex::Real) -> amrex::Real
    {
        const FireEnvironmentSample sample =
            environment.sample(position_m.x, position_m.y);
        const FireVec2 wind_mps = sample.horizontal_wind_mps;
        const amrex::Real speed_mps = norm(wind_mps);

        if (!std::isfinite(speed_mps)) {
            throw std::overflow_error(
                "fire spread sampled wind magnitude is not finite");
        }

        const RothermelResult behavior =
            evaluate_rothermel(
                config_.fuel,
                RothermelInputs{
                    config_.dead_fuel_moisture_fraction,
                    speed_mps,
                    amrex::Real(0.0)});

        const RichardsDirectionalSpread spread =
            make_richards_directional_spread(
                behavior,
                wind_push_unit(wind_mps, speed_mps),
                FireVec2{amrex::Real(1.0), amrex::Real(0.0)});

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
        burned_update.burned_area_m2};

    perimeter_ = std::move(remeshed.perimeter);
    first_arrival_ = std::move(next_arrival);
    burned_fraction_ = std::move(next_burned);
    current_time_s_ = end_time_s;

    return diagnostics;
}

} // namespace ERFFire
