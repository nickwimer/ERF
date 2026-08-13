#ifndef ERF_FIRE_RUNTIME_OPTIONS_H_
#define ERF_FIRE_RUNTIME_OPTIONS_H_

#include <AMReX_REAL.H>

#include <string>

namespace ERFFire
{

enum class ERFFireCouplingMode
{
    OneWay,
    TwoWay
};

/**
 * ERF-to-Fire wind-input policy.
 *
 * DirectReference is the existing compatibility/control mode: horizontal wind
 * sampled at reference_height_agl_m is used unchanged as the Rothermel
 * model/midflame wind. It is intentionally not a WAF model.
 */
enum class ERFFireWindMode
{
    DirectReference
};

/**
 * ERF-facing runtime configuration for level-0 Fire coupling.
 *
 * Coupling mode is driver policy. ERFFireSpreadRuntime remains coupling
 * neutral so the exact same fire evolution path can later serve one-way and
 * two-way simulations.
 */
struct ERFFireRuntimeOptions
{
    bool enabled{false};
    ERFFireCouplingMode coupling_mode{ERFFireCouplingMode::OneWay};
    ERFFireWindMode wind_mode{ERFFireWindMode::DirectReference};

    // User-configured AGL sampling height for the ERF horizontal reference
    // wind. No 20-ft, 10-m, or midflame standard is implied by this field.
    amrex::Real reference_height_agl_m{};

    std::string fuel_model{"FM1"};
    amrex::Real dead_fuel_moisture_fraction{};

    amrex::Real ignition_center_x_m{};
    amrex::Real ignition_center_y_m{};
    amrex::Real ignition_radius_m{};
    int ignition_vertex_count{64};

    amrex::Real remesh_min_edge_length_m{};
    amrex::Real remesh_max_edge_length_m{};
    amrex::Real remesh_max_chord_error_m{};

    amrex::Real arrival_time_tolerance_s{amrex::Real(1.0e-7)};
    int combustion_temporal_substeps{16};
    amrex::Real feedback_extinction_depth_m{amrex::Real(50.0)};

    std::string output_dir{"fire_output"};
    int output_interval_steps{1};
};

} // namespace ERFFire

#endif
