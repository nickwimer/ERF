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

enum class ERFFireWindMode
{
    DirectReference
};

/**
 * ERF-facing runtime configuration for the flat-grid Fire checkpoint.
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

    std::string output_dir{"fire_output"};
    int output_interval_steps{1};
};

} // namespace ERFFire

#endif
