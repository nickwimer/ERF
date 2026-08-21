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
 * DirectReference is the compatibility/control mode: horizontal wind sampled
 * at reference_height_agl_m is used unchanged as the Rothermel model/midflame
 * wind. ExplicitWaf20ft samples the horizontal reference wind at exactly
 * 20 ft = 6.096 m local AGL and applies wind_adjustment_factor to the complete
 * vector before Rothermel/Richards evaluation.
 */
enum class ERFFireWindMode
{
    DirectReference,
    ExplicitWaf20ft
};

inline constexpr amrex::Real explicit_waf_20ft_reference_height_agl_m =
    amrex::Real(6.096);

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

    // DirectReference-only AGL sampling height for the ERF horizontal
    // reference wind. It does not redefine the explicit 20-ft WAF height.
    amrex::Real reference_height_agl_m{};

    // ExplicitWaf20ft-only scalar mapping from the sampled 20-ft reference
    // vector to the Rothermel model/midflame-wind vector.
    amrex::Real wind_adjustment_factor{amrex::Real(1.0)};

    std::string fuel_model{"FM1"};
    amrex::Real dead_fuel_moisture_fraction{};

    // Optional Fire horizontal cell counts over the same physical domain as
    // atmospheric level 0. A zero pair preserves the legacy 1:1 L0 grid.
    int n_cell_x{0};
    int n_cell_y{0};

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
    // Diagnostic/development CSV snapshots only. Zero disables them; native
    // ERF plotfiles/checkpoints are the production science/restart paths.
    int output_interval_steps{0};
};

} // namespace ERFFire

#endif
