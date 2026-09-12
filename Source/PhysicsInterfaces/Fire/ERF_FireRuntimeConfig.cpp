#include <ERF_FireRuntimeConfig.H>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Vector.H>

#include <cmath>
#include <string>

namespace ERFFire
{

ERFFireRuntimeOptions
read_erf_fire_runtime_options(
    const ERFFireHostCapabilities& host)
{
    using amrex::Error;
    using amrex::ParmParse;
    using amrex::Real;
    using amrex::Vector;

    ERFFireRuntimeOptions options{};

    ParmParse pp_fire("fire");

    bool legacy_environment_read = false;
    if (pp_fire.query(
            "environment_read",
            legacy_environment_read)) {
        Error(
            "fire.environment_read is obsolete; use "
            "fire.enabled and fire.coupling_mode");
    }

    pp_fire.query(
        "enabled",
        options.enabled);

    if (!options.enabled) {
        return options;
    }

    if (host.max_level != 0) {
        Error(
            "Fire runtime currently requires amr.max_level = 0");
    }

    std::string coupling_mode;
    if (!pp_fire.query("coupling_mode", coupling_mode)) {
        Error(
            "fire.enabled requires fire.coupling_mode");
    }

    if (coupling_mode == "one_way") {
        options.coupling_mode =
            ERFFireCouplingMode::OneWay;
    } else if (coupling_mode == "two_way") {
        options.coupling_mode =
            ERFFireCouplingMode::TwoWay;
    } else {
        Error(
            "fire.coupling_mode must be one_way or two_way");
    }

    if (host.variable_dz
        && !host.static_fitted_mesh) {
        Error(
            "Fire VariableDz coupling requires "
            "erf.terrain_type = StaticFittedMesh");
    }

    if (options.coupling_mode
            == ERFFireCouplingMode::TwoWay) {
        if (!host.moist_no_condensation) {
            Error(
                "Fire.coupling_mode = two_way requires "
                "erf.moisture_model = MoistNoCondensation");
        }

        if (host.anelastic_level0) {
            Error(
                "Fire.coupling_mode = two_way currently requires "
                "compressible ERF");
        }
    }

    std::string wind_mode;
    if (!pp_fire.query("wind_mode", wind_mode)) {
        Error(
            "fire.enabled requires fire.wind_mode");
    }

    if (wind_mode == "direct_reference") {
        options.wind_mode =
            ERFFireWindMode::DirectReference;

        if (!pp_fire.query(
                "reference_height_agl_m",
                options.reference_height_agl_m)) {
            Error(
                "fire.wind_mode = direct_reference requires "
                "fire.reference_height_agl_m");
        }

        Real unused_wind_adjustment_factor{};
        if (pp_fire.query(
                "wind_adjustment_factor",
                unused_wind_adjustment_factor)) {
            Error(
                "fire.wind_adjustment_factor is valid only with "
                "fire.wind_mode = explicit_waf_20ft");
        }
    } else if (wind_mode == "explicit_waf_20ft") {
        options.wind_mode =
            ERFFireWindMode::ExplicitWaf20ft;

        if (!pp_fire.query(
                "wind_adjustment_factor",
                options.wind_adjustment_factor)) {
            Error(
                "fire.wind_mode = explicit_waf_20ft requires "
                "fire.wind_adjustment_factor");
        }

        Real unused_reference_height_agl_m{};
        if (pp_fire.query(
                "reference_height_agl_m",
                unused_reference_height_agl_m)) {
            Error(
                "fire.reference_height_agl_m is valid only with "
                "fire.wind_mode = direct_reference; explicit_waf_20ft "
                "always samples 20 ft = 6.096 m local AGL");
        }
    } else {
        Error(
            "fire.wind_mode must be direct_reference or "
            "explicit_waf_20ft");
    }

    if (!pp_fire.query(
            "fuel_model",
            options.fuel_model)) {
        Error(
            "fire.enabled requires fire.fuel_model");
    }

    if (options.fuel_model != "FM1") {
        Error(
            "Supports only fire.fuel_model = FM1");
    }

    if (!pp_fire.query(
            "dead_fuel_moisture_fraction",
            options.dead_fuel_moisture_fraction)) {
        Error(
            "fire.enabled requires fire.dead_fuel_moisture_fraction");
    }

    if (pp_fire.contains("n_cell")) {
        Vector<int> fire_n_cell(2);

        if (pp_fire.countval("n_cell") != 2
            || !pp_fire.queryarr(
                "n_cell",
                fire_n_cell,
                0,
                2)) {
            Error(
                "fire.n_cell must contain exactly two horizontal cell counts");
        }

        options.n_cell_x = fire_n_cell[0];
        options.n_cell_y = fire_n_cell[1];
    }

    Vector<Real> ignition_center(2);
    if (!pp_fire.queryarr(
            "ignition_center_m",
            ignition_center,
            0,
            2)) {
        Error(
            "fire.enabled requires two values in fire.ignition_center_m");
    }

    options.ignition_center_x_m =
        ignition_center[0];
    options.ignition_center_y_m =
        ignition_center[1];

    if (!pp_fire.query(
            "ignition_radius_m",
            options.ignition_radius_m)) {
        Error(
            "fire.enabled requires fire.ignition_radius_m");
    }

    pp_fire.query(
        "ignition_vertex_count",
        options.ignition_vertex_count);

    if (!pp_fire.query(
            "remesh_min_edge_length_m",
            options.remesh_min_edge_length_m)
        || !pp_fire.query(
            "remesh_max_edge_length_m",
            options.remesh_max_edge_length_m)
        || !pp_fire.query(
            "remesh_max_chord_error_m",
            options.remesh_max_chord_error_m)) {
        Error(
            "fire.enabled requires remesh_min_edge_length_m, "
            "remesh_max_edge_length_m, and remesh_max_chord_error_m");
    }

    pp_fire.query(
        "arrival_time_tolerance_s",
        options.arrival_time_tolerance_s);

    pp_fire.query(
        "combustion_temporal_substeps",
        options.combustion_temporal_substeps);

    pp_fire.query(
        "feedback_extinction_depth_m",
        options.feedback_extinction_depth_m);

    pp_fire.query(
        "output_dir",
        options.output_dir);

    pp_fire.query(
        "output_interval_steps",
        options.output_interval_steps);

    const auto finite_nonnegative = [] (Real value) {
        return std::isfinite(value)
            && value >= Real(0.0);
    };

    const auto finite_positive = [] (Real value) {
        return std::isfinite(value)
            && value > Real(0.0);
    };

    if (options.wind_mode
            == ERFFireWindMode::DirectReference) {
        if (!finite_nonnegative(
                options.reference_height_agl_m)) {
            Error(
                "fire.reference_height_agl_m must be finite and nonnegative");
        }
    } else {
        if (!std::isfinite(
                options.wind_adjustment_factor)
            || options.wind_adjustment_factor
                < Real(0.0)
            || options.wind_adjustment_factor
                > Real(1.0)) {
            Error(
                "fire.wind_adjustment_factor must be finite and in [0,1]");
        }
    }

    if (!finite_nonnegative(
            options.dead_fuel_moisture_fraction)) {
        Error(
            "fire.dead_fuel_moisture_fraction must be finite and nonnegative");
    }

    if ((options.n_cell_x != 0
            || options.n_cell_y != 0)
        && (options.n_cell_x <= 0
            || options.n_cell_y <= 0)) {
        Error(
            "fire.n_cell must contain two positive horizontal cell counts");
    }

    if (!std::isfinite(
            options.ignition_center_x_m)
        || !std::isfinite(
            options.ignition_center_y_m)
        || !finite_positive(
            options.ignition_radius_m)) {
        Error(
            "fire ignition center must be finite and radius must be positive");
    }

    if (options.ignition_vertex_count < 8) {
        Error(
            "fire.ignition_vertex_count must be at least 8");
    }

    if (!finite_positive(
            options.remesh_min_edge_length_m)
        || !finite_positive(
            options.remesh_max_edge_length_m)
        || options.remesh_min_edge_length_m
            > Real(0.5)
                * options.remesh_max_edge_length_m
        || !finite_nonnegative(
            options.remesh_max_chord_error_m)) {
        Error(
            "invalid fire remesh length/chord-error configuration");
    }

    if (!finite_positive(
            options.arrival_time_tolerance_s)) {
        Error(
            "fire.arrival_time_tolerance_s must be finite and positive");
    }

    if (options.combustion_temporal_substeps <= 0) {
        Error(
            "fire.combustion_temporal_substeps must be positive");
    }

    if (!finite_positive(
            options.feedback_extinction_depth_m)) {
        Error(
            "fire.feedback_extinction_depth_m must be finite and positive");
    }

    if (options.output_interval_steps < 0) {
        Error(
            "fire.output_interval_steps must be nonnegative; "
            "zero disables diagnostic CSV snapshots");
    }

    if (options.output_interval_steps > 0
        && options.output_dir.empty()) {
        Error(
            "fire.output_dir must not be empty when diagnostic CSV snapshots are enabled");
    }

    return options;
}

} // namespace ERFFire
