#ifndef ERF_ROTHERMEL_MODEL_H
#define ERF_ROTHERMEL_MODEL_H

#include <ERF_RothermelFuel.H>

namespace ERFFire
{

/**
 * Environmental inputs to the scalar Rothermel heading-spread calculation.
 *
 * model_wind_speed_mps is the non-negative wind speed consumed by the
 * Rothermel wind-factor equation. Rothermel (1972) defines that input at
 * midflame height. This behavior module performs no wind-adjustment-factor or
 * ERF-height conversion; upstream Fire-side wind preprocessing must provide
 * the appropriate model wind before this calculation.
 *
 * slope_tangent_magnitude is |rise/run|. Direction/sign is deliberately not
 * collapsed into this scalar behavior object; the later directional-spread
 * layer owns the upslope vector and combines directional effects. When both
 * scalar wind and slope are nonzero here, aligned_heading_ros_mps is the
 * published same-direction scalar amplification, not a cross-slope vector
 * solution.
 */
struct RothermelInputs
{
    amrex::Real dead_fuel_moisture_fraction{};
    amrex::Real model_wind_speed_mps{};
    amrex::Real slope_tangent_magnitude{};
};

/**
 * Audit-oriented diagnostics from the Rothermel calculation.
 *
 * Public dimensional quantities are converted back to SI even though the
 * published Rothermel/Albini correlations are evaluated internally in their
 * native English-unit basis.
 */
struct RothermelResult
{
    amrex::Real net_fuel_loading_kg_m2{};
    amrex::Real bulk_density_kg_m3{};
    amrex::Real packing_ratio{};
    amrex::Real optimum_packing_ratio{};
    amrex::Real reaction_velocity_exponent{};
    amrex::Real max_reaction_velocity_s_inv{};
    amrex::Real reaction_velocity_s_inv{};
    amrex::Real moisture_damping{};
    amrex::Real mineral_damping{};
    amrex::Real reaction_intensity_w_m2{};
    amrex::Real propagating_flux_ratio{};
    amrex::Real effective_heating_number{};
    amrex::Real heat_of_preignition_j_kg{};
    amrex::Real heat_sink_j_m3{};
    amrex::Real no_wind_no_slope_ros_mps{};

    // SI power-law diagnostics for the Rothermel wind factor:
    // Phi_w = wind_factor_coefficient_si * U_mps^wind_factor_exponent.
    amrex::Real wind_factor_coefficient_si{};
    amrex::Real wind_factor_exponent{};

    amrex::Real wind_factor{};
    amrex::Real slope_factor{};
    amrex::Real aligned_heading_ros_mps{};
};

/**
 * Evaluate the single-dead-class Rothermel surface-fire spread model.
 *
 * Equation basis: Rothermel (1972), especially equations 12, 14, 27, 29-31,
 * 36-39, 42, and 47-52, with the Albini (1976) combustible-loading and
 * reaction-velocity-exponent modifications used by the operational
 * computer-based formulation.
 */
[[nodiscard]] RothermelResult evaluate_rothermel(
    const RothermelFuelParameters& fuel,
    const RothermelInputs& inputs);

/**
 * Invert the same Rothermel wind-factor power law used by
 * evaluate_rothermel. The returned SI speed is the model/midflame wind
 * that would produce target_wind_factor for this evaluated fuel bed.
 * No wind-speed cap or WAF policy is applied here.
 */
[[nodiscard]] amrex::Real rothermel_model_wind_speed_for_factor_mps(
    const RothermelResult& result,
    amrex::Real target_wind_factor);

} // namespace ERFFire

#endif // ERF_ROTHERMEL_MODEL_H
