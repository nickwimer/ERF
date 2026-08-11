#include <ERF_RothermelModel.H>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ERFFire
{
namespace
{

constexpr amrex::Real foot_m = 0.3048;
constexpr amrex::Real pound_kg = 0.45359237;
constexpr amrex::Real btu_j = 1055.05585262;
constexpr amrex::Real seconds_per_minute = 60.0;

constexpr amrex::Real kg_m2_to_lb_ft2 =
    (foot_m * foot_m) / pound_kg;
constexpr amrex::Real m_inv_to_ft_inv = foot_m;
constexpr amrex::Real m_to_ft = amrex::Real(1.0) / foot_m;
constexpr amrex::Real j_kg_to_btu_lb = pound_kg / btu_j;
constexpr amrex::Real kg_m3_to_lb_ft3 =
    (foot_m * foot_m * foot_m) / pound_kg;
constexpr amrex::Real mps_to_ft_min =
    seconds_per_minute / foot_m;

constexpr amrex::Real btu_ft2_min_to_w_m2 =
    btu_j / (foot_m * foot_m * seconds_per_minute);
constexpr amrex::Real btu_ft3_to_j_m3 =
    btu_j / (foot_m * foot_m * foot_m);
constexpr amrex::Real btu_lb_to_j_kg = btu_j / pound_kg;
constexpr amrex::Real ft_min_to_mps = foot_m / seconds_per_minute;
constexpr amrex::Real lb_ft2_to_kg_m2 =
    pound_kg / (foot_m * foot_m);
constexpr amrex::Real lb_ft3_to_kg_m3 =
    pound_kg / (foot_m * foot_m * foot_m);

bool
finite (amrex::Real value) noexcept
{
    return std::isfinite(value);
}

void
validate_fuel (const RothermelFuelParameters& fuel)
{
    if (!finite(fuel.dead_1h_load_kg_m2)
        || fuel.dead_1h_load_kg_m2 <= amrex::Real(0.0)) {
        throw std::invalid_argument(
            "Rothermel dead 1-h fuel loading must be finite and positive");
    }

    if (!finite(fuel.dead_1h_sav_m_inv)
        || fuel.dead_1h_sav_m_inv <= amrex::Real(0.0)) {
        throw std::invalid_argument(
            "Rothermel dead 1-h SAV must be finite and positive");
    }

    if (!finite(fuel.fuel_bed_depth_m)
        || fuel.fuel_bed_depth_m <= amrex::Real(0.0)) {
        throw std::invalid_argument(
            "Rothermel fuel-bed depth must be finite and positive");
    }

    if (!finite(fuel.dead_heat_content_j_kg)
        || fuel.dead_heat_content_j_kg <= amrex::Real(0.0)) {
        throw std::invalid_argument(
            "Rothermel heat content must be finite and positive");
    }

    if (!finite(fuel.particle_density_kg_m3)
        || fuel.particle_density_kg_m3 <= amrex::Real(0.0)) {
        throw std::invalid_argument(
            "Rothermel particle density must be finite and positive");
    }

    if (!finite(fuel.total_mineral_fraction)
        || fuel.total_mineral_fraction < amrex::Real(0.0)
        || fuel.total_mineral_fraction >= amrex::Real(1.0)) {
        throw std::invalid_argument(
            "Rothermel total mineral fraction must lie in [0,1)");
    }

    if (!finite(fuel.effective_mineral_fraction)
        || fuel.effective_mineral_fraction <= amrex::Real(0.0)
        || fuel.effective_mineral_fraction > amrex::Real(1.0)) {
        throw std::invalid_argument(
            "Rothermel effective mineral fraction must lie in (0,1]");
    }

    if (!finite(fuel.dead_moisture_of_extinction)
        || fuel.dead_moisture_of_extinction <= amrex::Real(0.0)
        || fuel.dead_moisture_of_extinction > amrex::Real(1.0)) {
        throw std::invalid_argument(
            "Rothermel dead moisture of extinction must lie in (0,1]");
    }
}

void
validate_inputs (const RothermelInputs& inputs)
{
    if (!finite(inputs.dead_fuel_moisture_fraction)
        || inputs.dead_fuel_moisture_fraction < amrex::Real(0.0)) {
        throw std::invalid_argument(
            "Rothermel dead fuel moisture must be finite and non-negative");
    }

    if (!finite(inputs.model_wind_speed_mps)
        || inputs.model_wind_speed_mps < amrex::Real(0.0)) {
        throw std::invalid_argument(
            "Rothermel model wind speed must be finite and non-negative");
    }

    if (!finite(inputs.slope_tangent_magnitude)
        || inputs.slope_tangent_magnitude < amrex::Real(0.0)) {
        throw std::invalid_argument(
            "Rothermel slope tangent magnitude must be finite and non-negative");
    }
}

amrex::Real
moisture_damping (
    amrex::Real moisture,
    amrex::Real moisture_of_extinction) noexcept
{
    const amrex::Real ratio = moisture / moisture_of_extinction;
    if (ratio >= amrex::Real(1.0)) {
        return amrex::Real(0.0);
    }

    const amrex::Real ratio2 = ratio * ratio;
    const amrex::Real ratio3 = ratio2 * ratio;
    return std::clamp(
        amrex::Real(1.0)
            - amrex::Real(2.59) * ratio
            + amrex::Real(5.11) * ratio2
            - amrex::Real(3.52) * ratio3,
        amrex::Real(0.0),
        amrex::Real(1.0));
}

} // namespace

RothermelResult
evaluate_rothermel (
    const RothermelFuelParameters& fuel,
    const RothermelInputs& inputs)
{
    validate_fuel(fuel);
    validate_inputs(inputs);

    // Single audited conversion boundary: evaluate the published correlations
    // in the English units in which their empirical constants were fitted.
    const amrex::Real w0 = fuel.dead_1h_load_kg_m2 * kg_m2_to_lb_ft2;
    const amrex::Real sigma = fuel.dead_1h_sav_m_inv * m_inv_to_ft_inv;
    const amrex::Real depth = fuel.fuel_bed_depth_m * m_to_ft;
    const amrex::Real heat = fuel.dead_heat_content_j_kg * j_kg_to_btu_lb;
    const amrex::Real particle_density =
        fuel.particle_density_kg_m3 * kg_m3_to_lb_ft3;
    const amrex::Real wind_ft_min =
        inputs.model_wind_speed_mps * mps_to_ft_min;

    // Albini (1976) modification 1: the combustible loading excludes the
    // noncombustible total-mineral fraction by multiplication, not division.
    const amrex::Real net_loading =
        w0 * (amrex::Real(1.0) - fuel.total_mineral_fraction);

    const amrex::Real bulk_density = w0 / depth;
    const amrex::Real beta = bulk_density / particle_density;
    const amrex::Real beta_op =
        amrex::Real(3.348) * std::pow(sigma, amrex::Real(-0.8189));
    const amrex::Real beta_ratio = beta / beta_op;

    // Albini (1976) modification 2 to Rothermel equation 39.
    const amrex::Real reaction_exponent =
        amrex::Real(133.0) * std::pow(sigma, amrex::Real(-0.7913));

    const amrex::Real sigma_1p5 = std::pow(sigma, amrex::Real(1.5));
    const amrex::Real gamma_max =
        sigma_1p5
        / (amrex::Real(495.0) + amrex::Real(0.0594) * sigma_1p5);
    const amrex::Real gamma =
        gamma_max
        * std::pow(beta_ratio, reaction_exponent)
        * std::exp(
            reaction_exponent * (amrex::Real(1.0) - beta_ratio));

    const amrex::Real eta_m = moisture_damping(
        inputs.dead_fuel_moisture_fraction,
        fuel.dead_moisture_of_extinction);
    const amrex::Real eta_s = std::min(
        amrex::Real(1.0),
        amrex::Real(0.174)
            * std::pow(
                fuel.effective_mineral_fraction,
                amrex::Real(-0.19)));

    const amrex::Real reaction_intensity =
        gamma * net_loading * heat * eta_m * eta_s;

    const amrex::Real propagating_flux_ratio =
        std::exp(
            (amrex::Real(0.792)
             + amrex::Real(0.681) * std::sqrt(sigma))
            * (beta + amrex::Real(0.1)))
        / (amrex::Real(192.0) + amrex::Real(0.2595) * sigma);

    const amrex::Real effective_heating_number =
        std::exp(amrex::Real(-138.0) / sigma);
    const amrex::Real heat_of_preignition =
        amrex::Real(250.0)
        + amrex::Real(1116.0) * inputs.dead_fuel_moisture_fraction;
    const amrex::Real heat_sink =
        bulk_density * effective_heating_number * heat_of_preignition;

    const amrex::Real r0_ft_min =
        reaction_intensity == amrex::Real(0.0)
        ? amrex::Real(0.0)
        : reaction_intensity * propagating_flux_ratio / heat_sink;

    const amrex::Real wind_c =
        amrex::Real(7.47)
        * std::exp(
            amrex::Real(-0.133)
            * std::pow(sigma, amrex::Real(0.55)));
    const amrex::Real wind_b =
        amrex::Real(0.02526)
        * std::pow(sigma, amrex::Real(0.54));
    const amrex::Real wind_e =
        amrex::Real(0.715)
        * std::exp(amrex::Real(-3.59e-4) * sigma);

    // Equivalent SI representation of the same native-unit wind law.
    const amrex::Real wind_factor_coefficient_si =
        wind_c
        * std::pow(mps_to_ft_min, wind_b)
        * std::pow(beta_ratio, -wind_e);

    const amrex::Real phi_w =
        wind_ft_min == amrex::Real(0.0)
        ? amrex::Real(0.0)
        : wind_c
            * std::pow(wind_ft_min, wind_b)
            * std::pow(beta_ratio, -wind_e);

    const amrex::Real phi_s =
        amrex::Real(5.275)
        * std::pow(beta, amrex::Real(-0.3))
        * inputs.slope_tangent_magnitude
        * inputs.slope_tangent_magnitude;

    // This intentionally implements no legacy wind-speed cap. The direct
    // Rothermel factors are exposed so any later cap/policy can be a separate,
    // explicitly tested behavior decision rather than hidden inside this core.
    const amrex::Real final_ros_ft_min =
        r0_ft_min * (amrex::Real(1.0) + phi_w + phi_s);

    return {
        net_loading * lb_ft2_to_kg_m2,
        bulk_density * lb_ft3_to_kg_m3,
        beta,
        beta_op,
        reaction_exponent,
        gamma_max / seconds_per_minute,
        gamma / seconds_per_minute,
        eta_m,
        eta_s,
        reaction_intensity * btu_ft2_min_to_w_m2,
        propagating_flux_ratio,
        effective_heating_number,
        heat_of_preignition * btu_lb_to_j_kg,
        heat_sink * btu_ft3_to_j_m3,
        r0_ft_min * ft_min_to_mps,
        wind_factor_coefficient_si,
        wind_b,
        phi_w,
        phi_s,
        final_ros_ft_min * ft_min_to_mps
    };
}

amrex::Real
rothermel_model_wind_speed_for_factor_mps (
    const RothermelResult& result,
    amrex::Real target_wind_factor)
{
    if (!finite(target_wind_factor)
        || target_wind_factor < amrex::Real(0.0)) {
        throw std::invalid_argument(
            "Rothermel target wind factor must be finite and non-negative");
    }

    if (!finite(result.wind_factor_coefficient_si)
        || result.wind_factor_coefficient_si <= amrex::Real(0.0)
        || !finite(result.wind_factor_exponent)
        || result.wind_factor_exponent <= amrex::Real(0.0)) {
        throw std::invalid_argument(
            "Rothermel wind-factor inversion parameters are invalid");
    }

    if (target_wind_factor == amrex::Real(0.0)) {
        return amrex::Real(0.0);
    }

    const amrex::Real wind_mps = std::pow(
        target_wind_factor / result.wind_factor_coefficient_si,
        amrex::Real(1.0) / result.wind_factor_exponent);

    if (!finite(wind_mps)) {
        throw std::overflow_error(
            "Rothermel equivalent model wind is not finite");
    }

    return wind_mps;
}

} // namespace ERFFire
