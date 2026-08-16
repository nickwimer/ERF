#include "ERF_FireCombustion.H"

#include <ERF_RothermelFuel.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ERFFire
{
namespace
{

constexpr amrex::Real sfire_sensible_heat_release_j_kg_dry =
    amrex::Real(17.433e6);
constexpr amrex::Real sfire_burn_weight_rate_per_s =
    amrex::Real(0.85);

void require(bool condition, const char* message)
{
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

amrex::Real scaled_tolerance(amrex::Real scale)
{
    return amrex::Real(512)
        * std::numeric_limits<amrex::Real>::epsilon()
        * std::max(amrex::Real(1), std::abs(scale));
}

void validate_parameters(const FireCombustionParameters& p)
{
    require(
        std::isfinite(p.dry_fuel_load_kg_m2)
            && p.dry_fuel_load_kg_m2 > amrex::Real(0),
        "fire combustion dry fuel load must be finite and positive");
    require(
        std::isfinite(p.sensible_heat_release_j_kg_dry)
            && p.sensible_heat_release_j_kg_dry > amrex::Real(0),
        "fire combustion sensible heat release must be finite and positive");
    require(
        std::isfinite(p.fuel_moisture_fraction)
            && p.fuel_moisture_fraction >= amrex::Real(0),
        "fire combustion fuel moisture must be finite and nonnegative");
    require(
        std::isfinite(p.burn_time_constant_s)
            && p.burn_time_constant_s > amrex::Real(0),
        "fire combustion burn time must be finite and positive");
    require(
        std::isfinite(p.combustion_water_yield_kg_per_kg_dry)
            && p.combustion_water_yield_kg_per_kg_dry >= amrex::Real(0),
        "fire combustion water yield must be finite and nonnegative");
}

void validate_state(
    const FireCombustionState& s,
    const FireCombustionParameters& p)
{
    validate_parameters(p);

    require(
        std::isfinite(s.ignited_area_fraction)
            && s.ignited_area_fraction >= amrex::Real(0)
            && s.ignited_area_fraction <= amrex::Real(1),
        "fire combustion ignited area fraction must be finite in [0,1]");
    require(
        std::isfinite(s.remaining_dry_fuel_kg_m2)
            && s.remaining_dry_fuel_kg_m2 >= amrex::Real(0),
        "fire combustion remaining dry fuel must be finite and nonnegative");
    require(
        std::isfinite(s.consumed_dry_fuel_kg_m2)
            && s.consumed_dry_fuel_kg_m2 >= amrex::Real(0),
        "fire combustion consumed dry fuel must be finite and nonnegative");
    require(
        std::isfinite(s.sensible_energy_j_m2)
            && s.sensible_energy_j_m2 >= amrex::Real(0),
        "fire combustion sensible energy must be finite and nonnegative");
    require(
        std::isfinite(s.water_released_kg_m2)
            && s.water_released_kg_m2 >= amrex::Real(0),
        "fire combustion released water must be finite and nonnegative");

    const amrex::Real expected_mass =
        s.ignited_area_fraction * p.dry_fuel_load_kg_m2;
    const amrex::Real represented_mass =
        s.remaining_dry_fuel_kg_m2 + s.consumed_dry_fuel_kg_m2;

    require(
        std::abs(expected_mass - represented_mass)
            <= scaled_tolerance(expected_mass),
        "fire combustion state violates dry-fuel mass accounting");

    const amrex::Real expected_energy =
        s.consumed_dry_fuel_kg_m2 * p.sensible_heat_release_j_kg_dry;
    require(
        std::abs(expected_energy - s.sensible_energy_j_m2)
            <= scaled_tolerance(expected_energy),
        "fire combustion state violates sensible-energy accounting");

    const amrex::Real expected_water =
        s.consumed_dry_fuel_kg_m2
        * (p.fuel_moisture_fraction
           + p.combustion_water_yield_kg_per_kg_dry);
    require(
        std::abs(expected_water - s.water_released_kg_m2)
            <= scaled_tolerance(expected_water),
        "fire combustion state violates water accounting");
}

} // namespace

amrex::Real
fire_burn_time_constant_from_weight(amrex::Real weight)
{
    require(
        std::isfinite(weight) && weight > amrex::Real(0),
        "fire combustion fuel weight must be finite and positive");

    const amrex::Real result =
        weight / sfire_burn_weight_rate_per_s;

    if (!std::isfinite(result) || !(result > amrex::Real(0))) {
        throw std::overflow_error(
            "fire combustion burn-time conversion is not finite and positive");
    }
    return result;
}

FireCombustionParameters
make_fm1_combustion_parameters(
    amrex::Real dead_fuel_moisture_fraction)
{
    require(
        std::isfinite(dead_fuel_moisture_fraction)
            && dead_fuel_moisture_fraction >= amrex::Real(0),
        "FM1 combustion fuel moisture must be finite and nonnegative");

    const auto fuel = make_fm1_fuel_parameters();

    FireCombustionParameters p{
        fuel.dead_1h_load_kg_m2,
        sfire_sensible_heat_release_j_kg_dry,
        dead_fuel_moisture_fraction,
        fire_burn_time_constant_from_weight(amrex::Real(7)),
        amrex::Real(0.56)};

    validate_parameters(p);
    return p;
}

FireCombustionState
add_fire_combustion_ignition(
    const FireCombustionState& state,
    const FireCombustionParameters& p,
    amrex::Real newly_ignited_area_fraction)
{
    validate_state(state, p);

    require(
        std::isfinite(newly_ignited_area_fraction)
            && newly_ignited_area_fraction >= amrex::Real(0)
            && newly_ignited_area_fraction <= amrex::Real(1),
        "newly ignited fire area fraction must be finite in [0,1]");

    const amrex::Real next_fraction =
        state.ignited_area_fraction + newly_ignited_area_fraction;

    require(
        next_fraction <= amrex::Real(1)
            + scaled_tolerance(amrex::Real(1)),
        "cumulative ignited fire area fraction exceeds one");

    FireCombustionState next =
        detail::add_fire_combustion_ignition_unchecked(
            state,
            p,
            newly_ignited_area_fraction);

    if (!std::isfinite(next.remaining_dry_fuel_kg_m2)) {
        throw std::overflow_error(
            "fire combustion ignition dry-fuel mass is not finite");
    }

    validate_state(next, p);
    return next;
}

FireCombustionAdvance
advance_fire_combustion(
    const FireCombustionState& state,
    const FireCombustionParameters& p,
    amrex::Real dt_s)
{
    validate_state(state, p);

    require(
        std::isfinite(dt_s) && dt_s >= amrex::Real(0),
        "fire combustion dt must be finite and nonnegative");

    const FireCombustionAdvance result =
        detail::advance_fire_combustion_unchecked(
            state,
            p,
            dt_s);

    if (!std::isfinite(result.state.remaining_dry_fuel_kg_m2)
        || !std::isfinite(result.state.consumed_dry_fuel_kg_m2)
        || !std::isfinite(result.state.sensible_energy_j_m2)
        || !std::isfinite(result.state.water_released_kg_m2)) {
        throw std::overflow_error(
            "fire combustion advance produced non-finite accounting");
    }

    validate_state(result.state, p);
    return result;
}

} // namespace ERFFire
