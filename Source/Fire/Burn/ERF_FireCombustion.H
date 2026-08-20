#ifndef ERF_FIRE_COMBUSTION_H_
#define ERF_FIRE_COMBUSTION_H_

#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <limits>

namespace ERFFire
{

struct FireCombustionParameters
{
    amrex::Real dry_fuel_load_kg_m2{};
    amrex::Real sensible_heat_release_j_kg_dry{};
    amrex::Real fuel_moisture_fraction{};
    amrex::Real burn_time_constant_s{};
    amrex::Real combustion_water_yield_kg_per_kg_dry{
        amrex::Real(0.56)};
};

struct FireCombustionState
{
    amrex::Real ignited_area_fraction{};
    amrex::Real remaining_dry_fuel_kg_m2{};
    amrex::Real consumed_dry_fuel_kg_m2{};
    amrex::Real sensible_energy_j_m2{};
    amrex::Real water_released_kg_m2{};
};

struct FireCombustionAdvance
{
    FireCombustionState state{};
    amrex::Real newly_consumed_dry_fuel_kg_m2{};
    amrex::Real sensible_energy_increment_j_m2{};
    amrex::Real water_released_increment_kg_m2{};
};

enum class FireCombustionStatus : int
{
    success = 0,
    invalid_argument = 1,
    overflow_error = 2
};

namespace detail
{

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
combustion_scaled_tolerance(amrex::Real scale) noexcept
{
    const amrex::Real magnitude = amrex::Math::abs(scale);
    const amrex::Real bounded =
        magnitude > amrex::Real(1) ? magnitude : amrex::Real(1);
    return amrex::Real(512)
        * std::numeric_limits<amrex::Real>::epsilon()
        * bounded;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool
combustion_parameters_valid(
    const FireCombustionParameters& p) noexcept
{
    return amrex::Math::isfinite(p.dry_fuel_load_kg_m2)
        && p.dry_fuel_load_kg_m2 > amrex::Real(0)
        && amrex::Math::isfinite(p.sensible_heat_release_j_kg_dry)
        && p.sensible_heat_release_j_kg_dry > amrex::Real(0)
        && amrex::Math::isfinite(p.fuel_moisture_fraction)
        && p.fuel_moisture_fraction >= amrex::Real(0)
        && amrex::Math::isfinite(p.burn_time_constant_s)
        && p.burn_time_constant_s > amrex::Real(0)
        && amrex::Math::isfinite(
            p.combustion_water_yield_kg_per_kg_dry)
        && p.combustion_water_yield_kg_per_kg_dry
            >= amrex::Real(0);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool
combustion_state_valid(
    const FireCombustionState& s,
    const FireCombustionParameters& p) noexcept
{
    if (!combustion_parameters_valid(p)
        || !amrex::Math::isfinite(s.ignited_area_fraction)
        || s.ignited_area_fraction < amrex::Real(0)
        || s.ignited_area_fraction > amrex::Real(1)
        || !amrex::Math::isfinite(s.remaining_dry_fuel_kg_m2)
        || s.remaining_dry_fuel_kg_m2 < amrex::Real(0)
        || !amrex::Math::isfinite(s.consumed_dry_fuel_kg_m2)
        || s.consumed_dry_fuel_kg_m2 < amrex::Real(0)
        || !amrex::Math::isfinite(s.sensible_energy_j_m2)
        || s.sensible_energy_j_m2 < amrex::Real(0)
        || !amrex::Math::isfinite(s.water_released_kg_m2)
        || s.water_released_kg_m2 < amrex::Real(0)) {
        return false;
    }

    const amrex::Real expected_mass =
        s.ignited_area_fraction * p.dry_fuel_load_kg_m2;
    const amrex::Real represented_mass =
        s.remaining_dry_fuel_kg_m2
        + s.consumed_dry_fuel_kg_m2;
    if (amrex::Math::abs(expected_mass - represented_mass)
        > combustion_scaled_tolerance(expected_mass)) {
        return false;
    }

    const amrex::Real expected_energy =
        s.consumed_dry_fuel_kg_m2
        * p.sensible_heat_release_j_kg_dry;
    if (amrex::Math::abs(
            expected_energy - s.sensible_energy_j_m2)
        > combustion_scaled_tolerance(expected_energy)) {
        return false;
    }

    const amrex::Real expected_water =
        s.consumed_dry_fuel_kg_m2
        * (p.fuel_moisture_fraction
           + p.combustion_water_yield_kg_per_kg_dry);
    return amrex::Math::abs(
               expected_water - s.water_released_kg_m2)
        <= combustion_scaled_tolerance(expected_water);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
FireCombustionState
add_fire_combustion_ignition_unchecked(
    const FireCombustionState& state,
    const FireCombustionParameters& p,
    amrex::Real newly_ignited_area_fraction) noexcept
{
    const amrex::Real next_fraction =
        state.ignited_area_fraction
        + newly_ignited_area_fraction;

    FireCombustionState next = state;
    next.ignited_area_fraction =
        amrex::Real(1) < next_fraction
            ? amrex::Real(1)
            : next_fraction;
    next.remaining_dry_fuel_kg_m2 +=
        newly_ignited_area_fraction
        * p.dry_fuel_load_kg_m2;
    return next;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
FireCombustionAdvance
advance_fire_combustion_unchecked(
    const FireCombustionState& state,
    const FireCombustionParameters& p,
    amrex::Real dt_s) noexcept
{
    const amrex::Real exponent =
        -dt_s / p.burn_time_constant_s;
    const amrex::Real newly_consumed =
        -state.remaining_dry_fuel_kg_m2
        * std::expm1(exponent);
    const amrex::Real remaining =
        state.remaining_dry_fuel_kg_m2
        - newly_consumed;
    const amrex::Real energy_increment =
        newly_consumed
        * p.sensible_heat_release_j_kg_dry;
    const amrex::Real water_release_kg_per_kg_dry =
        p.fuel_moisture_fraction
        + p.combustion_water_yield_kg_per_kg_dry;
    const amrex::Real water_increment =
        newly_consumed
        * water_release_kg_per_kg_dry;

    FireCombustionState next = state;
    next.remaining_dry_fuel_kg_m2 =
        remaining < amrex::Real(0)
            ? amrex::Real(0)
            : remaining;
    next.consumed_dry_fuel_kg_m2 += newly_consumed;

    // Sensible energy and released water are algebraically
    // derived from cumulative consumed dry fuel.  Reconstruct
    // them after every positive-time advance instead of
    // independently accumulating roundoff in redundant state.
    // Preserve exact zero-dt idempotence.
    if (dt_s > amrex::Real(0)) {
        next.sensible_energy_j_m2 =
            next.consumed_dry_fuel_kg_m2
            * p.sensible_heat_release_j_kg_dry;
        next.water_released_kg_m2 =
            next.consumed_dry_fuel_kg_m2
            * water_release_kg_per_kg_dry;
    }

    return {
        next,
        newly_consumed,
        energy_increment,
        water_increment};
}

} // namespace detail

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
FireCombustionStatus
try_add_fire_combustion_ignition(
    const FireCombustionState& state,
    const FireCombustionParameters& p,
    amrex::Real newly_ignited_area_fraction,
    FireCombustionState& result) noexcept
{
    result = {};
    if (!detail::combustion_state_valid(state, p)
        || !amrex::Math::isfinite(newly_ignited_area_fraction)
        || newly_ignited_area_fraction < amrex::Real(0)
        || newly_ignited_area_fraction > amrex::Real(1)) {
        return FireCombustionStatus::invalid_argument;
    }

    const amrex::Real next_fraction =
        state.ignited_area_fraction
        + newly_ignited_area_fraction;
    if (next_fraction
        > amrex::Real(1)
            + detail::combustion_scaled_tolerance(
                amrex::Real(1))) {
        return FireCombustionStatus::invalid_argument;
    }

    result = detail::add_fire_combustion_ignition_unchecked(
        state, p, newly_ignited_area_fraction);
    if (!amrex::Math::isfinite(
            result.remaining_dry_fuel_kg_m2)) {
        result = {};
        return FireCombustionStatus::overflow_error;
    }
    if (!detail::combustion_state_valid(result, p)) {
        result = {};
        return FireCombustionStatus::invalid_argument;
    }
    return FireCombustionStatus::success;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
FireCombustionStatus
try_advance_fire_combustion(
    const FireCombustionState& state,
    const FireCombustionParameters& p,
    amrex::Real dt_s,
    FireCombustionAdvance& result) noexcept
{
    result = {};
    if (!detail::combustion_state_valid(state, p)
        || !amrex::Math::isfinite(dt_s)
        || dt_s < amrex::Real(0)) {
        return FireCombustionStatus::invalid_argument;
    }

    result = detail::advance_fire_combustion_unchecked(
        state, p, dt_s);
    if (!amrex::Math::isfinite(
            result.state.remaining_dry_fuel_kg_m2)
        || !amrex::Math::isfinite(
            result.state.consumed_dry_fuel_kg_m2)
        || !amrex::Math::isfinite(
            result.state.sensible_energy_j_m2)
        || !amrex::Math::isfinite(
            result.state.water_released_kg_m2)) {
        result = {};
        return FireCombustionStatus::overflow_error;
    }
    if (!detail::combustion_state_valid(result.state, p)) {
        result = {};
        return FireCombustionStatus::invalid_argument;
    }
    return FireCombustionStatus::success;
}

/**
 * Convert a fuel-burn calibration weight to the corresponding exponential
 * 1/e burn time. The scaling is similar to the WRF-Fire/SFIRE implementation:
 *
 *   fuel_time_s = weight / 0.85
 *
 * weight=1000 gives approximately a 40 percent fuel-mass decrease in 600 s.
 */
[[nodiscard]] amrex::Real
fire_burn_time_constant_from_weight(amrex::Real weight);

/**
 * Controlled FM1 combustion parameters for the first combustion checkpoint.
 *
 * Dry loading reuses ERF-Fire's validated FM1 fuel definition. Combustion
 * sensible heat release (17.433 MJ/kg dry fuel), exponential burn-time
 * calibration weight 7, and combustion-water yield 0.56 kg/kg dry fuel are
 * selected with guidance from the WRF-Fire/SFIRE heat-flux and fuel-time
 * treatment.
 *
 * The combustion sensible-heat constant is intentionally distinct from the
 * 8000 Btu/lb Rothermel heat-content parameter used by the spread model.
 */
[[nodiscard]] FireCombustionParameters
make_fm1_combustion_parameters(
    amrex::Real dead_fuel_moisture_fraction);

/**
 * Add an instantaneous newly ignited represented-area fraction.
 *
 * Ignition adds oven-dry fuel to the active reservoir but consumes none
 * instantaneously. The caller owns the physical ignition timing.
 */
[[nodiscard]] FireCombustionState
add_fire_combustion_ignition(
    const FireCombustionState& state,
    const FireCombustionParameters& parameters,
    amrex::Real newly_ignited_area_fraction);

/**
 * Advance all currently ignited fuel by exact exponential decay.
 *
 * This state deliberately contains no first-arrival clock. Partial-area
 * ignition events are supplied explicitly by the later raster/runtime layer.
 */
[[nodiscard]] FireCombustionAdvance
advance_fire_combustion(
    const FireCombustionState& state,
    const FireCombustionParameters& parameters,
    amrex::Real dt_s);

} // namespace ERFFire

#endif
