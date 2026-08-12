#ifndef ERF_FIRE_COMBUSTION_H_
#define ERF_FIRE_COMBUSTION_H_

#include <AMReX_REAL.H>

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
