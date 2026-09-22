#ifndef ERF_FIRE_FUEL_COMBUSTION_H_
#define ERF_FIRE_FUEL_COMBUSTION_H_

#include <ERF_FireCombustion.H>
#include <ERF_FireFuelBurnTime.H>
#include <ERF_FireFuelRaster.H>
#include <ERF_RothermelFuel.H>

#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>

#include <stdexcept>

namespace ERFFire
{

/**
 * Result of resolving one spatial fuel cell into the current semi-empirical
 * combustion parameterization.
 *
 * NonBurnable remains a distinct no-combustion material. UnsupportedModel
 * keeps the compatibility FM1 resolver narrow. The production spatial
 * resolver below handles FM2-FM13 with explicit model-dependent burn times.
 */
enum class FireFuelCombustionParameterStatus : int
{
    success = 0,
    invalid_base_parameters = 1,
    nonburnable = 2,
    unsupported_model = 3,
    missing_dead_1h_moisture = 4
};

/**
 * Status for resolving an original-13 cell into the scalar semi-empirical
 * combustion reservoir. Accounting alone does not select the production
 * burn-time policy; the spatial resolver below does so explicitly.
 */
enum class FireFuelCombustionAccountingStatus : int
{
    success = 0,
    invalid_base_parameters = 1,
    nonburnable = 2,
    invalid_model = 3,
    missing_required_moisture = 4,
    overflow_error = 5
};

struct FireFuelCombustionAccounting
{
    FireCombustionParameters parameters{};
    amrex::Real prescribed_water_load_kg_m2{};
};

namespace detail
{

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
fire_combustion_required_moisture_or_zero(
    const FireFuelMoisture& moisture,
    unsigned int required_mask,
    FireFuelMoistureClass component) noexcept
{
    const unsigned int bit =
        fire_fuel_moisture_component_bit(component);
    if ((required_mask & bit) == 0u) {
        return amrex::Real(0);
    }

    amrex::Real value{};
    (void)moisture.try_get(component, value);
    return value;
}

} // namespace detail

/**
 * Resolve an Anderson/Albini material cell into conservative scalar combustion
 * accounting with the caller's explicit burn-time policy.
 *
 * The scalar reservoir cannot retain separate particle classes. Therefore its
 * dry load is the sum of the published 1-h, 10-h, 100-h, and live dry loads,
 * while fuel_moisture_fraction is the dry-mass-weighted prescribed water/dry
 * mass ratio. This preserves both total oven-dry fuel and prescribed initial
 * water exactly at ignition. Sensible heat release per kg dry fuel, exponential
 * burn time, and combustion-water yield are inherited from base; those remain
 * the documented semi-empirical combustion policy rather than being inferred
 * from the Rothermel spread catalog.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
FireFuelCombustionAccountingStatus
try_make_anderson13_fire_combustion_accounting(
    const FireCombustionParameters& base,
    const FireFuelRasterCell& fuel_cell,
    FireFuelCombustionAccounting& result) noexcept
{
    result = {};

    if (!detail::combustion_parameters_valid(base)) {
        return FireFuelCombustionAccountingStatus::
            invalid_base_parameters;
    }

    const auto contract =
        fire_fuel_model_moisture_contract(
            fuel_cell.model_id);
    if (!contract.valid_model) {
        return FireFuelCombustionAccountingStatus::
            invalid_model;
    }
    if (!contract.burnable) {
        return FireFuelCombustionAccountingStatus::
            nonburnable;
    }
    if (!fire_fuel_model_moisture_complete(
            fuel_cell.model_id,
            fuel_cell.moisture)) {
        return FireFuelCombustionAccountingStatus::
            missing_required_moisture;
    }

    Anderson13FuelParameters fuel{};
    const int model_number =
        fire_fuel_model_number(fuel_cell.model_id);
    if (!try_make_anderson13_fuel_parameters(
            model_number,
            fuel)) {
        return FireFuelCombustionAccountingStatus::
            invalid_model;
    }

    const amrex::Real dead_1h_moisture =
        detail::fire_combustion_required_moisture_or_zero(
            fuel_cell.moisture,
            contract.required_mask,
            FireFuelMoistureClass::Dead1h);
    const amrex::Real dead_10h_moisture =
        detail::fire_combustion_required_moisture_or_zero(
            fuel_cell.moisture,
            contract.required_mask,
            FireFuelMoistureClass::Dead10h);
    const amrex::Real dead_100h_moisture =
        detail::fire_combustion_required_moisture_or_zero(
            fuel_cell.moisture,
            contract.required_mask,
            FireFuelMoistureClass::Dead100h);
    const amrex::Real live_moisture =
        detail::fire_combustion_required_moisture_or_zero(
            fuel_cell.moisture,
            contract.required_mask,
            FireFuelMoistureClass::LiveHerbaceous)
        + detail::fire_combustion_required_moisture_or_zero(
            fuel_cell.moisture,
            contract.required_mask,
            FireFuelMoistureClass::LiveWoody);

    const amrex::Real total_dry_load =
        fuel.dead_1h.dry_load_kg_m2
        + fuel.dead_10h.dry_load_kg_m2
        + fuel.dead_100h.dry_load_kg_m2
        + fuel.live_foliage.dry_load_kg_m2;

    const amrex::Real prescribed_water_load =
        fuel.dead_1h.dry_load_kg_m2
            * dead_1h_moisture
        + fuel.dead_10h.dry_load_kg_m2
            * dead_10h_moisture
        + fuel.dead_100h.dry_load_kg_m2
            * dead_100h_moisture
        + fuel.live_foliage.dry_load_kg_m2
            * live_moisture;

    if (!amrex::Math::isfinite(total_dry_load)
        || !(total_dry_load > amrex::Real(0))
        || !amrex::Math::isfinite(prescribed_water_load)
        || prescribed_water_load < amrex::Real(0)) {
        return FireFuelCombustionAccountingStatus::
            overflow_error;
    }

    result.parameters = base;
    result.parameters.dry_fuel_load_kg_m2 =
        total_dry_load;
    result.parameters.fuel_moisture_fraction =
        prescribed_water_load / total_dry_load;
    result.prescribed_water_load_kg_m2 =
        prescribed_water_load;

    if (!amrex::Math::isfinite(
            result.parameters.fuel_moisture_fraction)
        || !detail::combustion_parameters_valid(
            result.parameters)) {
        result = {};
        return FireFuelCombustionAccountingStatus::
            overflow_error;
    }

    return FireFuelCombustionAccountingStatus::success;
}

/** Host convenience wrapper for the conservative original-13 accounting API. */
[[nodiscard]] inline FireFuelCombustionAccounting
make_anderson13_fire_combustion_accounting(
    const FireCombustionParameters& base,
    const FireFuelRasterCell& fuel_cell)
{
    FireFuelCombustionAccounting result{};
    switch (try_make_anderson13_fire_combustion_accounting(
                base,
                fuel_cell,
                result)) {
    case FireFuelCombustionAccountingStatus::success:
        return result;
    case FireFuelCombustionAccountingStatus::invalid_base_parameters:
        throw std::invalid_argument(
            "Fire Anderson-13 combustion base parameters are invalid");
    case FireFuelCombustionAccountingStatus::nonburnable:
        throw std::invalid_argument(
            "Fire Anderson-13 combustion accounting does not apply to NonBurnable material");
    case FireFuelCombustionAccountingStatus::invalid_model:
        throw std::invalid_argument(
            "Fire Anderson-13 combustion accounting has an invalid fuel model");
    case FireFuelCombustionAccountingStatus::missing_required_moisture:
        throw std::invalid_argument(
            "Fire Anderson-13 combustion accounting is missing required prescribed moisture");
    case FireFuelCombustionAccountingStatus::overflow_error:
        throw std::overflow_error(
            "Fire Anderson-13 combustion accounting is not finite");
    }

    throw std::logic_error(
        "Fire Anderson-13 combustion accounting status is invalid");
}

/**
 * Resolve local combustion parameters for one material cell without throwing.
 *
 * This is intentionally narrower than a fuel-model catalog. For FM1 the
 * existing semi-empirical combustion constants remain authoritative except
 * that the local prescribed dead 1-h moisture replaces the legacy global
 * moisture. Live moisture is retained in the material cell for future
 * multi-class fuel models but does not affect FM1 combustion.
 *
 * The output is reset on failure. The function is allocation-free and safe to
 * call inside CUDA/HIP/SYCL cell kernels.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
FireFuelCombustionParameterStatus
try_make_fire_combustion_parameters_for_fuel_cell(
    const FireCombustionParameters& base,
    const FireFuelRasterCell& fuel_cell,
    FireCombustionParameters& result) noexcept
{
    result = {};

    if (!detail::combustion_parameters_valid(base)) {
        return FireFuelCombustionParameterStatus::invalid_base_parameters;
    }

    if (fuel_cell.model_id == FireFuelModelId::NonBurnable) {
        return FireFuelCombustionParameterStatus::nonburnable;
    }

    if (fuel_cell.model_id != FireFuelModelId::FM1) {
        return FireFuelCombustionParameterStatus::unsupported_model;
    }

    amrex::Real dead_1h_moisture{};
    if (!fuel_cell.moisture.try_get(
            FireFuelMoistureClass::Dead1h,
            dead_1h_moisture)) {
        return FireFuelCombustionParameterStatus::missing_dead_1h_moisture;
    }

    result = base;
    result.fuel_moisture_fraction = dead_1h_moisture;

    if (!detail::combustion_parameters_valid(result)) {
        result = {};
        return FireFuelCombustionParameterStatus::invalid_base_parameters;
    }

    return FireFuelCombustionParameterStatus::success;
}

/** Host convenience wrapper with explicit failure messages. */
[[nodiscard]] inline FireCombustionParameters
make_fire_combustion_parameters_for_fuel_cell(
    const FireCombustionParameters& base,
    const FireFuelRasterCell& fuel_cell)
{
    FireCombustionParameters result{};
    const auto status =
        try_make_fire_combustion_parameters_for_fuel_cell(
            base,
            fuel_cell,
            result);

    switch (status) {
    case FireFuelCombustionParameterStatus::success:
        return result;
    case FireFuelCombustionParameterStatus::invalid_base_parameters:
        throw std::invalid_argument(
            "Fire local combustion base parameters are invalid");
    case FireFuelCombustionParameterStatus::nonburnable:
        throw std::invalid_argument(
            "Fire local combustion parameters do not apply to NonBurnable material");
    case FireFuelCombustionParameterStatus::unsupported_model:
        throw std::invalid_argument(
            "Fire local combustion fuel model is not implemented");
    case FireFuelCombustionParameterStatus::missing_dead_1h_moisture:
        throw std::invalid_argument(
            "Fire FM1 local combustion requires prescribed dead 1-h moisture");
    }

    throw std::logic_error(
        "Fire local combustion parameter status is invalid");
}


/**
 * Resolve one spatial fuel cell for the production combustion runtime.
 *
 * FM1 preserves the established ERF-Fire compatibility contract: the base
 * dry load, sensible heat, burn time, and combustion-water yield remain
 * authoritative, while only prescribed dead 1-h moisture varies spatially.
 * FM2-FM13 use the conservative Anderson-13 dry-load and moisture accounting
 * above, but select the fuel-dependent SFIRE burn time, not FM1's grass time.
 * Heat per dry mass and water yield still inherit base. These defaults are
 * semi-empirical, not validation of class-resolved combustion. NonBurnable
 * remains a canonical zero-combustion material.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
FireFuelCombustionAccountingStatus
try_make_spatial_fire_combustion_accounting(
    const FireCombustionParameters& base,
    const FireFuelRasterCell& fuel_cell,
    FireFuelCombustionAccounting& result) noexcept
{
    result = {};

    FireCombustionParameters legacy{};
    const auto legacy_status =
        try_make_fire_combustion_parameters_for_fuel_cell(
            base,
            fuel_cell,
            legacy);

    switch (legacy_status) {
    case FireFuelCombustionParameterStatus::success:
        result.parameters = legacy;
        result.prescribed_water_load_kg_m2 =
            legacy.dry_fuel_load_kg_m2
            * legacy.fuel_moisture_fraction;
        if (!amrex::Math::isfinite(
                result.prescribed_water_load_kg_m2)
            || result.prescribed_water_load_kg_m2
                < amrex::Real(0)) {
            result = {};
            return FireFuelCombustionAccountingStatus::
                overflow_error;
        }
        return FireFuelCombustionAccountingStatus::success;

    case FireFuelCombustionParameterStatus::invalid_base_parameters:
        return FireFuelCombustionAccountingStatus::
            invalid_base_parameters;

    case FireFuelCombustionParameterStatus::nonburnable:
        return FireFuelCombustionAccountingStatus::nonburnable;

    case FireFuelCombustionParameterStatus::missing_dead_1h_moisture:
        return FireFuelCombustionAccountingStatus::
            missing_required_moisture;

    case FireFuelCombustionParameterStatus::unsupported_model:
        break;
    }

    const auto status = try_make_anderson13_fire_combustion_accounting(
        base,
        fuel_cell,
        result);
    if (status == FireFuelCombustionAccountingStatus::success) {
        const amrex::Real tau = anderson13_sfire_burn_time_s(
            fire_fuel_model_number(fuel_cell.model_id));
        if (!(tau > amrex::Real(0))) {
            result = {};
            return FireFuelCombustionAccountingStatus::invalid_model;
        }
        result.parameters.burn_time_constant_s = tau;
    }
    return status;
}

/** Host convenience wrapper for the production spatial accounting policy. */
[[nodiscard]] inline FireFuelCombustionAccounting
make_spatial_fire_combustion_accounting(
    const FireCombustionParameters& base,
    const FireFuelRasterCell& fuel_cell)
{
    FireFuelCombustionAccounting result{};
    switch (try_make_spatial_fire_combustion_accounting(
                base,
                fuel_cell,
                result)) {
    case FireFuelCombustionAccountingStatus::success:
        return result;
    case FireFuelCombustionAccountingStatus::invalid_base_parameters:
        throw std::invalid_argument(
            "Fire spatial combustion base parameters are invalid");
    case FireFuelCombustionAccountingStatus::nonburnable:
        throw std::invalid_argument(
            "Fire spatial combustion accounting does not apply to NonBurnable material");
    case FireFuelCombustionAccountingStatus::invalid_model:
        throw std::invalid_argument(
            "Fire spatial combustion accounting has an invalid fuel model");
    case FireFuelCombustionAccountingStatus::missing_required_moisture:
        throw std::invalid_argument(
            "Fire spatial combustion accounting is missing required prescribed moisture");
    case FireFuelCombustionAccountingStatus::overflow_error:
        throw std::overflow_error(
            "Fire spatial combustion accounting is not finite");
    }

    throw std::logic_error(
        "Fire spatial combustion accounting status is invalid");
}

} // namespace ERFFire

#endif // ERF_FIRE_FUEL_COMBUSTION_H_
