#ifndef ERF_FIRE_FUEL_SPREAD_H_
#define ERF_FIRE_FUEL_SPREAD_H_

#include <ERF_FireFuelModel.H>
#include <ERF_RothermelModel.H>

#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>

#include <stdexcept>

namespace ERFFire
{

/** Resolution status for one categorical material into Rothermel inputs. */
enum class FireFuelSpreadInputStatus : int
{
    success = 0,
    nonburnable = 1,
    invalid_model = 2,
    missing_required_moisture = 3
};

struct FireFuelSpreadInputs
{
    int anderson13_model_number{};
    RothermelMulticlassInputs rothermel{};
};

/**
 * Resolve one categorical material into original-13 multi-class inputs.
 *
 * This operation is catalog-parameter-free: it maps only stable model identity,
 * prescribed moisture channels, wind, and slope. The caller may then obtain
 * the immutable Anderson/Albini catalog parameters by model number and evaluate
 * the validated multi-class Rothermel equations.
 *
 * The function deliberately returns NonBurnable separately and never maps it
 * to zero ROS. Missing required moisture is distinct from numerical zero
 * moisture. Extra prescribed channels are ignored.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
FireFuelSpreadInputStatus
try_make_fire_fuel_spread_inputs(
    FireFuelModelId model_id,
    const FireFuelMoisture& moisture,
    amrex::Real model_wind_speed_mps,
    amrex::Real slope_tangent_magnitude,
    FireFuelSpreadInputs& result) noexcept
{
    result = {};

    const auto contract =
        fire_fuel_model_moisture_contract(model_id);
    if (!contract.valid_model) {
        return FireFuelSpreadInputStatus::invalid_model;
    }
    if (!contract.burnable) {
        return FireFuelSpreadInputStatus::nonburnable;
    }
    if (!fire_fuel_model_moisture_complete(
            model_id, moisture)) {
        return FireFuelSpreadInputStatus::missing_required_moisture;
    }

    const auto required_value =
        [&moisture, required_mask = contract.required_mask]
        AMREX_GPU_HOST_DEVICE (
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
        };

    result.anderson13_model_number =
        fire_fuel_model_number(model_id);
    result.rothermel = {
        required_value(FireFuelMoistureClass::Dead1h),
        required_value(FireFuelMoistureClass::Dead10h),
        required_value(FireFuelMoistureClass::Dead100h),
        required_value(FireFuelMoistureClass::LiveHerbaceous)
            + required_value(FireFuelMoistureClass::LiveWoody),
        model_wind_speed_mps,
        slope_tangent_magnitude
    };

    return FireFuelSpreadInputStatus::success;
}

/** Host convenience wrapper with explicit material-contract failures. */
[[nodiscard]] inline FireFuelSpreadInputs
make_fire_fuel_spread_inputs(
    FireFuelModelId model_id,
    const FireFuelMoisture& moisture,
    amrex::Real model_wind_speed_mps,
    amrex::Real slope_tangent_magnitude)
{
    FireFuelSpreadInputs result{};
    switch (try_make_fire_fuel_spread_inputs(
                model_id,
                moisture,
                model_wind_speed_mps,
                slope_tangent_magnitude,
                result)) {
    case FireFuelSpreadInputStatus::success:
        return result;
    case FireFuelSpreadInputStatus::nonburnable:
        throw std::invalid_argument(
            "Fire spread inputs cannot be constructed for NonBurnable material");
    case FireFuelSpreadInputStatus::invalid_model:
        throw std::invalid_argument(
            "Fire spread inputs contain an invalid fuel model");
    case FireFuelSpreadInputStatus::missing_required_moisture:
        throw std::invalid_argument(
            "Fire spread inputs are missing required prescribed fuel moisture");
    }

    throw std::logic_error(
        "Fire fuel spread-input status is invalid");
}

} // namespace ERFFire

#endif // ERF_FIRE_FUEL_SPREAD_H_
