#ifndef ERF_FIRE_FUEL_COMBUSTION_H_
#define ERF_FIRE_FUEL_COMBUSTION_H_

#include <ERF_FireCombustion.H>
#include <ERF_FireFuelRaster.H>

#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>

#include <stdexcept>

namespace ERFFire
{

/**
 * Result of resolving one spatial fuel cell into the current semi-empirical
 * combustion parameterization.
 *
 * NonBurnable is deliberately rejected here until front/barrier evolution can
 * guarantee that such cells are never ignited. UnsupportedModel is reserved
 * for future catalog identifiers whose combustion policy has not been defined.
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
            "Fire local combustion cannot ignite NonBurnable material before barrier dynamics are enabled");
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

} // namespace ERFFire

#endif // ERF_FIRE_FUEL_COMBUSTION_H_
