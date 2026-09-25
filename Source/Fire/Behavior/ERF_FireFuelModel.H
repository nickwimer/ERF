#ifndef ERF_FIRE_FUEL_MODEL_H_
#define ERF_FIRE_FUEL_MODEL_H_

#include <ERF_FireFuelMoisture.H>

#include <AMReX_GpuQualifiers.H>

namespace ERFFire
{

/**
 * Stable categorical identifiers for spatial Fire material.
 *
 * Zero is reserved for explicit nonburnable material. Values 1-13 are the
 * original Anderson/Albini fire-behavior fuel-model numbers and are kept
 * numerically identical to the published catalog.
 */
enum class FireFuelModelId : int
{
    NonBurnable = 0,
    FM1 = 1,
    FM2 = 2,
    FM3 = 3,
    FM4 = 4,
    FM5 = 5,
    FM6 = 6,
    FM7 = 7,
    FM8 = 8,
    FM9 = 9,
    FM10 = 10,
    FM11 = 11,
    FM12 = 12,
    FM13 = 13
};

/**
 * Model-neutral resolved surface-fire material.
 *
 * Uniform and spatial production paths should use this same categorical
 * representation. Storage location (uniform versus raster) is deliberately
 * not part of material identity.
 */
struct FireFuelMaterial
{
    FireFuelModelId model_id{FireFuelModelId::FM1};
    FireFuelMoisture moisture{};
};

struct FireFuelModelMoistureContract
{
    bool valid_model{};
    bool burnable{};
    unsigned int required_mask{};
};

[[nodiscard]] AMREX_GPU_HOST_DEVICE
constexpr unsigned int
fire_fuel_moisture_component_bit(
    FireFuelMoistureClass component) noexcept
{
    const int index = static_cast<int>(component);
    return index >= 0
            && index < FireFuelMoisture::component_count
        ? (1u << static_cast<unsigned int>(index))
        : 0u;
}

/**
 * Required prescribed moisture components for one original-13 model.
 *
 * A moisture component is required exactly when its corresponding published
 * particle class has positive loading. The original live class is herbaceous
 * only for FM2; FM4, FM5, FM7, and FM10 use the live-woody moisture channel.
 * Other original models contain no live loading.
 *
 * This contract describes material completeness only. It does not by itself
 * enable a fuel model in spread, combustion, input parsing, or restart policy.
 */
[[nodiscard]] AMREX_GPU_HOST_DEVICE
constexpr FireFuelModelMoistureContract
fire_fuel_model_moisture_contract(FireFuelModelId id) noexcept
{
    constexpr unsigned int d1 =
        fire_fuel_moisture_component_bit(
            FireFuelMoistureClass::Dead1h);
    constexpr unsigned int d10 =
        fire_fuel_moisture_component_bit(
            FireFuelMoistureClass::Dead10h);
    constexpr unsigned int d100 =
        fire_fuel_moisture_component_bit(
            FireFuelMoistureClass::Dead100h);
    constexpr unsigned int live_herb =
        fire_fuel_moisture_component_bit(
            FireFuelMoistureClass::LiveHerbaceous);
    constexpr unsigned int live_woody =
        fire_fuel_moisture_component_bit(
            FireFuelMoistureClass::LiveWoody);

    switch (id) {
    case FireFuelModelId::NonBurnable:
        return {true, false, 0u};
    case FireFuelModelId::FM1:
        return {true, true, d1};
    case FireFuelModelId::FM2:
        return {true, true, d1 | d10 | d100 | live_herb};
    case FireFuelModelId::FM3:
        return {true, true, d1};
    case FireFuelModelId::FM4:
        return {true, true, d1 | d10 | d100 | live_woody};
    case FireFuelModelId::FM5:
        return {true, true, d1 | d10 | live_woody};
    case FireFuelModelId::FM6:
        return {true, true, d1 | d10 | d100};
    case FireFuelModelId::FM7:
        return {true, true, d1 | d10 | d100 | live_woody};
    case FireFuelModelId::FM8:
    case FireFuelModelId::FM9:
        return {true, true, d1 | d10 | d100};
    case FireFuelModelId::FM10:
        return {true, true, d1 | d10 | d100 | live_woody};
    case FireFuelModelId::FM11:
    case FireFuelModelId::FM12:
    case FireFuelModelId::FM13:
        return {true, true, d1 | d10 | d100};
    }

    return {};
}

[[nodiscard]] AMREX_GPU_HOST_DEVICE
constexpr bool
fire_fuel_model_id_valid(FireFuelModelId id) noexcept
{
    return fire_fuel_model_moisture_contract(id).valid_model;
}

[[nodiscard]] AMREX_GPU_HOST_DEVICE
constexpr bool
fire_fuel_model_is_anderson13(FireFuelModelId id) noexcept
{
    const int raw = static_cast<int>(id);
    return raw >= static_cast<int>(FireFuelModelId::FM1)
        && raw <= static_cast<int>(FireFuelModelId::FM13);
}

/**
 * Return published Anderson/Albini number 1-13, or zero for NonBurnable or an
 * invalid enum value.
 */
[[nodiscard]] AMREX_GPU_HOST_DEVICE
constexpr int
fire_fuel_model_number(FireFuelModelId id) noexcept
{
    return fire_fuel_model_is_anderson13(id)
        ? static_cast<int>(id)
        : 0;
}

[[nodiscard]] AMREX_GPU_HOST_DEVICE
inline unsigned int
fire_fuel_missing_required_moisture_mask(
    FireFuelModelId id,
    const FireFuelMoisture& moisture) noexcept
{
    const auto contract =
        fire_fuel_model_moisture_contract(id);
    if (!contract.valid_model) {
        return 0u;
    }

    unsigned int missing = 0u;
    for (int component = 0;
         component < FireFuelMoisture::component_count;
         ++component) {
        const auto moisture_class =
            static_cast<FireFuelMoistureClass>(component);
        const unsigned int bit =
            fire_fuel_moisture_component_bit(
                moisture_class);
        if ((contract.required_mask & bit) != 0u
            && !moisture.has(moisture_class)) {
            missing |= bit;
        }
    }
    return missing;
}

[[nodiscard]] AMREX_GPU_HOST_DEVICE
inline bool
fire_fuel_model_moisture_complete(
    FireFuelModelId id,
    const FireFuelMoisture& moisture) noexcept
{
    const auto contract =
        fire_fuel_model_moisture_contract(id);
    return contract.valid_model
        && fire_fuel_missing_required_moisture_mask(
               id, moisture)
            == 0u;
}

} // namespace ERFFire

#endif // ERF_FIRE_FUEL_MODEL_H_
