#ifndef ERF_FIRE_FUEL_MOISTURE_H_
#define ERF_FIRE_FUEL_MOISTURE_H_

#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <stdexcept>
#include <type_traits>

namespace ERFFire
{

/** Moisture components; these are not fuel-model catalog identifiers. */
enum class FireFuelMoistureClass : int
{
    Dead1h = 0,
    Dead10h,
    Dead100h,
    LiveHerbaceous,
    LiveWoody
};

/**
 * Prescribed water / oven-dry-fuel mass ratios, in kg/kg (not percent).
 *
 * Presence is independent of value: an absent component is not zero moisture.
 * Values must be finite and nonnegative, but may exceed one. This type stores
 * conditions only; it implements neither curing nor prognostic drying/wetting.
 * Physics consumers must require the components of their populated fuel classes.
 *
 * Fixed-size, pointer-free storage can be copied by value into a device kernel.
 * The try_* accessors are nonthrowing and preserve outputs/state on failure.
 */
class FireFuelMoisture
{
public:
    static constexpr int component_count = 5;

    AMREX_GPU_HOST_DEVICE
    constexpr FireFuelMoisture() noexcept = default;

    [[nodiscard]] AMREX_GPU_HOST_DEVICE
    bool has(FireFuelMoistureClass component) const noexcept
    {
        const int index = static_cast<int>(component);
        return valid_index(index)
            && (present_ & (1u << index)) != 0u;
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE
    bool try_get(
        FireFuelMoistureClass component,
        amrex::Real& value) const noexcept
    {
        if (!has(component)) {
            return false;
        }
        value = values_[static_cast<int>(component)];
        return true;
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE
    bool try_set(
        FireFuelMoistureClass component,
        amrex::Real value) noexcept
    {
        const int index = static_cast<int>(component);
        // AMReX supplies a backend-safe finite predicate for CPU/CUDA/HIP/SYCL.
        // Check finiteness before ordering comparisons (including with FPE
        // traps enabled). There is deliberately no upper bound of one.
        if (!valid_index(index)
            || !amrex::Math::isfinite(value)
            || value < amrex::Real(0)) {
            return false;
        }
        values_[index] = value;
        present_ |= (1u << index);
        return true;
    }

    [[nodiscard]] amrex::Real get(FireFuelMoistureClass component) const
    {
        amrex::Real value{};
        if (!try_get(component, value)) {
            throw std::invalid_argument(
                "Fire fuel moisture component is missing or invalid");
        }
        return value;
    }

    void set(FireFuelMoistureClass component, amrex::Real value)
    {
        if (!try_set(component, value)) {
            throw std::invalid_argument(
                "Fire fuel moisture requires a valid component and a finite "
                "nonnegative kg/kg value");
        }
    }

    /** Legacy scalar convenience: populate dead classes, never live classes. */
    [[nodiscard]] static FireFuelMoisture uniform_dead(amrex::Real value)
    {
        FireFuelMoisture result;
        result.set(FireFuelMoistureClass::Dead1h, value);
        result.set(FireFuelMoistureClass::Dead10h, value);
        result.set(FireFuelMoistureClass::Dead100h, value);
        return result;
    }

private:
    [[nodiscard]] AMREX_GPU_HOST_DEVICE
    static constexpr bool valid_index(int index) noexcept
    {
        return index >= 0 && index < component_count;
    }

    amrex::Real values_[component_count]{};
    unsigned int present_{0u};
};

static_assert(std::is_trivially_copyable_v<FireFuelMoisture>);
static_assert(std::is_standard_layout_v<FireFuelMoisture>);

} // namespace ERFFire

#endif
