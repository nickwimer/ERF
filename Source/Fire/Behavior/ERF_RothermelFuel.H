#ifndef ERF_ROTHERMEL_FUEL_H
#define ERF_ROTHERMEL_FUEL_H

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace ERFFire
{

/** One published fuel-particle class in an Anderson/Albini fuel model. */
struct RothermelFuelClassParameters
{
    amrex::Real dry_load_kg_m2{};
    amrex::Real sav_m_inv{};
};

/**
 * Original 13 fire-behavior fuel-model description in SI.
 *
 * Albini (1976), Appendix III, table 7 defines separate dead 1-h, 10-h,
 * 100-h, and live-foliage loading/SAV classes, plus fuel-bed depth and dead
 * moisture of extinction. Particle density, heat content, total mineral
 * fraction, and effective mineral fraction are common to all 13 models.
 *
 * A zero class loading means that particle class is absent; its SAV is then
 * stored as zero.
 */
struct Anderson13FuelParameters
{
    RothermelFuelClassParameters dead_1h{};
    RothermelFuelClassParameters dead_10h{};
    RothermelFuelClassParameters dead_100h{};
    RothermelFuelClassParameters live_foliage{};
    amrex::Real fuel_bed_depth_m{};
    amrex::Real heat_content_j_kg{};
    amrex::Real particle_density_kg_m3{};
    amrex::Real total_mineral_fraction{};
    amrex::Real effective_mineral_fraction{};
    amrex::Real dead_moisture_of_extinction{};
};

namespace detail
{

constexpr amrex::Real rothermel_foot_m = amrex::Real(0.3048);
constexpr amrex::Real rothermel_pound_kg = amrex::Real(0.45359237);
constexpr amrex::Real rothermel_btu_j = amrex::Real(1055.05585262);

[[nodiscard]] AMREX_GPU_HOST_DEVICE
constexpr amrex::Real
rothermel_lb_ft2_to_kg_m2(amrex::Real value) noexcept
{
    return value * rothermel_pound_kg
        / (rothermel_foot_m * rothermel_foot_m);
}

[[nodiscard]] AMREX_GPU_HOST_DEVICE
constexpr amrex::Real
rothermel_ft_inv_to_m_inv(amrex::Real value) noexcept
{
    return value / rothermel_foot_m;
}

[[nodiscard]] AMREX_GPU_HOST_DEVICE
constexpr amrex::Real
rothermel_btu_lb_to_j_kg(amrex::Real value) noexcept
{
    return value * rothermel_btu_j
        / rothermel_pound_kg;
}

[[nodiscard]] AMREX_GPU_HOST_DEVICE
constexpr amrex::Real
rothermel_lb_ft3_to_kg_m3(amrex::Real value) noexcept
{
    return value * rothermel_pound_kg
        / (rothermel_foot_m
           * rothermel_foot_m
           * rothermel_foot_m);
}

[[nodiscard]] AMREX_GPU_HOST_DEVICE
constexpr RothermelFuelClassParameters
anderson13_fuel_class(
    amrex::Real sav_ft_inv,
    amrex::Real load_lb_ft2) noexcept
{
    return load_lb_ft2 == amrex::Real(0)
        ? RothermelFuelClassParameters{}
        : RothermelFuelClassParameters{
            rothermel_lb_ft2_to_kg_m2(load_lb_ft2),
            rothermel_ft_inv_to_m_inv(sav_ft_inv)};
}

[[nodiscard]] AMREX_GPU_HOST_DEVICE
constexpr Anderson13FuelParameters
anderson13_fuel(
    amrex::Real dead_1h_sav_ft_inv,
    amrex::Real dead_1h_load_lb_ft2,
    amrex::Real dead_10h_load_lb_ft2,
    amrex::Real dead_100h_load_lb_ft2,
    amrex::Real live_sav_ft_inv,
    amrex::Real live_load_lb_ft2,
    amrex::Real depth_ft,
    amrex::Real dead_moisture_of_extinction) noexcept
{
    return {
        anderson13_fuel_class(
            dead_1h_sav_ft_inv,
            dead_1h_load_lb_ft2),
        anderson13_fuel_class(
            dead_10h_load_lb_ft2 > amrex::Real(0)
                ? amrex::Real(109)
                : amrex::Real(0),
            dead_10h_load_lb_ft2),
        anderson13_fuel_class(
            dead_100h_load_lb_ft2 > amrex::Real(0)
                ? amrex::Real(30)
                : amrex::Real(0),
            dead_100h_load_lb_ft2),
        anderson13_fuel_class(
            live_sav_ft_inv,
            live_load_lb_ft2),
        depth_ft * rothermel_foot_m,
        rothermel_btu_lb_to_j_kg(amrex::Real(8000)),
        rothermel_lb_ft3_to_kg_m3(amrex::Real(32)),
        amrex::Real(0.0555),
        amrex::Real(0.01),
        dead_moisture_of_extinction
    };
}

} // namespace detail

/**
 * Nonthrowing original-13 catalog lookup suitable for host and device code.
 *
 * The result is reset on failure. model_number uses the published range 1-13.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool
try_make_anderson13_fuel_parameters(
    int model_number,
    Anderson13FuelParameters& result) noexcept
{
    using detail::anderson13_fuel;

    result = {};
    switch (model_number) {
    case 1:
        result = anderson13_fuel(
            3500, 0.034, 0.000, 0.000, 0, 0.000, 1.0, 0.12);
        return true;
    case 2:
        result = anderson13_fuel(
            3000, 0.092, 0.046, 0.023, 1500, 0.023, 1.0, 0.15);
        return true;
    case 3:
        result = anderson13_fuel(
            1500, 0.138, 0.000, 0.000, 0, 0.000, 2.5, 0.25);
        return true;
    case 4:
        result = anderson13_fuel(
            2000, 0.230, 0.184, 0.092, 1500, 0.230, 6.0, 0.20);
        return true;
    case 5:
        result = anderson13_fuel(
            2000, 0.046, 0.023, 0.000, 1500, 0.092, 2.0, 0.20);
        return true;
    case 6:
        result = anderson13_fuel(
            1750, 0.069, 0.115, 0.092, 0, 0.000, 2.5, 0.25);
        return true;
    case 7:
        result = anderson13_fuel(
            1750, 0.052, 0.086, 0.069, 1550, 0.017, 2.5, 0.40);
        return true;
    case 8:
        result = anderson13_fuel(
            2000, 0.069, 0.046, 0.115, 0, 0.000, 0.2, 0.30);
        return true;
    case 9:
        result = anderson13_fuel(
            2500, 0.134, 0.019, 0.007, 0, 0.000, 0.2, 0.25);
        return true;
    case 10:
        result = anderson13_fuel(
            2000, 0.138, 0.092, 0.230, 1500, 0.092, 1.0, 0.25);
        return true;
    case 11:
        result = anderson13_fuel(
            1500, 0.069, 0.207, 0.253, 0, 0.000, 1.0, 0.15);
        return true;
    case 12:
        result = anderson13_fuel(
            1500, 0.184, 0.644, 0.759, 0, 0.000, 2.3, 0.20);
        return true;
    case 13:
        result = anderson13_fuel(
            1500, 0.322, 1.058, 1.288, 0, 0.000, 3.0, 0.25);
        return true;
    }

    return false;
}

/**
 * Return one of the original Anderson/Albini fire-behavior fuel models 1-13.
 *
 * model_number follows the published numbering. Values outside [1,13] are
 * rejected rather than mapped to a default model.
 */
[[nodiscard]] Anderson13FuelParameters
make_anderson13_fuel_parameters(int model_number);

/**
 * Single dead-fuel-class inputs for Rothermel implementation.
 *
 * This implementation intentionally supports only the homogeneous
 * dead fine-fuel path required by Anderson/Albini Fuel Model 1. This type is
 * not intended to imply that one scalar loading describes a general
 * multi-class Rothermel fuel bed.
 */
struct RothermelFuelParameters
{
    amrex::Real dead_1h_load_kg_m2{};
    amrex::Real dead_1h_sav_m_inv{};
    amrex::Real fuel_bed_depth_m{};
    amrex::Real dead_heat_content_j_kg{};
    amrex::Real particle_density_kg_m3{};
    amrex::Real total_mineral_fraction{};
    amrex::Real effective_mineral_fraction{};
    amrex::Real dead_moisture_of_extinction{};
};

/**
 * Original fire-behavior Fuel Model 1 (short grass), expressed in SI.
 *
 * Source values are the Albini (1976) original-13 model description and the
 * Anderson (1982) FM1 description. Constants common to the original models
 * are the published 8000 Btu/lb heat content, 32 lb/ft^3 particle density,
 * 5.55% total mineral content, and 1.00% effective mineral content.
 */
[[nodiscard]] RothermelFuelParameters make_fm1_fuel_parameters() noexcept;

} // namespace ERFFire

#endif // ERF_ROTHERMEL_FUEL_H
