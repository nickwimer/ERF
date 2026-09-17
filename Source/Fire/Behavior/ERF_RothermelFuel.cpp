#include <ERF_RothermelFuel.H>

#include <array>
#include <stdexcept>

namespace ERFFire
{
namespace
{

constexpr amrex::Real foot_m = 0.3048;
constexpr amrex::Real pound_kg = 0.45359237;
constexpr amrex::Real btu_j = 1055.05585262;

constexpr amrex::Real
lb_ft2_to_kg_m2 (amrex::Real value) noexcept
{
    return value * pound_kg / (foot_m * foot_m);
}

constexpr amrex::Real
ft_inv_to_m_inv (amrex::Real value) noexcept
{
    return value / foot_m;
}

constexpr amrex::Real
btu_lb_to_j_kg (amrex::Real value) noexcept
{
    return value * btu_j / pound_kg;
}

constexpr amrex::Real
lb_ft3_to_kg_m3 (amrex::Real value) noexcept
{
    return value * pound_kg / (foot_m * foot_m * foot_m);
}

struct NativeFuelClass
{
    amrex::Real sav_ft_inv{};
    amrex::Real load_lb_ft2{};
};

struct NativeAnderson13Fuel
{
    NativeFuelClass dead_1h{};
    NativeFuelClass dead_10h{};
    NativeFuelClass dead_100h{};
    NativeFuelClass live_foliage{};
    amrex::Real depth_ft{};
    amrex::Real dead_moisture_of_extinction{};
};

// Albini (1976), Appendix III, table 7. Table entries are SAV [1/ft]
// paired with dry loading [lb/ft^2]. Blank particle classes are represented
// by zero loading/SAV. Models 6, 7, and 11-13 retain their published 10-h
// and 100-h classes instead of collapsing them into a weighted scalar load.
constexpr std::array<NativeAnderson13Fuel, 13> anderson13_native{{
    {{3500.0, 0.034}, {0.0, 0.0},     {0.0, 0.0},     {0.0, 0.0},       1.0, 0.12},
    {{3000.0, 0.092}, {109.0, 0.046}, {30.0, 0.023},  {1500.0, 0.023},  1.0, 0.15},
    {{1500.0, 0.138}, {0.0, 0.0},     {0.0, 0.0},     {0.0, 0.0},       2.5, 0.25},
    {{2000.0, 0.230}, {109.0, 0.184}, {30.0, 0.092},  {1500.0, 0.230},  6.0, 0.20},
    {{2000.0, 0.046}, {109.0, 0.023}, {0.0, 0.0},     {1500.0, 0.092},  2.0, 0.20},
    {{1750.0, 0.069}, {109.0, 0.115}, {30.0, 0.092},  {0.0, 0.0},       2.5, 0.25},
    {{1750.0, 0.052}, {109.0, 0.086}, {30.0, 0.069},  {1550.0, 0.017},  2.5, 0.40},
    {{2000.0, 0.069}, {109.0, 0.046}, {30.0, 0.115},  {0.0, 0.0},       0.2, 0.30},
    {{2500.0, 0.134}, {109.0, 0.019}, {30.0, 0.007},  {0.0, 0.0},       0.2, 0.25},
    {{2000.0, 0.138}, {109.0, 0.092}, {30.0, 0.230},  {1500.0, 0.092},  1.0, 0.25},
    {{1500.0, 0.069}, {109.0, 0.207}, {30.0, 0.253},  {0.0, 0.0},       1.0, 0.15},
    {{1500.0, 0.184}, {109.0, 0.644}, {30.0, 0.759},  {0.0, 0.0},       2.3, 0.20},
    {{1500.0, 0.322}, {109.0, 1.058}, {30.0, 1.288},  {0.0, 0.0},       3.0, 0.25}
}};

constexpr RothermelFuelClassParameters
to_si (NativeFuelClass native) noexcept
{
    if (native.load_lb_ft2 == amrex::Real(0.0)) {
        return {};
    }
    return {
        lb_ft2_to_kg_m2(native.load_lb_ft2),
        ft_inv_to_m_inv(native.sav_ft_inv)
    };
}

constexpr Anderson13FuelParameters
to_si (const NativeAnderson13Fuel& native) noexcept
{
    return {
        to_si(native.dead_1h),
        to_si(native.dead_10h),
        to_si(native.dead_100h),
        to_si(native.live_foliage),
        native.depth_ft * foot_m,
        btu_lb_to_j_kg(8000.0),
        lb_ft3_to_kg_m3(32.0),
        0.0555,
        0.01,
        native.dead_moisture_of_extinction
    };
}

} // namespace

Anderson13FuelParameters
make_anderson13_fuel_parameters (int model_number)
{
    if (model_number < 1 || model_number > 13) {
        throw std::invalid_argument(
            "Anderson/Albini fuel model number must lie in [1,13]");
    }

    return to_si(
        anderson13_native[
            static_cast<std::size_t>(model_number - 1)]);
}

RothermelFuelParameters
make_fm1_fuel_parameters () noexcept
{
    const Anderson13FuelParameters fm1 =
        to_si(anderson13_native[0]);

    // Preserve the existing single-dead-class API as an exact projection of
    // model 1. Models 2-13 require the generalized multi-class evaluator and
    // must never be silently collapsed into this type.
    return {
        fm1.dead_1h.dry_load_kg_m2,
        fm1.dead_1h.sav_m_inv,
        fm1.fuel_bed_depth_m,
        fm1.heat_content_j_kg,
        fm1.particle_density_kg_m3,
        fm1.total_mineral_fraction,
        fm1.effective_mineral_fraction,
        fm1.dead_moisture_of_extinction
    };
}

} // namespace ERFFire
