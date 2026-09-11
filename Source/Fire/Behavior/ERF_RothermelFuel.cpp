#include <ERF_RothermelFuel.H>

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

} // namespace

RothermelFuelParameters
make_fm1_fuel_parameters () noexcept
{
    // Albini (1976), Appendix III, table 7:
    //   sigma = 3500 1/ft, w0 = 0.034 lb/ft^2, depth = 1.0 ft,
    //   dead moisture of extinction = 12%.
    // Common original-model particle properties:
    //   h = 8000 Btu/lb, rho_p = 32 lb/ft^3,
    //   S_T = 0.0555, S_e = 0.01.
    return {
        lb_ft2_to_kg_m2(0.034),
        ft_inv_to_m_inv(3500.0),
        foot_m,
        btu_lb_to_j_kg(8000.0),
        lb_ft3_to_kg_m3(32.0),
        0.0555,
        0.01,
        0.12
    };
}

} // namespace ERFFire
