#ifndef ERF_FIRE_FUEL_BURN_TIME_H_
#define ERF_FIRE_FUEL_BURN_TIME_H_

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

namespace ERFFire
{

/**
 * Semi-empirical SFIRE surface-fuel burn weights, indexed by Anderson model.
 *
 * Source: openwfm/wrf-fire, wrfv2_fire/phys/module_fr_sfire_phys.F,
 * DATA weight (blob 4cbb42fa7b5fd36faf4df0edfbbc50e7ee3c24b5).
 * The exponential 1/e time is weight / 0.85 seconds, NOT the particle's
 * 1-h/10-h/100-h moisture-response time. These are model defaults, not
 * validation against measured heat-release histories. One reservoir per
 * cell still cannot distinguish flaming and residual smouldering classes.
 *
 * Zero means an invalid/nonburnable model; callers must not divide by it.
 */
[[nodiscard]] AMREX_GPU_HOST_DEVICE
constexpr amrex::Real anderson13_sfire_burn_weight(int model) noexcept
{
    switch (model) {
    case 1:
    case 2:
    case 3: return amrex::Real(7);
    case 4: return amrex::Real(180);
    case 5:
    case 6:
    case 7: return amrex::Real(100);
    case 8:
    case 9:
    case 10:
    case 11:
    case 12:
    case 13: return amrex::Real(900);
    default: return amrex::Real(0);
    }
}

[[nodiscard]] AMREX_GPU_HOST_DEVICE
constexpr amrex::Real anderson13_sfire_burn_time_s(int model) noexcept
{
    return anderson13_sfire_burn_weight(model) / amrex::Real(0.85);
}

} // namespace ERFFire

#endif
