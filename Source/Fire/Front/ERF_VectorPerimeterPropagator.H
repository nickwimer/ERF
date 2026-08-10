#ifndef ERF_VECTOR_PERIMETER_PROPAGATOR_H
#define ERF_VECTOR_PERIMETER_PROPAGATOR_H

#include <ERF_FirePerimeter.H>

#include <functional>

namespace ERFFire
{

/**
 * Prescribed local normal-spread-speed callback.
 */
using NormalSpeedFunction = std::function<amrex::Real(
    const FireVec2& position_m,
    const FireVec2& outward_normal,
    amrex::Real time_s)>;

/**
 * Advance a fixed-topology perimeter by one explicit midpoint/RK2 step.
 */
[[nodiscard]] FirePerimeter advance_perimeter_rk2(
    const FirePerimeter& perimeter,
    amrex::Real time_s,
    amrex::Real dt_s,
    const NormalSpeedFunction& normal_speed_mps);

} // namespace ERFFire

#endif // ERF_VECTOR_PERIMETER_PROPAGATOR_H
