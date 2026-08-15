#ifndef ERF_VECTOR_PERIMETER_PROPAGATOR_H
#define ERF_VECTOR_PERIMETER_PROPAGATOR_H

#include <ERF_FirePerimeter.H>

#include <functional>
#include <vector>

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
 * Batched prescribed local normal-spread-speed callback.
 *
 * positions_m and outward_normals have identical lengths and canonical
 * perimeter-vertex order. The returned speed vector must have that same
 * length. This lets a coupling layer resolve one complete RK stage with one
 * collective environment-sampling operation.
 */
using NormalSpeedBatchFunction = std::function<std::vector<amrex::Real>(
    const std::vector<FireVec2>& positions_m,
    const std::vector<FireVec2>& outward_normals,
    amrex::Real time_s)>;

/**
 * Advance a fixed-topology perimeter by one explicit midpoint/RK2 step.
 */
[[nodiscard]] FirePerimeter advance_perimeter_rk2(
    const FirePerimeter& perimeter,
    amrex::Real time_s,
    amrex::Real dt_s,
    const NormalSpeedFunction& normal_speed_mps);

/**
 * Advance the same RK2 method while evaluating all vertices in each stage as
 * one batch. The science and vertex ordering are identical to
 * advance_perimeter_rk2(); only callback granularity differs.
 */
[[nodiscard]] FirePerimeter advance_perimeter_rk2_batched(
    const FirePerimeter& perimeter,
    amrex::Real time_s,
    amrex::Real dt_s,
    const NormalSpeedBatchFunction& normal_speed_mps);

} // namespace ERFFire

#endif // ERF_VECTOR_PERIMETER_PROPAGATOR_H
