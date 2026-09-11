#ifndef ERF_FIRE_BURNED_FRACTION_H
#define ERF_FIRE_BURNED_FRACTION_H

#include <AMReX_REAL.H>

namespace ERFFire
{

/**
 * Result of one monotone burned-fraction state transition.
 *
 * burned_fraction is the updated persistent history state.
 * newly_burned_fraction is the non-negative increment created by this update
 * and is intended to become the clean input to the later burn/energy layer.
 */
struct FireBurnedFractionUpdate
{
    amrex::Real burned_fraction{};
    amrex::Real newly_burned_fraction{};
};

/**
 * Advance one cell's burned-area fraction monotonically.
 *
 * Both previous_burned_fraction and current_coverage_fraction must be finite
 * values in [0,1]. No geometric tolerance or threshold is applied here:
 *
 *   burned_new = max(burned_old, coverage_current)
 *   newly_burned = burned_new - burned_old
 *
 * A later perimeter that covers less of the cell therefore cannot erase burn
 * history.
 */
[[nodiscard]] FireBurnedFractionUpdate update_fire_burned_fraction(
    amrex::Real previous_burned_fraction,
    amrex::Real current_coverage_fraction);

} // namespace ERFFire

#endif // ERF_FIRE_BURNED_FRACTION_H
