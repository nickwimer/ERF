#ifndef ERF_FIRE_TYPES_H
#define ERF_FIRE_TYPES_H

#include <AMReX_REAL.H>

namespace ERFFire
{

/**
 * Two-dimensional vector in physical horizontal coordinates.
 */
struct FireVec2 {
    amrex::Real x{};
    amrex::Real y{};
};

} // namespace ERFFire

#endif // ERF_FIRE_TYPES_H
