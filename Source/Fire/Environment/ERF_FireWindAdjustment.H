#ifndef ERF_FIRE_WIND_ADJUSTMENT_H
#define ERF_FIRE_WIND_ADJUSTMENT_H

#include "ERF_FireTypes.H"

#include <cmath>
#include <stdexcept>

namespace ERFFire
{

/**
 * Convert a horizontal 20-ft reference-wind vector to model/midflame wind
 * using an explicit scalar wind adjustment factor (WAF).
 *
 * The input and output vectors are map-plane push-to velocities in m/s. The
 * factor must be finite and lie in [0,1]. WAF=1 is explicitly returned as the
 * unmodified input so the identity endpoint can remain bitwise identical;
 * WAF=0 returns the canonical +0,+0 vector. Intermediate factors scale the
 * complete vector, preserving direction apart from ordinary floating-point
 * component multiplication.
 *
 * This coupling-neutral Fire-core operation owns no ERF sampling-height,
 * ParmParse, terrain, or one-way/two-way policy. Its input is specifically a
 * 20-ft reference wind; selection and sampling of that reference wind belongs
 * to the later ERF-facing integration layer.
 */
[[nodiscard]] inline FireVec2
fire_midflame_wind_from_20ft_reference (
    const FireVec2& reference_wind_20ft_mps,
    amrex::Real wind_adjustment_factor)
{
    if (!std::isfinite(wind_adjustment_factor) ||
        wind_adjustment_factor < amrex::Real(0.0) ||
        wind_adjustment_factor > amrex::Real(1.0)) {
        throw std::invalid_argument(
            "fire 20-ft wind adjustment factor must be finite and in [0,1]");
    }

    if (wind_adjustment_factor == amrex::Real(1.0)) {
        return reference_wind_20ft_mps;
    }
    if (wind_adjustment_factor == amrex::Real(0.0)) {
        return {};
    }

    return reference_wind_20ft_mps * wind_adjustment_factor;
}

} // namespace ERFFire

#endif // ERF_FIRE_WIND_ADJUSTMENT_H
