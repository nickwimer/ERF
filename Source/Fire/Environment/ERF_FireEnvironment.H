#ifndef ERF_FIRE_ENVIRONMENT_H_
#define ERF_FIRE_ENVIRONMENT_H_

#include <ERF_FireTypes.H>

namespace ERFFire
{

/**
 * Coupling-neutral horizontal Fire-environment sample.
 *
 * horizontal_wind_mps is a map-plane reference-wind vector in SI units.
 * The sampler/adapter defines its AGL sampling height. It is not intrinsically
 * a 20-ft, 10-m, or Rothermel midflame wind. Wind adjustment, when requested,
 * belongs in Fire-side preprocessing before Rothermel evaluation.
 */
struct FireEnvironmentSample
{
    FireVec2 horizontal_wind_mps{};
};

} // namespace ERFFire

#endif
