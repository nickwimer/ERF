#ifndef ERF_FIRE_ENVIRONMENT_H_
#define ERF_FIRE_ENVIRONMENT_H_

#include <ERF_FireTypes.H>

#include <functional>
#include <vector>

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

/**
 * Coupling-neutral ordered batch sampler.
 *
 * The caller supplies map-plane positions in canonical perimeter order and
 * receives one environment sample per position in that same order. A concrete
 * implementation may be collective (for example, an ERF/AMReX distributed
 * field adapter), so Fire propagation evaluates this callback once per RK
 * stage rather than once per vertex.
 */
using FireEnvironmentBatchFunction =
    std::function<std::vector<FireEnvironmentSample>(
        const std::vector<FireVec2>& positions_m)>;

} // namespace ERFFire

#endif
