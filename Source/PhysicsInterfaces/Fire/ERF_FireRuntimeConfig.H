#ifndef ERF_FIRE_RUNTIME_CONFIG_H_
#define ERF_FIRE_RUNTIME_CONFIG_H_

#include <ERF_FireRuntimeOptions.H>

namespace ERFFire
{

/**
 * ERF capabilities required to validate Fire runtime configuration.
 *
 * These are deliberately expressed as simple Fire-facing facts instead
 * of ERF solver enums.  Fire configuration therefore does not depend on
 * SolverChoice, MeshType, TerrainType, or MoistureType definitions.
 */
struct ERFFireHostCapabilities
{
    int max_level{0};
    bool variable_dz{false};
    bool static_fitted_mesh{false};
    bool moist_no_condensation{false};
    bool anelastic_level0{false};
};

/**
 * Read and validate the fire.* runtime configuration.
 *
 * ERF supplies only the host capabilities needed for cross-component
 * validation.  All fire.* parameter names, defaults, validation rules,
 * and error messages are owned here.
 */
[[nodiscard]] ERFFireRuntimeOptions
read_erf_fire_runtime_options(
    const ERFFireHostCapabilities& host);

} // namespace ERFFire

#endif
