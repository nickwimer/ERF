#ifndef ERF_FIRE_LEVEL0_ADVANCE_H_
#define ERF_FIRE_LEVEL0_ADVANCE_H_

#include <ERF_FireLevel0Environment.H>

#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include <functional>

class ERFTerrainSource;

namespace ERFFire
{

/**
 * Lazy terrain-source lookup used by the level-0 Fire coupling driver.
 *
 * The provider is invoked collectively only when a VariableDz Fire terrain
 * surface must be constructed or reconstructed. Keeping the lookup lazy
 * preserves the existing collective-I/O ordering while avoiding any ERF or
 * ProblemBase backpointer in the Fire context.
 */
using ERFFireTerrainSourceProvider =
    std::function<const ERFTerrainSource*()>;

/**
 * Read-only ERF level-0 inputs required for one explicit Fire coupling step.
 *
 * environment contains the already-fillpatched t^n velocity/geometry views.
 * conserved_state_tn is the already-fillpatched atmospheric conserved state
 * used for pressure diagnosis in two-way source construction.
 *
 * detJ_cc may be null outside static-fitted VariableDz two-way coupling.
 */
struct ERFFireLevel0AdvanceInputs
{
    ERFFireLevel0EnvironmentInputs environment;

    const amrex::MultiFab& conserved_state_tn;
    const amrex::MultiFab* detJ_cc{nullptr};

    MoistureType moisture_type{MoistureType::None};

    ERFFireTerrainSourceProvider terrain_source_provider{};

    double time_s{};
    amrex::Real dt_s{};
};

} // namespace ERFFire

#endif
