#ifndef ERF_FIRE_LEVEL0_ADVANCE_H_
#define ERF_FIRE_LEVEL0_ADVANCE_H_

#include <ERF_FireLevel0Environment.H>

#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>



namespace ERFFire
{

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


    double time_s{};
    amrex::Real dt_s{};
};

} // namespace ERFFire

#endif
