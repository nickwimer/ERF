#ifndef ERF_FIRE_CONTEXT_H_
#define ERF_FIRE_CONTEXT_H_

#include <ERF_FireRuntimeConfig.H>
#include <ERF_FireSpreadRuntime.H>
#include <ERF_FireTerrainSurface.H>

#include <AMReX_MultiFab.H>

#include <limits>
#include <memory>

namespace ERFFire
{

struct ERFFireCheckpointRestoreInputs;
struct ERFFireCheckpointWriteInputs;
struct ERFFireLevel0AdvanceInputs;
struct ERFFirePlotfile2DAvailabilityInputs;
struct ERFFirePlotfile2DFillInputs;
struct ERFFirePlotfile2DPrepareInputs;
struct ERFFirePlotfile2DState;

/**
 * ERF-facing owner for Fire coupling state.
 *
 * ERFFireSpreadRuntime remains coupling-neutral. This context owns the
 * adapter state needed to connect that runtime to ERF.
 *
 * The low-level accessors below are transitional and remain only for
 * callers not yet moved behind subsystem-level Fire interface methods.
 */
class ERFFireContext
{
public:
    ERFFireContext() = default;

    ERFFireContext(const ERFFireContext&) = delete;
    ERFFireContext& operator=(const ERFFireContext&) = delete;
    ERFFireContext(ERFFireContext&&) = delete;
    ERFFireContext& operator=(ERFFireContext&&) = delete;

    void configure_from_inputs(
        const ERFFireHostCapabilities& host);

    void advance_level0(
        const ERFFireLevel0AdvanceInputs& inputs);

    /**
     * Add the prepared Fire atmospheric tendency to ERF conserved sources.
     *
     * This is a no-op away from level 0 or when no Fire source is installed.
     * The source is additive: ordinary ERF source construction remains
     * authoritative and executes before this hook.
     */
    void add_atmospheric_source(
        int level,
        amrex::MultiFab& conserved_source) const;

    void write_checkpoint(
        const ERFFireCheckpointWriteInputs& inputs) const;

    void restore_checkpoint(
        const ERFFireCheckpointRestoreInputs& inputs);

    void append_available_plotfile2d_diagnostics(
        const ERFFirePlotfile2DAvailabilityInputs& inputs) const;

    [[nodiscard]] ERFFirePlotfile2DState
    prepare_plotfile2d_diagnostics(
        const ERFFirePlotfile2DPrepareInputs& inputs) const;

    void fill_plotfile2d_diagnostics(
        const ERFFirePlotfile2DFillInputs& inputs) const;

    [[nodiscard]] ERFFireRuntimeOptions&
    runtime_options() noexcept
    {
        return runtime_options_;
    }

    [[nodiscard]] const ERFFireRuntimeOptions&
    runtime_options() const noexcept
    {
        return runtime_options_;
    }

    [[nodiscard]] double&
    environment_snapshot_time() noexcept
    {
        return environment_snapshot_time_;
    }

    [[nodiscard]] double
    environment_snapshot_time() const noexcept
    {
        return environment_snapshot_time_;
    }

    [[nodiscard]] double&
    environment_reference_height_agl_m() noexcept
    {
        return environment_reference_height_agl_m_;
    }

    [[nodiscard]] double
    environment_reference_height_agl_m() const noexcept
    {
        return environment_reference_height_agl_m_;
    }

    [[nodiscard]] std::unique_ptr<ERFFireSpreadRuntime>&
    spread_runtime() noexcept
    {
        return spread_runtime_;
    }

    [[nodiscard]] const std::unique_ptr<ERFFireSpreadRuntime>&
    spread_runtime() const noexcept
    {
        return spread_runtime_;
    }

    [[nodiscard]] std::unique_ptr<FireTerrainSurface>&
    terrain_surface() noexcept
    {
        return terrain_surface_;
    }

    [[nodiscard]] const std::unique_ptr<FireTerrainSurface>&
    terrain_surface() const noexcept
    {
        return terrain_surface_;
    }

    [[nodiscard]] std::unique_ptr<amrex::MultiFab>&
    atmospheric_source_tendency() noexcept
    {
        return atmospheric_source_tendency_;
    }

    [[nodiscard]] const std::unique_ptr<amrex::MultiFab>&
    atmospheric_source_tendency() const noexcept
    {
        return atmospheric_source_tendency_;
    }

    [[nodiscard]] double&
    atmospheric_source_time() noexcept
    {
        return atmospheric_source_time_;
    }

    [[nodiscard]] double
    atmospheric_source_time() const noexcept
    {
        return atmospheric_source_time_;
    }

    [[nodiscard]] int&
    step_index() noexcept
    {
        return step_index_;
    }

    [[nodiscard]] int
    step_index() const noexcept
    {
        return step_index_;
    }

private:
    ERFFireRuntimeOptions runtime_options_{};

    double environment_snapshot_time_ =
        std::numeric_limits<double>::quiet_NaN();

    double environment_reference_height_agl_m_ =
        std::numeric_limits<double>::quiet_NaN();

    std::unique_ptr<ERFFireSpreadRuntime> spread_runtime_;
    std::unique_ptr<FireTerrainSurface> terrain_surface_;
    std::unique_ptr<amrex::MultiFab> atmospheric_source_tendency_;

    double atmospheric_source_time_ =
        std::numeric_limits<double>::quiet_NaN();

    int step_index_ = 0;
};

} // namespace ERFFire

#endif
