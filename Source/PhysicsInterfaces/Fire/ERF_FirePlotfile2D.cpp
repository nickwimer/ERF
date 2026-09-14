#include <ERF_FirePlotfile2D.H>

#include <ERF_FireContext.H>
#include <ERF_FireRuntimeInit.H>
#include <ERF_FireSpreadOutput.H>

#include <ERF_Plotfile2DCatalog.H>

#include <AMReX.H>

#include <memory>

namespace
{

bool
is_fire_diagnostic(
    const plotfile2d::DiagnosticDescriptor* descriptor) noexcept
{
    return descriptor
        && descriptor->category
            == plotfile2d::DiagnosticCategory::Fire;
}

bool
fire_raster_matches_level0_geometry(
    const ERFFire::FireCartesianRasterGeometry2D& fire_geometry,
    const amrex::Geometry& level0_geometry) noexcept
{
    const amrex::Box& domain = level0_geometry.Domain();
    const auto prob_lo = level0_geometry.ProbLoArray();
    const auto cell_size = level0_geometry.CellSizeArray();

    return fire_geometry.nx
            == static_cast<std::size_t>(domain.length(0))
        && fire_geometry.ny
            == static_cast<std::size_t>(domain.length(1))
        && fire_geometry.xlo_m == prob_lo[0]
        && fire_geometry.ylo_m == prob_lo[1]
        && fire_geometry.dx_m == cell_size[0]
        && fire_geometry.dy_m == cell_size[1];
}

int
fire_checkpoint_raster_component(
    plotfile2d::DiagnosticID id) noexcept
{
    using Components =
        ERFFire::ERFFireCheckpointRasterComponents;

    switch (id) {
    case plotfile2d::DiagnosticID::FireBurnedFraction:
        return Components::burned_fraction;
    case plotfile2d::DiagnosticID::FireHasArrived:
        return Components::arrived;
    case plotfile2d::DiagnosticID::FireFirstArrivalTime:
        return Components::first_arrival_time_s;
    case plotfile2d::DiagnosticID::FireIgnitedAreaFraction:
        return Components::ignited_area_fraction;
    case plotfile2d::DiagnosticID::FireRemainingDryFuel:
        return Components::remaining_dry_fuel_kg_m2;
    case plotfile2d::DiagnosticID::FireConsumedDryFuel:
        return Components::consumed_dry_fuel_kg_m2;
    case plotfile2d::DiagnosticID::FireSensibleEnergy:
        return Components::sensible_energy_j_m2;
    case plotfile2d::DiagnosticID::FireWaterReleased:
        return Components::water_released_kg_m2;
    default:
        return -1;
    }
}

} // namespace

namespace ERFFire
{

void
ERFFireContext::append_available_plotfile2d_diagnostics(
    const ERFFirePlotfile2DAvailabilityInputs& inputs) const
{
    if (!runtime_options_.enabled) {
        return;
    }

    // Runtime Fire diagnostics follow dynamic land-surface fields so every
    // existing non-Fire component index remains unchanged.
    for (const auto& descriptor :
         plotfile2d::diagnostic_catalog()) {
        if (descriptor.category
            == plotfile2d::DiagnosticCategory::Fire) {
            inputs.available_names.push_back(
                descriptor.name);
        }
    }
}

ERFFirePlotfile2DState
ERFFireContext::prepare_plotfile2d_diagnostics(
    const ERFFirePlotfile2DPrepareInputs& inputs) const
{
    ERFFirePlotfile2DState state;

    for (const auto& name : inputs.plot_var_names) {
        state.requested =
            state.requested
            || is_fire_diagnostic(
                plotfile2d::find_diagnostic(name));
    }

    if (!state.requested) {
        return state;
    }

    if (!runtime_options_.enabled) {
        amrex::Abort(
            "Fire 2D diagnostics were selected while fire.enabled is false");
    }

    const ERFFireSpreadRuntime* fire_runtime =
        spread_runtime_.get();

    std::unique_ptr<ERFFireSpreadRuntime>
        initial_fire_runtime;

    // Initial-time output can precede construction by the coupling driver.
    // Build an equivalent temporary runtime without mutating solver state.
    if (fire_runtime == nullptr) {
        if (inputs.level0_step != 0
            || inputs.level0_time_s
                != amrex::Real(0.0)) {
            amrex::Abort(
                "Fire 2D diagnostics requested but the Fire runtime is missing");
        }

        initial_fire_runtime =
            make_erf_fire_spread_runtime(
                runtime_options_,
                inputs.level0_geometry,
                inputs.level0_time_s);

        fire_runtime =
            initial_fire_runtime.get();
    }

    // Reuse the version-2 checkpoint raster: this is the canonical
    // distributed eight-component persistent Fire surface schema.
    const auto& fire_geometry =
        fire_runtime->config().raster_geometry;

    if (!fire_raster_matches_level0_geometry(
            fire_geometry,
            inputs.level0_geometry)) {
        amrex::Abort(
            "Fire diagnostics in the combined ERF 2D plotfile currently require fire.n_cell to match the level-0 atmospheric grid; independent native Fire-grid plot output has not yet been enabled");
    }

    state.raster =
        make_erf_fire_checkpoint_v2_raster(
            *fire_runtime);

    return state;
}

void
ERFFireContext::fill_plotfile2d_diagnostics(
    const ERFFirePlotfile2DFillInputs& inputs) const
{
    if (!inputs.state.requested) {
        return;
    }

    // Fire currently requires amr.max_level = 0, so its surface raster is
    // copied directly into the level-0 2D plot slab.
    AMREX_ALWAYS_ASSERT(inputs.level == 0);

    for (const auto& name : inputs.plot_var_names) {
        const auto* descriptor =
            plotfile2d::find_diagnostic(name);

        if (!is_fire_diagnostic(descriptor)) {
            continue;
        }

        const int src_comp =
            fire_checkpoint_raster_component(
                descriptor->id);

        AMREX_ALWAYS_ASSERT(src_comp >= 0);
        AMREX_ALWAYS_ASSERT(
            src_comp
            < ERFFireCheckpointRasterComponents::
                component_count);

        inputs.output.ParallelCopy(
            inputs.state.raster,
            src_comp,
            inputs.output_component,
            1,
            0,
            0);

        ++inputs.output_component;
    }
}

} // namespace ERFFire
