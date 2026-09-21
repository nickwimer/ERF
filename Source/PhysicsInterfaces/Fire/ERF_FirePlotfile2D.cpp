#include <ERF.H>
#include <ERF_FirePlotfile2D.H>

#include <ERF_FireContext.H>
#include <ERF_FireRuntimeInit.H>
#include <ERF_FireSpreadOutput.H>

#include <AMReX.H>

#include <cstddef>
#include <memory>
#include <utility>

namespace
{

struct FirePlotfile2DDiagnosticSpec
{
    const char* name;
    const char* long_name;
    const char* units;
    int checkpoint_component;
};

const amrex::Vector<FirePlotfile2DDiagnosticSpec>&
fire_plotfile2d_catalog()
{
    using Components =
        ERFFire::ERFFireCheckpointRasterComponents;

    static const amrex::Vector<FirePlotfile2DDiagnosticSpec> catalog{
        {
            "fire_burned_fraction",
            "Persistent Fire burned-area fraction",
            "1",
            Components::burned_fraction
        },
        {
            "fire_has_arrived",
            "Fire first-arrival validity mask",
            "1",
            Components::arrived
        },
        {
            "fire_first_arrival_time_s",
            "First positive-area Fire arrival time; valid where fire_has_arrived is 1",
            "s",
            Components::first_arrival_time_s
        },
        {
            "fire_ignited_area_fraction",
            "Fire combustion ignited-area fraction",
            "1",
            Components::ignited_area_fraction
        },
        {
            "fire_remaining_dry_fuel_kg_m2",
            "Remaining oven-dry Fire fuel load",
            "kg/m^2",
            Components::remaining_dry_fuel_kg_m2
        },
        {
            "fire_consumed_dry_fuel_kg_m2",
            "Cumulative consumed oven-dry Fire fuel load",
            "kg/m^2",
            Components::consumed_dry_fuel_kg_m2
        },
        {
            "fire_sensible_energy_j_m2",
            "Cumulative Fire sensible energy released per unit surface area",
            "J/m^2",
            Components::sensible_energy_j_m2
        },
        {
            "fire_water_released_kg_m2",
            "Cumulative Fire combustion water released per unit surface area",
            "kg/m^2",
            Components::water_released_kg_m2
        }
    };

    return catalog;
}

const FirePlotfile2DDiagnosticSpec*
find_fire_plotfile2d_diagnostic(
    const std::string& name) noexcept
{
    for (const auto& descriptor : fire_plotfile2d_catalog()) {
        if (name == descriptor.name) {
            return &descriptor;
        }
    }

    return nullptr;
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

} // namespace

namespace ERFFire
{

ERFFirePlotfile2DDescriptorSelection
select_fire_plotfile2d_output_descriptors(
    const amrex::Vector<std::string>& plot_var_names)
{
    ERFFirePlotfile2DDescriptorSelection result;

    result.non_fire_plot_var_names.reserve(
        plot_var_names.size());

    result.fire_descriptors.reserve(
        fire_plotfile2d_catalog().size());

    for (const auto& name : plot_var_names) {
        const auto* fire_descriptor =
            find_fire_plotfile2d_diagnostic(name);

        if (fire_descriptor == nullptr) {
            result.non_fire_plot_var_names.push_back(name);
            continue;
        }

        plotfile2d::Plotfile2DOutputDescriptor descriptor;
        descriptor.name = fire_descriptor->name;
        descriptor.long_name = fire_descriptor->long_name;
        descriptor.units = fire_descriptor->units;
        descriptor.category =
            plotfile2d::DiagnosticCategory::SurfaceState;
        descriptor.missing_policy =
            plotfile2d::MissingPolicy::AlwaysAvailable;
        descriptor.missing_value = amrex::Real(0.0);

        result.fire_descriptors.push_back(
            std::move(descriptor));
    }

    return result;
}

void
ERFFireContext::append_available_plotfile2d_diagnostics(
    const ERFFirePlotfile2DAvailabilityInputs& inputs) const
{
    if (!runtime_options_.enabled) {
        return;
    }

    // Fire fields are appended after all generic ERF diagnostics. This
    // preserves every existing non-Fire component index.
    for (const auto& descriptor :
         fire_plotfile2d_catalog()) {
        inputs.available_names.push_back(
            descriptor.name);
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
            || find_fire_plotfile2d_diagnostic(name)
                != nullptr;
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
            find_fire_plotfile2d_diagnostic(name);

        if (descriptor == nullptr) {
            continue;
        }

        const int src_comp =
            descriptor->checkpoint_component;

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


#ifdef ERF_USE_FIRE

void
ERF::append_fire_plotfile2d_available_names (
    amrex::Vector<std::string>& available_names) const
{
    if (!m_fire) {
        return;
    }

    const ERFFire::ERFFirePlotfile2DAvailabilityInputs inputs{
        available_names
    };

    m_fire->append_available_plotfile2d_diagnostics(inputs);
}

void
ERF::build_fire_plotfile2d_output_descriptors (
    int which,
    const amrex::Vector<std::string>& plot_var_names,
    amrex::Vector<plotfile2d::Plotfile2DOutputDescriptor>&
        output_descriptors) const
{
    const auto selection =
        ERFFire::select_fire_plotfile2d_output_descriptors(
            plot_var_names);

    output_descriptors =
        plotfile2d::build_sampled_level_output_descriptors(
            pp_prefix,
            which,
            selection.non_fire_plot_var_names,
            solverChoice);

    const auto insert_position =
        output_descriptors.begin()
        + static_cast<std::ptrdiff_t>(
            selection.non_fire_plot_var_names.size());

    output_descriptors.insert(
        insert_position,
        selection.fire_descriptors.begin(),
        selection.fire_descriptors.end());
}

void
ERF::fill_fire_plotfile2d_diagnostics (
    const amrex::Vector<std::string>& plot_var_names,
    int level,
    amrex::MultiFab& output,
    int& output_component) const
{
    if (!m_fire) {
        return;
    }

    const ERFFire::ERFFirePlotfile2DPrepareInputs
        prepare_inputs{
            plot_var_names,
            geom[0],
            istep[0],
            static_cast<amrex::Real>(t_new[0])
        };

    const auto state =
        m_fire->prepare_plotfile2d_diagnostics(
            prepare_inputs);

    const ERFFire::ERFFirePlotfile2DFillInputs
        fill_inputs{
            plot_var_names,
            state,
            level,
            output,
            output_component
        };

    m_fire->fill_plotfile2d_diagnostics(
        fill_inputs);
}

#endif
