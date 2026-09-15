#include <ERF_FireContext.H>
#include <ERF_FireLevel0Advance.H>

#include <ERF_FireLevel0Environment.H>
#include <ERF_FireLevel0SourceCoupling.H>
#include <ERF_FireLevel0TerrainWindSampler.H>
#include <ERF_FireRuntimeInit.H>
#include <ERF_FireSpreadOutput.H>
#include <ERF_FireSurfaceFeedback.H>

#include <AMReX.H>

#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace ERFFire
{

void
ERFFireContext::advance_level0(
    const ERFFireLevel0AdvanceInputs& inputs)
{
    using amrex::Error;
    using amrex::MultiFab;
    using amrex::Real;

    if (!runtime_options_.enabled) {
        return;
    }

    const ERFFireLevel0EnvironmentInputs& environment_inputs =
        inputs.environment;

    // Explicit Fire coupling sequencing:
    //
    //   already-fillpatched atmosphere at t^n
    //   -> bind one immutable t^n distributed environment view
    //   -> advance one candidate coupling-neutral Fire state across dt
    //   -> for two-way only, difference combustion history and construct
    //      the native ERF source tendency from the same t^n atmosphere
    //   -> atomically commit the Fire candidate/source.
    //
    // The caller invokes this method before ordinary ERF Advance.

    std::unique_ptr<ERFFireLevel0TerrainWindSampler>
        next_terrain_sampler;
    std::unique_ptr<ERFFireLevel0FlatWindSampler>
        next_flat_sampler;

    const Real fire_reference_height_agl_m =
        runtime_options_.wind_mode
                == ERFFireWindMode::ExplicitWaf20ft
            ? explicit_waf_20ft_reference_height_agl_m
            : runtime_options_.reference_height_agl_m;

    if (environment_inputs.mesh_type == MeshType::VariableDz) {
        next_terrain_sampler =
            std::make_unique<ERFFireLevel0TerrainWindSampler>(
                environment_inputs,
                fire_reference_height_agl_m);
    } else {
        next_flat_sampler =
            std::make_unique<ERFFireLevel0FlatWindSampler>(
                environment_inputs,
                fire_reference_height_agl_m);
    }

    environment_snapshot_time_ = inputs.time_s;
    environment_reference_height_agl_m_ =
        static_cast<double>(
            fire_reference_height_agl_m);

    if (!spread_runtime_) {
        spread_runtime_ =
            make_erf_fire_spread_runtime(
                runtime_options_,
                environment_inputs.geometry,
                static_cast<Real>(inputs.time_s));

        step_index_ = 0;

        if (runtime_options_.output_interval_steps > 0) {
            write_erf_fire_spread_snapshot(
                *spread_runtime_,
                runtime_options_.output_dir,
                step_index_);
        }
    }

    if (next_terrain_sampler
        && !terrain_surface_) {

        const FireCartesianRasterGeometry2D
            fire_terrain_geometry =
                spread_runtime_
                    ->config()
                    .raster_geometry;

        // This lookup may trigger collective regular-text terrain loading.
        // It is intentionally performed by all ranks here, at the same point in
        // the coupling sequence as the pre-extraction timestep implementation.
        const ERFTerrainSource* shared_terrain_source =
            resolve_terrain_source();

        if (shared_terrain_source != nullptr) {
            terrain_surface_ =
                std::make_unique<FireTerrainSurface>(
                    make_erf_terrain_source_surface_on_geometry(
                        *shared_terrain_source,
                        fire_terrain_geometry));
        } else {
            terrain_surface_ =
                std::make_unique<FireTerrainSurface>(
                    make_erf_level0_terrain_surface_on_geometry(
                        environment_inputs,
                        fire_terrain_geometry));
        }
    }

    if (spread_runtime_->current_time_s()
        != static_cast<Real>(inputs.time_s)) {
        Error(
            "ERF-Fire runtime clock is not synchronized with level-0 t^n");
    }

    ERFFireSpreadRuntime next_fire_runtime =
        *spread_runtime_;

    const FireEnvironmentBatchFunction
        flat_environment =
            [&next_flat_sampler](
                const std::vector<FireVec2>& positions_m) {
                return next_flat_sampler->sample_points(
                    positions_m);
            };

    const FireEnvironmentBatchFunction
        terrain_environment =
            [&next_terrain_sampler](
                const std::vector<FireVec2>& positions_m) {
                return next_terrain_sampler->sample_points(
                    positions_m);
            };

    if (runtime_options_.wind_mode
            == ERFFireWindMode::DirectReference) {
        if (next_terrain_sampler) {
            (void)next_fire_runtime
                .advance_direct_reference_wind_batched(
                    terrain_environment,
                    *terrain_surface_,
                    inputs.dt_s);
        } else {
            (void)next_fire_runtime
                .advance_direct_reference_wind_batched(
                    flat_environment,
                    inputs.dt_s);
        }
    } else if (runtime_options_.wind_mode
               == ERFFireWindMode::ExplicitWaf20ft) {
        if (next_terrain_sampler) {
            (void)next_fire_runtime
                .advance_explicit_waf_20ft_batched(
                    terrain_environment,
                    *terrain_surface_,
                    runtime_options_.wind_adjustment_factor,
                    inputs.dt_s);
        } else {
            (void)next_fire_runtime
                .advance_explicit_waf_20ft_batched(
                    flat_environment,
                    runtime_options_.wind_adjustment_factor,
                    inputs.dt_s);
        }
    } else {
        Error("unsupported ERF-Fire wind mode");
    }

    std::unique_ptr<MultiFab> next_fire_source;
    double next_fire_source_time =
        std::numeric_limits<double>::quiet_NaN();

    if (runtime_options_.coupling_mode
        == ERFFireCouplingMode::TwoWay) {
        const FireSurfaceFeedbackRaster feedback =
            make_fire_surface_feedback_increment(
                spread_runtime_->combustion_raster(),
                next_fire_runtime.combustion_raster());

        const ERFFireAtmosphericSourceOptions source_options{
            runtime_options_.feedback_extinction_depth_m};

        if (environment_inputs.mesh_type
            == MeshType::VariableDz) {
            if (inputs.detJ_cc == nullptr) {
                Error(
                    "ERF-Fire VariableDz two-way coupling requires detJ_cc");
            }

            next_fire_source =
                make_erf_fire_level0_terrain_source_tendency(
                    feedback,
                    environment_inputs,
                    *inputs.detJ_cc,
                    inputs.conserved_state_tn,
                    inputs.moisture_type,
                    inputs.dt_s,
                    source_options);
        } else {
            next_fire_source =
                make_erf_fire_level0_source_tendency(
                    feedback,
                    environment_inputs,
                    inputs.conserved_state_tn,
                    inputs.moisture_type,
                    inputs.dt_s,
                    source_options);
        }

        next_fire_source_time = inputs.time_s;
    }

    // Commit only after sampling, Fire evolution, and optional source
    // construction have all completed successfully.
    *spread_runtime_ =
        std::move(next_fire_runtime);

    atmospheric_source_tendency_ =
        std::move(next_fire_source);

    atmospheric_source_time_ =
        next_fire_source_time;

    ++step_index_;

    if (runtime_options_.output_interval_steps > 0
        && step_index_
               % runtime_options_.output_interval_steps
            == 0) {
        write_erf_fire_spread_snapshot(
            *spread_runtime_,
            runtime_options_.output_dir,
            step_index_);
    }
}

} // namespace ERFFire
