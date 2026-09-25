#ifndef ERF_FIRE_SPREAD_RUNTIME_H_
#define ERF_FIRE_SPREAD_RUNTIME_H_

#include <ERF_FireBurnedFractionRaster.H>
#include <ERF_FireCombustionRaster.H>
#include <ERF_FireFirstArrivalRaster.H>
#include <ERF_FireFront.H>
#include <ERF_FireFuelField.H>
#include <ERF_FireFuelRaster.H>
#include <ERF_FireFlatEnvironmentSampler.H>
#include <ERF_FireTerrainSurface.H>
#include <ERF_FirePerimeterRemesher.H>
#include <ERF_RothermelFuel.H>

#include <AMReX_REAL.H>

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ERFFire
{
struct ERFFireFrontComponentState
{
    FireFrontRole role{FireFrontRole::Outer};
    std::vector<FireVec2> vertices_m;
};


/**
 * Map-plane fire-spread configuration.
 *
 * The direct-reference-wind advance path deliberately performs no wind
 * adjustment factor or hidden height conversion. It is valid only when the
 * configured atmospheric reference height is itself the intended Rothermel
 * model-wind height for the controlled case.
 */
struct ERFFireSpreadConfig
{
    RothermelFuelParameters fuel{};
    amrex::Real dead_fuel_moisture_fraction{};
    FireCombustionParameters combustion_parameters{};
    FireCombustionRasterOptions combustion_options{};
    FirePerimeterRemeshOptions remesh_options{};
    FireCartesianRasterGeometry2D raster_geometry{};
    amrex::Real arrival_time_tolerance_s{amrex::Real(1.0e-6)};

    // Canonical categorical material for the ERF-facing uniform path.
    // Runtime construction sources categorical Dead1h moisture into the exact
    // legacy FM1 compatibility field; the spread arithmetic itself deliberately
    // remains unchanged so FM1 stays bitwise stable. A default FM1 material
    // with no moisture identifies legacy checkpoint/aggregate state, while
    // explicitly altered single-class fuel/moisture remains a compatibility
    // path. Keeping this field last preserves existing aggregate initializers.
    FireFuelMaterial uniform_material{};
};

struct ERFFireSpreadRuntimeState
{
    ERFFireSpreadConfig config{};
    std::vector<FireVec2> perimeter_vertices_m;
    FireBurnedFractionRasterState burned_fraction;
    FireFirstArrivalRasterState first_arrival;
    FireCombustionRasterState combustion;
    double current_time_s{};
    std::vector<ERFFireFrontComponentState> front_components;
};

struct ERFFireStepDiagnostics
{
    double start_time_s{};
    double end_time_s{};

    std::size_t pre_remesh_vertex_count{};
    std::size_t post_remesh_vertex_count{};
    std::size_t vertices_removed{};
    std::size_t vertices_added{};

    std::size_t newly_arrived_cell_count{};
    std::size_t arrived_cell_count{};

    amrex::Real newly_burned_area_m2{};
    amrex::Real burned_area_m2{};

    amrex::Real newly_consumed_dry_fuel_kg{};
    amrex::Real remaining_dry_fuel_kg{};
    amrex::Real consumed_dry_fuel_kg{};

    amrex::Real sensible_energy_increment_j{};
    amrex::Real sensible_energy_j{};

    amrex::Real water_released_increment_kg{};
    amrex::Real water_released_kg{};
};

/**
 * Owning map-plane Fire spread runtime state.
 *
 * FirePerimeter is authoritative. Burned fraction and first arrival are
 * persistent history derived from the propagated pre-remesh perimeter.
 * Cell-resolved combustion state is synchronized with burned history and is
 * advanced from the same fixed-topology physical sweep before remeshing.
 *
 * Each advance call consumes one immutable atmospheric snapshot for the whole
 * fire step. The RK2 normal-speed callback samples that same snapshot at each
 * current and midpoint vertex location. Public entry points make wind
 * preprocessing explicit: DirectReference is identity, while ExplicitWaf20ft
 * applies the scalar 20-ft-to-midflame transform before Rothermel evaluation.
 * No atmospheric state is retained or modified.
 *
 * Coupling direction is deliberately not encoded in this class. ERF may use
 * the same fire-state evolution path in one-way or two-way mode; atmospheric
 * feedback/source deposition belongs to the ERF coupling driver, not to a
 * separate fire-propagation implementation.
 */
class ERFFireSpreadRuntime
{
public:
    ERFFireSpreadRuntime(
        FirePerimeter initial_perimeter,
        double initial_time_s,
        ERFFireSpreadConfig config);

    ERFFireSpreadRuntime(
        FireFront initial_front,
        double initial_time_s,
        ERFFireSpreadConfig config);

    /**
     * Construct an FM1 runtime with prescribed cellwise material.
     *
     * The spatial raster is immutable/shareable and must exactly match the
     * Fire runtime geometry. This first production spatial path supports only
     * canonical FM1 spread/combustion parameters with cellwise dead 1-h
     * moisture. NonBurnable remains rejected until barrier dynamics exist.
     * Spatial material is consumed only by batched RK advance entry points.
     */
    ERFFireSpreadRuntime(
        FirePerimeter initial_perimeter,
        double initial_time_s,
        ERFFireSpreadConfig config,
        FireFuelRaster spatial_fuel_raster);

    ERFFireSpreadRuntime(
        FireFront initial_front,
        double initial_time_s,
        ERFFireSpreadConfig config,
        FireFuelRaster spatial_fuel_raster);

    [[nodiscard]] const FireFront&
    front() const;

    [[nodiscard]] const FirePerimeter& perimeter() const
    {
        const auto& components =
            front_.components();
        if (components.size() != 1
            || components.front().role
                != FireFrontRole::Outer) {
            throw std::logic_error(
                "Fire runtime perimeter view requires one Outer component");
        }
        return components.front().perimeter;
    }

    [[nodiscard]] const FireBurnedFractionRaster&
    burned_fraction_raster() const noexcept
    {
        return burned_fraction_;
    }

    [[nodiscard]] const FireFirstArrivalRaster&
    first_arrival_raster() const noexcept
    {
        return first_arrival_;
    }

    [[nodiscard]] const FireCombustionRaster&
    combustion_raster() const noexcept
    {
        return combustion_;
    }

    [[nodiscard]] double current_time_s() const noexcept
    {
        return current_time_s_;
    }

    [[nodiscard]] const ERFFireSpreadConfig&
    config() const noexcept
    {
        return config_;
    }

    /**
     * Read-only uniform compatibility material derived from config().
     *
     * Uniform runtimes evaluate this field directly. Spatial runtimes retain
     * it only as their legacy/config compatibility view; production spread and
     * combustion consume spatial_fuel_raster() instead.
     */
    [[nodiscard]] const FireFuelField&
    fuel_field() const noexcept
    {
        return fuel_field_;
    }

    [[nodiscard]] bool
    has_spatial_fuel() const noexcept
    {
        return spatial_fuel_raster_.has_value();
    }

    /**
     * Immutable distributed spatial material, or nullptr for a uniform runtime.
     * Copies of a spatial runtime share the static raster storage.
     */
    [[nodiscard]] const FireFuelRaster*
    spatial_fuel_raster() const noexcept
    {
        return spatial_fuel_raster_
            ? &*spatial_fuel_raster_
            : nullptr;
    }

    /**
     * Return one canonical runtime snapshot in a single-rank runtime.
     *
     * This accessor is noncollective. Parallel checkpoint/output code must use
     * collective_snapshot_state_to_io_rank() so ordinary runtime access never
     * hides full-domain communication. Spatial-fuel runtimes reject this legacy
     * canonical snapshot because v4 persists material in a distributed raster.
     */
    [[nodiscard]] ERFFireSpreadRuntimeState
    snapshot_state() const;

    /**
     * Collectively pack legacy canonical raster state onto the AMReX IO rank.
     *
     * Every rank must call this method in the same collective order. The
     * returned perimeter/configuration/scalars are valid on every rank, while
     * full canonical raster vectors are populated only on IOProcessor().
     * Spatial-fuel runtimes reject this legacy packing path; v4 checkpointing
     * preserves spatial material in its distributed checkpoint raster instead.
     */
    [[nodiscard]] ERFFireSpreadRuntimeState
    collective_snapshot_state_to_io_rank() const;

    [[nodiscard]] static ERFFireSpreadRuntime
    restore_from_state(ERFFireSpreadRuntimeState state);

    /**
     * Collectively restore a version-1 state whose full canonical raster
     * vectors and perimeter are present only on IOProcessor(). Config and
     * current_time_s must already be valid and identical on every rank.
     */
    [[nodiscard]] static ERFFireSpreadRuntime
    collective_restore_from_io_rank_state(
        ERFFireSpreadRuntimeState state);

    /**
     * Collectively restore version-2/3 dynamic state from a distributed
     * checkpoint MultiFab. Config/current_time_s must already be identical on
     * every rank; perimeter and scalar raster metadata may be populated only on
     * IOProcessor(). Spatial v4 material is attached only after its independent
     * schema/fingerprint validation succeeds.
     */
    [[nodiscard]] static ERFFireSpreadRuntime
    collective_restore_from_checkpoint_raster(
        ERFFireSpreadRuntimeState state,
        const amrex::MultiFab& checkpoint_raster,
        bool combustion_state_already_validated = false);

    /**
     * Attach already validated spatial material to restored dynamic state.
     *
     * Every rank must participate. Geometry and canonical FM1 base physics are
     * checked, then the raw restored combustion components are validated
     * against the immutable cellwise material before the runtime is exposed.
     */
    [[nodiscard]] static ERFFireSpreadRuntime
    attach_spatial_fuel_after_checkpoint_restore(
        ERFFireSpreadRuntime restored_runtime,
        FireFuelRaster spatial_fuel_raster)
    {
        if (restored_runtime.spatial_fuel_raster_) {
            throw std::logic_error(
                "restored Fire runtime already owns spatial fuel");
        }

        const auto& runtime_geometry =
            restored_runtime.config_.raster_geometry;
        const auto& fuel_geometry =
            spatial_fuel_raster.geometry();
        if (runtime_geometry.nx != fuel_geometry.nx
            || runtime_geometry.ny != fuel_geometry.ny
            || runtime_geometry.xlo_m != fuel_geometry.xlo_m
            || runtime_geometry.ylo_m != fuel_geometry.ylo_m
            || runtime_geometry.dx_m != fuel_geometry.dx_m
            || runtime_geometry.dy_m != fuel_geometry.dy_m) {
            throw std::invalid_argument(
                "restored spatial Fire fuel geometry does not match runtime geometry");
        }

        const RothermelFuelParameters expected_fuel =
            make_fm1_fuel_parameters();
        const auto& fuel = restored_runtime.config_.fuel;
        const bool same_fuel =
            fuel.dead_1h_load_kg_m2 == expected_fuel.dead_1h_load_kg_m2
            && fuel.dead_1h_sav_m_inv == expected_fuel.dead_1h_sav_m_inv
            && fuel.fuel_bed_depth_m == expected_fuel.fuel_bed_depth_m
            && fuel.dead_heat_content_j_kg == expected_fuel.dead_heat_content_j_kg
            && fuel.particle_density_kg_m3 == expected_fuel.particle_density_kg_m3
            && fuel.total_mineral_fraction == expected_fuel.total_mineral_fraction
            && fuel.effective_mineral_fraction == expected_fuel.effective_mineral_fraction
            && fuel.dead_moisture_of_extinction == expected_fuel.dead_moisture_of_extinction;
        if (!same_fuel) {
            throw std::invalid_argument(
                "restored spatial Fire runtime requires canonical FM1 spread parameters");
        }

        const FireCombustionParameters expected_combustion =
            make_fm1_combustion_parameters(
                restored_runtime.config_
                    .dead_fuel_moisture_fraction);
        const auto& combustion =
            restored_runtime.config_.combustion_parameters;
        const bool same_combustion =
            combustion.dry_fuel_load_kg_m2
                == expected_combustion.dry_fuel_load_kg_m2
            && combustion.sensible_heat_release_j_kg_dry
                == expected_combustion.sensible_heat_release_j_kg_dry
            && combustion.fuel_moisture_fraction
                == expected_combustion.fuel_moisture_fraction
            && combustion.burn_time_constant_s
                == expected_combustion.burn_time_constant_s
            && combustion.combustion_water_yield_kg_per_kg_dry
                == expected_combustion.combustion_water_yield_kg_per_kg_dry;
        if (!same_combustion) {
            throw std::invalid_argument(
                "restored spatial Fire runtime requires canonical FM1 combustion parameters");
        }

        restored_runtime.combustion_
            .validate_restored_spatial_accounting(
                spatial_fuel_raster);
        restored_runtime.spatial_fuel_raster_.emplace(
            std::move(spatial_fuel_raster));
        return restored_runtime;
    }

    /**
     * Advance one fire step using sampled reference wind directly as the
     * Rothermel model wind.
     *
     * This is an explicit controlled-case compatibility mode, not a WAF model:
     * the sampled reference-wind vector is an identity mapping to the Rothermel
     * model/midflame-wind input. This overload uses exactly zero terrain slope.
     * The atmospheric wind vector is the map-plane push-to direction. Any wind
     * adjustment must occur before Rothermel/Richards evaluation.
     *
     * Strong exception guarantee: persistent perimeter/raster/combustion/time
     * state is committed only after propagation, arrival, burned-history,
     * combustion, and remeshing all succeed.
     */
    [[nodiscard]] ERFFireStepDiagnostics
    advance_direct_reference_wind(
        const FireFlatEnvironmentSampler& environment,
        amrex::Real dt_s);

    /**
     * Direct-reference advance with one ordered environment batch per RK
     * stage. The batch sample carries both reference wind and the represented
     * map-plane terrain gradient; flat samplers supply exact zero gradient.
     * Spatial fuel is sampled collectively once for the same ordered batch.
     */
    [[nodiscard]] ERFFireStepDiagnostics
    advance_direct_reference_wind_batched(
        const FireEnvironmentBatchFunction& environment,
        amrex::Real dt_s);

    [[nodiscard]] ERFFireStepDiagnostics
    advance_direct_reference_wind_batched(
        const FireEnvironmentBatchFunction& environment,
        const FireTerrainSurface& terrain,
        amrex::Real dt_s);

    /**
     * Advance one Fire step with direct reference wind plus terrain slope.
     *
     * Fire remains a map-plane model: terrain contributes the local
     * horizontal gradient (dh/dx, dh/dy), its magnitude is the Rothermel
     * slope tangent, and its normalized direction is the Richards upslope
     * direction. No terrain-surface arclength rescaling or WAF is applied;
     * the sampled reference wind is used unchanged as model/midflame wind.
     */
    [[nodiscard]] ERFFireStepDiagnostics
    advance_direct_reference_wind(
        const FireFlatEnvironmentSampler& environment,
        const FireTerrainSurface& terrain,
        amrex::Real dt_s);

    /**
     * Advance one Fire step from a 20-ft reference-wind snapshot using an
     * explicit scalar WAF before Rothermel/Richards evaluation.
     *
     * The environment must have been sampled at 20 ft = 6.096 m local AGL by
     * the ERF-facing integration layer. The complete map-plane reference-wind
     * vector is scaled; terrain slope, when supplied, remains independent.
     */
    [[nodiscard]] ERFFireStepDiagnostics
    advance_explicit_waf_20ft(
        const FireFlatEnvironmentSampler& environment,
        amrex::Real wind_adjustment_factor,
        amrex::Real dt_s);

    /**
     * Explicit-WAF advance with batched wind and terrain-gradient sampling.
     * Flat samplers supply exact zero terrain gradient. Spatial fuel uses the
     * same ordered batch contract and one collective material sample per call.
     */
    [[nodiscard]] ERFFireStepDiagnostics
    advance_explicit_waf_20ft_batched(
        const FireEnvironmentBatchFunction& environment,
        amrex::Real wind_adjustment_factor,
        amrex::Real dt_s);

    [[nodiscard]] ERFFireStepDiagnostics
    advance_explicit_waf_20ft_batched(
        const FireEnvironmentBatchFunction& environment,
        const FireTerrainSurface& terrain,
        amrex::Real wind_adjustment_factor,
        amrex::Real dt_s);

    [[nodiscard]] ERFFireStepDiagnostics
    advance_explicit_waf_20ft(
        const FireFlatEnvironmentSampler& environment,
        const FireTerrainSurface& terrain,
        amrex::Real wind_adjustment_factor,
        amrex::Real dt_s);

private:
    struct RestoreStateTag {};
    struct CollectiveRestoreStateTag {};

    ERFFireSpreadRuntime(
        ERFFireSpreadRuntimeState state,
        RestoreStateTag);

    ERFFireSpreadRuntime(
        ERFFireSpreadConfig config,
        FireFront front,
        FireBurnedFractionRaster burned_fraction,
        FireFirstArrivalRaster first_arrival,
        FireCombustionRaster combustion,
        double current_time_s,
        CollectiveRestoreStateTag);

    enum class WindInputMode
    {
        DirectReference,
        ExplicitWaf20ft
    };

    [[nodiscard]] ERFFireStepDiagnostics
    advance_wind_impl(
        const FireFlatEnvironmentSampler& environment,
        const FireTerrainSurface* terrain_surface,
        WindInputMode wind_input_mode,
        amrex::Real wind_adjustment_factor,
        amrex::Real dt_s);

    [[nodiscard]] ERFFireStepDiagnostics
    advance_wind_batched_impl(
        const FireEnvironmentBatchFunction& environment,
        WindInputMode wind_input_mode,
        amrex::Real wind_adjustment_factor,
        amrex::Real dt_s);

    ERFFireSpreadConfig config_;
    // Value-owned and read-only to callers; ordinary uniform runtime copies
    // retain no references into another runtime.
    FireFuelField fuel_field_;
    // Static distributed material is immutable/shareable. Optional keeps the
    // existing uniform runtime O(1) and legacy checkpoint schema unchanged.
    std::optional<FireFuelRaster> spatial_fuel_raster_;
    FireFront front_;
    FireBurnedFractionRaster burned_fraction_;
    FireFirstArrivalRaster first_arrival_;
    FireCombustionRaster combustion_;
    double current_time_s_{};
};

} // namespace ERFFire

#endif
