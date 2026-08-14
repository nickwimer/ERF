#ifndef ERF_FIRE_SPREAD_RUNTIME_H_
#define ERF_FIRE_SPREAD_RUNTIME_H_

#include <ERF_FireBurnedFractionRaster.H>
#include <ERF_FireCombustionRaster.H>
#include <ERF_FireFirstArrivalRaster.H>
#include <ERF_FireFlatEnvironmentSampler.H>
#include <ERF_FireTerrainSurface.H>
#include <ERF_FirePerimeterRemesher.H>
#include <ERF_RothermelFuel.H>

#include <AMReX_REAL.H>

#include <cstddef>
#include <vector>

namespace ERFFire
{

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
};

struct ERFFireSpreadRuntimeState
{
    ERFFireSpreadConfig config{};
    std::vector<FireVec2> perimeter_vertices_m;
    FireBurnedFractionRasterState burned_fraction;
    FireFirstArrivalRasterState first_arrival;
    FireCombustionRasterState combustion;
    amrex::Real current_time_s{};
};

struct ERFFireStepDiagnostics
{
    amrex::Real start_time_s{};
    amrex::Real end_time_s{};

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
        amrex::Real initial_time_s,
        ERFFireSpreadConfig config);

    [[nodiscard]] const FirePerimeter& perimeter() const noexcept
    {
        return perimeter_;
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

    [[nodiscard]] amrex::Real current_time_s() const noexcept
    {
        return current_time_s_;
    }

    [[nodiscard]] const ERFFireSpreadConfig&
    config() const noexcept
    {
        return config_;
    }

    [[nodiscard]] ERFFireSpreadRuntimeState
    snapshot_state() const;

    [[nodiscard]] static ERFFireSpreadRuntime
    restore_from_state(ERFFireSpreadRuntimeState state);

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

    [[nodiscard]] ERFFireStepDiagnostics
    advance_explicit_waf_20ft(
        const FireFlatEnvironmentSampler& environment,
        const FireTerrainSurface& terrain,
        amrex::Real wind_adjustment_factor,
        amrex::Real dt_s);

private:
    struct RestoreStateTag {};

    ERFFireSpreadRuntime(
        ERFFireSpreadRuntimeState state,
        RestoreStateTag);

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

    ERFFireSpreadConfig config_;
    FirePerimeter perimeter_;
    FireBurnedFractionRaster burned_fraction_;
    FireFirstArrivalRaster first_arrival_;
    FireCombustionRaster combustion_;
    amrex::Real current_time_s_{};
};

} // namespace ERFFire

#endif
