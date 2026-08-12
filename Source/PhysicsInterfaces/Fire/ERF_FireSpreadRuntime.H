#ifndef ERF_FIRE_SPREAD_RUNTIME_H_
#define ERF_FIRE_SPREAD_RUNTIME_H_

#include <ERF_FireBurnedFractionRaster.H>
#include <ERF_FireCombustionRaster.H>
#include <ERF_FireFirstArrivalRaster.H>
#include <ERF_FireFlatEnvironmentSampler.H>
#include <ERF_FirePerimeterRemesher.H>
#include <ERF_RothermelFuel.H>

#include <AMReX_REAL.H>

#include <cstddef>

namespace ERFFire
{

/**
 * Flat-grid fire-spread configuration.
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
 * Owning flat-fire spread runtime state.
 *
 * FirePerimeter is authoritative. Burned fraction and first arrival are
 * persistent history derived from the propagated pre-remesh perimeter.
 * Cell-resolved combustion state is synchronized with burned history and is
 * advanced from the same fixed-topology physical sweep before remeshing.
 *
 * Each advance_direct_reference_wind() call consumes one immutable atmospheric
 * snapshot for the whole fire step. The RK2 normal-speed callback samples that
 * same snapshot at each current and midpoint vertex location. No atmospheric
 * state is retained or modified.
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

    /**
     * Advance one fire step using sampled reference wind directly as the
     * Rothermel model wind.
     *
     * This is an explicit controlled-case coupling mode, not a WAF model.
     * Flat terrain implies zero slope. The atmospheric wind vector is the
     * push-to direction.
     *
     * Strong exception guarantee: persistent perimeter/raster/combustion/time
     * state is committed only after propagation, arrival, burned-history,
     * combustion, and remeshing all succeed.
     */
    [[nodiscard]] ERFFireStepDiagnostics
    advance_direct_reference_wind(
        const FireFlatEnvironmentSampler& environment,
        amrex::Real dt_s);

private:
    ERFFireSpreadConfig config_;
    FirePerimeter perimeter_;
    FireBurnedFractionRaster burned_fraction_;
    FireFirstArrivalRaster first_arrival_;
    FireCombustionRaster combustion_;
    amrex::Real current_time_s_{};
};

} // namespace ERFFire

#endif
