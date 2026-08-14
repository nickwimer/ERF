#ifndef ERF_FIRE_COMBUSTION_RASTER_H_
#define ERF_FIRE_COMBUSTION_RASTER_H_

#include <ERF_FireBurnedFractionRaster.H>
#include <ERF_FireCombustion.H>
#include <ERF_FirePerimeter.H>

#include <AMReX_REAL.H>

#include <cstddef>
#include <vector>

namespace ERFFire
{

struct FireCombustionRasterOptions
{
    /**
     * Number of equal temporal bins used to resolve newly ignited fractional
     * area during one physical perimeter sweep.
     *
     * Each bin's newly burned area is represented as an ignition cohort at
     * that bin's temporal midpoint. Existing fuel is advanced exactly with the
     * exponential law on each half-bin. This is a convergent temporal
     * quadrature for partial-area ignition, not a first-arrival clock.
     */
    std::size_t temporal_substeps{16};
};

struct FireCombustionRasterTotals
{
    amrex::Real remaining_dry_fuel_kg{};
    amrex::Real consumed_dry_fuel_kg{};
    amrex::Real sensible_energy_j{};
    amrex::Real water_released_kg{};
};

struct FireCombustionRasterAdvance
{
    FireCombustionRasterTotals totals{};
    amrex::Real newly_consumed_dry_fuel_kg{};
    amrex::Real sensible_energy_increment_j{};
    amrex::Real water_released_increment_kg{};
};

struct FireCombustionRasterState
{
    std::vector<FireCombustionState> cells;
    bool initialized{false};
};

/**
 * Cell-resolved combustion state on the same Cartesian geometry as burn
 * history.
 *
 * The combustion ignited-area fraction is required to equal the persistent
 * burned-fraction history at every committed state. The raster does not use
 * first-arrival time as its combustion clock. Instead, during a fixed-topology
 * sweep it samples the continuous vector perimeter at temporal bin endpoints,
 * obtains each cell's incremental burned area, and places that increment at
 * the corresponding bin midpoint before exact exponential decay.
 *
 * This class owns combustion state only. FirePerimeter remains authoritative
 * geometry and FireBurnedFractionRaster remains authoritative area history.
 */
class FireCombustionRaster
{
public:
    FireCombustionRaster(
        FireCartesianRasterGeometry2D geometry,
        FireCombustionParameters parameters,
        FireCombustionRasterOptions options = {});

    FireCombustionRaster(
        FireCartesianRasterGeometry2D geometry,
        FireCombustionParameters parameters,
        FireCombustionRasterOptions options,
        FireCombustionRasterState state);

    [[nodiscard]] FireCombustionRasterState
    snapshot_state() const
    {
        return {states_, initialized_};
    }

    [[nodiscard]] const FireCartesianRasterGeometry2D&
    geometry() const noexcept
    {
        return geometry_;
    }

    [[nodiscard]] const FireCombustionParameters&
    parameters() const noexcept
    {
        return parameters_;
    }

    [[nodiscard]] const FireCombustionRasterOptions&
    options() const noexcept
    {
        return options_;
    }

    [[nodiscard]] bool initialized() const noexcept
    {
        return initialized_;
    }

    [[nodiscard]] const FireCombustionState&
    state(std::size_t i, std::size_t j) const;

    [[nodiscard]] FireCombustionRasterTotals totals() const noexcept;

    /**
     * Seed initial ignition from existing burned-fraction history.
     *
     * All initially burned fractional area is ignited at the caller's initial
     * model time with no instantaneous consumption. May be called exactly once.
     */
    [[nodiscard]] FireCombustionRasterTotals
    initialize_from_burned_fraction(
        const FireBurnedFractionRaster& burned_fraction);

    /**
     * Advance combustion across one pre-remesh physical sweep.
     *
     * burned_before and burned_after are the persistent area histories at the
     * sweep endpoints. The temporal sampling must reproduce burned_after; a
     * sampled transient coverage larger than the endpoint history is rejected
     * because that would violate the current outward/monotone raster contract.
     *
     * Strong exception guarantee: persistent combustion state changes only
     * after the complete sweep, endpoint-history consistency checks, and all
     * extensive accounting succeed.
     */
    [[nodiscard]] FireCombustionRasterAdvance
    advance_from_linear_sweep(
        const FirePerimeter& start_perimeter,
        const FirePerimeter& end_perimeter,
        const FireBurnedFractionRaster& burned_before,
        const FireBurnedFractionRaster& burned_after,
        amrex::Real dt_s);

private:
    [[nodiscard]] std::size_t flat_index(
        std::size_t i,
        std::size_t j) const;

    [[nodiscard]] FireCombustionRasterTotals
    totals_for(
        const std::vector<FireCombustionState>& states) const noexcept;

    FireCartesianRasterGeometry2D geometry_{};
    FireCombustionParameters parameters_{};
    FireCombustionRasterOptions options_{};
    std::vector<FireCombustionState> states_;
    bool initialized_{false};
};

} // namespace ERFFire

#endif
