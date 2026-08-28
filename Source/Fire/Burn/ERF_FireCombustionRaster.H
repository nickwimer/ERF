#ifndef ERF_FIRE_COMBUSTION_RASTER_H_
#define ERF_FIRE_COMBUSTION_RASTER_H_

#include <ERF_FireBurnedFractionRaster.H>
#include <ERF_FireCombustion.H>
#include <ERF_FirePerimeter.H>
#include <ERF_FireSurfaceLayout.H>

#include <AMReX_MultiFab.H>
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

class FireFront;
class FireCombustionRaster;

namespace detail
{

[[nodiscard]] FireCombustionRasterAdvance
advance_fire_combustion_vertex_sweep(
    FireCombustionRaster& raster,
    const FirePerimeter& start_perimeter,
    const std::vector<FireVec2>& end_vertices_m,
    const FireFront* event_front,
    const FireFront* front_sweep_start,
    const FireFront* front_sweep_end,
    const FireBurnedFractionRaster& burned_before,
    const FireBurnedFractionRaster& burned_after,
    amrex::Real dt_s);

} // namespace detail

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
 * Combustion evolution is authoritative on the distributed AMReX Fire
 * surface layout. No persistent canonical row-major mirror is retained. Full
 * canonical cells are materialized only by explicit serialization/snapshot
 * boundaries; extensive totals remain replicated O(1) metadata.
 *
 * This class owns combustion state only. FirePerimeter remains authoritative
 * geometry and FireBurnedFractionRaster remains authoritative area history.
 */
class FireCombustionRaster
{
public:
    static constexpr int ignited_area_fraction_comp = 0;
    static constexpr int remaining_dry_fuel_comp = 1;
    static constexpr int consumed_dry_fuel_comp = 2;
    static constexpr int sensible_energy_comp = 3;
    static constexpr int water_released_comp = 4;
    static constexpr int component_count = 5;

    FireCombustionRaster(
        FireCartesianRasterGeometry2D geometry,
        FireCombustionParameters parameters,
        FireCombustionRasterOptions options = {});

    FireCombustionRaster(
        FireCartesianRasterGeometry2D geometry,
        FireCombustionParameters parameters,
        FireCombustionRasterOptions options,
        FireCombustionRasterState state);

    /**
     * Collectively restore version-1 canonical state owned only by IOProcessor().
     * Non-IO ranks may pass an empty canonical cell vector.
     */
    [[nodiscard]] static FireCombustionRaster
    collective_restore_from_io_rank_state(
        FireCartesianRasterGeometry2D geometry,
        FireCombustionParameters parameters,
        FireCombustionRasterOptions options,
        const FireCombustionRasterState& state);

    /**
     * Collectively restore five combustion components from a distributed
     * checkpoint raster.
     */
    [[nodiscard]] static FireCombustionRaster
    collective_restore_from_checkpoint_raster(
        FireCartesianRasterGeometry2D geometry,
        FireCombustionParameters parameters,
        FireCombustionRasterOptions options,
        const amrex::MultiFab& checkpoint,
        int source_comp,
        bool initialized);

    FireCombustionRaster(
        const FireCombustionRaster& other);
    FireCombustionRaster& operator=(
        const FireCombustionRaster& other);
    FireCombustionRaster(
        FireCombustionRaster&&) noexcept = default;
    FireCombustionRaster& operator=(
        FireCombustionRaster&&) noexcept = default;

    /**
     * Return one canonical row-major snapshot in a single-rank runtime.
     *
     * This accessor is deliberately noncollective. Multi-rank callers must
     * use collective_snapshot_state_to_io_rank() so ordinary state access
     * never hides full-domain communication.
     */
    [[nodiscard]] FireCombustionRasterState
    snapshot_state() const;

    /**
     * Collectively pack distributed combustion state onto the AMReX IO rank.
     *
     * Every rank must participate. The canonical cell vector is populated
     * only on IOProcessor(); the initialized flag is returned on every rank.
     */
    [[nodiscard]] FireCombustionRasterState
    collective_snapshot_state_to_io_rank() const;

    [[nodiscard]] const FireCartesianRasterGeometry2D&
    geometry() const noexcept
    {
        return geometry_;
    }

    [[nodiscard]] const FireSurfaceLayout&
    surface_layout() const noexcept
    {
        return surface_layout_;
    }

    /**
     * Read-only authoritative distributed combustion state.
     *
     * Co-located Fire physics should consume this view directly instead of
     * routing local cell reads through the canonical compatibility vector.
     */
    [[nodiscard]] const amrex::MultiFab&
    distributed_states() const noexcept
    {
        return states_mf_;
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

    /**
     * Return one combustion cell in a single-rank runtime.
     *
     * Parallel Fire physics consumes distributed_states() directly; this
     * accessor deliberately performs no hidden communication.
     */
    [[nodiscard]] FireCombustionState
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

    [[nodiscard]] FireCombustionRasterAdvance
    advance_from_front_linear_sweep(
        const FireFront& start_front,
        const FireFront& end_front,
        const FireBurnedFractionRaster& burned_before,
        const FireBurnedFractionRaster& burned_after,
        amrex::Real dt_s);

    [[nodiscard]] FireCombustionRasterAdvance
    advance_from_topology_event_sweep(
        const FirePerimeter& start_perimeter,
        const std::vector<FireVec2>& event_vertices_m,
        const FireFront& event_front,
        const FireBurnedFractionRaster& burned_before,
        const FireBurnedFractionRaster& burned_after,
        amrex::Real dt_s);

private:
    friend FireCombustionRasterAdvance
    detail::advance_fire_combustion_vertex_sweep(
        FireCombustionRaster& raster,
        const FirePerimeter& start_perimeter,
        const std::vector<FireVec2>& end_vertices_m,
        const FireFront* event_front,
        const FireFront* front_sweep_start,
        const FireFront* front_sweep_end,
        const FireBurnedFractionRaster& burned_before,
        const FireBurnedFractionRaster& burned_after,
        amrex::Real dt_s);

    [[nodiscard]] std::size_t flat_index(
        std::size_t i,
        std::size_t j) const;

    void scatter_canonical_to_distributed(
        const std::vector<FireCombustionState>& canonical,
        amrex::MultiFab& distributed) const;

    void copy_distributed_state(
        const amrex::MultiFab& source,
        amrex::MultiFab& destination) const;

    [[nodiscard]] std::vector<FireCombustionState>
    gather_distributed_to_canonical(
        const amrex::MultiFab& distributed) const;

    [[nodiscard]] FireCombustionRasterTotals
    totals_for(
        const std::vector<FireCombustionState>& states) const noexcept;

    FireCartesianRasterGeometry2D geometry_{};
    FireCombustionParameters parameters_{};
    FireCombustionRasterOptions options_{};
    FireSurfaceLayout surface_layout_;
    amrex::MultiFab states_mf_;
    FireCombustionRasterTotals totals_{};
    bool initialized_{false};
};

} // namespace ERFFire

#endif
