#ifndef ERF_FIRE_FIRST_ARRIVAL_RASTER_H
#define ERF_FIRE_FIRST_ARRIVAL_RASTER_H

#include <ERF_FireRasterGeometry.H>
#include <ERF_FireSurfaceLayout.H>

#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>
#include <AMReX_iMultiFab.H>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace ERFFire
{

class FirePerimeter;

/**
 * Aggregate result of one persistent first-arrival raster update.
 */
struct FireFirstArrivalRasterUpdate
{
    std::size_t arrived_cell_count{};
    std::size_t newly_arrived_cell_count{};
};

struct FireFirstArrivalRasterState
{
    std::vector<std::uint8_t> arrived;
    std::vector<amrex::Real> first_arrival_time_s;
    bool has_initial_condition{false};
    amrex::Real initial_condition_time_s{};
    bool has_committed_sweep{false};
    amrex::Real last_sweep_end_time_s{};
};

/**
 * Persistent first-positive-area arrival history on a Cartesian raster.
 *
 * The uint8_t mask is authoritative for whether a stored time is meaningful.
 * No floating-point sentinel value represents "not arrived".
 *
 * This is diagnostic/history state. It is deliberately separate from burned
 * fraction and is not the sole temporal state for later combustion physics.
 *
 * update_from_sweep() consumes one chronological, fixed-topology physical
 * perimeter sweep before remeshing. After the first successful update, each
 * subsequent sweep must start exactly at the previous committed sweep end
 * time; gaps, overlaps, and out-of-order updates are rejected.
 *
 * Expensive cell-arrival queries are distributed over the AMReX Fire surface
 * layout. During this migration stage, canonical row-major vectors remain as
 * synchronized compatibility state for checkpoint/output and existing scalar
 * accessors. Distributed and canonical state are committed atomically.
 */
class FireFirstArrivalRaster
{
public:
    explicit FireFirstArrivalRaster(
        const FireCartesianRasterGeometry2D& geometry);

    FireFirstArrivalRaster(
        const FireCartesianRasterGeometry2D& geometry,
        FireFirstArrivalRasterState state);

    FireFirstArrivalRaster(
        const FireFirstArrivalRaster& other);
    FireFirstArrivalRaster& operator=(
        const FireFirstArrivalRaster& other);
    FireFirstArrivalRaster(
        FireFirstArrivalRaster&&) noexcept = default;
    FireFirstArrivalRaster& operator=(
        FireFirstArrivalRaster&&) noexcept = default;

    [[nodiscard]] FireFirstArrivalRasterState
    snapshot_state() const
    {
        return {
            arrived_,
            first_arrival_time_s_,
            has_initial_condition_,
            initial_condition_time_s_,
            has_committed_sweep_,
            last_sweep_end_time_s_};
    }

    [[nodiscard]] const FireCartesianRasterGeometry2D&
    geometry () const noexcept
    {
        return geometry_;
    }

    [[nodiscard]] const FireSurfaceLayout&
    surface_layout() const noexcept
    {
        return surface_layout_;
    }

    [[nodiscard]] std::size_t cell_count () const noexcept
    {
        return arrived_.size();
    }

    [[nodiscard]] FireCartesianCell2D cell_bounds(
        std::size_t i,
        std::size_t j) const;

    [[nodiscard]] bool has_arrived(
        std::size_t i,
        std::size_t j) const;

    /**
     * Return the immutable first-arrival time for one arrived cell.
     *
     * Throws std::logic_error when the cell has not yet arrived.
     */
    [[nodiscard]] amrex::Real first_arrival_time_s(
        std::size_t i,
        std::size_t j) const;

    [[nodiscard]] std::size_t arrived_cell_count() const noexcept;

    /**
     * Seed ignition-area arrival history at one exact initial time.
     *
     * This may be called exactly once before any sweep update. Cells with
     * positive perimeter intersection area are marked arrived at time_s.
     * Boundary-only contact remains zero-area and is not marked.
     *
     * The first subsequent sweep must begin exactly at time_s.
     */
    [[nodiscard]] FireFirstArrivalRasterUpdate initialize_from_perimeter(
        const FirePerimeter& perimeter,
        amrex::Real time_s);

    /**
     * Update first-arrival history from one pre-remesh physical sweep.
     *
     * A cell whose mask is already set is never
     * overwritten. The update has strong exception safety: no persistent
     * arrival history changes unless every unresolved cell query succeeds.
     */
    [[nodiscard]] FireFirstArrivalRasterUpdate update_from_sweep(
        const FirePerimeter& start_perimeter,
        const FirePerimeter& end_perimeter,
        amrex::Real start_time_s,
        amrex::Real end_time_s,
        amrex::Real time_tolerance_s);

private:
    [[nodiscard]] std::size_t flat_index(
        std::size_t i,
        std::size_t j) const;

    void scatter_canonical_to_distributed(
        const std::vector<std::uint8_t>& arrived,
        const std::vector<amrex::Real>& first_arrival_time_s,
        amrex::iMultiFab& arrived_mf,
        amrex::MultiFab& first_arrival_time_mf) const;

    [[nodiscard]] std::pair<
        std::vector<std::uint8_t>,
        std::vector<amrex::Real>>
    gather_distributed_to_canonical(
        const amrex::iMultiFab& arrived_mf,
        const amrex::MultiFab& first_arrival_time_mf) const;

    FireCartesianRasterGeometry2D geometry_{};
    FireSurfaceLayout surface_layout_;
    amrex::iMultiFab arrived_mf_;
    amrex::MultiFab first_arrival_time_mf_;
    std::vector<std::uint8_t> arrived_;
    std::vector<amrex::Real> first_arrival_time_s_;
    bool has_initial_condition_{false};
    amrex::Real initial_condition_time_s_{};
    bool has_committed_sweep_{false};
    amrex::Real last_sweep_end_time_s_{};
};

} // namespace ERFFire

#endif // ERF_FIRE_FIRST_ARRIVAL_RASTER_H
