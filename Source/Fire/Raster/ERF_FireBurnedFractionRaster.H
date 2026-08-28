#ifndef ERF_FIRE_BURNED_FRACTION_RASTER_H
#define ERF_FIRE_BURNED_FRACTION_RASTER_H

#include <ERF_FireTypes.H>
#include <ERF_FireBurnedFraction.H>
#include <ERF_FireRasterGeometry.H>
#include <ERF_FireSurfaceLayout.H>

#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include <cstddef>
#include <vector>

namespace ERFFire
{

class FirePerimeter;
class FireFront;

/**
 * Aggregate result of updating the persistent burned-fraction raster.
 *
 * burned_area_m2 is the total persistent burned area after the update.
 * newly_burned_area_m2 is the extensive area increment created by this update.
 */
struct FireRasterBurnedAreaUpdate
{
    amrex::Real burned_area_m2{};
    amrex::Real newly_burned_area_m2{};
};

struct FireBurnedFractionRasterState
{
    std::vector<amrex::Real> burned_fraction;
};

/**
 * Persistent burned-area fraction on a uniform Cartesian raster.
 *
 * FirePerimeter remains authoritative. This raster stores only monotone burn
 * history derived from exact polygon-cell coverage. It deliberately stores no
 * arrival time and no copy of instantaneous coverage.
 *
 * Expensive perimeter/cell coverage work is distributed over an AMReX
 * MultiFab surface layout. During this migration stage, a canonical row-major
 * vector is retained as a synchronized compatibility representation for the
 * existing checkpoint/output/combustion APIs. The MultiFab and canonical
 * vector are committed atomically after a successful distributed update.
 */
class FireBurnedFractionRaster
{
public:
    explicit FireBurnedFractionRaster(
        const FireCartesianRasterGeometry2D& geometry);

    FireBurnedFractionRaster(
        const FireCartesianRasterGeometry2D& geometry,
        FireBurnedFractionRasterState state);

    /**
     * Collectively restore version-1 canonical state owned only by IOProcessor().
     * Non-IO ranks may pass an empty state vector.
     */
    [[nodiscard]] static FireBurnedFractionRaster
    collective_restore_from_io_rank_state(
        const FireCartesianRasterGeometry2D& geometry,
        const FireBurnedFractionRasterState& state);

    /**
     * Collectively restore one component from a distributed checkpoint raster.
     */
    [[nodiscard]] static FireBurnedFractionRaster
    collective_restore_from_checkpoint_raster(
        const FireCartesianRasterGeometry2D& geometry,
        const amrex::MultiFab& checkpoint,
        int source_comp);

    FireBurnedFractionRaster(
        const FireBurnedFractionRaster& other);
    FireBurnedFractionRaster& operator=(
        const FireBurnedFractionRaster& other);
    FireBurnedFractionRaster(
        FireBurnedFractionRaster&&) noexcept = default;
    FireBurnedFractionRaster& operator=(
        FireBurnedFractionRaster&&) noexcept = default;

    /**
     * Return one canonical row-major snapshot in a single-rank runtime.
     *
     * This accessor is deliberately noncollective. Multi-rank callers must
     * use collective_snapshot_state_to_io_rank() instead of hiding an O(N)
     * communication inside an ordinary state accessor.
     */
    [[nodiscard]] FireBurnedFractionRasterState
    snapshot_state() const;

    /**
     * Collectively pack distributed state onto the AMReX IO rank.
     *
     * Every rank in the active communicator must call this method. The
     * returned canonical vector is populated only on IOProcessor(); non-IO
     * ranks return an empty vector. This keeps full-raster serialization
     * storage root-owned rather than replicated.
     */
    [[nodiscard]] FireBurnedFractionRasterState
    collective_snapshot_state_to_io_rank() const;

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

    /**
     * Read-only authoritative distributed burned-fraction state.
     *
     * Fire physics with the same surface layout should consume this view
     * directly instead of routing local cell reads through the replicated
     * canonical compatibility vector.
     */
    [[nodiscard]] const amrex::MultiFab&
    distributed_burned_fraction() const noexcept
    {
        return burned_fraction_mf_;
    }

    [[nodiscard]] std::size_t cell_count () const noexcept
    {
        return geometry_.nx * geometry_.ny;
    }

    [[nodiscard]] FireCartesianCell2D cell_bounds(
        std::size_t i,
        std::size_t j) const;

    [[nodiscard]] amrex::Real burned_fraction(
        std::size_t i,
        std::size_t j) const;

    [[nodiscard]] amrex::Real burned_area_m2() const noexcept;

    [[nodiscard]] FireRasterBurnedAreaUpdate update_from_perimeter(
        const FirePerimeter& perimeter);

    [[nodiscard]] FireRasterBurnedAreaUpdate update_from_front(
        const FireFront& front);

    [[nodiscard]] FireRasterBurnedAreaUpdate
    update_from_front_linear_sweep(
        const FireFront& start_front,
        const FireFront& end_front,
        std::size_t temporal_substeps);

    /**
     * Update monotone burn history across the pre-event linear motion from
     * one simple perimeter to its first topology-changing contact.
     *
     * event_vertices_m preserves the start-perimeter vertex correspondence
     * at the contact instant. event_front is the resolved Outer/Hole topology
     * at that same instant.
     */
    [[nodiscard]] FireRasterBurnedAreaUpdate
    update_from_topology_event_sweep(
        const FirePerimeter& start_perimeter,
        const std::vector<FireVec2>& event_vertices_m,
        const FireFront& event_front,
        std::size_t temporal_substeps);

    [[nodiscard]] FireRasterBurnedAreaUpdate update_from_linear_sweep(
        const FirePerimeter& start_perimeter,
        const FirePerimeter& end_perimeter,
        std::size_t temporal_substeps);

private:
    [[nodiscard]] std::size_t flat_index(
        std::size_t i,
        std::size_t j) const;

    void scatter_canonical_to_distributed(
        const std::vector<amrex::Real>& canonical,
        amrex::MultiFab& distributed) const;

    void copy_distributed_state(
        const amrex::MultiFab& source,
        amrex::MultiFab& destination) const;

    FireCartesianRasterGeometry2D geometry_{};
    FireSurfaceLayout surface_layout_;
    amrex::MultiFab burned_fraction_mf_;
    amrex::Real burned_area_m2_{};
};

} // namespace ERFFire

#endif // ERF_FIRE_BURNED_FRACTION_RASTER_H
