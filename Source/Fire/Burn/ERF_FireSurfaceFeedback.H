#ifndef ERF_FIRE_SURFACE_FEEDBACK_H_
#define ERF_FIRE_SURFACE_FEEDBACK_H_

#include <ERF_FireCombustionRaster.H>
#include <ERF_FireRasterGeometry.H>
#include <ERF_FireSurfaceLayout.H>

#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include <cstddef>
#include <vector>

namespace ERFFire
{

/**
 * Extensive atmospheric-feedback candidate from one fire step in one
 * horizontal fire/ERF cell.
 *
 * These are step-integrated quantities, not tendencies and not yet ERF source
 * terms. Coupling direction is deliberately absent from this type.
 */
struct FireSurfaceFeedbackCell
{
    amrex::Real consumed_dry_fuel_kg{};
    amrex::Real sensible_energy_j{};
    amrex::Real water_released_kg{};
};

struct FireSurfaceFeedbackTotals
{
    amrex::Real consumed_dry_fuel_kg{};
    amrex::Real sensible_energy_j{};
    amrex::Real water_released_kg{};
};

/**
 * Cell-resolved surface feedback increment derived from two committed
 * combustion states.
 *
 * Geometry is the same uniform Cartesian raster used by burn/combustion
 * history. Values are extensive over each represented horizontal cell.
 *
 * This object contains no vertical deposition policy, no ERF thermodynamic
 * conversion, and no one-way/two-way switch. Those remain responsibilities of
 * the later ERF coupling layer.
 *
 * Cell values are authoritative only on the distributed AMReX Fire surface
 * layout. No persistent canonical row-major mirror is retained. Scalar totals
 * are replicated O(1) metadata; arbitrary cell access is a single-rank
 * compatibility operation.
 */
class FireSurfaceFeedbackRaster
{
public:
    static constexpr int consumed_dry_fuel_comp = 0;
    static constexpr int sensible_energy_comp = 1;
    static constexpr int water_released_comp = 2;
    static constexpr int component_count = 3;

    explicit FireSurfaceFeedbackRaster(
        FireCartesianRasterGeometry2D geometry);

    FireSurfaceFeedbackRaster(
        const FireSurfaceFeedbackRaster& other);
    FireSurfaceFeedbackRaster& operator=(
        const FireSurfaceFeedbackRaster& other);
    FireSurfaceFeedbackRaster(
        FireSurfaceFeedbackRaster&&) noexcept = default;
    FireSurfaceFeedbackRaster& operator=(
        FireSurfaceFeedbackRaster&&) noexcept = default;

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

    [[nodiscard]] const amrex::MultiFab&
    distributed_values() const noexcept
    {
        return cells_mf_;
    }

    [[nodiscard]] std::size_t cell_count() const noexcept
    {
        return geometry_.nx * geometry_.ny;
    }

    /**
     * Return one cell in a single-rank runtime.
     *
     * Parallel coupling consumes distributed_values() directly; this accessor
     * deliberately performs no hidden communication.
     */
    [[nodiscard]] FireSurfaceFeedbackCell
    cell(std::size_t i, std::size_t j) const;

    [[nodiscard]] FireSurfaceFeedbackTotals totals() const noexcept;

private:
    friend FireSurfaceFeedbackRaster
    make_fire_surface_feedback_increment(
        const FireCombustionRaster& before,
        const FireCombustionRaster& after);

    friend FireSurfaceFeedbackRaster
    conservatively_regrid_fire_surface_feedback(
        const FireSurfaceFeedbackRaster& source,
        FireCartesianRasterGeometry2D target_geometry);

    [[nodiscard]] std::size_t flat_index(
        std::size_t i,
        std::size_t j) const;

    void scatter_canonical_to_distributed(
        const std::vector<FireSurfaceFeedbackCell>& canonical,
        amrex::MultiFab& distributed) const;

    [[nodiscard]] std::vector<FireSurfaceFeedbackCell>
    gather_distributed_to_canonical(
        const amrex::MultiFab& distributed) const;

    FireCartesianRasterGeometry2D geometry_{};
    FireSurfaceLayout surface_layout_;
    amrex::MultiFab cells_mf_;
    FireSurfaceFeedbackTotals totals_{};
};

/**
 * Difference two synchronized combustion rasters into one step-integrated
 * surface-feedback raster.
 *
 * Preconditions:
 *   * before/after are initialized,
 *   * geometry and combustion parameters are identical,
 *   * ignited area and cumulative consumed/energy/water histories are
 *     nondecreasing cell-by-cell.
 *
 * Strong exception guarantee follows naturally because neither input is
 * mutated and the result is returned only after complete validation.
 */
[[nodiscard]] FireSurfaceFeedbackRaster
make_fire_surface_feedback_increment(
    const FireCombustionRaster& before,
    const FireCombustionRaster& after);

/**
 * Conservatively transfer step-integrated extensive Fire feedback between
 * aligned, integer-nested horizontal rasters representing the same physical
 * map-plane domain.
 *
 * Fine-to-coarse transfer sums child-cell kg/J quantities. Coarse-to-fine
 * transfer partitions each parent quantity uniformly by represented overlap
 * area. The operation therefore preserves total consumed dry fuel, sensible
 * energy, and released water up to floating-point arithmetic.
 *
 * Each horizontal axis may have its own integer ratio, but both axes must
 * refine in the same direction (with equality allowed on either axis).
 */
[[nodiscard]] FireSurfaceFeedbackRaster
conservatively_regrid_fire_surface_feedback(
    const FireSurfaceFeedbackRaster& source,
    FireCartesianRasterGeometry2D target_geometry);

} // namespace ERFFire

#endif
