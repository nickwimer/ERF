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
 * Cell values are stored on the distributed AMReX Fire surface layout. A
 * canonical row-major vector remains synchronized during this migration stage
 * for scalar accessors, deterministic totals, and the existing checkpoint/test
 * interfaces.
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
        return cells_.size();
    }

    [[nodiscard]] const FireSurfaceFeedbackCell&
    cell(std::size_t i, std::size_t j) const;

    [[nodiscard]] FireSurfaceFeedbackTotals totals() const noexcept;

private:
    friend FireSurfaceFeedbackRaster
    make_fire_surface_feedback_increment(
        const FireCombustionRaster& before,
        const FireCombustionRaster& after);

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
    std::vector<FireSurfaceFeedbackCell> cells_;
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

} // namespace ERFFire

#endif
