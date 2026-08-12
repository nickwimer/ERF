#include "ERF_FireSurfaceFeedback.H"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ERFFire
{
namespace
{

void
require(bool condition, const char* message)
{
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

bool
same_geometry(
    const FireCartesianRasterGeometry2D& a,
    const FireCartesianRasterGeometry2D& b) noexcept
{
    return a.nx == b.nx
        && a.ny == b.ny
        && a.xlo_m == b.xlo_m
        && a.ylo_m == b.ylo_m
        && a.dx_m == b.dx_m
        && a.dy_m == b.dy_m;
}

bool
same_parameters(
    const FireCombustionParameters& a,
    const FireCombustionParameters& b) noexcept
{
    return a.dry_fuel_load_kg_m2
            == b.dry_fuel_load_kg_m2
        && a.sensible_heat_release_j_kg_dry
            == b.sensible_heat_release_j_kg_dry
        && a.fuel_moisture_fraction
            == b.fuel_moisture_fraction
        && a.burn_time_constant_s
            == b.burn_time_constant_s
        && a.combustion_water_yield_kg_per_kg_dry
            == b.combustion_water_yield_kg_per_kg_dry;
}

amrex::Real
history_tolerance(amrex::Real scale) noexcept
{
    return amrex::Real(1024)
        * std::numeric_limits<amrex::Real>::epsilon()
        * std::max(amrex::Real(1), std::abs(scale));
}

amrex::Real
nonnegative_increment(
    amrex::Real before,
    amrex::Real after,
    const char* message)
{
    const amrex::Real increment = after - before;
    const amrex::Real tolerance =
        history_tolerance(
            std::max(std::abs(before), std::abs(after)));

    if (!std::isfinite(increment)
        || increment < -tolerance) {
        throw std::invalid_argument(message);
    }

    return std::max(increment, amrex::Real(0));
}

void
require_finite_nonnegative(
    amrex::Real value,
    const char* message)
{
    if (!std::isfinite(value) || value < amrex::Real(0)) {
        throw std::overflow_error(message);
    }
}

} // namespace

FireSurfaceFeedbackRaster::FireSurfaceFeedbackRaster(
    FireCartesianRasterGeometry2D geometry)
    : geometry_(geometry)
{
    const std::size_t count =
        detail::validate_fire_cartesian_raster_geometry(
            geometry_);
    cells_.assign(count, FireSurfaceFeedbackCell{});
}

std::size_t
FireSurfaceFeedbackRaster::flat_index(
    std::size_t i,
    std::size_t j) const
{
    return detail::fire_cartesian_raster_flat_index(
        geometry_, i, j);
}

const FireSurfaceFeedbackCell&
FireSurfaceFeedbackRaster::cell(
    std::size_t i,
    std::size_t j) const
{
    return cells_[flat_index(i, j)];
}

FireSurfaceFeedbackTotals
FireSurfaceFeedbackRaster::totals() const noexcept
{
    FireSurfaceFeedbackTotals result{};
    for (const auto& cell_value : cells_) {
        result.consumed_dry_fuel_kg +=
            cell_value.consumed_dry_fuel_kg;
        result.sensible_energy_j +=
            cell_value.sensible_energy_j;
        result.water_released_kg +=
            cell_value.water_released_kg;
    }
    return result;
}

FireSurfaceFeedbackRaster
make_fire_surface_feedback_increment(
    const FireCombustionRaster& before,
    const FireCombustionRaster& after)
{
    require(
        before.initialized() && after.initialized(),
        "fire surface feedback requires initialized combustion rasters");
    require(
        same_geometry(before.geometry(), after.geometry()),
        "fire surface feedback combustion geometry mismatch");
    require(
        same_parameters(before.parameters(), after.parameters()),
        "fire surface feedback combustion parameter mismatch");

    FireSurfaceFeedbackRaster result(before.geometry());

    const auto& geometry = before.geometry();
    const amrex::Real cell_area_m2 =
        geometry.dx_m * geometry.dy_m;

    for (std::size_t j = 0; j < geometry.ny; ++j) {
        for (std::size_t i = 0; i < geometry.nx; ++i) {
            const FireCombustionState& before_state =
                before.state(i, j);
            const FireCombustionState& after_state =
                after.state(i, j);

            (void)nonnegative_increment(
                before_state.ignited_area_fraction,
                after_state.ignited_area_fraction,
                "fire surface feedback ignited-area history decreased");

            const amrex::Real consumed_kg_m2 =
                nonnegative_increment(
                    before_state.consumed_dry_fuel_kg_m2,
                    after_state.consumed_dry_fuel_kg_m2,
                    "fire surface feedback consumed-fuel history decreased");
            const amrex::Real energy_j_m2 =
                nonnegative_increment(
                    before_state.sensible_energy_j_m2,
                    after_state.sensible_energy_j_m2,
                    "fire surface feedback sensible-energy history decreased");
            const amrex::Real water_kg_m2 =
                nonnegative_increment(
                    before_state.water_released_kg_m2,
                    after_state.water_released_kg_m2,
                    "fire surface feedback released-water history decreased");

            FireSurfaceFeedbackCell& output =
                result.cells_[result.flat_index(i, j)];
            output.consumed_dry_fuel_kg =
                consumed_kg_m2 * cell_area_m2;
            output.sensible_energy_j =
                energy_j_m2 * cell_area_m2;
            output.water_released_kg =
                water_kg_m2 * cell_area_m2;

            require_finite_nonnegative(
                output.consumed_dry_fuel_kg,
                "fire surface feedback consumed fuel is not finite");
            require_finite_nonnegative(
                output.sensible_energy_j,
                "fire surface feedback sensible energy is not finite");
            require_finite_nonnegative(
                output.water_released_kg,
                "fire surface feedback released water is not finite");
        }
    }

    const FireSurfaceFeedbackTotals result_totals =
        result.totals();
    require_finite_nonnegative(
        result_totals.consumed_dry_fuel_kg,
        "fire surface feedback total consumed fuel is not finite");
    require_finite_nonnegative(
        result_totals.sensible_energy_j,
        "fire surface feedback total sensible energy is not finite");
    require_finite_nonnegative(
        result_totals.water_released_kg,
        "fire surface feedback total released water is not finite");

    return result;
}

} // namespace ERFFire
