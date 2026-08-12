#include "ERF_FireCombustionRaster.H"

#include <ERF_FireCellCoverage.H>
#include <ERF_FirePerimeterSweep.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

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

amrex::Real
fraction_tolerance(amrex::Real scale) noexcept
{
    return amrex::Real(1024)
        * std::numeric_limits<amrex::Real>::epsilon()
        * std::max(amrex::Real(1), std::abs(scale));
}

bool
fraction_equal(amrex::Real a, amrex::Real b) noexcept
{
    return std::abs(a - b)
        <= fraction_tolerance(std::max(std::abs(a), std::abs(b)));
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

FireCombustionRaster::FireCombustionRaster(
    FireCartesianRasterGeometry2D geometry,
    FireCombustionParameters parameters,
    FireCombustionRasterOptions options)
    : geometry_(geometry),
      parameters_(parameters),
      options_(options)
{
    const std::size_t cell_count =
        detail::validate_fire_cartesian_raster_geometry(geometry_);

    require(
        options_.temporal_substeps > 0,
        "fire combustion raster temporal_substeps must be positive");

    // Reuse the scalar combustion layer as the authoritative parameter/state
    // validator without changing the zero initial state.
    (void)advance_fire_combustion(
        FireCombustionState{},
        parameters_,
        amrex::Real(0));

    states_.assign(cell_count, FireCombustionState{});
}

std::size_t
FireCombustionRaster::flat_index(
    std::size_t i,
    std::size_t j) const
{
    return detail::fire_cartesian_raster_flat_index(
        geometry_, i, j);
}

const FireCombustionState&
FireCombustionRaster::state(
    std::size_t i,
    std::size_t j) const
{
    return states_[flat_index(i, j)];
}

FireCombustionRasterTotals
FireCombustionRaster::totals_for(
    const std::vector<FireCombustionState>& states) const noexcept
{
    const amrex::Real cell_area_m2 =
        geometry_.dx_m * geometry_.dy_m;

    FireCombustionRasterTotals totals{};

    for (const auto& state_value : states) {
        totals.remaining_dry_fuel_kg +=
            state_value.remaining_dry_fuel_kg_m2 * cell_area_m2;
        totals.consumed_dry_fuel_kg +=
            state_value.consumed_dry_fuel_kg_m2 * cell_area_m2;
        totals.sensible_energy_j +=
            state_value.sensible_energy_j_m2 * cell_area_m2;
        totals.water_released_kg +=
            state_value.water_released_kg_m2 * cell_area_m2;
    }

    return totals;
}

FireCombustionRasterTotals
FireCombustionRaster::totals() const noexcept
{
    return totals_for(states_);
}

FireCombustionRasterTotals
FireCombustionRaster::initialize_from_burned_fraction(
    const FireBurnedFractionRaster& burned_fraction)
{
    if (initialized_) {
        throw std::logic_error(
            "fire combustion raster may be initialized only once");
    }

    require(
        same_geometry(geometry_, burned_fraction.geometry()),
        "fire combustion raster initialization geometry mismatch");

    std::vector<FireCombustionState> next_states = states_;

    for (std::size_t j = 0; j < geometry_.ny; ++j) {
        for (std::size_t i = 0; i < geometry_.nx; ++i) {
            const std::size_t index = flat_index(i, j);
            next_states[index] =
                add_fire_combustion_ignition(
                    next_states[index],
                    parameters_,
                    burned_fraction.burned_fraction(i, j));
        }
    }

    const FireCombustionRasterTotals next_totals =
        totals_for(next_states);

    require_finite_nonnegative(
        next_totals.remaining_dry_fuel_kg,
        "fire combustion raster initial remaining fuel is not finite");
    require_finite_nonnegative(
        next_totals.consumed_dry_fuel_kg,
        "fire combustion raster initial consumed fuel is not finite");
    require_finite_nonnegative(
        next_totals.sensible_energy_j,
        "fire combustion raster initial energy is not finite");
    require_finite_nonnegative(
        next_totals.water_released_kg,
        "fire combustion raster initial water is not finite");

    states_.swap(next_states);
    initialized_ = true;

    return next_totals;
}

FireCombustionRasterAdvance
FireCombustionRaster::advance_from_linear_sweep(
    const FirePerimeter& start_perimeter,
    const FirePerimeter& end_perimeter,
    const FireBurnedFractionRaster& burned_before,
    const FireBurnedFractionRaster& burned_after,
    amrex::Real dt_s)
{
    if (!initialized_) {
        throw std::logic_error(
            "fire combustion raster must be initialized before advance");
    }

    require(
        same_geometry(geometry_, burned_before.geometry())
            && same_geometry(geometry_, burned_after.geometry()),
        "fire combustion raster advance geometry mismatch");
    require(
        start_perimeter.size() == end_perimeter.size(),
        "fire combustion raster sweep requires matching perimeter vertex counts");
    require(
        std::isfinite(dt_s) && dt_s > amrex::Real(0),
        "fire combustion raster dt must be finite and positive");

    const amrex::Real substep_dt_s =
        dt_s
        / static_cast<amrex::Real>(options_.temporal_substeps);
    const amrex::Real half_substep_dt_s =
        amrex::Real(0.5) * substep_dt_s;

    if (!std::isfinite(substep_dt_s)
        || !(substep_dt_s > amrex::Real(0))
        || !(half_substep_dt_s > amrex::Real(0))) {
        throw std::invalid_argument(
            "fire combustion raster temporal substep is not representable");
    }

    std::vector<FireCombustionState> next_states = states_;
    std::vector<amrex::Real> running_burned_fraction(
        states_.size(), amrex::Real(0));

    for (std::size_t j = 0; j < geometry_.ny; ++j) {
        for (std::size_t i = 0; i < geometry_.nx; ++i) {
            const std::size_t index = flat_index(i, j);
            const amrex::Real before =
                burned_before.burned_fraction(i, j);
            const amrex::Real after =
                burned_after.burned_fraction(i, j);

            require(
                after + fraction_tolerance(after) >= before,
                "fire combustion raster burned history is not monotone");
            require(
                fraction_equal(
                    next_states[index].ignited_area_fraction,
                    before),
                "fire combustion raster state is not synchronized with burned history");

            running_burned_fraction[index] = before;
        }
    }

    const amrex::Real cell_area_m2 =
        geometry_.dx_m * geometry_.dy_m;

    amrex::Real newly_consumed_dry_fuel_kg = 0;
    amrex::Real sensible_energy_increment_j = 0;
    amrex::Real water_released_increment_kg = 0;

    for (std::size_t substep = 1;
         substep <= options_.temporal_substeps;
         ++substep) {
        const amrex::Real alpha =
            static_cast<amrex::Real>(substep)
            / static_cast<amrex::Real>(
                options_.temporal_substeps);

        const FirePerimeter sample_perimeter =
            interpolate_fire_perimeter_linear_sweep(
                start_perimeter,
                end_perimeter,
                alpha);

        for (std::size_t j = 0; j < geometry_.ny; ++j) {
            for (std::size_t i = 0; i < geometry_.nx; ++i) {
                const std::size_t index = flat_index(i, j);
                const FireCartesianCell2D cell =
                    burned_before.cell_bounds(i, j);

                const amrex::Real coverage =
                    fire_perimeter_cell_coverage_fraction(
                        sample_perimeter,
                        cell);
                const amrex::Real next_burned_fraction =
                    std::max(
                        running_burned_fraction[index],
                        coverage);
                const amrex::Real newly_ignited_fraction =
                    next_burned_fraction
                    - running_burned_fraction[index];

                const FireCombustionAdvance first_half =
                    advance_fire_combustion(
                        next_states[index],
                        parameters_,
                        half_substep_dt_s);

                const FireCombustionState with_ignition =
                    add_fire_combustion_ignition(
                        first_half.state,
                        parameters_,
                        newly_ignited_fraction);

                const FireCombustionAdvance second_half =
                    advance_fire_combustion(
                        with_ignition,
                        parameters_,
                        half_substep_dt_s);

                newly_consumed_dry_fuel_kg +=
                    (first_half.newly_consumed_dry_fuel_kg_m2
                     + second_half.newly_consumed_dry_fuel_kg_m2)
                    * cell_area_m2;
                sensible_energy_increment_j +=
                    (first_half.sensible_energy_increment_j_m2
                     + second_half.sensible_energy_increment_j_m2)
                    * cell_area_m2;
                water_released_increment_kg +=
                    (first_half.water_released_increment_kg_m2
                     + second_half.water_released_increment_kg_m2)
                    * cell_area_m2;

                next_states[index] = second_half.state;
                running_burned_fraction[index] =
                    next_burned_fraction;
            }
        }
    }

    for (std::size_t j = 0; j < geometry_.ny; ++j) {
        for (std::size_t i = 0; i < geometry_.nx; ++i) {
            const std::size_t index = flat_index(i, j);
            const amrex::Real target =
                burned_after.burned_fraction(i, j);

            require(
                fraction_equal(
                    running_burned_fraction[index],
                    target),
                "fire combustion temporal sampling does not reproduce endpoint burned history");
            require(
                fraction_equal(
                    next_states[index].ignited_area_fraction,
                    target),
                "fire combustion endpoint state is not synchronized with burned history");
        }
    }

    const FireCombustionRasterTotals next_totals =
        totals_for(next_states);

    require_finite_nonnegative(
        newly_consumed_dry_fuel_kg,
        "fire combustion raster consumed-fuel increment is not finite");
    require_finite_nonnegative(
        sensible_energy_increment_j,
        "fire combustion raster energy increment is not finite");
    require_finite_nonnegative(
        water_released_increment_kg,
        "fire combustion raster water increment is not finite");
    require_finite_nonnegative(
        next_totals.remaining_dry_fuel_kg,
        "fire combustion raster remaining fuel is not finite");
    require_finite_nonnegative(
        next_totals.consumed_dry_fuel_kg,
        "fire combustion raster consumed fuel is not finite");
    require_finite_nonnegative(
        next_totals.sensible_energy_j,
        "fire combustion raster cumulative energy is not finite");
    require_finite_nonnegative(
        next_totals.water_released_kg,
        "fire combustion raster cumulative water is not finite");

    states_.swap(next_states);

    return {
        next_totals,
        newly_consumed_dry_fuel_kg,
        sensible_energy_increment_j,
        water_released_increment_kg
    };
}

} // namespace ERFFire
