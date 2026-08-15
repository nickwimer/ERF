#include "ERF_FireSurfaceFeedback.H"

#include <AMReX_Arena.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ERFFire
{
namespace
{

enum class DistributedFailure : int
{
    none = 0,
    invalid_argument = 1,
    overflow_error = 2,
    runtime_error = 3
};

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

amrex::MFInfo
fire_surface_mf_info()
{
    amrex::MFInfo info;
    info.SetArena(amrex::The_Pinned_Arena());
    return info;
}

[[noreturn]] void
throw_distributed_failure(
    DistributedFailure failure,
    const std::string& local_error)
{
    const std::string message =
        local_error.empty()
        ? "distributed Fire surface feedback failed on another MPI rank"
        : "distributed Fire surface feedback failed: " + local_error;

    switch (failure) {
    case DistributedFailure::invalid_argument:
        throw std::invalid_argument(message);
    case DistributedFailure::overflow_error:
        throw std::overflow_error(message);
    case DistributedFailure::runtime_error:
        throw std::runtime_error(message);
    case DistributedFailure::none:
        break;
    }

    throw std::runtime_error(
        "distributed Fire surface feedback failed with an invalid error code");
}

void
synchronize_distributed_failure(
    DistributedFailure local_failure,
    const std::string& local_error)
{
    int failure = static_cast<int>(local_failure);
    amrex::ParallelDescriptor::ReduceIntMax(failure);

    if (failure != static_cast<int>(DistributedFailure::none)) {
        throw_distributed_failure(
            static_cast<DistributedFailure>(failure),
            local_error);
    }
}

} // namespace

FireSurfaceFeedbackRaster::FireSurfaceFeedbackRaster(
    FireCartesianRasterGeometry2D geometry)
    : geometry_(geometry),
      surface_layout_(geometry_),
      cells_mf_(
          surface_layout_.box_array(),
          surface_layout_.distribution_map(),
          component_count,
          0,
          fire_surface_mf_info())
{
    const std::size_t count =
        detail::validate_fire_cartesian_raster_geometry(
            geometry_);
    cells_.assign(count, FireSurfaceFeedbackCell{});
    scatter_canonical_to_distributed(
        cells_,
        cells_mf_);
}

FireSurfaceFeedbackRaster::FireSurfaceFeedbackRaster(
    const FireSurfaceFeedbackRaster& other)
    : geometry_(other.geometry_),
      surface_layout_(other.surface_layout_),
      cells_mf_(
          surface_layout_.box_array(),
          surface_layout_.distribution_map(),
          component_count,
          0,
          fire_surface_mf_info()),
      cells_(other.cells_)
{
    scatter_canonical_to_distributed(
        cells_,
        cells_mf_);
}

FireSurfaceFeedbackRaster&
FireSurfaceFeedbackRaster::operator=(
    const FireSurfaceFeedbackRaster& other)
{
    if (this != &other) {
        FireSurfaceFeedbackRaster copy(other);
        *this = std::move(copy);
    }
    return *this;
}

std::size_t
FireSurfaceFeedbackRaster::flat_index(
    std::size_t i,
    std::size_t j) const
{
    return detail::fire_cartesian_raster_flat_index(
        geometry_, i, j);
}

void
FireSurfaceFeedbackRaster::scatter_canonical_to_distributed(
    const std::vector<FireSurfaceFeedbackCell>& canonical,
    amrex::MultiFab& distributed) const
{
    if (canonical.size() != cells_.size()) {
        throw std::invalid_argument(
            "Fire surface-feedback canonical state has the wrong cell count");
    }
    if (distributed.boxArray() != surface_layout_.box_array()
        || distributed.DistributionMap()
            != surface_layout_.distribution_map()
        || distributed.nComp() != component_count
        || distributed.nGrow() != 0) {
        throw std::invalid_argument(
            "Fire surface-feedback MultiFab does not match its surface layout");
    }

    for (amrex::MFIter mfi(distributed); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.validbox();
        const auto values = distributed.array(mfi);

        for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
            for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                const auto& cell_value =
                    canonical[
                        flat_index(
                            static_cast<std::size_t>(i),
                            static_cast<std::size_t>(j))];

                values(i, j, 0, consumed_dry_fuel_comp) =
                    cell_value.consumed_dry_fuel_kg;
                values(i, j, 0, sensible_energy_comp) =
                    cell_value.sensible_energy_j;
                values(i, j, 0, water_released_comp) =
                    cell_value.water_released_kg;
            }
        }
    }
}

std::vector<FireSurfaceFeedbackCell>
FireSurfaceFeedbackRaster::gather_distributed_to_canonical(
    const amrex::MultiFab& distributed) const
{
    if (distributed.boxArray() != surface_layout_.box_array()
        || distributed.DistributionMap()
            != surface_layout_.distribution_map()
        || distributed.nComp() != component_count
        || distributed.nGrow() != 0) {
        throw std::invalid_argument(
            "Fire surface-feedback MultiFab does not match its surface layout");
    }

    amrex::FArrayBox gathered(
        surface_layout_.cell_domain(),
        component_count,
        amrex::The_Pinned_Arena());
    distributed.copyTo(
        gathered,
        0,
        0,
        component_count,
        0);
    const auto values = gathered.const_array();

    std::vector<FireSurfaceFeedbackCell> canonical(
        cells_.size(),
        FireSurfaceFeedbackCell{});

    for (std::size_t j = 0; j < geometry_.ny; ++j) {
        for (std::size_t i = 0; i < geometry_.nx; ++i) {
            auto& cell_value =
                canonical[flat_index(i, j)];
            cell_value.consumed_dry_fuel_kg =
                values(
                    static_cast<int>(i),
                    static_cast<int>(j),
                    0,
                    consumed_dry_fuel_comp);
            cell_value.sensible_energy_j =
                values(
                    static_cast<int>(i),
                    static_cast<int>(j),
                    0,
                    sensible_energy_comp);
            cell_value.water_released_kg =
                values(
                    static_cast<int>(i),
                    static_cast<int>(j),
                    0,
                    water_released_comp);
        }
    }

    return canonical;
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

    DistributedFailure local_failure =
        DistributedFailure::none;
    std::string local_error;

    try {
        for (amrex::MFIter mfi(result.cells_mf_);
             mfi.isValid(); ++mfi) {
            const amrex::Box& box = mfi.validbox();
            const auto values = result.cells_mf_.array(mfi);

            for (int j = box.smallEnd(1);
                 j <= box.bigEnd(1);
                 ++j) {
                for (int i = box.smallEnd(0);
                     i <= box.bigEnd(0);
                     ++i) {
                    const std::size_t ii =
                        static_cast<std::size_t>(i);
                    const std::size_t jj =
                        static_cast<std::size_t>(j);
                    const FireCombustionState& before_state =
                        before.state(ii, jj);
                    const FireCombustionState& after_state =
                        after.state(ii, jj);

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

                    const amrex::Real consumed_kg =
                        consumed_kg_m2 * cell_area_m2;
                    const amrex::Real energy_j =
                        energy_j_m2 * cell_area_m2;
                    const amrex::Real water_kg =
                        water_kg_m2 * cell_area_m2;

                    require_finite_nonnegative(
                        consumed_kg,
                        "fire surface feedback consumed fuel is not finite");
                    require_finite_nonnegative(
                        energy_j,
                        "fire surface feedback sensible energy is not finite");
                    require_finite_nonnegative(
                        water_kg,
                        "fire surface feedback released water is not finite");

                    values(
                        i, j, 0,
                        FireSurfaceFeedbackRaster::consumed_dry_fuel_comp) =
                        consumed_kg;
                    values(
                        i, j, 0,
                        FireSurfaceFeedbackRaster::sensible_energy_comp) =
                        energy_j;
                    values(
                        i, j, 0,
                        FireSurfaceFeedbackRaster::water_released_comp) =
                        water_kg;
                }
            }
        }
    } catch (const std::invalid_argument& error) {
        local_failure = DistributedFailure::invalid_argument;
        local_error = error.what();
    } catch (const std::overflow_error& error) {
        local_failure = DistributedFailure::overflow_error;
        local_error = error.what();
    } catch (const std::exception& error) {
        local_failure = DistributedFailure::runtime_error;
        local_error = error.what();
    } catch (...) {
        local_failure = DistributedFailure::runtime_error;
        local_error = "unknown local Fire surface-feedback error";
    }

    synchronize_distributed_failure(
        local_failure,
        local_error);

    result.cells_ =
        result.gather_distributed_to_canonical(
            result.cells_mf_);

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
