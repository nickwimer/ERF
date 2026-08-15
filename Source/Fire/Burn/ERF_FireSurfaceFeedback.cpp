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
    (void)detail::validate_fire_cartesian_raster_geometry(
        geometry_);
    cells_mf_.setVal(amrex::Real(0));
    totals_ = {};
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
      totals_(other.totals_)
{
    amrex::MultiFab::Copy(
        cells_mf_,
        other.cells_mf_,
        0,
        0,
        component_count,
        0);
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
    if (canonical.size() != geometry_.nx * geometry_.ny) {
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
        geometry_.nx * geometry_.ny,
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

FireSurfaceFeedbackCell
FireSurfaceFeedbackRaster::cell(
    std::size_t i,
    std::size_t j) const
{
    (void)flat_index(i, j);

    if (amrex::ParallelDescriptor::NProcs() != 1) {
        throw std::logic_error(
            "Fire surface-feedback scalar cell access is single-rank only; "
            "parallel coupling must use distributed_values");
    }

    const amrex::IntVect cell_index(
        static_cast<int>(i),
        static_cast<int>(j),
        0);
    for (amrex::MFIter mfi(cells_mf_);
         mfi.isValid();
         ++mfi) {
        if (!mfi.validbox().contains(cell_index)) {
            continue;
        }

        const auto values = cells_mf_.const_array(mfi);
        return {
            values(
                cell_index[0], cell_index[1], cell_index[2],
                consumed_dry_fuel_comp),
            values(
                cell_index[0], cell_index[1], cell_index[2],
                sensible_energy_comp),
            values(
                cell_index[0], cell_index[1], cell_index[2],
                water_released_comp)};
    }

    throw std::logic_error(
        "single-rank Fire surface-feedback cell is not locally represented");
}

FireSurfaceFeedbackTotals
FireSurfaceFeedbackRaster::totals() const noexcept
{
    return totals_;
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

    const amrex::MultiFab& before_states =
        before.distributed_states();
    const amrex::MultiFab& after_states =
        after.distributed_states();
    const auto matches_combustion_layout =
        [&result](const amrex::MultiFab& field) {
            return field.boxArray()
                    == result.surface_layout_.box_array()
                && field.DistributionMap()
                    == result.surface_layout_.distribution_map()
                && field.nComp()
                    == FireCombustionRaster::component_count
                && field.nGrow() == 0;
        };
    require(
        matches_combustion_layout(before_states)
            && matches_combustion_layout(after_states),
        "fire surface feedback requires co-located distributed combustion state");

    const auto& geometry = before.geometry();
    const amrex::Real cell_area_m2 =
        geometry.dx_m * geometry.dy_m;

    amrex::Real local_totals[3]{
        amrex::Real(0),
        amrex::Real(0),
        amrex::Real(0)};

    DistributedFailure local_failure =
        DistributedFailure::none;
    std::string local_error;

    try {
        for (amrex::MFIter mfi(result.cells_mf_);
             mfi.isValid(); ++mfi) {
            const amrex::Box& box = mfi.validbox();
            const auto values = result.cells_mf_.array(mfi);
            const auto before_values =
                before_states.const_array(mfi);
            const auto after_values =
                after_states.const_array(mfi);

            for (int j = box.smallEnd(1);
                 j <= box.bigEnd(1);
                 ++j) {
                for (int i = box.smallEnd(0);
                     i <= box.bigEnd(0);
                     ++i) {
                    (void)nonnegative_increment(
                        before_values(
                            i, j, 0,
                            FireCombustionRaster::ignited_area_fraction_comp),
                        after_values(
                            i, j, 0,
                            FireCombustionRaster::ignited_area_fraction_comp),
                        "fire surface feedback ignited-area history decreased");

                    const amrex::Real consumed_kg_m2 =
                        nonnegative_increment(
                            before_values(
                                i, j, 0,
                                FireCombustionRaster::consumed_dry_fuel_comp),
                            after_values(
                                i, j, 0,
                                FireCombustionRaster::consumed_dry_fuel_comp),
                            "fire surface feedback consumed-fuel history decreased");
                    const amrex::Real energy_j_m2 =
                        nonnegative_increment(
                            before_values(
                                i, j, 0,
                                FireCombustionRaster::sensible_energy_comp),
                            after_values(
                                i, j, 0,
                                FireCombustionRaster::sensible_energy_comp),
                            "fire surface feedback sensible-energy history decreased");
                    const amrex::Real water_kg_m2 =
                        nonnegative_increment(
                            before_values(
                                i, j, 0,
                                FireCombustionRaster::water_released_comp),
                            after_values(
                                i, j, 0,
                                FireCombustionRaster::water_released_comp),
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

                    local_totals[0] += consumed_kg;
                    local_totals[1] += energy_j;
                    local_totals[2] += water_kg;
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

    amrex::ParallelDescriptor::ReduceRealSum(
        local_totals,
        3);

    result.totals_ = {
        local_totals[0],
        local_totals[1],
        local_totals[2]};

    require_finite_nonnegative(
        result.totals_.consumed_dry_fuel_kg,
        "fire surface feedback total consumed fuel is not finite");
    require_finite_nonnegative(
        result.totals_.sensible_energy_j,
        "fire surface feedback total sensible energy is not finite");
    require_finite_nonnegative(
        result.totals_.water_released_kg,
        "fire surface feedback total released water is not finite");

    return result;
}

} // namespace ERFFire
