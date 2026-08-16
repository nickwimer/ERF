#include "ERF_FireCombustionRaster.H"

#include <ERF_FireCellCoverage.H>
#include <ERF_FirePerimeterSweep.H>

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
#include <vector>

namespace ERFFire
{
namespace
{

constexpr int ignited_area_fraction_comp = 0;
constexpr int remaining_dry_fuel_comp = 1;
constexpr int consumed_dry_fuel_comp = 2;
constexpr int sensible_energy_comp = 3;
constexpr int water_released_comp = 4;
constexpr int combustion_component_count = 5;

enum class DistributedFailure : int
{
    none = 0,
    invalid_argument = 1,
    logic_error = 2,
    overflow_error = 3,
    runtime_error = 4
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
    const std::string& local_error,
    const char* operation)
{
    const std::string message =
        local_error.empty()
        ? std::string(operation) + " failed on another MPI rank"
        : std::string(operation) + " failed: " + local_error;

    switch (failure) {
    case DistributedFailure::invalid_argument:
        throw std::invalid_argument(message);
    case DistributedFailure::logic_error:
        throw std::logic_error(message);
    case DistributedFailure::overflow_error:
        throw std::overflow_error(message);
    case DistributedFailure::runtime_error:
        throw std::runtime_error(message);
    case DistributedFailure::none:
        break;
    }

    throw std::runtime_error(
        std::string(operation) + " failed with an invalid error code");
}

void
synchronize_distributed_failure(
    DistributedFailure local_failure,
    const std::string& local_error,
    const char* operation)
{
    int failure = static_cast<int>(local_failure);
    amrex::ParallelDescriptor::ReduceIntMax(failure);

    if (failure != static_cast<int>(DistributedFailure::none)) {
        throw_distributed_failure(
            static_cast<DistributedFailure>(failure),
            local_error,
            operation);
    }
}

template <typename T>
FireCombustionState
load_combustion_state(
    const amrex::Array4<T>& values,
    int i,
    int j)
{
    return {
        values(i, j, 0, ignited_area_fraction_comp),
        values(i, j, 0, remaining_dry_fuel_comp),
        values(i, j, 0, consumed_dry_fuel_comp),
        values(i, j, 0, sensible_energy_comp),
        values(i, j, 0, water_released_comp)};
}

void
store_combustion_state(
    const amrex::Array4<amrex::Real>& values,
    int i,
    int j,
    const FireCombustionState& state)
{
    values(i, j, 0, ignited_area_fraction_comp) =
        state.ignited_area_fraction;
    values(i, j, 0, remaining_dry_fuel_comp) =
        state.remaining_dry_fuel_kg_m2;
    values(i, j, 0, consumed_dry_fuel_comp) =
        state.consumed_dry_fuel_kg_m2;
    values(i, j, 0, sensible_energy_comp) =
        state.sensible_energy_j_m2;
    values(i, j, 0, water_released_comp) =
        state.water_released_kg_m2;
}

FireCombustionRasterTotals
local_distributed_totals(
    const amrex::MultiFab& states,
    const FireCartesianRasterGeometry2D& geometry) noexcept
{
    const amrex::Real cell_area_m2 =
        geometry.dx_m * geometry.dy_m;

    FireCombustionRasterTotals totals{};

    for (amrex::MFIter mfi(states); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.validbox();
        const auto values = states.const_array(mfi);

        for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
            for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                const FireCombustionState state =
                    load_combustion_state(values, i, j);
                totals.remaining_dry_fuel_kg +=
                    state.remaining_dry_fuel_kg_m2 * cell_area_m2;
                totals.consumed_dry_fuel_kg +=
                    state.consumed_dry_fuel_kg_m2 * cell_area_m2;
                totals.sensible_energy_j +=
                    state.sensible_energy_j_m2 * cell_area_m2;
                totals.water_released_kg +=
                    state.water_released_kg_m2 * cell_area_m2;
            }
        }
    }

    return totals;
}

void
reduce_distributed_totals(
    FireCombustionRasterTotals& totals)
{
    amrex::Real values[4]{
        totals.remaining_dry_fuel_kg,
        totals.consumed_dry_fuel_kg,
        totals.sensible_energy_j,
        totals.water_released_kg};

    amrex::ParallelDescriptor::ReduceRealSum(values, 4);

    totals = {
        values[0],
        values[1],
        values[2],
        values[3]};
}

} // namespace

FireCombustionRaster::FireCombustionRaster(
    FireCartesianRasterGeometry2D geometry,
    FireCombustionParameters parameters,
    FireCombustionRasterOptions options)
    : geometry_(geometry),
      parameters_(parameters),
      options_(options),
      surface_layout_(geometry_),
      states_mf_(
          surface_layout_.box_array(),
          surface_layout_.distribution_map(),
          combustion_component_count,
          0,
          fire_surface_mf_info())
{
    (void)detail::validate_fire_cartesian_raster_geometry(
        geometry_);

    require(
        options_.temporal_substeps > 0,
        "fire combustion raster temporal_substeps must be positive");

    // Reuse the scalar combustion layer as the authoritative parameter/state
    // validator without changing the zero initial state.
    (void)advance_fire_combustion(
        FireCombustionState{},
        parameters_,
        amrex::Real(0));

    states_mf_.setVal(amrex::Real(0));
    totals_ = {};
}

FireCombustionRaster::FireCombustionRaster(
    FireCartesianRasterGeometry2D geometry,
    FireCombustionParameters parameters,
    FireCombustionRasterOptions options,
    FireCombustionRasterState state)
    : FireCombustionRaster(
          geometry,
          parameters,
          options)
{
    if (state.cells.size()
        != geometry_.nx * geometry_.ny) {
        throw std::invalid_argument(
            "restored fire combustion state has the wrong cell count");
    }

    for (const FireCombustionState& cell : state.cells) {
        // Zero-time advance is the authoritative scalar state validator.
        (void)advance_fire_combustion(
            cell,
            parameters_,
            amrex::Real(0));

        if (!state.initialized
            && (cell.ignited_area_fraction != amrex::Real(0.0)
                || cell.remaining_dry_fuel_kg_m2 != amrex::Real(0.0)
                || cell.consumed_dry_fuel_kg_m2 != amrex::Real(0.0)
                || cell.sensible_energy_j_m2 != amrex::Real(0.0)
                || cell.water_released_kg_m2 != amrex::Real(0.0))) {
            throw std::invalid_argument(
                "uninitialized restored fire combustion state must be zero");
        }
    }

    totals_ = totals_for(state.cells);
    require_finite_nonnegative(
        totals_.remaining_dry_fuel_kg,
        "restored fire combustion remaining fuel is not finite");
    require_finite_nonnegative(
        totals_.consumed_dry_fuel_kg,
        "restored fire combustion consumed fuel is not finite");
    require_finite_nonnegative(
        totals_.sensible_energy_j,
        "restored fire combustion sensible energy is not finite");
    require_finite_nonnegative(
        totals_.water_released_kg,
        "restored fire combustion released water is not finite");

    initialized_ = state.initialized;
    scatter_canonical_to_distributed(
        state.cells,
        states_mf_);
}

FireCombustionRaster
FireCombustionRaster::collective_restore_from_io_rank_state(
    FireCartesianRasterGeometry2D geometry,
    FireCombustionParameters parameters,
    FireCombustionRasterOptions options,
    const FireCombustionRasterState& state)
{
    FireCombustionRaster result(
        geometry, parameters, options);
    const int io_rank =
        amrex::ParallelDescriptor::IOProcessorNumber();

    int invalid_state = 0;
    int initialized = 0;
    amrex::Real totals[4]{
        amrex::Real(0), amrex::Real(0),
        amrex::Real(0), amrex::Real(0)};

    if (amrex::ParallelDescriptor::IOProcessor()) {
        try {
            if (state.cells.size() != geometry.nx * geometry.ny) {
                throw std::invalid_argument(
                    "restored fire combustion state has the wrong cell count");
            }
            for (const FireCombustionState& cell : state.cells) {
                (void)advance_fire_combustion(
                    cell, parameters, amrex::Real(0));
                if (!state.initialized
                    && (cell.ignited_area_fraction != amrex::Real(0.0)
                        || cell.remaining_dry_fuel_kg_m2 != amrex::Real(0.0)
                        || cell.consumed_dry_fuel_kg_m2 != amrex::Real(0.0)
                        || cell.sensible_energy_j_m2 != amrex::Real(0.0)
                        || cell.water_released_kg_m2 != amrex::Real(0.0))) {
                    throw std::invalid_argument(
                        "uninitialized restored fire combustion state must be zero");
                }
            }

            const FireCombustionRasterTotals restored_totals =
                result.totals_for(state.cells);
            require_finite_nonnegative(
                restored_totals.remaining_dry_fuel_kg,
                "restored fire combustion remaining fuel is not finite");
            require_finite_nonnegative(
                restored_totals.consumed_dry_fuel_kg,
                "restored fire combustion consumed fuel is not finite");
            require_finite_nonnegative(
                restored_totals.sensible_energy_j,
                "restored fire combustion sensible energy is not finite");
            require_finite_nonnegative(
                restored_totals.water_released_kg,
                "restored fire combustion released water is not finite");

            initialized = state.initialized ? 1 : 0;
            totals[0] = restored_totals.remaining_dry_fuel_kg;
            totals[1] = restored_totals.consumed_dry_fuel_kg;
            totals[2] = restored_totals.sensible_energy_j;
            totals[3] = restored_totals.water_released_kg;
        } catch (...) {
            invalid_state = 1;
        }
    }

    amrex::ParallelDescriptor::Bcast(&invalid_state, 1, io_rank);
    if (invalid_state != 0) {
        throw std::invalid_argument(
            "collective Fire combustion restore rejected IO-rank state");
    }
    amrex::ParallelDescriptor::Bcast(&initialized, 1, io_rank);
    amrex::ParallelDescriptor::Bcast(totals, 4, io_rank);

    amrex::BoxArray io_boxes{result.surface_layout_.cell_domain()};
    amrex::Vector<int> processor_map(1, io_rank);
    const amrex::DistributionMapping io_dm(std::move(processor_map));
    amrex::MultiFab io_state(
        io_boxes, io_dm, combustion_component_count, 0,
        fire_surface_mf_info());

    if (amrex::ParallelDescriptor::IOProcessor()) {
        for (amrex::MFIter mfi(io_state); mfi.isValid(); ++mfi) {
            const amrex::Box& box = mfi.validbox();
            const auto values = io_state.array(mfi);
            for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
                for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                    store_combustion_state(
                        values, i, j,
                        state.cells[
                            result.flat_index(
                                static_cast<std::size_t>(i),
                                static_cast<std::size_t>(j))]);
                }
            }
        }
    }

    result.states_mf_.ParallelCopy(
        io_state, 0, 0, combustion_component_count, 0, 0);
    result.totals_ = {totals[0], totals[1], totals[2], totals[3]};
    result.initialized_ = initialized != 0;
    return result;
}

FireCombustionRaster::FireCombustionRaster(
    const FireCombustionRaster& other)
    : geometry_(other.geometry_),
      parameters_(other.parameters_),
      options_(other.options_),
      surface_layout_(other.surface_layout_),
      states_mf_(
          surface_layout_.box_array(),
          surface_layout_.distribution_map(),
          combustion_component_count,
          0,
          fire_surface_mf_info()),
      totals_(other.totals_),
      initialized_(other.initialized_)
{
    copy_distributed_state(
        other.states_mf_,
        states_mf_);
}

FireCombustionRaster&
FireCombustionRaster::operator=(
    const FireCombustionRaster& other)
{
    if (this != &other) {
        FireCombustionRaster copy(other);
        *this = std::move(copy);
    }
    return *this;
}

FireCombustionRasterState
FireCombustionRaster::snapshot_state() const
{
    if (amrex::ParallelDescriptor::NProcs() != 1) {
        throw std::logic_error(
            "Fire combustion snapshot_state is single-rank only; "
            "use collective_snapshot_state_to_io_rank in parallel");
    }

    return {
        gather_distributed_to_canonical(states_mf_),
        initialized_};
}

FireCombustionRasterState
FireCombustionRaster::
collective_snapshot_state_to_io_rank() const
{
    amrex::BoxArray io_boxes{
        surface_layout_.cell_domain()};
    amrex::Vector<int> processor_map(
        1,
        amrex::ParallelDescriptor::IOProcessorNumber());
    const amrex::DistributionMapping io_dm(
        std::move(processor_map));

    amrex::MultiFab gathered(
        io_boxes,
        io_dm,
        combustion_component_count,
        0,
        fire_surface_mf_info());
    gathered.ParallelCopy(
        states_mf_,
        0,
        0,
        combustion_component_count,
        0,
        0);

    FireCombustionRasterState state;
    state.initialized = initialized_;

    if (!amrex::ParallelDescriptor::IOProcessor()) {
        return state;
    }

    state.cells.assign(
        geometry_.nx * geometry_.ny,
        FireCombustionState{});

    for (amrex::MFIter mfi(gathered);
         mfi.isValid();
         ++mfi) {
        const amrex::Box& box = mfi.validbox();
        const auto values = gathered.const_array(mfi);

        for (int j = box.smallEnd(1);
             j <= box.bigEnd(1);
             ++j) {
            for (int i = box.smallEnd(0);
                 i <= box.bigEnd(0);
                 ++i) {
                state.cells[
                    flat_index(
                        static_cast<std::size_t>(i),
                        static_cast<std::size_t>(j))] =
                            load_combustion_state(
                                values,
                                i,
                                j);
            }
        }
    }

    return state;
}

std::size_t
FireCombustionRaster::flat_index(
    std::size_t i,
    std::size_t j) const
{
    return detail::fire_cartesian_raster_flat_index(
        geometry_, i, j);
}

FireCombustionState
FireCombustionRaster::state(
    std::size_t i,
    std::size_t j) const
{
    (void)flat_index(i, j);

    if (amrex::ParallelDescriptor::NProcs() != 1) {
        throw std::logic_error(
            "Fire combustion scalar cell access is single-rank only; "
            "parallel Fire physics must use distributed_states");
    }

    const amrex::IntVect cell(
        static_cast<int>(i),
        static_cast<int>(j),
        0);
    for (amrex::MFIter mfi(states_mf_); mfi.isValid(); ++mfi) {
        if (mfi.validbox().contains(cell)) {
            return load_combustion_state(
                states_mf_.const_array(mfi),
                cell[0],
                cell[1]);
        }
    }

    throw std::logic_error(
        "single-rank Fire combustion cell is not locally represented");
}

void
FireCombustionRaster::scatter_canonical_to_distributed(
    const std::vector<FireCombustionState>& canonical,
    amrex::MultiFab& distributed) const
{
    if (canonical.size()
        != geometry_.nx * geometry_.ny) {
        throw std::invalid_argument(
            "Fire combustion canonical state has the wrong cell count");
    }
    if (distributed.boxArray() != surface_layout_.box_array()
        || distributed.DistributionMap()
            != surface_layout_.distribution_map()
        || distributed.nComp() != combustion_component_count
        || distributed.nGrow() != 0) {
        throw std::invalid_argument(
            "Fire combustion MultiFab does not match its surface layout");
    }

    for (amrex::MFIter mfi(distributed); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.validbox();
        const auto values = distributed.array(mfi);

        for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
            for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                const FireCombustionState& state_value =
                    canonical[
                        flat_index(
                            static_cast<std::size_t>(i),
                            static_cast<std::size_t>(j))];
                store_combustion_state(
                    values,
                    i,
                    j,
                    state_value);
            }
        }
    }
}

void
FireCombustionRaster::copy_distributed_state(
    const amrex::MultiFab& source,
    amrex::MultiFab& destination) const
{
    const auto matches_layout =
        [this](const amrex::MultiFab& field) {
            return field.boxArray() == surface_layout_.box_array()
                && field.DistributionMap()
                    == surface_layout_.distribution_map()
                && field.nComp() == combustion_component_count
                && field.nGrow() == 0;
        };

    if (!matches_layout(source)
        || !matches_layout(destination)) {
        throw std::invalid_argument(
            "Fire combustion MultiFab does not match its surface layout");
    }

    for (amrex::MFIter mfi(destination); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.validbox();
        const auto src = source.const_array(mfi);
        const auto dst = destination.array(mfi);

        for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
            for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                for (int component = 0;
                     component < combustion_component_count;
                     ++component) {
                    dst(i, j, 0, component) =
                        src(i, j, 0, component);
                }
            }
        }
    }
}

std::vector<FireCombustionState>
FireCombustionRaster::gather_distributed_to_canonical(
    const amrex::MultiFab& distributed) const
{
    if (distributed.boxArray() != surface_layout_.box_array()
        || distributed.DistributionMap()
            != surface_layout_.distribution_map()
        || distributed.nComp() != combustion_component_count
        || distributed.nGrow() != 0) {
        throw std::invalid_argument(
            "Fire combustion MultiFab does not match its surface layout");
    }

    amrex::FArrayBox gathered(
        surface_layout_.cell_domain(),
        combustion_component_count,
        amrex::The_Pinned_Arena());
    distributed.copyTo(
        gathered,
        0,
        0,
        combustion_component_count,
        0);
    const auto values = gathered.const_array();

    std::vector<FireCombustionState> canonical(
        geometry_.nx * geometry_.ny,
        FireCombustionState{});
    for (std::size_t j = 0; j < geometry_.ny; ++j) {
        for (std::size_t i = 0; i < geometry_.nx; ++i) {
            canonical[flat_index(i, j)] =
                load_combustion_state(
                    values,
                    static_cast<int>(i),
                    static_cast<int>(j));
        }
    }

    return canonical;
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
    return totals_;
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

    const amrex::MultiFab& burned_fraction_mf =
        burned_fraction.distributed_burned_fraction();
    require(
        burned_fraction_mf.boxArray()
                == surface_layout_.box_array()
            && burned_fraction_mf.DistributionMap()
                == surface_layout_.distribution_map()
            && burned_fraction_mf.nComp() == 1
            && burned_fraction_mf.nGrow() == 0,
        "fire combustion initialization requires co-located distributed burned history");

    amrex::MultiFab next_states(
        surface_layout_.box_array(),
        surface_layout_.distribution_map(),
        combustion_component_count,
        0,
        fire_surface_mf_info());
    copy_distributed_state(
        states_mf_,
        next_states);

    DistributedFailure local_failure =
        DistributedFailure::none;
    std::string local_error;

    try {
        for (amrex::MFIter mfi(next_states);
             mfi.isValid(); ++mfi) {
            const amrex::Box& box = mfi.validbox();
            const auto values = next_states.array(mfi);
            const auto burned =
                burned_fraction_mf.const_array(mfi);

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
                    const FireCombustionState current =
                        load_combustion_state(
                            values,
                            i,
                            j);
                    const FireCombustionState next =
                        add_fire_combustion_ignition(
                            current,
                            parameters_,
                            burned(i, j, 0));
                    store_combustion_state(
                        values,
                        i,
                        j,
                        next);
                }
            }
        }
    } catch (const std::invalid_argument& error) {
        local_failure = DistributedFailure::invalid_argument;
        local_error = error.what();
    } catch (const std::logic_error& error) {
        local_failure = DistributedFailure::logic_error;
        local_error = error.what();
    } catch (const std::overflow_error& error) {
        local_failure = DistributedFailure::overflow_error;
        local_error = error.what();
    } catch (const std::exception& error) {
        local_failure = DistributedFailure::runtime_error;
        local_error = error.what();
    } catch (...) {
        local_failure = DistributedFailure::runtime_error;
        local_error = "unknown local Fire combustion initialization error";
    }

    synchronize_distributed_failure(
        local_failure,
        local_error,
        "distributed Fire combustion initialization");

    FireCombustionRasterTotals next_totals =
        local_distributed_totals(
            next_states,
            geometry_);
    reduce_distributed_totals(next_totals);

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

    states_mf_ = std::move(next_states);
    totals_ = next_totals;
    initialized_ = true;

    return totals_;
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

    const amrex::MultiFab& burned_before_mf =
        burned_before.distributed_burned_fraction();
    const amrex::MultiFab& burned_after_mf =
        burned_after.distributed_burned_fraction();
    const auto matches_burned_layout =
        [this](const amrex::MultiFab& field) {
            return field.boxArray()
                    == surface_layout_.box_array()
                && field.DistributionMap()
                    == surface_layout_.distribution_map()
                && field.nComp() == 1
                && field.nGrow() == 0;
        };
    require(
        matches_burned_layout(burned_before_mf)
            && matches_burned_layout(burned_after_mf),
        "fire combustion advance requires co-located distributed burned history");

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

    amrex::MultiFab next_states(
        surface_layout_.box_array(),
        surface_layout_.distribution_map(),
        combustion_component_count,
        0,
        fire_surface_mf_info());
    amrex::MultiFab running_burned_fraction(
        surface_layout_.box_array(),
        surface_layout_.distribution_map(),
        1,
        0,
        fire_surface_mf_info());

    copy_distributed_state(
        states_mf_,
        next_states);

    DistributedFailure local_failure =
        DistributedFailure::none;
    std::string local_error;

    try {
        for (amrex::MFIter mfi(next_states);
             mfi.isValid(); ++mfi) {
            const amrex::Box& box = mfi.validbox();
            const auto values = next_states.const_array(mfi);
            const auto running =
                running_burned_fraction.array(mfi);
            const auto before_values =
                burned_before_mf.const_array(mfi);
            const auto after_values =
                burned_after_mf.const_array(mfi);

            for (int j = box.smallEnd(1);
                 j <= box.bigEnd(1);
                 ++j) {
                for (int i = box.smallEnd(0);
                     i <= box.bigEnd(0);
                     ++i) {
                    const amrex::Real before =
                        before_values(i, j, 0);
                    const amrex::Real after =
                        after_values(i, j, 0);
                    const FireCombustionState current =
                        load_combustion_state(
                            values,
                            i,
                            j);

                    require(
                        after + fraction_tolerance(after)
                            >= before,
                        "fire combustion raster burned history is not monotone");
                    require(
                        fraction_equal(
                            current.ignited_area_fraction,
                            before),
                        "fire combustion raster state is not synchronized with burned history");

                    running(i, j, 0) = before;
                }
            }
        }

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

            for (amrex::MFIter mfi(next_states);
                 mfi.isValid(); ++mfi) {
                const amrex::Box& box = mfi.validbox();
                const auto values = next_states.array(mfi);
                const auto running =
                    running_burned_fraction.array(mfi);

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
                        const FireCartesianCell2D cell =
                            burned_before.cell_bounds(ii, jj);

                        const amrex::Real coverage =
                            fire_perimeter_cell_coverage_fraction(
                                sample_perimeter,
                                cell);
                        const amrex::Real next_burned_fraction =
                            std::max(
                                running(i, j, 0),
                                coverage);
                        const amrex::Real newly_ignited_fraction =
                            next_burned_fraction
                            - running(i, j, 0);

                        const FireCombustionState current =
                            load_combustion_state(
                                values,
                                i,
                                j);
                        const FireCombustionAdvance first_half =
                            advance_fire_combustion(
                                current,
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

                        store_combustion_state(
                            values,
                            i,
                            j,
                            second_half.state);
                        running(i, j, 0) =
                            next_burned_fraction;
                    }
                }
            }
        }

        for (amrex::MFIter mfi(next_states);
             mfi.isValid(); ++mfi) {
            const amrex::Box& box = mfi.validbox();
            const auto values = next_states.const_array(mfi);
            const auto running =
                running_burned_fraction.const_array(mfi);
            const auto target_values =
                burned_after_mf.const_array(mfi);

            for (int j = box.smallEnd(1);
                 j <= box.bigEnd(1);
                 ++j) {
                for (int i = box.smallEnd(0);
                     i <= box.bigEnd(0);
                     ++i) {
                    const amrex::Real target =
                        target_values(i, j, 0);
                    const FireCombustionState current =
                        load_combustion_state(
                            values,
                            i,
                            j);

                    require(
                        fraction_equal(
                            running(i, j, 0),
                            target),
                        "fire combustion temporal sampling does not reproduce endpoint burned history");
                    require(
                        fraction_equal(
                            current.ignited_area_fraction,
                            target),
                        "fire combustion endpoint state is not synchronized with burned history");
                }
            }
        }
    } catch (const std::invalid_argument& error) {
        local_failure = DistributedFailure::invalid_argument;
        local_error = error.what();
    } catch (const std::logic_error& error) {
        local_failure = DistributedFailure::logic_error;
        local_error = error.what();
    } catch (const std::overflow_error& error) {
        local_failure = DistributedFailure::overflow_error;
        local_error = error.what();
    } catch (const std::exception& error) {
        local_failure = DistributedFailure::runtime_error;
        local_error = error.what();
    } catch (...) {
        local_failure = DistributedFailure::runtime_error;
        local_error = "unknown local Fire combustion sweep error";
    }

    synchronize_distributed_failure(
        local_failure,
        local_error,
        "distributed Fire combustion sweep");

    const FireCombustionRasterTotals local_next_totals =
        local_distributed_totals(
            next_states,
            geometry_);

    const amrex::Real cell_area_m2 =
        geometry_.dx_m * geometry_.dy_m;
    amrex::Real accounting[7]{
        local_next_totals.remaining_dry_fuel_kg,
        local_next_totals.consumed_dry_fuel_kg,
        local_next_totals.sensible_energy_j,
        local_next_totals.water_released_kg,
        amrex::Real(0),
        amrex::Real(0),
        amrex::Real(0)};

    for (amrex::MFIter mfi(next_states); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.validbox();
        const auto previous = states_mf_.const_array(mfi);
        const auto next = next_states.const_array(mfi);

        for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
            for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                accounting[4] +=
                    (next(i, j, 0, consumed_dry_fuel_comp)
                     - previous(i, j, 0, consumed_dry_fuel_comp))
                    * cell_area_m2;
                accounting[5] +=
                    (next(i, j, 0, sensible_energy_comp)
                     - previous(i, j, 0, sensible_energy_comp))
                    * cell_area_m2;
                accounting[6] +=
                    (next(i, j, 0, water_released_comp)
                     - previous(i, j, 0, water_released_comp))
                    * cell_area_m2;
            }
        }
    }

    amrex::ParallelDescriptor::ReduceRealSum(accounting, 7);

    const FireCombustionRasterTotals next_totals{
        accounting[0],
        accounting[1],
        accounting[2],
        accounting[3]};
    const amrex::Real newly_consumed_dry_fuel_kg = accounting[4];
    const amrex::Real sensible_energy_increment_j = accounting[5];
    const amrex::Real water_released_increment_kg = accounting[6];

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

    states_mf_ = std::move(next_states);
    totals_ = next_totals;

    return {
        totals_,
        newly_consumed_dry_fuel_kg,
        sensible_energy_increment_j,
        water_released_increment_kg
    };
}

} // namespace ERFFire
