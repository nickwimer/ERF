#ifndef ERF_FIRE_FUEL_RASTER_H_
#define ERF_FIRE_FUEL_RASTER_H_

#include <ERF_FireFuelMoisture.H>
#include <ERF_FireRasterGeometry.H>
#include <ERF_FireSurfaceLayout.H>

#include <AMReX_Arena.H>
#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Gpu.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_Math.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParReduce.H>
#include <AMReX_ParallelDescriptor.H>

#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ERFFire
{

/**
 * Stable categorical identifiers for the spatial material raster.
 *
 * Only explicit no-fuel and FM1 are admitted in this storage slice. Additional
 * Anderson identifiers are added only when their multi-class behavior is
 * implemented. Storing NonBurnable here does not yet enable barrier dynamics.
 */
enum class FireFuelModelId : int
{
    NonBurnable = 0,
    FM1 = 1
};

struct FireFuelRasterCell
{
    FireFuelModelId model_id{FireFuelModelId::FM1};
    FireFuelMoisture moisture{};
};

struct FireFuelRasterState
{
    std::vector<FireFuelRasterCell> cells;
};

/**
 * Real-valued distributed storage schema.
 *
 * Model id and moisture-presence mask are small exact integers encoded as
 * amrex::Real so one MultiFab owns the complete immutable device-resident
 * material raster. Values 0..13 and masks 0..31 are exactly representable in
 * both supported ERF precisions. Missing moisture is represented by the mask,
 * never by a numerical sentinel.
 */
struct FireFuelRasterComponents
{
    static constexpr int model_id = 0;
    static constexpr int moisture_presence_mask = 1;
    static constexpr int dead_1h_moisture = 2;
    static constexpr int dead_10h_moisture = 3;
    static constexpr int dead_100h_moisture = 4;
    static constexpr int live_herbaceous_moisture = 5;
    static constexpr int live_woody_moisture = 6;
    static constexpr int component_count = 7;
};

namespace detail
{

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool
fire_fuel_model_id_valid(FireFuelModelId id) noexcept
{
    return id == FireFuelModelId::NonBurnable
        || id == FireFuelModelId::FM1;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
FireFuelMoistureClass
fire_fuel_moisture_class_from_index(int index) noexcept
{
    return static_cast<FireFuelMoistureClass>(index);
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
int
fire_fuel_moisture_component_from_index(int index) noexcept
{
    return FireFuelRasterComponents::dead_1h_moisture + index;
}

/**
 * Decode one cell without allocation or exceptions. Suitable for cell-local
 * CPU/GPU physics. The output is reset on failure and is valid only when this
 * function returns true.
 */
template <typename T>
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
bool
try_decode_fire_fuel_raster_cell(
    const amrex::Array4<T>& values,
    int i,
    int j,
    FireFuelRasterCell& result) noexcept
{
    result = {};

    const amrex::Real raw_model =
        values(i, j, 0, FireFuelRasterComponents::model_id);
    const amrex::Real raw_mask =
        values(i, j, 0, FireFuelRasterComponents::moisture_presence_mask);

    if (!amrex::Math::isfinite(raw_model)
        || !amrex::Math::isfinite(raw_mask)) {
        return false;
    }

    FireFuelModelId model_id{};
    if (raw_model == amrex::Real(0)) {
        model_id = FireFuelModelId::NonBurnable;
    } else if (raw_model == amrex::Real(1)) {
        model_id = FireFuelModelId::FM1;
    } else {
        return false;
    }

    constexpr int valid_mask =
        (1 << FireFuelMoisture::component_count) - 1;
    if (raw_mask < amrex::Real(0)
        || raw_mask > static_cast<amrex::Real>(valid_mask)) {
        return false;
    }
    const int mask = static_cast<int>(raw_mask);
    if (static_cast<amrex::Real>(mask) != raw_mask) {
        return false;
    }

    FireFuelMoisture moisture;
    for (int component = 0;
         component < FireFuelMoisture::component_count;
         ++component) {
        const amrex::Real value =
            values(
                i,
                j,
                0,
                fire_fuel_moisture_component_from_index(component));
        if (!amrex::Math::isfinite(value)) {
            return false;
        }
        if ((mask & (1 << component)) != 0
            && !moisture.try_set(
                fire_fuel_moisture_class_from_index(component),
                value)) {
            return false;
        }
    }

    result.model_id = model_id;
    result.moisture = moisture;
    return true;
}

inline
bool
fire_fuel_raster_cell_valid(const FireFuelRasterCell& cell) noexcept
{
    if (!fire_fuel_model_id_valid(cell.model_id)) {
        return false;
    }

    for (int component = 0;
         component < FireFuelMoisture::component_count;
         ++component) {
        amrex::Real value{};
        if (cell.moisture.try_get(
                fire_fuel_moisture_class_from_index(component),
                value)
            && (!std::isfinite(value)
                || value < amrex::Real(0))) {
            return false;
        }
    }
    return true;
}

inline
void
store_fire_fuel_raster_cell(
    const amrex::Array4<amrex::Real>& values,
    int i,
    int j,
    const FireFuelRasterCell& cell)
{
    if (!fire_fuel_raster_cell_valid(cell)) {
        throw std::invalid_argument(
            "Fire fuel raster cell contains an unsupported model or invalid moisture");
    }

    values(i, j, 0, FireFuelRasterComponents::model_id) =
        static_cast<amrex::Real>(
            static_cast<int>(cell.model_id));

    int presence_mask = 0;
    for (int component = 0;
         component < FireFuelMoisture::component_count;
         ++component) {
        amrex::Real value = amrex::Real(0);
        if (cell.moisture.try_get(
                fire_fuel_moisture_class_from_index(component),
                value)) {
            presence_mask |= (1 << component);
        }
        values(
            i,
            j,
            0,
            fire_fuel_moisture_component_from_index(component)) = value;
    }

    values(i, j, 0, FireFuelRasterComponents::moisture_presence_mask) =
        static_cast<amrex::Real>(presence_mask);
}

inline
amrex::MFInfo
fire_fuel_host_mf_info()
{
    amrex::MFInfo info;
    info.SetArena(amrex::The_Pinned_Arena());
    return info;
}

} // namespace detail

/**
 * Immutable distributed categorical fuel/moisture raster.
 *
 * Persistent storage uses the same FireSurfaceLayout as the existing Fire
 * history rasters and is device-resident in GPU builds. A full canonical
 * row-major mirror is not retained. Construction/restoration accepts canonical
 * cells on IOProcessor only, then scatters collectively. Copies share immutable
 * storage, so copying ERFFire runtime candidates will not deep-copy a static
 * landscape once this raster is wired into production.
 *
 * This class is storage only. It does not open files, map source resolutions,
 * evaluate Rothermel, construct combustion parameters, or enforce barriers.
 */
class FireFuelRaster
{
private:
    struct Storage
    {
        explicit Storage(FireCartesianRasterGeometry2D geometry_in)
            : geometry(geometry_in),
              surface_layout(geometry),
              values(
                  surface_layout.box_array(),
                  surface_layout.distribution_map(),
                  FireFuelRasterComponents::component_count,
                  0)
        {}

        FireCartesianRasterGeometry2D geometry{};
        FireSurfaceLayout surface_layout;
        amrex::MultiFab values;
    };

public:
    /**
     * Collectively construct from row-major cells owned only by IOProcessor().
     * Non-IO ranks must pass an empty vector. No canonical mirror is retained.
     */
    [[nodiscard]] static FireFuelRaster
    collective_from_io_rank_state(
        FireCartesianRasterGeometry2D geometry,
        const FireFuelRasterState& state)
    {
        (void)detail::validate_fire_cartesian_raster_geometry(geometry);

        auto storage = std::make_shared<Storage>(geometry);
        const int io_rank =
            amrex::ParallelDescriptor::IOProcessorNumber();

        int invalid = 0;
        if (amrex::ParallelDescriptor::IOProcessor()) {
            if (state.cells.size() != geometry.nx * geometry.ny) {
                invalid = 1;
            } else {
                for (const auto& cell : state.cells) {
                    if (!detail::fire_fuel_raster_cell_valid(cell)) {
                        invalid = 1;
                        break;
                    }
                }
            }
        } else if (!state.cells.empty()) {
            invalid = 1;
        }

        amrex::ParallelDescriptor::ReduceIntMax(invalid);
        if (invalid != 0) {
            throw std::invalid_argument(
                "collective Fire fuel raster state is invalid or not IO-rank owned");
        }

        amrex::BoxArray io_boxes{storage->surface_layout.cell_domain()};
        amrex::Vector<int> processor_map(1, io_rank);
        const amrex::DistributionMapping io_dm(std::move(processor_map));
        amrex::MultiFab io_values(
            io_boxes,
            io_dm,
            FireFuelRasterComponents::component_count,
            0,
            detail::fire_fuel_host_mf_info());
        io_values.setVal(amrex::Real(0));

        if (amrex::ParallelDescriptor::IOProcessor()) {
            for (amrex::MFIter mfi(io_values); mfi.isValid(); ++mfi) {
                const amrex::Box& box = mfi.validbox();
                const auto values = io_values.array(mfi);
                for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
                    for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                        const std::size_t index =
                            detail::fire_cartesian_raster_flat_index(
                                geometry,
                                static_cast<std::size_t>(i),
                                static_cast<std::size_t>(j));
                        detail::store_fire_fuel_raster_cell(
                            values,
                            i,
                            j,
                            state.cells[index]);
                    }
                }
            }
        }

        storage->values.ParallelCopy(
            io_values,
            0,
            0,
            FireFuelRasterComponents::component_count,
            0,
            0);
        amrex::Gpu::streamSynchronize();

        const auto arrays = storage->values.const_arrays();
        const auto validation =
            amrex::ParReduce(
                amrex::TypeList<amrex::ReduceOpMax>{},
                amrex::TypeList<int>{},
                storage->values,
                [=] AMREX_GPU_DEVICE (
                    int box_no,
                    int i,
                    int j,
                    int) noexcept -> amrex::GpuTuple<int>
                {
                    FireFuelRasterCell cell;
                    return {
                        detail::try_decode_fire_fuel_raster_cell(
                            arrays[box_no], i, j, cell)
                            ? 0
                            : 1};
                });

        int distributed_invalid = validation;
        amrex::ParallelDescriptor::ReduceIntMax(distributed_invalid);
        if (distributed_invalid != 0) {
            throw std::invalid_argument(
                "distributed Fire fuel raster encoding is invalid");
        }

        return FireFuelRaster(std::move(storage));
    }

    [[nodiscard]] const FireCartesianRasterGeometry2D&
    geometry() const noexcept
    {
        return storage_->geometry;
    }

    [[nodiscard]] const FireSurfaceLayout&
    surface_layout() const noexcept
    {
        return storage_->surface_layout;
    }

    [[nodiscard]] const amrex::MultiFab&
    distributed_values() const noexcept
    {
        return storage_->values;
    }

    /** Single-rank scalar access; performs no hidden MPI communication. */
    [[nodiscard]] FireFuelRasterCell
    state(std::size_t i, std::size_t j) const
    {
        (void)detail::fire_cartesian_raster_flat_index(
            geometry(), i, j);
        if (amrex::ParallelDescriptor::NProcs() != 1) {
            throw std::logic_error(
                "Fire fuel raster scalar access is single-rank only");
        }

        const amrex::IntVect index(
            static_cast<int>(i),
            static_cast<int>(j),
            0);
        amrex::FArrayBox host_cell(
            amrex::Box(index, index),
            FireFuelRasterComponents::component_count,
            amrex::The_Pinned_Arena());
        storage_->values.copyTo(
            host_cell,
            0,
            0,
            FireFuelRasterComponents::component_count,
            0);
        amrex::Gpu::streamSynchronize();

        FireFuelRasterCell result;
        if (!detail::try_decode_fire_fuel_raster_cell(
                host_cell.const_array(),
                index[0],
                index[1],
                result)) {
            throw std::logic_error(
                "Fire fuel raster contains an invalid encoded cell");
        }
        return result;
    }

    [[nodiscard]] FireFuelRasterState
    snapshot_state() const
    {
        if (amrex::ParallelDescriptor::NProcs() != 1) {
            throw std::logic_error(
                "Fire fuel raster snapshot_state is single-rank only; use collective snapshot in parallel");
        }

        amrex::FArrayBox gathered(
            storage_->surface_layout.cell_domain(),
            FireFuelRasterComponents::component_count,
            amrex::The_Pinned_Arena());
        storage_->values.copyTo(
            gathered,
            0,
            0,
            FireFuelRasterComponents::component_count,
            0);
        amrex::Gpu::streamSynchronize();

        FireFuelRasterState result;
        result.cells.resize(geometry().nx * geometry().ny);
        const auto values = gathered.const_array();
        for (std::size_t j = 0; j < geometry().ny; ++j) {
            for (std::size_t i = 0; i < geometry().nx; ++i) {
                FireFuelRasterCell cell;
                if (!detail::try_decode_fire_fuel_raster_cell(
                        values,
                        static_cast<int>(i),
                        static_cast<int>(j),
                        cell)) {
                    throw std::logic_error(
                        "Fire fuel raster contains an invalid encoded cell");
                }
                result.cells[
                    detail::fire_cartesian_raster_flat_index(
                        geometry(), i, j)] = cell;
            }
        }
        return result;
    }

    /**
     * Collectively gather canonical cells onto IOProcessor only.
     * Other ranks return an empty cell vector.
     */
    [[nodiscard]] FireFuelRasterState
    collective_snapshot_state_to_io_rank() const
    {
        const int io_rank =
            amrex::ParallelDescriptor::IOProcessorNumber();
        amrex::BoxArray io_boxes{storage_->surface_layout.cell_domain()};
        amrex::Vector<int> processor_map(1, io_rank);
        const amrex::DistributionMapping io_dm(std::move(processor_map));
        amrex::MultiFab gathered(
            io_boxes,
            io_dm,
            FireFuelRasterComponents::component_count,
            0,
            detail::fire_fuel_host_mf_info());
        gathered.ParallelCopy(
            storage_->values,
            0,
            0,
            FireFuelRasterComponents::component_count,
            0,
            0);
        amrex::Gpu::streamSynchronize();

        FireFuelRasterState result;
        if (!amrex::ParallelDescriptor::IOProcessor()) {
            return result;
        }

        result.cells.resize(geometry().nx * geometry().ny);
        for (amrex::MFIter mfi(gathered); mfi.isValid(); ++mfi) {
            const amrex::Box& box = mfi.validbox();
            const auto values = gathered.const_array(mfi);
            for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
                for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                    FireFuelRasterCell cell;
                    if (!detail::try_decode_fire_fuel_raster_cell(
                            values, i, j, cell)) {
                        throw std::logic_error(
                            "Fire fuel raster contains an invalid encoded cell");
                    }
                    result.cells[
                        detail::fire_cartesian_raster_flat_index(
                            geometry(),
                            static_cast<std::size_t>(i),
                            static_cast<std::size_t>(j))] = cell;
                }
            }
        }
        return result;
    }

private:
    explicit FireFuelRaster(std::shared_ptr<const Storage> storage)
        : storage_(std::move(storage))
    {}

    std::shared_ptr<const Storage> storage_;
};

} // namespace ERFFire

#endif // ERF_FIRE_FUEL_RASTER_H_
