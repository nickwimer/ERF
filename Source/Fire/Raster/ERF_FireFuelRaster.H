#ifndef ERF_FIRE_FUEL_RASTER_H_
#define ERF_FIRE_FUEL_RASTER_H_

#include <ERF_FireFuelModel.H>
#include <ERF_FireFuelMoisture.H>
#include <ERF_FireRasterGeometry.H>
#include <ERF_FireSurfaceLayout.H>
#include <ERF_FireTypes.H>

#include <AMReX_Arena.H>
#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuBuffer.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_Math.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParReduce.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ERFFire
{

/**
 * Distributed storage accepts the stable material IDs defined by
 * ERF_FireFuelModel.H. Storage validity is intentionally separate from the
 * required-moisture and production-physics contracts.
 */
struct FireFuelRasterCell
{
    FireFuelModelId model_id{FireFuelModelId::FM1};
    FireFuelMoisture moisture{};
};

/**
 * Model-neutral name for a resolved surface-fire material.
 *
 * This is intentionally an alias of the established raster-cell type so the
 * first material-typing step changes no runtime object layout or type identity.
 */
using FireFuelMaterial = FireFuelRasterCell;

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

    if (raw_model < amrex::Real(0)
        || raw_model > amrex::Real(13)) {
        return false;
    }
    const int model_number = static_cast<int>(raw_model);
    if (static_cast<amrex::Real>(model_number) != raw_model) {
        return false;
    }
    const FireFuelModelId model_id =
        static_cast<FireFuelModelId>(model_number);
    if (!fire_fuel_model_id_valid(model_id)) {
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

struct FireFuelRasterPointQuery
{
    int local_box{};
    int i{};
    int j{};
    int output_index{};
};

inline FireFuelRasterCell
decode_collective_fire_fuel_sample(
    const amrex::Real* raw)
{
    FireFuelRasterCell result;

    const amrex::Real raw_model =
        raw[FireFuelRasterComponents::model_id];
    const amrex::Real raw_mask =
        raw[FireFuelRasterComponents::moisture_presence_mask];

    if (!std::isfinite(raw_model)
        || !std::isfinite(raw_mask)) {
        throw std::logic_error(
            "collective Fire fuel point sample contains non-finite encoding");
    }

    if (raw_model < amrex::Real(0)
        || raw_model > amrex::Real(13)) {
        throw std::logic_error(
            "collective Fire fuel point sample contains invalid model id");
    }
    const int model_number = static_cast<int>(raw_model);
    if (static_cast<amrex::Real>(model_number) != raw_model) {
        throw std::logic_error(
            "collective Fire fuel point sample contains non-integral model id");
    }
    result.model_id =
        static_cast<FireFuelModelId>(model_number);
    if (!fire_fuel_model_id_valid(result.model_id)) {
        throw std::logic_error(
            "collective Fire fuel point sample contains invalid model id");
    }

    constexpr int valid_mask =
        (1 << FireFuelMoisture::component_count) - 1;
    if (raw_mask < amrex::Real(0)
        || raw_mask > static_cast<amrex::Real>(valid_mask)) {
        throw std::logic_error(
            "collective Fire fuel point sample contains invalid moisture mask");
    }
    const int mask = static_cast<int>(raw_mask);
    if (static_cast<amrex::Real>(mask) != raw_mask) {
        throw std::logic_error(
            "collective Fire fuel point sample contains non-integral moisture mask");
    }

    FireFuelMoisture moisture;
    for (int component = 0;
         component < FireFuelMoisture::component_count;
         ++component) {
        const amrex::Real value =
            raw[fire_fuel_moisture_component_from_index(component)];
        if (!std::isfinite(value)) {
            throw std::logic_error(
                "collective Fire fuel point sample contains non-finite moisture");
        }
        if ((mask & (1 << component)) != 0
            && !moisture.try_set(
                fire_fuel_moisture_class_from_index(component),
                value)) {
            throw std::logic_error(
                "collective Fire fuel point sample contains invalid moisture");
        }
    }

    result.moisture = moisture;
    return result;
}

/**
 * Collectively sample distributed raw material values at ordered map points.
 *
 * Every rank must provide the same ordered positions. Each rank gathers only
 * queries owned by its local FireSurfaceLayout boxes into compact GPU buffers;
 * one all-reduce then assembles those sparse results on every rank. No full
 * material raster is gathered and no per-point MPI operation is performed.
 */
inline std::vector<FireFuelRasterCell>
collective_sample_fire_fuel_raster_points(
    const FireCartesianRasterGeometry2D& geometry,
    const amrex::MultiFab& values,
    const std::vector<FireVec2>& positions_m)
{
    if (positions_m.empty()) {
        return {};
    }

    constexpr int packed_stride =
        FireFuelRasterComponents::component_count + 1;
    const std::size_t max_points =
        static_cast<std::size_t>(
            std::numeric_limits<int>::max() / packed_stride);
    if (positions_m.size() > max_points) {
        throw std::length_error(
            "collective Fire fuel point sample is too large for one MPI reduction");
    }

    std::vector<FireCartesianRasterIndex2D> cells;
    cells.reserve(positions_m.size());
    for (const FireVec2& position : positions_m) {
        cells.push_back(
            fire_cartesian_raster_locate_cell(
                geometry,
                position.x,
                position.y));
    }

    std::vector<FireFuelRasterPointQuery> local_queries;
    local_queries.reserve(positions_m.size());

    for (amrex::MFIter mfi(values); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.validbox();
        const int local_box = mfi.LocalIndex();

        for (std::size_t index = 0;
             index < cells.size();
             ++index) {
            if (cells[index].i
                    > static_cast<std::size_t>(
                        std::numeric_limits<int>::max())
                || cells[index].j
                    > static_cast<std::size_t>(
                        std::numeric_limits<int>::max())) {
                throw std::overflow_error(
                    "Fire fuel point-sample cell index is not representable");
            }

            const int i = static_cast<int>(cells[index].i);
            const int j = static_cast<int>(cells[index].j);
            const amrex::IntVect cell(i, j, 0);
            if (box.contains(cell)) {
                local_queries.push_back({
                    local_box,
                    i,
                    j,
                    static_cast<int>(index)});
            }
        }
    }

    const std::size_t packed_count =
        positions_m.size()
        * static_cast<std::size_t>(packed_stride);
    amrex::Gpu::Buffer<amrex::Real> packed(packed_count);
    std::fill_n(
        packed.hostData(),
        packed_count,
        amrex::Real(0));
    (void)packed.copyToDeviceAsync();

    amrex::Gpu::Buffer<FireFuelRasterPointQuery> queries(
        local_queries.data(),
        local_queries.size());

    if (!local_queries.empty()) {
        const auto arrays = values.const_arrays();
        const auto* query_data = queries.data();
        auto* packed_data = packed.data();
        const int query_count =
            static_cast<int>(local_queries.size());

        amrex::ParallelFor(
            query_count,
            [=] AMREX_GPU_DEVICE (int query_index) noexcept
            {
                const FireFuelRasterPointQuery query =
                    query_data[query_index];
                const auto cell_values =
                    arrays[query.local_box];
                const int base =
                    query.output_index * packed_stride;

                packed_data[base] = amrex::Real(1);
                for (int component = 0;
                     component
                         < FireFuelRasterComponents::component_count;
                     ++component) {
                    packed_data[base + 1 + component] =
                        cell_values(
                            query.i,
                            query.j,
                            0,
                            component);
                }
            });
    }

    (void)packed.copyToHost();

    amrex::ParallelDescriptor::ReduceRealSum(
        packed.hostData(),
        static_cast<int>(packed_count));

    std::vector<FireFuelRasterCell> result;
    result.reserve(positions_m.size());
    for (std::size_t index = 0;
         index < positions_m.size();
         ++index) {
        const std::size_t base =
            index * static_cast<std::size_t>(packed_stride);
        if (packed[base] != amrex::Real(1)) {
            throw std::logic_error(
                "collective Fire fuel point sample did not resolve exactly one distributed owner");
        }
        result.push_back(
            decode_collective_fire_fuel_sample(
                packed.hostData() + base + 1));
    }

    return result;
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

    /**
     * Collectively sample material at ordered physical map-plane positions.
     *
     * Every rank must call this method with the same ordered position vector.
     * Internal/upper-boundary ownership follows the authoritative Cartesian
     * raster locator. The result has the same length and ordering as input.
     * One sparse all-reduce is used for a nonempty batch; there is no full-
     * domain gather and no per-point MPI communication.
     */
    [[nodiscard]] std::vector<FireFuelRasterCell>
    collective_sample_points(
        const std::vector<FireVec2>& positions_m) const
    {
        return detail::collective_sample_fire_fuel_raster_points(
            geometry(),
            storage_->values,
            positions_m);
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
