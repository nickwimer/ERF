#ifndef ERF_FIRE_CHECKPOINT_V4_H_
#define ERF_FIRE_CHECKPOINT_V4_H_

#include <ERF_FireFuelRaster.H>
#include <ERF_FireSpreadOutput.H>
#include <ERF_FireSpreadRuntime.H>
#include <ERF_FireSurfaceLayout.H>

#include <AMReX_Arena.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>

#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace ERFFire
{

/** Distributed checkpoint schema for spatial-material runtime format v4. */
struct ERFFireCheckpointV4RasterComponents
{
    static constexpr int burned_fraction =
        ERFFireCheckpointRasterComponents::burned_fraction;
    static constexpr int arrived =
        ERFFireCheckpointRasterComponents::arrived;
    static constexpr int first_arrival_time_s =
        ERFFireCheckpointRasterComponents::first_arrival_time_s;
    static constexpr int ignited_area_fraction =
        ERFFireCheckpointRasterComponents::ignited_area_fraction;
    static constexpr int remaining_dry_fuel_kg_m2 =
        ERFFireCheckpointRasterComponents::remaining_dry_fuel_kg_m2;
    static constexpr int consumed_dry_fuel_kg_m2 =
        ERFFireCheckpointRasterComponents::consumed_dry_fuel_kg_m2;
    static constexpr int sensible_energy_j_m2 =
        ERFFireCheckpointRasterComponents::sensible_energy_j_m2;
    static constexpr int water_released_kg_m2 =
        ERFFireCheckpointRasterComponents::water_released_kg_m2;

    static constexpr int spatial_fuel =
        ERFFireCheckpointRasterComponents::component_count;
    static constexpr int component_count =
        spatial_fuel + FireFuelRasterComponents::component_count;
};

struct ERFFireCheckpointV4Metadata
{
    ERFFireCheckpointState checkpoint{};
    int spatial_fuel_schema_version{1};
    std::uint64_t spatial_fuel_fingerprint_fnv1a64{};
};

namespace detail
{

inline constexpr int fire_spatial_fuel_checkpoint_schema_version = 1;
inline constexpr std::uint64_t fire_fuel_fnv1a64_offset_basis =
    UINT64_C(14695981039346656037);
inline constexpr std::uint64_t fire_fuel_fnv1a64_prime =
    UINT64_C(1099511628211);

inline void
fire_fuel_fnv1a64_append_bytes(
    std::uint64_t& hash,
    const void* data,
    std::size_t byte_count) noexcept
{
    const auto* bytes =
        static_cast<const unsigned char*>(data);
    for (std::size_t index = 0;
         index < byte_count;
         ++index) {
        hash ^= static_cast<std::uint64_t>(bytes[index]);
        hash *= fire_fuel_fnv1a64_prime;
    }
}

template <typename T>
inline void
fire_fuel_fnv1a64_append_value(
    std::uint64_t& hash,
    const T& value) noexcept
{
    fire_fuel_fnv1a64_append_bytes(
        hash,
        &value,
        sizeof(T));
}

inline std::uint64_t
fire_fuel_raster_state_fingerprint_fnv1a64(
    const FireCartesianRasterGeometry2D& geometry,
    const FireFuelRasterState& state)
{
    const std::size_t cell_count =
        validate_fire_cartesian_raster_geometry(geometry);
    if (state.cells.size() != cell_count) {
        throw std::invalid_argument(
            "Fire fuel fingerprint requires one canonical cell per raster cell");
    }

    std::uint64_t hash = fire_fuel_fnv1a64_offset_basis;
    const std::uint32_t schema =
        static_cast<std::uint32_t>(
            fire_spatial_fuel_checkpoint_schema_version);
    const std::uint32_t real_bytes =
        static_cast<std::uint32_t>(sizeof(amrex::Real));
    const std::uint64_t nx =
        static_cast<std::uint64_t>(geometry.nx);
    const std::uint64_t ny =
        static_cast<std::uint64_t>(geometry.ny);

    fire_fuel_fnv1a64_append_value(hash, schema);
    fire_fuel_fnv1a64_append_value(hash, real_bytes);
    fire_fuel_fnv1a64_append_value(hash, nx);
    fire_fuel_fnv1a64_append_value(hash, ny);
    fire_fuel_fnv1a64_append_value(hash, geometry.xlo_m);
    fire_fuel_fnv1a64_append_value(hash, geometry.ylo_m);
    fire_fuel_fnv1a64_append_value(hash, geometry.dx_m);
    fire_fuel_fnv1a64_append_value(hash, geometry.dy_m);

    for (const FireFuelRasterCell& cell : state.cells) {
        if (!fire_fuel_raster_cell_valid(cell)) {
            throw std::invalid_argument(
                "Fire fuel fingerprint received invalid resolved material");
        }

        const std::int32_t model_id =
            static_cast<std::int32_t>(cell.model_id);
        fire_fuel_fnv1a64_append_value(hash, model_id);

        for (int component = 0;
             component < FireFuelMoisture::component_count;
             ++component) {
            const auto moisture_class =
                fire_fuel_moisture_class_from_index(component);
            amrex::Real value{};
            const bool present =
                cell.moisture.try_get(
                    moisture_class,
                    value);
            const std::uint8_t present_byte =
                present ? UINT8_C(1) : UINT8_C(0);
            fire_fuel_fnv1a64_append_value(
                hash,
                present_byte);
            if (present) {
                fire_fuel_fnv1a64_append_value(
                    hash,
                    value);
            }
        }
    }

    return hash;
}

inline FireFuelRaster
collective_restore_fire_fuel_raster_from_v4(
    const FireCartesianRasterGeometry2D& geometry,
    const amrex::MultiFab& checkpoint_raster)
{
    if (checkpoint_raster.nComp()
            != ERFFireCheckpointV4RasterComponents::component_count
        || checkpoint_raster.nGrow() != 0) {
        throw std::invalid_argument(
            "ERF-Fire v4 checkpoint raster has incompatible components or ghosts");
    }

    const FireSurfaceLayout layout(geometry);
    const int io_rank =
        amrex::ParallelDescriptor::IOProcessorNumber();
    amrex::BoxArray io_boxes{layout.cell_domain()};
    amrex::Vector<int> processor_map(1, io_rank);
    const amrex::DistributionMapping io_dm(
        std::move(processor_map));
    amrex::MFInfo host_info;
    host_info.SetArena(amrex::The_Pinned_Arena());
    amrex::MultiFab gathered(
        io_boxes,
        io_dm,
        FireFuelRasterComponents::component_count,
        0,
        host_info);

    gathered.ParallelCopy(
        checkpoint_raster,
        ERFFireCheckpointV4RasterComponents::spatial_fuel,
        0,
        FireFuelRasterComponents::component_count,
        0,
        0);
    amrex::Gpu::streamSynchronize();

    FireFuelRasterState state;
    int invalid = 0;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        state.cells.resize(geometry.nx * geometry.ny);
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
                    FireFuelRasterCell cell;
                    if (!try_decode_fire_fuel_raster_cell(
                            values,
                            i,
                            j,
                            cell)) {
                        invalid = 1;
                        continue;
                    }
                    state.cells[
                        fire_cartesian_raster_flat_index(
                            geometry,
                            static_cast<std::size_t>(i),
                            static_cast<std::size_t>(j))] =
                                cell;
                }
            }
        }
    }

    amrex::ParallelDescriptor::Bcast(
        &invalid,
        1,
        io_rank);
    if (invalid != 0) {
        throw std::invalid_argument(
            "ERF-Fire v4 checkpoint contains invalid spatial fuel encoding");
    }

    return FireFuelRaster::collective_from_io_rank_state(
        geometry,
        state);
}

} // namespace detail

/**
 * Decomposition-independent fingerprint of the resolved spatial material.
 *
 * The fingerprint is defined over raster geometry and canonical row-major
 * model/moisture values, not AMReX box ordering or rank ownership. The full
 * canonical material is gathered only at this explicit checkpoint boundary.
 */
[[nodiscard]] inline std::uint64_t
collective_fire_fuel_raster_fingerprint_fnv1a64(
    const FireFuelRaster& raster)
{
    const FireFuelRasterState state =
        raster.collective_snapshot_state_to_io_rank();

    unsigned long long packed = 0ULL;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        packed =
            static_cast<unsigned long long>(
                detail::fire_fuel_raster_state_fingerprint_fnv1a64(
                    raster.geometry(),
                    state));
    }

    amrex::ParallelDescriptor::Bcast(
        &packed,
        1,
        amrex::ParallelDescriptor::IOProcessorNumber());

    return static_cast<std::uint64_t>(packed);
}

/** Build v4 distributed state: legacy eight state components plus material. */
[[nodiscard]] inline amrex::MultiFab
make_erf_fire_checkpoint_v4_raster(
    const ERFFireSpreadRuntime& runtime)
{
    const FireFuelRaster* fuel =
        runtime.spatial_fuel_raster();
    if (fuel == nullptr) {
        throw std::invalid_argument(
            "ERF-Fire checkpoint v4 requires a spatial-fuel runtime");
    }

    amrex::MultiFab legacy =
        make_erf_fire_checkpoint_v2_raster(runtime);
    amrex::MultiFab checkpoint(
        legacy.boxArray(),
        legacy.DistributionMap(),
        ERFFireCheckpointV4RasterComponents::component_count,
        0);
    checkpoint.setVal(
        std::numeric_limits<amrex::Real>::quiet_NaN());

    checkpoint.ParallelCopy(
        legacy,
        0,
        0,
        ERFFireCheckpointRasterComponents::component_count,
        0,
        0);
    checkpoint.ParallelCopy(
        fuel->distributed_values(),
        0,
        ERFFireCheckpointV4RasterComponents::spatial_fuel,
        FireFuelRasterComponents::component_count,
        0,
        0);

    return checkpoint;
}

/**
 * Write v4 metadata while reusing the validated v3 base schema byte-for-byte.
 *
 * The caller computes the material fingerprint collectively before entering an
 * IO-rank-only metadata section. This function itself performs no collectives.
 */
inline void
write_erf_fire_checkpoint_v4_metadata(
    const ERFFireSpreadRuntime& runtime,
    const ERFFireRuntimeOptions& options,
    std::uint64_t spatial_fuel_fingerprint_fnv1a64,
    std::ostream& stream)
{
    if (!runtime.has_spatial_fuel()) {
        throw std::invalid_argument(
            "ERF-Fire checkpoint v4 requires a spatial-fuel runtime");
    }
    if (!stream.good()) {
        throw std::runtime_error(
            "ERF-Fire checkpoint v4 output stream is not writable");
    }

    std::ostringstream legacy_stream;
    write_erf_fire_checkpoint_v3_metadata(
        runtime,
        options,
        legacy_stream);
    const std::string legacy = legacy_stream.str();

    const std::string prefix =
        "ERF_FIRE_RUNTIME_STATE 3\n";
    const std::string suffix =
        "raster_components 8\n"
        "END_ERF_FIRE_RUNTIME_STATE\n";

    if (legacy.size() < prefix.size() + suffix.size()
        || legacy.compare(0, prefix.size(), prefix) != 0
        || legacy.compare(
               legacy.size() - suffix.size(),
               suffix.size(),
               suffix) != 0) {
        throw std::logic_error(
            "ERF-Fire v3 metadata layout changed underneath v4 adapter");
    }

    stream << "ERF_FIRE_RUNTIME_STATE 4\n";
    stream.write(
        legacy.data() + prefix.size(),
        static_cast<std::streamsize>(
            legacy.size() - prefix.size() - suffix.size()));
    stream
        << "spatial_fuel_provenance resolved_raster\n"
        << "spatial_fuel_schema_version "
        << detail::fire_spatial_fuel_checkpoint_schema_version
        << "\n"
        << "spatial_fuel_fingerprint_fnv1a64 "
        << spatial_fuel_fingerprint_fnv1a64
        << "\n"
        << "raster_components "
        << ERFFireCheckpointV4RasterComponents::component_count
        << "\n"
        << "END_ERF_FIRE_RUNTIME_STATE\n";

    if (!stream.good()) {
        throw std::runtime_error(
            "failed while writing ERF-Fire checkpoint v4 metadata");
    }
}

[[nodiscard]] inline ERFFireCheckpointV4Metadata
read_erf_fire_checkpoint_v4_metadata(
    std::istream& stream)
{
    const std::string text{
        std::istreambuf_iterator<char>(stream),
        std::istreambuf_iterator<char>()};
    const std::string prefix =
        "ERF_FIRE_RUNTIME_STATE 4\n";
    const std::string marker =
        "spatial_fuel_provenance ";

    if (text.size() < prefix.size()
        || text.compare(0, prefix.size(), prefix) != 0) {
        throw std::runtime_error(
            "ERF-Fire v4 metadata reader received another format");
    }

    const std::size_t marker_position =
        text.rfind(marker);
    if (marker_position == std::string::npos
        || marker_position < prefix.size()) {
        throw std::runtime_error(
            "ERF-Fire v4 metadata is missing spatial fuel provenance");
    }

    std::istringstream spatial_tail(
        text.substr(marker_position));
    std::string token;
    std::string provenance;
    int schema_version = 0;
    unsigned long long fingerprint = 0ULL;
    std::size_t raster_components = 0;

    if (!(spatial_tail >> token)
        || token != "spatial_fuel_provenance"
        || !(spatial_tail >> provenance)
        || provenance != "resolved_raster"
        || !(spatial_tail >> token)
        || token != "spatial_fuel_schema_version"
        || !(spatial_tail >> schema_version)
        || !(spatial_tail >> token)
        || token != "spatial_fuel_fingerprint_fnv1a64"
        || !(spatial_tail >> fingerprint)
        || !(spatial_tail >> token)
        || token != "raster_components"
        || !(spatial_tail >> raster_components)
        || !(spatial_tail >> token)
        || token != "END_ERF_FIRE_RUNTIME_STATE") {
        throw std::runtime_error(
            "invalid ERF-Fire v4 spatial fuel metadata");
    }

    std::string trailing;
    if (spatial_tail >> trailing) {
        throw std::runtime_error(
            "ERF-Fire v4 metadata contains trailing data");
    }
    if (schema_version
        != detail::fire_spatial_fuel_checkpoint_schema_version) {
        throw std::runtime_error(
            "unsupported ERF-Fire spatial fuel checkpoint schema version");
    }
    if (raster_components
        != static_cast<std::size_t>(
            ERFFireCheckpointV4RasterComponents::component_count)) {
        throw std::runtime_error(
            "ERF-Fire v4 checkpoint raster component count mismatch");
    }

    std::ostringstream legacy_text;
    legacy_text << "ERF_FIRE_RUNTIME_STATE 3\n";
    legacy_text.write(
        text.data() + prefix.size(),
        static_cast<std::streamsize>(
            marker_position - prefix.size()));
    legacy_text
        << "raster_components "
        << ERFFireCheckpointRasterComponents::component_count
        << "\n"
        << "END_ERF_FIRE_RUNTIME_STATE\n";

    std::istringstream legacy_stream(
        legacy_text.str());
    ERFFireCheckpointV3Metadata legacy =
        read_erf_fire_checkpoint_v3_metadata(
            legacy_stream);

    ERFFireCheckpointV4Metadata result;
    result.checkpoint =
        std::move(legacy.checkpoint);
    result.spatial_fuel_schema_version =
        schema_version;
    result.spatial_fuel_fingerprint_fnv1a64 =
        static_cast<std::uint64_t>(fingerprint);
    return result;
}

/**
 * Restore state and exact resolved material from one v4 distributed raster.
 * The material fingerprint and cellwise combustion invariants are checked
 * before the restored runtime is exposed.
 */
[[nodiscard]] inline ERFFireSpreadRuntime
collective_restore_erf_fire_checkpoint_v4(
    ERFFireCheckpointV4Metadata metadata,
    const amrex::MultiFab& checkpoint_raster)
{
    if (metadata.spatial_fuel_schema_version
        != detail::fire_spatial_fuel_checkpoint_schema_version) {
        throw std::invalid_argument(
            "ERF-Fire v4 spatial fuel schema is unsupported");
    }
    if (checkpoint_raster.nComp()
            != ERFFireCheckpointV4RasterComponents::component_count
        || checkpoint_raster.nGrow() != 0) {
        throw std::invalid_argument(
            "ERF-Fire v4 checkpoint raster has incompatible components or ghosts");
    }

    const auto geometry =
        metadata.checkpoint.runtime_state.config.raster_geometry;
    FireFuelRaster spatial_fuel =
        detail::collective_restore_fire_fuel_raster_from_v4(
            geometry,
            checkpoint_raster);

    const std::uint64_t actual_fingerprint =
        collective_fire_fuel_raster_fingerprint_fnv1a64(
            spatial_fuel);
    if (actual_fingerprint
        != metadata.spatial_fuel_fingerprint_fnv1a64) {
        throw std::invalid_argument(
            "ERF-Fire v4 spatial fuel fingerprint does not match checkpoint metadata");
    }

    amrex::MultiFab legacy_raster(
        checkpoint_raster.boxArray(),
        checkpoint_raster.DistributionMap(),
        ERFFireCheckpointRasterComponents::component_count,
        0);
    legacy_raster.ParallelCopy(
        checkpoint_raster,
        0,
        0,
        ERFFireCheckpointRasterComponents::component_count,
        0,
        0);
    amrex::Gpu::streamSynchronize();

    // Spatial combustion stores cumulative water using each cell's local
    // moisture, while the legacy v2/v3 restore validator expects the base
    // scalar moisture. First validate the checkpoint exactly against its
    // resolved spatial material on the active FireSurfaceLayout.
    const amrex::MultiFab& fuel_values =
        spatial_fuel.distributed_values();
    amrex::MultiFab spatial_combustion_state(
        fuel_values.boxArray(),
        fuel_values.DistributionMap(),
        FireCombustionRaster::component_count,
        0);
    spatial_combustion_state.ParallelCopy(
        legacy_raster,
        ERFFireCheckpointRasterComponents::ignited_area_fraction,
        0,
        FireCombustionRaster::component_count,
        0,
        0);
    amrex::Gpu::streamSynchronize();

    detail::validate_spatial_fuel_impl(
        geometry,
        metadata.checkpoint.runtime_state.config
            .combustion_parameters,
        spatial_combustion_state,
        spatial_fuel,
        true);

    // The raw combustion components have already been validated against the
    // fingerprint-checked spatial material above. Preserve them byte-for-byte
    // through dynamic restore and defer uniform-base validation.
    ERFFireSpreadRuntime restored =
        ERFFireSpreadRuntime::
            collective_restore_from_checkpoint_raster(
                std::move(
                    metadata.checkpoint.runtime_state),
                legacy_raster,
                true);

    return ERFFireSpreadRuntime::
        attach_spatial_fuel_after_checkpoint_restore(
            std::move(restored),
            std::move(spatial_fuel));
}

} // namespace ERFFire

#endif // ERF_FIRE_CHECKPOINT_V4_H_
