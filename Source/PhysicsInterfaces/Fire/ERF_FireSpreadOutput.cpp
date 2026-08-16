#include "ERF_FireSpreadOutput.H"

#include <ERF_FireSpreadRuntime.H>
#include <ERF_FireSurfaceLayout.H>

#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace ERFFire
{
namespace
{

constexpr int fire_checkpoint_max_grid_size = 64;
constexpr std::size_t fire_output_chunk_max_cells = 4096;

std::string
indexed_name(const char* stem, int step_index)
{
    std::ostringstream name;
    name << stem
         << "_"
         << std::setw(6)
         << std::setfill('0')
         << step_index
         << ".csv";
    return name.str();
}

std::ofstream
open_output(const std::filesystem::path& path)
{
    std::ofstream stream(path, std::ios::out | std::ios::trunc);
    if (!stream.is_open()) {
        throw std::runtime_error(
            "unable to open ERF-Fire visualization output " + path.string());
    }
    stream << std::setprecision(17);
    return stream;
}

void
expect_token(std::istream& stream, const char* expected)
{
    std::string token;
    if (!(stream >> token) || token != expected) {
        throw std::runtime_error(
            std::string("invalid ERF-Fire checkpoint field; expected ")
            + expected);
    }
}

std::size_t
read_size(std::istream& stream, const char* field)
{
    long long value = -1;
    if (!(stream >> value)
        || value < 0
        || static_cast<unsigned long long>(value)
            > static_cast<unsigned long long>(
                std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error(
            std::string("invalid ERF-Fire checkpoint size for ")
            + field);
    }
    return static_cast<std::size_t>(value);
}

bool
read_bool(std::istream& stream, const char* field)
{
    int value = -1;
    if (!(stream >> value) || (value != 0 && value != 1)) {
        throw std::runtime_error(
            std::string("invalid ERF-Fire checkpoint boolean for ")
            + field);
    }
    return value != 0;
}

ERFFireCouplingMode
read_coupling_mode(std::istream& stream)
{
    std::string token;
    if (!(stream >> token)) {
        throw std::runtime_error(
            "missing ERF-Fire checkpoint coupling mode");
    }
    if (token == "one_way") {
        return ERFFireCouplingMode::OneWay;
    }
    if (token == "two_way") {
        return ERFFireCouplingMode::TwoWay;
    }
    throw std::runtime_error(
        "invalid ERF-Fire checkpoint coupling mode");
}

ERFFireWindMode
read_wind_mode(std::istream& stream)
{
    std::string token;
    if (!(stream >> token)) {
        throw std::runtime_error(
            "missing ERF-Fire checkpoint wind mode");
    }
    if (token == "direct_reference") {
        return ERFFireWindMode::DirectReference;
    }
    if (token == "explicit_waf_20ft") {
        return ERFFireWindMode::ExplicitWaf20ft;
    }
    throw std::runtime_error(
        "invalid ERF-Fire checkpoint wind mode");
}

const char*
coupling_mode_token(ERFFireCouplingMode mode)
{
    if (mode == ERFFireCouplingMode::OneWay) {
        return "one_way";
    }
    if (mode == ERFFireCouplingMode::TwoWay) {
        return "two_way";
    }
    throw std::invalid_argument(
        "unsupported ERF-Fire checkpoint coupling mode");
}

const char*
wind_mode_token(ERFFireWindMode mode)
{
    if (mode == ERFFireWindMode::DirectReference) {
        return "direct_reference";
    }
    if (mode == ERFFireWindMode::ExplicitWaf20ft) {
        return "explicit_waf_20ft";
    }
    throw std::invalid_argument(
        "unsupported ERF-Fire checkpoint wind mode");
}

void
require_stream_read(bool condition, const char* field)
{
    if (!condition) {
        throw std::runtime_error(
            std::string("invalid ERF-Fire checkpoint data for ")
            + field);
    }
}

} // namespace

void
write_erf_fire_spread_snapshot(
    const ERFFireSpreadRuntime& runtime,
    const std::string& output_dir,
    int step_index)
{
    if (output_dir.empty()) {
        throw std::invalid_argument(
            "fire output directory must not be empty");
    }
    if (step_index < 0) {
        throw std::invalid_argument(
            "fire output step index must be nonnegative");
    }

    const auto& geometry =
        runtime.config().raster_geometry;
    (void)detail::validate_fire_cartesian_raster_geometry(
        geometry);

    const amrex::Real current_time_s =
        runtime.current_time_s();
    const auto& vertices =
        runtime.perimeter().vertices_m();

    const std::filesystem::path directory(output_dir);
    const std::string perimeter_name =
        indexed_name("perimeter", step_index);
    const std::string raster_name =
        indexed_name("raster", step_index);

    const int io_rank =
        amrex::ParallelDescriptor::IOProcessorNumber();

    std::ofstream raster_stream;
    int io_failed = 0;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        try {
            std::filesystem::create_directories(directory);

            {
                auto perimeter_stream =
                    open_output(directory / perimeter_name);
                perimeter_stream
                    << "time_s,vertex_index,x_m,y_m\n";
                for (std::size_t index = 0;
                     index < vertices.size();
                     ++index) {
                    perimeter_stream
                        << current_time_s << ","
                        << index << ","
                        << vertices[index].x << ","
                        << vertices[index].y << "\n";
                }
                perimeter_stream.close();
                if (perimeter_stream.fail()) {
                    throw std::runtime_error(
                        "failed while writing ERF-Fire perimeter visualization");
                }
            }

            raster_stream =
                open_output(directory / raster_name);
            raster_stream
                << "time_s,i,j,xlo_m,xhi_m,ylo_m,yhi_m,"
                << "burned_fraction,has_arrived,first_arrival_time_s,"
                << "ignited_area_fraction,remaining_dry_fuel_kg_m2,"
                << "consumed_dry_fuel_kg_m2,sensible_energy_j_m2,"
                << "water_released_kg_m2\n";
            if (!raster_stream.good()) {
                throw std::runtime_error(
                    "failed while writing ERF-Fire raster visualization header");
            }
        } catch (...) {
            io_failed = 1;
        }
    }

    amrex::ParallelDescriptor::Bcast(
        &io_failed, 1, io_rank);
    if (io_failed != 0) {
        throw std::runtime_error(
            "IO rank failed to initialize ERF-Fire visualization output");
    }

    // Reuse the distributed eight-component raster assembly used by the
    // native checkpoint path. This is O(Nxy/Nranks) on each rank and avoids
    // materializing canonical full-domain vectors on the IO rank.
    amrex::MultiFab distributed_raster =
        make_erf_fire_checkpoint_v2_raster(runtime);

    amrex::Real burned_area_m2 = amrex::Real(0.0);
    std::size_t arrived_cell_count = 0;
    FireCombustionRasterTotals combustion_totals{};
    const amrex::Real combustion_cell_area_m2 =
        geometry.dx_m * geometry.dy_m;

    const auto stream_chunk =
        [&](std::size_t ilo,
            std::size_t ihi,
            std::size_t jlo,
            std::size_t jhi) {
            const amrex::Box chunk_box(
                amrex::IntVect(
                    static_cast<int>(ilo),
                    static_cast<int>(jlo),
                    0),
                amrex::IntVect(
                    static_cast<int>(ihi - 1),
                    static_cast<int>(jhi - 1),
                    0));
            amrex::BoxArray chunk_boxes{chunk_box};
            amrex::Vector<int> processor_map(1, io_rank);
            const amrex::DistributionMapping chunk_dm(
                std::move(processor_map));
            amrex::MultiFab io_chunk(
                chunk_boxes,
                chunk_dm,
                ERFFireCheckpointRasterComponents::component_count,
                0);

            io_chunk.ParallelCopy(
                distributed_raster,
                0,
                0,
                ERFFireCheckpointRasterComponents::component_count,
                0,
                0);

            int chunk_io_failed = 0;
            if (amrex::ParallelDescriptor::IOProcessor()) {
                try {
                    for (amrex::MFIter mfi(io_chunk);
                         mfi.isValid();
                         ++mfi) {
                        const auto values =
                            io_chunk.const_array(mfi);

                        for (std::size_t j = jlo;
                             j < jhi;
                             ++j) {
                            for (std::size_t i = ilo;
                                 i < ihi;
                                 ++i) {
                                const int ii =
                                    static_cast<int>(i);
                                const int jj =
                                    static_cast<int>(j);
                                const auto cell =
                                    detail::fire_cartesian_raster_cell_bounds(
                                        geometry,
                                        i,
                                        j);

                                const amrex::Real burned_fraction =
                                    values(
                                        ii,
                                        jj,
                                        0,
                                        ERFFireCheckpointRasterComponents::
                                            burned_fraction);
                                const bool arrived =
                                    values(
                                        ii,
                                        jj,
                                        0,
                                        ERFFireCheckpointRasterComponents::
                                            arrived)
                                    != amrex::Real(0.0);

                                raster_stream
                                    << current_time_s << ","
                                    << i << ","
                                    << j << ","
                                    << cell.xlo_m << ","
                                    << cell.xhi_m << ","
                                    << cell.ylo_m << ","
                                    << cell.yhi_m << ","
                                    << burned_fraction << ","
                                    << (arrived ? 1 : 0) << ",";

                                if (arrived) {
                                    raster_stream
                                        << values(
                                            ii,
                                            jj,
                                            0,
                                            ERFFireCheckpointRasterComponents::
                                                first_arrival_time_s);
                                }

                                const amrex::Real ignited_area_fraction =
                                    values(
                                        ii,
                                        jj,
                                        0,
                                        ERFFireCheckpointRasterComponents::
                                            ignited_area_fraction);
                                const amrex::Real remaining_dry_fuel_kg_m2 =
                                    values(
                                        ii,
                                        jj,
                                        0,
                                        ERFFireCheckpointRasterComponents::
                                            remaining_dry_fuel_kg_m2);
                                const amrex::Real consumed_dry_fuel_kg_m2 =
                                    values(
                                        ii,
                                        jj,
                                        0,
                                        ERFFireCheckpointRasterComponents::
                                            consumed_dry_fuel_kg_m2);
                                const amrex::Real sensible_energy_j_m2 =
                                    values(
                                        ii,
                                        jj,
                                        0,
                                        ERFFireCheckpointRasterComponents::
                                            sensible_energy_j_m2);
                                const amrex::Real water_released_kg_m2 =
                                    values(
                                        ii,
                                        jj,
                                        0,
                                        ERFFireCheckpointRasterComponents::
                                            water_released_kg_m2);

                                raster_stream
                                    << ","
                                    << ignited_area_fraction << ","
                                    << remaining_dry_fuel_kg_m2 << ","
                                    << consumed_dry_fuel_kg_m2 << ","
                                    << sensible_energy_j_m2 << ","
                                    << water_released_kg_m2
                                    << "\n";

                                burned_area_m2 +=
                                    burned_fraction
                                    * detail::fire_cartesian_cell_area_m2(
                                        cell);
                                if (arrived) {
                                    ++arrived_cell_count;
                                }
                                combustion_totals
                                    .remaining_dry_fuel_kg +=
                                        remaining_dry_fuel_kg_m2
                                        * combustion_cell_area_m2;
                                combustion_totals
                                    .consumed_dry_fuel_kg +=
                                        consumed_dry_fuel_kg_m2
                                        * combustion_cell_area_m2;
                                combustion_totals
                                    .sensible_energy_j +=
                                        sensible_energy_j_m2
                                        * combustion_cell_area_m2;
                                combustion_totals
                                    .water_released_kg +=
                                        water_released_kg_m2
                                        * combustion_cell_area_m2;
                            }
                        }
                    }

                    if (!raster_stream.good()) {
                        chunk_io_failed = 1;
                    }
                } catch (...) {
                    chunk_io_failed = 1;
                }
            }

            amrex::ParallelDescriptor::Bcast(
                &chunk_io_failed, 1, io_rank);
            if (chunk_io_failed != 0) {
                throw std::runtime_error(
                    "IO rank failed while streaming ERF-Fire raster visualization");
            }
        };

    if (geometry.nx <= fire_output_chunk_max_cells) {
        std::size_t rows_per_chunk =
            fire_output_chunk_max_cells / geometry.nx;
        if (rows_per_chunk == 0) {
            rows_per_chunk = 1;
        }

        for (std::size_t jlo = 0;
             jlo < geometry.ny;) {
            std::size_t jhi =
                jlo + rows_per_chunk;
            if (jhi > geometry.ny) {
                jhi = geometry.ny;
            }
            stream_chunk(
                0,
                geometry.nx,
                jlo,
                jhi);
            jlo = jhi;
        }
    } else {
        for (std::size_t j = 0;
             j < geometry.ny;
             ++j) {
            for (std::size_t ilo = 0;
                 ilo < geometry.nx;) {
                std::size_t ihi =
                    ilo + fire_output_chunk_max_cells;
                if (ihi > geometry.nx) {
                    ihi = geometry.nx;
                }
                stream_chunk(
                    ilo,
                    ihi,
                    j,
                    j + 1);
                ilo = ihi;
            }
        }
    }

    int raster_close_failed = 0;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        raster_stream.close();
        if (raster_stream.fail()) {
            raster_close_failed = 1;
        }
    }
    amrex::ParallelDescriptor::Bcast(
        &raster_close_failed, 1, io_rank);
    if (raster_close_failed != 0) {
        throw std::runtime_error(
            "IO rank failed while closing ERF-Fire raster visualization");
    }

    int summary_io_failed = 0;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        try {
            const std::filesystem::path summary_path =
                directory / "summary.csv";
            std::ofstream summary;
            const bool write_summary_header =
                step_index == 0
                || !std::filesystem::exists(summary_path);
            if (step_index == 0) {
                summary.open(
                    summary_path,
                    std::ios::out | std::ios::trunc);
            } else {
                summary.open(
                    summary_path,
                    std::ios::out | std::ios::app);
            }

            if (!summary.is_open()) {
                throw std::runtime_error(
                    "unable to open ERF-Fire visualization manifest "
                    + summary_path.string());
            }

            summary << std::setprecision(17);
            if (write_summary_header) {
                summary
                    << "step,time_s,vertex_count,burned_area_m2,"
                    << "arrived_cell_count,remaining_dry_fuel_kg,"
                    << "consumed_dry_fuel_kg,sensible_energy_j,"
                    << "water_released_kg,perimeter_file,raster_file\n";
            }

            summary
                << step_index << ","
                << current_time_s << ","
                << vertices.size() << ","
                << burned_area_m2 << ","
                << arrived_cell_count << ","
                << combustion_totals.remaining_dry_fuel_kg << ","
                << combustion_totals.consumed_dry_fuel_kg << ","
                << combustion_totals.sensible_energy_j << ","
                << combustion_totals.water_released_kg << ","
                << perimeter_name << ","
                << raster_name << "\n";

            summary.close();
            if (summary.fail()) {
                throw std::runtime_error(
                    "failed while writing ERF-Fire visualization manifest");
            }
        } catch (...) {
            summary_io_failed = 1;
        }
    }

    amrex::ParallelDescriptor::Bcast(
        &summary_io_failed, 1, io_rank);
    if (summary_io_failed != 0) {
        throw std::runtime_error(
            "IO rank failed while writing ERF-Fire visualization manifest");
    }
}

int
read_erf_fire_checkpoint_version(std::istream& stream)
{
    expect_token(stream, "ERF_FIRE_RUNTIME_STATE");
    int version = 0;
    require_stream_read(
        static_cast<bool>(stream >> version),
        "format version");
    return version;
}

amrex::MultiFab
make_erf_fire_checkpoint_v2_raster(
    const ERFFireSpreadRuntime& runtime)
{
    const auto& geometry =
        runtime.config().raster_geometry;
    const FireSurfaceLayout domain_layout(geometry);

    amrex::BoxArray checkpoint_boxes{
        domain_layout.cell_domain()};
    checkpoint_boxes.maxSize(
        amrex::IntVect(
            fire_checkpoint_max_grid_size,
            fire_checkpoint_max_grid_size,
            1));
    const amrex::DistributionMapping checkpoint_dm{
        checkpoint_boxes,
        amrex::ParallelDescriptor::NProcs()};

    amrex::MultiFab checkpoint_raster(
        checkpoint_boxes,
        checkpoint_dm,
        ERFFireCheckpointRasterComponents::component_count,
        0);
    checkpoint_raster.setVal(
        std::numeric_limits<amrex::Real>::quiet_NaN());

    checkpoint_raster.ParallelCopy(
        runtime.burned_fraction_raster()
            .distributed_burned_fraction(),
        0,
        ERFFireCheckpointRasterComponents::burned_fraction,
        1,
        0,
        0);

    const auto& arrival =
        runtime.first_arrival_raster();
    const auto& arrived_mask =
        arrival.distributed_arrived();
    amrex::MultiFab arrived_real(
        arrived_mask.boxArray(),
        arrived_mask.DistributionMap(),
        1,
        0);

    int invalid_mask = 0;
    for (amrex::MFIter mfi(arrived_mask);
         mfi.isValid();
         ++mfi) {
        const amrex::Box& box = mfi.validbox();
        const auto source = arrived_mask.const_array(mfi);
        const auto destination = arrived_real.array(mfi);
        for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
            for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                const int value = source(i, j, 0);
                if (value != 0 && value != 1) {
                    invalid_mask = 1;
                }
                destination(i, j, 0) =
                    static_cast<amrex::Real>(value);
            }
        }
    }
    amrex::ParallelDescriptor::ReduceIntMax(invalid_mask);
    if (invalid_mask != 0) {
        throw std::logic_error(
            "distributed Fire first-arrival mask is not 0 or 1");
    }

    checkpoint_raster.ParallelCopy(
        arrived_real,
        0,
        ERFFireCheckpointRasterComponents::arrived,
        1,
        0,
        0);
    checkpoint_raster.ParallelCopy(
        arrival.distributed_first_arrival_time_s(),
        0,
        ERFFireCheckpointRasterComponents::first_arrival_time_s,
        1,
        0,
        0);
    checkpoint_raster.ParallelCopy(
        runtime.combustion_raster().distributed_states(),
        0,
        ERFFireCheckpointRasterComponents::ignited_area_fraction,
        FireCombustionRaster::component_count,
        0,
        0);

    return checkpoint_raster;
}

void
write_erf_fire_checkpoint_v2_metadata(
    const ERFFireSpreadRuntime& runtime,
    const ERFFireRuntimeOptions& options,
    std::ostream& stream)
{
    if (!stream.good()) {
        throw std::runtime_error(
            "ERF-Fire checkpoint output stream is not writable");
    }
    if (!options.enabled) {
        throw std::invalid_argument(
            "cannot checkpoint disabled ERF-Fire runtime");
    }

    const auto& config = runtime.config();
    const auto& fuel = config.fuel;
    const auto& combustion_parameters =
        config.combustion_parameters;
    const auto& geometry = config.raster_geometry;
    const auto& vertices =
        runtime.perimeter().vertices_m();
    const auto& arrival =
        runtime.first_arrival_raster();
    const auto& combustion =
        runtime.combustion_raster();

    stream
        << std::setprecision(
            std::numeric_limits<amrex::Real>::max_digits10);

    stream << "ERF_FIRE_RUNTIME_STATE 2\n";
    stream
        << "coupling_mode "
        << coupling_mode_token(options.coupling_mode)
        << "\n";
    stream
        << "wind_mode "
        << wind_mode_token(options.wind_mode)
        << "\n";
    stream
        << "reference_height_agl_m "
        << options.reference_height_agl_m
        << "\n";
    stream
        << "wind_adjustment_factor "
        << options.wind_adjustment_factor
        << "\n";
    stream
        << "feedback_extinction_depth_m "
        << options.feedback_extinction_depth_m
        << "\n";

    stream
        << "fuel "
        << fuel.dead_1h_load_kg_m2 << " "
        << fuel.dead_1h_sav_m_inv << " "
        << fuel.fuel_bed_depth_m << " "
        << fuel.dead_heat_content_j_kg << " "
        << fuel.particle_density_kg_m3 << " "
        << fuel.total_mineral_fraction << " "
        << fuel.effective_mineral_fraction << " "
        << fuel.dead_moisture_of_extinction
        << "\n";
    stream
        << "dead_fuel_moisture_fraction "
        << config.dead_fuel_moisture_fraction
        << "\n";
    stream
        << "combustion_parameters "
        << combustion_parameters.dry_fuel_load_kg_m2 << " "
        << combustion_parameters.sensible_heat_release_j_kg_dry << " "
        << combustion_parameters.fuel_moisture_fraction << " "
        << combustion_parameters.burn_time_constant_s << " "
        << combustion_parameters
               .combustion_water_yield_kg_per_kg_dry
        << "\n";
    stream
        << "combustion_temporal_substeps "
        << config.combustion_options.temporal_substeps
        << "\n";
    stream
        << "remesh_options "
        << config.remesh_options.min_edge_length_m << " "
        << config.remesh_options.max_edge_length_m << " "
        << config.remesh_options.max_chord_error_m
        << "\n";
    stream
        << "raster_geometry "
        << geometry.nx << " "
        << geometry.ny << " "
        << geometry.xlo_m << " "
        << geometry.ylo_m << " "
        << geometry.dx_m << " "
        << geometry.dy_m
        << "\n";
    stream
        << "arrival_time_tolerance_s "
        << config.arrival_time_tolerance_s
        << "\n";
    stream
        << "current_time_s "
        << runtime.current_time_s()
        << "\n";

    stream
        << "perimeter "
        << vertices.size()
        << "\n";
    for (const FireVec2& vertex : vertices) {
        stream << vertex.x << " " << vertex.y << "\n";
    }

    stream
        << "first_arrival_metadata "
        << (arrival.has_initial_condition() ? 1 : 0)
        << " "
        << arrival.initial_condition_time_s()
        << " "
        << (arrival.has_committed_sweep() ? 1 : 0)
        << " "
        << arrival.last_sweep_end_time_s()
        << "\n";
    stream
        << "combustion_metadata "
        << (combustion.initialized() ? 1 : 0)
        << "\n";
    stream
        << "raster_components "
        << ERFFireCheckpointRasterComponents::component_count
        << "\n";
    stream << "END_ERF_FIRE_RUNTIME_STATE\n";

    if (!stream.good()) {
        throw std::runtime_error(
            "failed while writing ERF-Fire checkpoint state");
    }
}

ERFFireCheckpointV2Metadata
read_erf_fire_checkpoint_v2_metadata(std::istream& stream)
{
    ERFFireCheckpointV2Metadata metadata;
    auto& checkpoint = metadata.checkpoint;

    expect_token(stream, "ERF_FIRE_RUNTIME_STATE");
    int version = 0;
    require_stream_read(
        static_cast<bool>(stream >> version),
        "format version");
    if (version != 2) {
        throw std::runtime_error(
            "ERF-Fire version-2 metadata reader received another format");
    }

    expect_token(stream, "coupling_mode");
    checkpoint.coupling_mode =
        read_coupling_mode(stream);

    expect_token(stream, "wind_mode");
    checkpoint.wind_mode =
        read_wind_mode(stream);

    expect_token(stream, "reference_height_agl_m");
    require_stream_read(
        static_cast<bool>(
            stream >> checkpoint.reference_height_agl_m),
        "reference_height_agl_m");

    expect_token(stream, "wind_adjustment_factor");
    require_stream_read(
        static_cast<bool>(
            stream >> checkpoint.wind_adjustment_factor),
        "wind_adjustment_factor");

    expect_token(stream, "feedback_extinction_depth_m");
    require_stream_read(
        static_cast<bool>(
            stream >> checkpoint.feedback_extinction_depth_m),
        "feedback_extinction_depth_m");

    auto& state = checkpoint.runtime_state;
    auto& config = state.config;
    auto& fuel = config.fuel;
    auto& combustion_parameters =
        config.combustion_parameters;
    auto& geometry = config.raster_geometry;

    expect_token(stream, "fuel");
    require_stream_read(
        static_cast<bool>(
            stream
            >> fuel.dead_1h_load_kg_m2
            >> fuel.dead_1h_sav_m_inv
            >> fuel.fuel_bed_depth_m
            >> fuel.dead_heat_content_j_kg
            >> fuel.particle_density_kg_m3
            >> fuel.total_mineral_fraction
            >> fuel.effective_mineral_fraction
            >> fuel.dead_moisture_of_extinction),
        "fuel");

    expect_token(stream, "dead_fuel_moisture_fraction");
    require_stream_read(
        static_cast<bool>(
            stream >> config.dead_fuel_moisture_fraction),
        "dead_fuel_moisture_fraction");

    expect_token(stream, "combustion_parameters");
    require_stream_read(
        static_cast<bool>(
            stream
            >> combustion_parameters.dry_fuel_load_kg_m2
            >> combustion_parameters.sensible_heat_release_j_kg_dry
            >> combustion_parameters.fuel_moisture_fraction
            >> combustion_parameters.burn_time_constant_s
            >> combustion_parameters
                   .combustion_water_yield_kg_per_kg_dry),
        "combustion_parameters");

    expect_token(stream, "combustion_temporal_substeps");
    config.combustion_options.temporal_substeps =
        read_size(
            stream,
            "combustion_temporal_substeps");

    expect_token(stream, "remesh_options");
    require_stream_read(
        static_cast<bool>(
            stream
            >> config.remesh_options.min_edge_length_m
            >> config.remesh_options.max_edge_length_m
            >> config.remesh_options.max_chord_error_m),
        "remesh_options");

    expect_token(stream, "raster_geometry");
    geometry.nx = read_size(stream, "raster nx");
    geometry.ny = read_size(stream, "raster ny");
    require_stream_read(
        static_cast<bool>(
            stream
            >> geometry.xlo_m
            >> geometry.ylo_m
            >> geometry.dx_m
            >> geometry.dy_m),
        "raster_geometry");

    if (geometry.nx != 0
        && geometry.ny
            > std::numeric_limits<std::size_t>::max()
                / geometry.nx) {
        throw std::runtime_error(
            "ERF-Fire checkpoint raster cell count overflows");
    }

    expect_token(stream, "arrival_time_tolerance_s");
    require_stream_read(
        static_cast<bool>(
            stream >> config.arrival_time_tolerance_s),
        "arrival_time_tolerance_s");

    expect_token(stream, "current_time_s");
    require_stream_read(
        static_cast<bool>(
            stream >> state.current_time_s),
        "current_time_s");

    expect_token(stream, "perimeter");
    const std::size_t perimeter_count =
        read_size(stream, "perimeter");
    if (perimeter_count < 3) {
        throw std::runtime_error(
            "ERF-Fire checkpoint perimeter has fewer than three vertices");
    }
    state.perimeter_vertices_m.resize(perimeter_count);
    for (FireVec2& vertex : state.perimeter_vertices_m) {
        require_stream_read(
            static_cast<bool>(
                stream >> vertex.x >> vertex.y),
            "perimeter vertex");
    }

    expect_token(stream, "first_arrival_metadata");
    state.first_arrival.has_initial_condition =
        read_bool(
            stream,
            "first-arrival initial-condition flag");
    require_stream_read(
        static_cast<bool>(
            stream
            >> state.first_arrival.initial_condition_time_s),
        "first-arrival initial-condition time");
    state.first_arrival.has_committed_sweep =
        read_bool(
            stream,
            "first-arrival committed-sweep flag");
    require_stream_read(
        static_cast<bool>(
            stream
            >> state.first_arrival.last_sweep_end_time_s),
        "first-arrival last-sweep time");

    expect_token(stream, "combustion_metadata");
    state.combustion.initialized =
        read_bool(stream, "combustion initialized flag");

    expect_token(stream, "raster_components");
    const std::size_t raster_components =
        read_size(stream, "raster_components");
    if (raster_components
        != static_cast<std::size_t>(
            ERFFireCheckpointRasterComponents::component_count)) {
        throw std::runtime_error(
            "ERF-Fire checkpoint raster component count mismatch");
    }

    expect_token(stream, "END_ERF_FIRE_RUNTIME_STATE");

    std::string trailing_token;
    if (stream >> trailing_token) {
        throw std::runtime_error(
            "ERF-Fire checkpoint contains trailing data");
    }

    return metadata;
}

void
write_erf_fire_checkpoint_state(
    const ERFFireSpreadRuntime& runtime,
    const ERFFireRuntimeOptions& options,
    std::ostream& stream)
{
    write_erf_fire_checkpoint_state(
        runtime.snapshot_state(),
        options,
        stream);
}

void
write_erf_fire_checkpoint_state(
    const ERFFireSpreadRuntimeState& state,
    const ERFFireRuntimeOptions& options,
    std::ostream& stream)
{
    if (!stream.good()) {
        throw std::runtime_error(
            "ERF-Fire checkpoint output stream is not writable");
    }
    if (!options.enabled) {
        throw std::invalid_argument(
            "cannot checkpoint disabled ERF-Fire runtime");
    }

    const auto& config = state.config;
    const auto& fuel = config.fuel;
    const auto& combustion_parameters =
        config.combustion_parameters;
    const auto& geometry = config.raster_geometry;

    stream
        << std::setprecision(
            std::numeric_limits<amrex::Real>::max_digits10);

    stream << "ERF_FIRE_RUNTIME_STATE 1\n";
    stream
        << "coupling_mode "
        << coupling_mode_token(options.coupling_mode)
        << "\n";
    stream
        << "wind_mode "
        << wind_mode_token(options.wind_mode)
        << "\n";
    stream
        << "reference_height_agl_m "
        << options.reference_height_agl_m
        << "\n";
    stream
        << "wind_adjustment_factor "
        << options.wind_adjustment_factor
        << "\n";
    stream
        << "feedback_extinction_depth_m "
        << options.feedback_extinction_depth_m
        << "\n";

    stream
        << "fuel "
        << fuel.dead_1h_load_kg_m2 << " "
        << fuel.dead_1h_sav_m_inv << " "
        << fuel.fuel_bed_depth_m << " "
        << fuel.dead_heat_content_j_kg << " "
        << fuel.particle_density_kg_m3 << " "
        << fuel.total_mineral_fraction << " "
        << fuel.effective_mineral_fraction << " "
        << fuel.dead_moisture_of_extinction
        << "\n";
    stream
        << "dead_fuel_moisture_fraction "
        << config.dead_fuel_moisture_fraction
        << "\n";
    stream
        << "combustion_parameters "
        << combustion_parameters.dry_fuel_load_kg_m2 << " "
        << combustion_parameters.sensible_heat_release_j_kg_dry << " "
        << combustion_parameters.fuel_moisture_fraction << " "
        << combustion_parameters.burn_time_constant_s << " "
        << combustion_parameters
               .combustion_water_yield_kg_per_kg_dry
        << "\n";
    stream
        << "combustion_temporal_substeps "
        << config.combustion_options.temporal_substeps
        << "\n";
    stream
        << "remesh_options "
        << config.remesh_options.min_edge_length_m << " "
        << config.remesh_options.max_edge_length_m << " "
        << config.remesh_options.max_chord_error_m
        << "\n";
    stream
        << "raster_geometry "
        << geometry.nx << " "
        << geometry.ny << " "
        << geometry.xlo_m << " "
        << geometry.ylo_m << " "
        << geometry.dx_m << " "
        << geometry.dy_m
        << "\n";
    stream
        << "arrival_time_tolerance_s "
        << config.arrival_time_tolerance_s
        << "\n";
    stream
        << "current_time_s "
        << state.current_time_s
        << "\n";

    stream
        << "perimeter "
        << state.perimeter_vertices_m.size()
        << "\n";
    for (const FireVec2& vertex : state.perimeter_vertices_m) {
        stream << vertex.x << " " << vertex.y << "\n";
    }

    stream
        << "burned_fraction "
        << state.burned_fraction.burned_fraction.size()
        << "\n";
    for (const amrex::Real value
         : state.burned_fraction.burned_fraction) {
        stream << value << "\n";
    }

    stream
        << "first_arrival_metadata "
        << (state.first_arrival.has_initial_condition ? 1 : 0)
        << " "
        << state.first_arrival.initial_condition_time_s
        << " "
        << (state.first_arrival.has_committed_sweep ? 1 : 0)
        << " "
        << state.first_arrival.last_sweep_end_time_s
        << "\n";
    stream
        << "first_arrival "
        << state.first_arrival.arrived.size()
        << "\n";
    if (state.first_arrival.arrived.size()
        != state.first_arrival.first_arrival_time_s.size()) {
        throw std::logic_error(
            "ERF-Fire first-arrival checkpoint vectors have different sizes");
    }
    for (std::size_t index = 0;
         index < state.first_arrival.arrived.size();
         ++index) {
        stream
            << static_cast<unsigned int>(
                   state.first_arrival.arrived[index])
            << " "
            << state.first_arrival.first_arrival_time_s[index]
            << "\n";
    }

    stream
        << "combustion "
        << (state.combustion.initialized ? 1 : 0)
        << " "
        << state.combustion.cells.size()
        << "\n";
    for (const FireCombustionState& cell
         : state.combustion.cells) {
        stream
            << cell.ignited_area_fraction << " "
            << cell.remaining_dry_fuel_kg_m2 << " "
            << cell.consumed_dry_fuel_kg_m2 << " "
            << cell.sensible_energy_j_m2 << " "
            << cell.water_released_kg_m2
            << "\n";
    }

    stream << "END_ERF_FIRE_RUNTIME_STATE\n";

    if (!stream.good()) {
        throw std::runtime_error(
            "failed while writing ERF-Fire checkpoint state");
    }
}

ERFFireCheckpointState
read_erf_fire_checkpoint_state(std::istream& stream)
{
    ERFFireCheckpointState checkpoint;

    expect_token(stream, "ERF_FIRE_RUNTIME_STATE");
    int version = 0;
    require_stream_read(
        static_cast<bool>(stream >> version),
        "format version");
    if (version != 1) {
        throw std::runtime_error(
            "unsupported ERF-Fire checkpoint format version");
    }

    expect_token(stream, "coupling_mode");
    checkpoint.coupling_mode =
        read_coupling_mode(stream);

    expect_token(stream, "wind_mode");
    checkpoint.wind_mode =
        read_wind_mode(stream);

    expect_token(stream, "reference_height_agl_m");
    require_stream_read(
        static_cast<bool>(
            stream >> checkpoint.reference_height_agl_m),
        "reference_height_agl_m");

    expect_token(stream, "wind_adjustment_factor");
    require_stream_read(
        static_cast<bool>(
            stream >> checkpoint.wind_adjustment_factor),
        "wind_adjustment_factor");

    expect_token(stream, "feedback_extinction_depth_m");
    require_stream_read(
        static_cast<bool>(
            stream >> checkpoint.feedback_extinction_depth_m),
        "feedback_extinction_depth_m");

    auto& state = checkpoint.runtime_state;
    auto& config = state.config;
    auto& fuel = config.fuel;
    auto& combustion_parameters =
        config.combustion_parameters;
    auto& geometry = config.raster_geometry;

    expect_token(stream, "fuel");
    require_stream_read(
        static_cast<bool>(
            stream
            >> fuel.dead_1h_load_kg_m2
            >> fuel.dead_1h_sav_m_inv
            >> fuel.fuel_bed_depth_m
            >> fuel.dead_heat_content_j_kg
            >> fuel.particle_density_kg_m3
            >> fuel.total_mineral_fraction
            >> fuel.effective_mineral_fraction
            >> fuel.dead_moisture_of_extinction),
        "fuel");

    expect_token(stream, "dead_fuel_moisture_fraction");
    require_stream_read(
        static_cast<bool>(
            stream >> config.dead_fuel_moisture_fraction),
        "dead_fuel_moisture_fraction");

    expect_token(stream, "combustion_parameters");
    require_stream_read(
        static_cast<bool>(
            stream
            >> combustion_parameters.dry_fuel_load_kg_m2
            >> combustion_parameters.sensible_heat_release_j_kg_dry
            >> combustion_parameters.fuel_moisture_fraction
            >> combustion_parameters.burn_time_constant_s
            >> combustion_parameters
                   .combustion_water_yield_kg_per_kg_dry),
        "combustion_parameters");

    expect_token(stream, "combustion_temporal_substeps");
    config.combustion_options.temporal_substeps =
        read_size(
            stream,
            "combustion_temporal_substeps");

    expect_token(stream, "remesh_options");
    require_stream_read(
        static_cast<bool>(
            stream
            >> config.remesh_options.min_edge_length_m
            >> config.remesh_options.max_edge_length_m
            >> config.remesh_options.max_chord_error_m),
        "remesh_options");

    expect_token(stream, "raster_geometry");
    geometry.nx = read_size(stream, "raster nx");
    geometry.ny = read_size(stream, "raster ny");
    require_stream_read(
        static_cast<bool>(
            stream
            >> geometry.xlo_m
            >> geometry.ylo_m
            >> geometry.dx_m
            >> geometry.dy_m),
        "raster_geometry");

    if (geometry.nx != 0
        && geometry.ny
            > std::numeric_limits<std::size_t>::max()
                / geometry.nx) {
        throw std::runtime_error(
            "ERF-Fire checkpoint raster cell count overflows");
    }
    const std::size_t raster_cell_count =
        geometry.nx * geometry.ny;

    expect_token(stream, "arrival_time_tolerance_s");
    require_stream_read(
        static_cast<bool>(
            stream >> config.arrival_time_tolerance_s),
        "arrival_time_tolerance_s");

    expect_token(stream, "current_time_s");
    require_stream_read(
        static_cast<bool>(
            stream >> state.current_time_s),
        "current_time_s");

    expect_token(stream, "perimeter");
    const std::size_t perimeter_count =
        read_size(stream, "perimeter");
    if (perimeter_count < 3) {
        throw std::runtime_error(
            "ERF-Fire checkpoint perimeter has fewer than three vertices");
    }
    state.perimeter_vertices_m.resize(perimeter_count);
    for (FireVec2& vertex : state.perimeter_vertices_m) {
        require_stream_read(
            static_cast<bool>(
                stream >> vertex.x >> vertex.y),
            "perimeter vertex");
    }

    expect_token(stream, "burned_fraction");
    const std::size_t burned_count =
        read_size(stream, "burned_fraction");
    if (burned_count != raster_cell_count) {
        throw std::runtime_error(
            "ERF-Fire checkpoint burned-fraction cell count mismatch");
    }
    state.burned_fraction.burned_fraction.resize(
        burned_count);
    for (amrex::Real& value
         : state.burned_fraction.burned_fraction) {
        require_stream_read(
            static_cast<bool>(stream >> value),
            "burned fraction");
    }

    expect_token(stream, "first_arrival_metadata");
    state.first_arrival.has_initial_condition =
        read_bool(stream, "first-arrival initial-condition flag");
    require_stream_read(
        static_cast<bool>(
            stream
            >> state.first_arrival.initial_condition_time_s),
        "first-arrival initial-condition time");
    state.first_arrival.has_committed_sweep =
        read_bool(stream, "first-arrival committed-sweep flag");
    require_stream_read(
        static_cast<bool>(
            stream
            >> state.first_arrival.last_sweep_end_time_s),
        "first-arrival last-sweep time");

    expect_token(stream, "first_arrival");
    const std::size_t arrival_count =
        read_size(stream, "first_arrival");
    if (arrival_count != raster_cell_count) {
        throw std::runtime_error(
            "ERF-Fire checkpoint first-arrival cell count mismatch");
    }
    state.first_arrival.arrived.resize(arrival_count);
    state.first_arrival.first_arrival_time_s.resize(
        arrival_count);
    for (std::size_t index = 0;
         index < arrival_count;
         ++index) {
        unsigned int mask = 0;
        require_stream_read(
            static_cast<bool>(
                stream
                >> mask
                >> state.first_arrival
                       .first_arrival_time_s[index]),
            "first-arrival cell");
        if (mask > 1) {
            throw std::runtime_error(
                "ERF-Fire checkpoint first-arrival mask is not 0 or 1");
        }
        state.first_arrival.arrived[index] =
            static_cast<std::uint8_t>(mask);
    }

    expect_token(stream, "combustion");
    state.combustion.initialized =
        read_bool(stream, "combustion initialized flag");
    const std::size_t combustion_count =
        read_size(stream, "combustion");
    if (combustion_count != raster_cell_count) {
        throw std::runtime_error(
            "ERF-Fire checkpoint combustion cell count mismatch");
    }
    state.combustion.cells.resize(combustion_count);
    for (FireCombustionState& cell
         : state.combustion.cells) {
        require_stream_read(
            static_cast<bool>(
                stream
                >> cell.ignited_area_fraction
                >> cell.remaining_dry_fuel_kg_m2
                >> cell.consumed_dry_fuel_kg_m2
                >> cell.sensible_energy_j_m2
                >> cell.water_released_kg_m2),
            "combustion cell");
    }

    expect_token(stream, "END_ERF_FIRE_RUNTIME_STATE");

    std::string trailing_token;
    if (stream >> trailing_token) {
        throw std::runtime_error(
            "ERF-Fire checkpoint contains trailing data");
    }

    return checkpoint;
}

} // namespace ERFFire
