#include <ERF.H>
#include <ERF_FireCheckpoint.H>
#include <ERF_FireCheckpointV4.H>
#include <ERF_FireFlatEnvironmentSampler.H>
#include <ERF_FireFuelRaster.H>
#include <ERF_FireRuntimeOptions.H>
#include <ERF_FireSpreadRuntime.H>

#include <AMReX_Arena.H>
#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>
#include <AMReX_VisMF.H>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

using amrex::Real;
using ERFFire::ERFFireRuntimeOptions;
using ERFFire::ERFFireSpreadConfig;
using ERFFire::ERFFireSpreadRuntime;
using ERFFire::FireCartesianRasterGeometry2D;
using ERFFire::FireEnvironmentBatchFunction;
using ERFFire::FireEnvironmentSample;
using ERFFire::FireFuelModelId;
using ERFFire::FireFuelMoistureClass;
using ERFFire::FireFuelRaster;
using ERFFire::FireFuelRasterState;
using ERFFire::FirePerimeter;
using ERFFire::FireVec2;

constexpr Real spatial_v4_dry_moisture = Real(0.08);
constexpr Real spatial_v4_wet_moisture = Real(0.20);
constexpr Real spatial_v4_dt_s = Real(0.5);

void
require_spatial_v4(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

ERFFireSpreadConfig
spatial_v4_config()
{
    const FireCartesianRasterGeometry2D geometry{
        128,
        8,
        Real(0),
        Real(0),
        Real(1),
        Real(1)};

    return {
        ERFFire::make_fm1_fuel_parameters(),
        spatial_v4_dry_moisture,
        ERFFire::make_fm1_combustion_parameters(
            spatial_v4_dry_moisture),
        ERFFire::FireCombustionRasterOptions{8},
        ERFFire::FirePerimeterRemeshOptions{
            Real(0.05), Real(0.5), Real(0.002)},
        geometry,
        Real(1.0e-7)};
}

FirePerimeter
spatial_v4_ignition()
{
    constexpr int vertex_count = 32;
    const Real pi =
        Real(3.141592653589793238462643383279502884L);
    std::vector<FireVec2> vertices;
    vertices.reserve(vertex_count);

    for (int index = 0; index < vertex_count; ++index) {
        const Real angle =
            Real(2) * pi * Real(index) / Real(vertex_count);
        vertices.push_back({
            Real(64) + Real(1.25) * std::cos(angle),
            Real(4) + Real(1.25) * std::sin(angle)});
    }

    return FirePerimeter(std::move(vertices));
}

FireFuelRaster
spatial_v4_fuel(const ERFFireSpreadConfig& config)
{
    const auto& geometry = config.raster_geometry;
    FireFuelRasterState state;

    if (amrex::ParallelDescriptor::IOProcessor()) {
        state.cells.resize(geometry.nx * geometry.ny);
        for (std::size_t j = 0; j < geometry.ny; ++j) {
            for (std::size_t i = 0; i < geometry.nx; ++i) {
                auto& cell = state.cells[j * geometry.nx + i];
                cell.model_id = FireFuelModelId::FM1;
                cell.moisture.set(
                    FireFuelMoistureClass::Dead1h,
                    i < geometry.nx / 2
                        ? spatial_v4_dry_moisture
                        : spatial_v4_wet_moisture);
            }
        }
    }

    return FireFuelRaster::collective_from_io_rank_state(
        geometry,
        state);
}

FireEnvironmentBatchFunction
spatial_v4_environment()
{
    return [](const std::vector<FireVec2>& positions) {
        return std::vector<FireEnvironmentSample>(
            positions.size(),
            FireEnvironmentSample{
                FireVec2{Real(0.5), Real(0.125)},
                FireVec2{Real(0), Real(0)}});
    };
}

ERFFireRuntimeOptions
spatial_v4_options()
{
    ERFFireRuntimeOptions options;
    options.enabled = true;
    options.coupling_mode =
        ERFFire::ERFFireCouplingMode::OneWay;
    options.wind_mode =
        ERFFire::ERFFireWindMode::DirectReference;
    options.reference_height_agl_m = Real(0.5);
    options.dead_fuel_moisture_fraction =
        spatial_v4_dry_moisture;
    options.feedback_extinction_depth_m = Real(50);
    return options;
}

std::string
spatial_v4_raster_prefix(
    const std::filesystem::path& directory)
{
    return amrex::MultiFabFileFullPrefix(
        0,
        directory.string(),
        "Level_",
        "FireStateRaster");
}

void
prepare_spatial_v4_directory(
    const std::filesystem::path& directory)
{
    int io_failed = 0;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        try {
            std::filesystem::remove_all(directory);
            std::filesystem::create_directories(
                directory / "Level_0");
        } catch (...) {
            io_failed = 1;
        }
    }

    amrex::ParallelDescriptor::Bcast(
        &io_failed,
        1,
        amrex::ParallelDescriptor::IOProcessorNumber());
    if (io_failed != 0) {
        throw std::runtime_error(
            "unable to prepare spatial-v4 checkpoint directory");
    }
    amrex::ParallelDescriptor::Barrier();
}

void
write_spatial_v4_checkpoint(
    const ERFFireSpreadRuntime& runtime,
    const std::filesystem::path& directory)
{
    require_spatial_v4(
        runtime.has_spatial_fuel(),
        "spatial-v4 checkpoint writer requires spatial material");

    prepare_spatial_v4_directory(directory);

    const FireFuelRaster* fuel =
        runtime.spatial_fuel_raster();
    require_spatial_v4(
        fuel != nullptr,
        "spatial-v4 runtime lost its fuel raster");

    const std::uint64_t fingerprint =
        ERFFire::collective_fire_fuel_raster_fingerprint_fnv1a64(
            *fuel);
    amrex::MultiFab checkpoint =
        ERFFire::make_erf_fire_checkpoint_v4_raster(runtime);

    amrex::VisMF::Write(
        checkpoint,
        spatial_v4_raster_prefix(directory));

    int io_failed = 0;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        try {
            std::ofstream metadata(
                directory / "FireState",
                std::ios::out
                    | std::ios::trunc
                    | std::ios::binary);
            if (!metadata.good()) {
                throw std::runtime_error(
                    "unable to open spatial-v4 FireState");
            }

            ERFFire::write_erf_fire_checkpoint_v4_metadata(
                runtime,
                spatial_v4_options(),
                fingerprint,
                metadata);
            metadata.close();
            if (metadata.fail()) {
                throw std::runtime_error(
                    "failed while closing spatial-v4 FireState");
            }
        } catch (const std::exception& error) {
            amrex::Print()
                << "spatial-v4 metadata write error: "
                << error.what() << "\n";
            io_failed = 1;
        }
    }

    amrex::ParallelDescriptor::Bcast(
        &io_failed,
        1,
        amrex::ParallelDescriptor::IOProcessorNumber());
    if (io_failed != 0) {
        throw std::runtime_error(
            "IO rank failed to write spatial-v4 metadata");
    }
    amrex::ParallelDescriptor::Barrier();
}

ERFFireSpreadRuntime
read_spatial_v4_checkpoint(
    const std::filesystem::path& directory,
    Real expected_time_s)
{
    const ERFFireSpreadConfig expected_config =
        spatial_v4_config();

    ERFFire::ERFFireCheckpointV4Metadata metadata;
    metadata.checkpoint.runtime_state.config =
        expected_config;
    metadata.checkpoint.runtime_state.current_time_s =
        expected_time_s;

    int io_failed = 0;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        try {
            std::ifstream stream(
                directory / "FireState",
                std::ios::in | std::ios::binary);
            if (!stream.good()) {
                throw std::runtime_error(
                    "unable to open spatial-v4 FireState");
            }

            const int version =
                ERFFire::read_erf_fire_checkpoint_version(stream);
            if (version != 4) {
                throw std::runtime_error(
                    "spatial-v4 file has the wrong checkpoint version");
            }

            stream.clear();
            stream.seekg(0);
            metadata =
                ERFFire::read_erf_fire_checkpoint_v4_metadata(stream);

            if (!ERFFire::same_fire_spread_config(
                    metadata.checkpoint.runtime_state.config,
                    expected_config)) {
                throw std::runtime_error(
                    "spatial-v4 file has the wrong spread configuration");
            }
            if (metadata.checkpoint.runtime_state.current_time_s
                != expected_time_s) {
                throw std::runtime_error(
                    "spatial-v4 file has the wrong runtime clock");
            }
        } catch (const std::exception& error) {
            amrex::Print()
                << "spatial-v4 metadata read error: "
                << error.what() << "\n";
            io_failed = 1;
        }
    }

    const int io_rank =
        amrex::ParallelDescriptor::IOProcessorNumber();
    amrex::ParallelDescriptor::Bcast(
        &io_failed,
        1,
        io_rank);
    if (io_failed != 0) {
        throw std::runtime_error(
            "IO rank failed to read spatial-v4 metadata");
    }

    int schema_version =
        amrex::ParallelDescriptor::IOProcessor()
            ? metadata.spatial_fuel_schema_version
            : 0;
    unsigned long long fingerprint =
        amrex::ParallelDescriptor::IOProcessor()
            ? static_cast<unsigned long long>(
                metadata.spatial_fuel_fingerprint_fnv1a64)
            : 0ULL;

    amrex::ParallelDescriptor::Bcast(
        &schema_version, 1, io_rank);
    amrex::ParallelDescriptor::Bcast(
        &fingerprint, 1, io_rank);

    metadata.spatial_fuel_schema_version =
        schema_version;
    metadata.spatial_fuel_fingerprint_fnv1a64 =
        static_cast<std::uint64_t>(fingerprint);

    amrex::MultiFab checkpoint;
    amrex::VisMF::Read(
        checkpoint,
        spatial_v4_raster_prefix(directory));

    ERFFireSpreadRuntime restored =
        ERFFire::collective_restore_erf_fire_checkpoint_v4(
            std::move(metadata),
            checkpoint);

    require_spatial_v4(
        restored.has_spatial_fuel(),
        "spatial-v4 restore lost spatial material");
    require_spatial_v4(
        restored.current_time_s() == expected_time_s,
        "spatial-v4 restore changed the runtime clock");

    const FireFuelRaster* restored_fuel =
        restored.spatial_fuel_raster();
    require_spatial_v4(
        restored_fuel != nullptr,
        "spatial-v4 restore has no material raster");

    const std::uint64_t restored_fingerprint =
        ERFFire::collective_fire_fuel_raster_fingerprint_fnv1a64(
            *restored_fuel);
    require_spatial_v4(
        restored_fingerprint
            == static_cast<std::uint64_t>(fingerprint),
        "spatial-v4 restored material fingerprint changed");

    return restored;
}

void
require_dry_and_wet_consumption(
    const ERFFireSpreadRuntime& runtime)
{
    const auto state =
        runtime.combustion_raster()
            .collective_snapshot_state_to_io_rank();

    int saw_dry = 0;
    int saw_wet = 0;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        const auto& geometry =
            runtime.config().raster_geometry;
        if (state.cells.size()
            != geometry.nx * geometry.ny) {
            throw std::runtime_error(
                "spatial-v4 combustion snapshot has the wrong cell count");
        }

        for (std::size_t j = 0; j < geometry.ny; ++j) {
            for (std::size_t i = 0; i < geometry.nx; ++i) {
                if (state.cells[j * geometry.nx + i]
                        .consumed_dry_fuel_kg_m2
                    > Real(0)) {
                    if (i < geometry.nx / 2) {
                        saw_dry = 1;
                    } else {
                        saw_wet = 1;
                    }
                }
            }
        }
    }

    const int io_rank =
        amrex::ParallelDescriptor::IOProcessorNumber();
    amrex::ParallelDescriptor::Bcast(
        &saw_dry, 1, io_rank);
    amrex::ParallelDescriptor::Bcast(
        &saw_wet, 1, io_rank);

    require_spatial_v4(
        saw_dry != 0 && saw_wet != 0,
        "spatial-v4 fixture did not consume fuel on both moisture regions");
}

std::string
read_file_bytes(const std::filesystem::path& path)
{
    std::ifstream stream(
        path,
        std::ios::in | std::ios::binary);
    if (!stream.good()) {
        throw std::runtime_error(
            "unable to open comparison file " + path.string());
    }

    return std::string(
        std::istreambuf_iterator<char>(stream),
        std::istreambuf_iterator<char>());
}

void
compare_spatial_v4_rasters(
    const std::filesystem::path& reference_directory,
    const std::filesystem::path& comparison_directory)
{
    require_spatial_v4(
        amrex::ParallelDescriptor::NProcs() == 1,
        "spatial-v4 final comparison must run on one MPI rank");

    require_spatial_v4(
        read_file_bytes(reference_directory / "FireState")
            == read_file_bytes(comparison_directory / "FireState"),
        "spatial-v4 final FireState metadata differ");

    amrex::MultiFab reference;
    amrex::MultiFab comparison;
    amrex::VisMF::Read(
        reference,
        spatial_v4_raster_prefix(reference_directory));
    amrex::VisMF::Read(
        comparison,
        spatial_v4_raster_prefix(comparison_directory));

    require_spatial_v4(
        reference.nComp()
                == ERFFire::ERFFireCheckpointV4RasterComponents::component_count
            && comparison.nComp() == reference.nComp()
            && reference.nGrow() == 0
            && comparison.nGrow() == 0,
        "spatial-v4 final checkpoint raster schema differs");
    require_spatial_v4(
        reference.boxArray() == comparison.boxArray(),
        "spatial-v4 final checkpoint BoxArrays differ");

    amrex::MultiFab aligned(
        reference.boxArray(),
        reference.DistributionMap(),
        reference.nComp(),
        0);
    aligned.ParallelCopy(
        comparison,
        0,
        0,
        comparison.nComp(),
        0,
        0);

    amrex::MFInfo host_info;
    host_info.SetArena(amrex::The_Pinned_Arena());
    amrex::MultiFab reference_host(
        reference.boxArray(),
        reference.DistributionMap(),
        reference.nComp(),
        0,
        host_info);
    amrex::MultiFab comparison_host(
        reference.boxArray(),
        reference.DistributionMap(),
        reference.nComp(),
        0,
        host_info);

    reference_host.ParallelCopy(
        reference,
        0,
        0,
        reference.nComp(),
        0,
        0);
    comparison_host.ParallelCopy(
        aligned,
        0,
        0,
        aligned.nComp(),
        0,
        0);
    amrex::Gpu::streamSynchronize();

    for (amrex::MFIter mfi(reference_host);
         mfi.isValid();
         ++mfi) {
        const amrex::Box& box = mfi.validbox();
        const auto a = reference_host.const_array(mfi);
        const auto b = comparison_host.const_array(mfi);

        for (int component = 0;
             component < reference_host.nComp();
             ++component) {
            for (int j = box.smallEnd(1);
                 j <= box.bigEnd(1);
                 ++j) {
                for (int i = box.smallEnd(0);
                     i <= box.bigEnd(0);
                     ++i) {
                    if (a(i, j, 0, component)
                        != b(i, j, 0, component)) {
                        throw std::runtime_error(
                            "spatial-v4 final checkpoint mismatch at ("
                            + std::to_string(i) + ","
                            + std::to_string(j) + ") component "
                            + std::to_string(component));
                    }
                }
            }
        }
    }
}

void
run_spatial_v4_mode(
    const std::string& mode,
    const std::filesystem::path& root)
{
    if (mode == "write_reference") {
        require_spatial_v4(
            amrex::ParallelDescriptor::NProcs() == 1,
            "spatial-v4 write_reference requires exactly one MPI rank");

        const ERFFireSpreadConfig config =
            spatial_v4_config();
        ERFFireSpreadRuntime runtime(
            spatial_v4_ignition(),
            Real(0),
            config,
            spatial_v4_fuel(config));
        const FireEnvironmentBatchFunction environment =
            spatial_v4_environment();

        (void)runtime.advance_direct_reference_wind_batched(
            environment,
            spatial_v4_dt_s);
        require_dry_and_wet_consumption(runtime);
        write_spatial_v4_checkpoint(
            runtime,
            root / "checkpoint");

        (void)runtime.advance_direct_reference_wind_batched(
            environment,
            spatial_v4_dt_s);
        write_spatial_v4_checkpoint(
            runtime,
            root / "reference_final");

        amrex::Print()
            << "ERF_FIRE_SPATIAL_V4_WRITE_REFERENCE_OK=1\n";
        return;
    }

    if (mode == "restore_continue") {
        require_spatial_v4(
            amrex::ParallelDescriptor::NProcs() == 2,
            "spatial-v4 restore_continue requires exactly two MPI ranks");

        ERFFireSpreadRuntime runtime =
            read_spatial_v4_checkpoint(
                root / "checkpoint",
                spatial_v4_dt_s);
        require_dry_and_wet_consumption(runtime);

        (void)runtime.advance_direct_reference_wind_batched(
            spatial_v4_environment(),
            spatial_v4_dt_s);
        write_spatial_v4_checkpoint(
            runtime,
            root / "restart_final");

        amrex::Print()
            << "ERF_FIRE_SPATIAL_V4_RESTORE_CONTINUE_OK=1\n";
        return;
    }

    if (mode == "compare") {
        compare_spatial_v4_rasters(
            root / "reference_final",
            root / "restart_final");
        amrex::Print()
            << "ERF_FIRE_SPATIAL_V4_COMPARE_OK=1\n";
        return;
    }

    throw std::invalid_argument(
        "unsupported fire_spatial_v4_test.mode");
}

} // namespace

int
main (int argc, char** argv)
{
    amrex::Initialize(argc, argv);

    // This NO_ERF_MAIN test driver bypasses Source/main.cpp::add_par(), where
    // ERF installs its AMR defaults. Mirror the vertical box-splitting default
    // required by implicit acoustic substepping while preserving any explicit
    // value supplied by the test input or command line.
    {
        amrex::ParmParse pp_amr("amr");
        int no_box_split_dir = 2;
        pp_amr.queryAdd("no_box_split_dir", no_box_split_dir);
    }

    int result = 0;
    try {
        amrex::ParmParse pp_spatial_v4("fire_spatial_v4_test");
        std::string spatial_v4_mode;
        const bool run_spatial_v4 =
            pp_spatial_v4.query("mode", spatial_v4_mode);

        if (run_spatial_v4) {
            std::string spatial_v4_root;
            pp_spatial_v4.get("root", spatial_v4_root);
            run_spatial_v4_mode(
                spatial_v4_mode,
                std::filesystem::path(spatial_v4_root));
        } else {
            ERF erf;
            erf.InitData();

            double time = 0.0;
            double last_dt = 0.0;
            constexpr int steps = 5;

            for (int step = 0; step < steps; ++step) {
                last_dt = erf.EvolveOneStep(time, 0.01);
                if (!(last_dt > 0.0)) {
                    throw std::runtime_error(
                        "ERF one-way identity run did not advance");
                }
                time += last_dt;
            }

            // Use ERF's existing public checkpoint writer as the atmospheric
            // identity surface. The enabled process advances a real Fire perimeter
            // for five coarse steps, while prognostic ERF checkpoint bytes must
            // remain identical to the disabled control.
            erf.WriteCheckpointFile();
            amrex::Print() << "ERF_FIRE_IDENTITY_CHILD_OK=1\n";
        }
    } catch (const std::exception& error) {
        amrex::Print() << "ERF fire integration error: "
                       << error.what() << "\n";
        result = 2;
    }

    amrex::Finalize();
    return result;
}
