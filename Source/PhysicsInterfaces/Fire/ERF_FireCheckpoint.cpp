#include <ERF_FireCheckpoint.H>
#include <ERF_FireCheckpointV4.H>
#include <ERF_FireFuelSource.H>

#include <ERF_FireContext.H>
#include <ERF_FireRuntimeInit.H>
#include <ERF_FireTerrainSource.H>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>

#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace ERFFire
{

void
ERFFireContext::write_checkpoint(
    const ERFFireCheckpointWriteInputs& inputs) const
{
    if (!runtime_options_.enabled) {
        return;
    }

    std::unique_ptr<ERFFireSpreadRuntime>
        initial_fire_runtime;

    const ERFFireSpreadRuntime*
        fire_runtime_for_checkpoint =
            spread_runtime_.get();

    if (fire_runtime_for_checkpoint == nullptr) {
        if (inputs.level0_step != 0
            || inputs.level0_time_s != amrex::Real(0.0)) {
            amrex::Error(
                "ERF-Fire runtime is missing while writing a noninitial checkpoint");
        }

        initial_fire_runtime =
            make_erf_fire_spread_runtime(
                runtime_options_,
                inputs.level0_geometry,
                inputs.level0_time_s);

        fire_runtime_for_checkpoint =
            initial_fire_runtime.get();
    }

    const bool spatial_checkpoint =
        fire_runtime_for_checkpoint->has_spatial_fuel();

    std::uint64_t spatial_fuel_fingerprint_fnv1a64 = 0;
    if (spatial_checkpoint) {
        const FireFuelRaster* spatial_fuel =
            fire_runtime_for_checkpoint->spatial_fuel_raster();
        if (spatial_fuel == nullptr) {
            amrex::Error(
                "ERF-Fire spatial checkpoint runtime lost its material raster");
        }

        spatial_fuel_fingerprint_fnv1a64 =
            collective_fire_fuel_raster_fingerprint_fnv1a64(
                *spatial_fuel);
    }

    amrex::MultiFab fire_checkpoint_raster =
        spatial_checkpoint
        ? make_erf_fire_checkpoint_v4_raster(
            *fire_runtime_for_checkpoint)
        : make_erf_fire_checkpoint_v2_raster(
            *fire_runtime_for_checkpoint);

    amrex::VisMF::Write(
        fire_checkpoint_raster,
        amrex::MultiFabFileFullPrefix(
            0,
            inputs.checkpoint_directory,
            "Level_",
            "FireStateRaster"));

    // This lookup may perform collective regular-text terrain I/O and
    // broadcasts. It must remain on all ranks and must occur before the
    // IO-rank-only metadata block below.
    const ERFFireTerrainSource*
        fire_terrain_source_for_checkpoint = nullptr;

    if (inputs.solver_choices.mesh_type
            == MeshType::VariableDz
        && inputs.solver_choices.terrain_type
            == TerrainType::StaticFittedMesh) {

        fire_terrain_source_for_checkpoint =
            resolve_terrain_source();
    }

    if (!amrex::ParallelDescriptor::IOProcessor()) {
        return;
    }

    const std::string fire_state_name =
        inputs.checkpoint_directory
        + "/FireState";

    std::ofstream fire_state(
        fire_state_name,
        std::ios::out
            | std::ios::trunc
            | std::ios::binary);

    if (!fire_state.good()) {
        amrex::FileOpenFailed(
            fire_state_name);
    }

    try {
        if (spatial_checkpoint) {
            write_erf_fire_checkpoint_v4_metadata(
                *fire_runtime_for_checkpoint,
                runtime_options_,
                spatial_fuel_fingerprint_fnv1a64,
                fire_state);
        } else {
            write_erf_fire_checkpoint_v3_metadata(
                *fire_runtime_for_checkpoint,
                runtime_options_,
                fire_state);
        }

        const std::string
            fire_terrain_policy_name =
                inputs.checkpoint_directory
                + "/FireTerrainSourcePolicy";

        std::ofstream fire_terrain_policy(
            fire_terrain_policy_name,
            std::ios::out
                | std::ios::trunc
                | std::ios::binary);

        if (!fire_terrain_policy.good()) {
            amrex::FileOpenFailed(
                fire_terrain_policy_name);
        }

        write_fire_terrain_source_policy(
            fire_terrain_policy,
            inputs.solver_choices,
            fire_terrain_source_for_checkpoint);

    } catch (const std::exception& error) {
        amrex::Error(
            std::string(
                "failed to write ERF-Fire checkpoint metadata: ")
            + error.what());
    }
}

void
ERFFireContext::restore_checkpoint(
    const ERFFireCheckpointRestoreInputs& inputs)
{
    if (!runtime_options_.enabled) {
        return;
    }

    const std::string fire_state_name =
        inputs.checkpoint_directory
        + "/FireState";

    if (!amrex::FileExists(fire_state_name)) {
        amrex::Error(
            "Fire-enabled restart requires persistent FireState in native checkpoint "
            + inputs.checkpoint_directory);
    }

    try {
        const ERFFireSpreadConfig expected_config =
            make_erf_fire_spread_config(
                runtime_options_,
                inputs.level0_geometry);

        int checkpoint_version = 0;
        int fire_version_read_failed = 0;

        if (amrex::ParallelDescriptor::IOProcessor()) {
            try {
                std::ifstream version_stream(
                    fire_state_name,
                    std::ios::in | std::ios::binary);

                if (!version_stream.good()) {
                    throw std::runtime_error(
                        "unable to open ERF-Fire checkpoint state "
                        + fire_state_name);
                }

                checkpoint_version =
                    read_erf_fire_checkpoint_version(
                        version_stream);

            } catch (const std::exception&) {
                fire_version_read_failed = 1;
            }
        }

        amrex::ParallelDescriptor::Bcast(
            &fire_version_read_failed,
            1,
            amrex::ParallelDescriptor::IOProcessorNumber());

        amrex::ParallelDescriptor::Bcast(
            &checkpoint_version,
            1,
            amrex::ParallelDescriptor::IOProcessorNumber());

        if (fire_version_read_failed != 0) {
            throw std::runtime_error(
                "IO rank failed to identify ERF-Fire checkpoint version");
        }

        if (checkpoint_version != 1
            && checkpoint_version != 2
            && checkpoint_version != 3
            && checkpoint_version != 4) {
            throw std::runtime_error(
                "unsupported ERF-Fire checkpoint format version");
        }

        if (checkpoint_version != 4
            && !runtime_options_.fuel_raster_file.empty()) {
            throw std::runtime_error(
                "fire.fuel_raster_file cannot be applied to ERF-Fire checkpoint formats v1-v3; restart without the spatial fuel file or use a v4 checkpoint");
        }

        ERFFireSpreadRuntimeState restore_state;
        restore_state.config = expected_config;
        restore_state.current_time_s =
            inputs.level0_time_s;

        ERFFireCheckpointV2Metadata v2_metadata;
        ERFFireCheckpointV3Metadata v3_metadata;
        ERFFireCheckpointV4Metadata v4_metadata;

        int v4_spatial_fuel_schema_version = 0;
        unsigned long long
            v4_spatial_fuel_fingerprint_fnv1a64 = 0ULL;

        // Preserve the existing collective restart ordering exactly.
        // resolve_terrain_source() may load a regular-text terrain source using
        // IO-rank file I/O followed by MPI broadcasts, so this resolution is
        // performed on every rank before entering the metadata IO-rank block.

        const ERFFireTerrainSource* current_terrain_source =
            resolve_terrain_source();

        int fire_state_read_failed = 0;

        if (amrex::ParallelDescriptor::IOProcessor()) {
            try {
                validate_fire_terrain_source_restart_policy(
                    inputs.checkpoint_directory,
                    inputs.solver_choices,
                    current_terrain_source);

                std::ifstream fire_state_stream(
                    fire_state_name,
                    std::ios::in | std::ios::binary);

                if (!fire_state_stream.good()) {
                    throw std::runtime_error(
                        "unable to open ERF-Fire checkpoint state "
                        + fire_state_name);
                }

                ERFFireCheckpointState* checkpoint =
                    nullptr;

                ERFFireCheckpointState v1_checkpoint;

                if (checkpoint_version == 1) {
                    v1_checkpoint =
                        read_erf_fire_checkpoint_state(
                            fire_state_stream);

                    checkpoint =
                        &v1_checkpoint;

                } else if (checkpoint_version == 2) {
                    v2_metadata =
                        read_erf_fire_checkpoint_v2_metadata(
                            fire_state_stream);

                    checkpoint =
                        &v2_metadata.checkpoint;

                } else if (checkpoint_version == 3) {
                    v3_metadata =
                        read_erf_fire_checkpoint_v3_metadata(
                            fire_state_stream);

                    checkpoint =
                        &v3_metadata.checkpoint;

                } else {
                    v4_metadata =
                        read_erf_fire_checkpoint_v4_metadata(
                            fire_state_stream);

                    checkpoint =
                        &v4_metadata.checkpoint;
                    v4_spatial_fuel_schema_version =
                        v4_metadata.spatial_fuel_schema_version;
                    v4_spatial_fuel_fingerprint_fnv1a64 =
                        static_cast<unsigned long long>(
                            v4_metadata
                                .spatial_fuel_fingerprint_fnv1a64);
                }

                validate_fire_checkpoint_policy(
                    *checkpoint,
                    runtime_options_);

                if (!same_fire_spread_config(
                        checkpoint->runtime_state.config,
                        expected_config)) {
                    throw std::runtime_error(
                        "ERF-Fire checkpoint spread configuration does not match current inputs or level-0 geometry");
                }

                if (checkpoint->runtime_state.current_time_s
                    != inputs.level0_time_s) {
                    throw std::runtime_error(
                        "ERF-Fire checkpoint clock does not match ERF level-0 checkpoint time");
                }

                restore_state =
                    std::move(
                        checkpoint->runtime_state);

            } catch (const std::exception& error) {
                amrex::Print()
                    << "ERF-Fire restart validation error: "
                    << error.what()
                    << "\n";

                fire_state_read_failed = 1;
            }
        }

        amrex::ParallelDescriptor::Bcast(
            &fire_state_read_failed,
            1,
            amrex::ParallelDescriptor::IOProcessorNumber());

        if (fire_state_read_failed != 0) {
            throw std::runtime_error(
                "IO rank failed to read or validate ERF-Fire checkpoint state");
        }

        if (checkpoint_version == 4) {
            amrex::ParallelDescriptor::Bcast(
                &v4_spatial_fuel_schema_version,
                1,
                amrex::ParallelDescriptor::IOProcessorNumber());
            amrex::ParallelDescriptor::Bcast(
                &v4_spatial_fuel_fingerprint_fnv1a64,
                1,
                amrex::ParallelDescriptor::IOProcessorNumber());

            if (!runtime_options_.fuel_raster_file.empty()) {
                const FireFuelRaster current_spatial_fuel =
                    read_erf_fire_aligned_fuel_raster_text_file(
                        runtime_options_.fuel_raster_file,
                        expected_config.raster_geometry);

                const std::uint64_t current_fingerprint =
                    collective_fire_fuel_raster_fingerprint_fnv1a64(
                        current_spatial_fuel);

                if (current_fingerprint
                    != static_cast<std::uint64_t>(
                        v4_spatial_fuel_fingerprint_fnv1a64)) {
                    throw std::runtime_error(
                        "ERF-Fire v4 checkpoint spatial fuel fingerprint does not match current fire.fuel_raster_file");
                }
            }
        }

        if (checkpoint_version == 1) {
            ERFFireSpreadRuntime restored =
                ERFFireSpreadRuntime::
                    collective_restore_from_io_rank_state(
                        std::move(restore_state));

            spread_runtime_ =
                std::make_unique<
                    ERFFireSpreadRuntime>(
                        std::move(restored));

        } else {
            const std::string fire_raster_name =
                amrex::MultiFabFileFullPrefix(
                    0,
                    inputs.checkpoint_directory,
                    "Level_",
                    "FireStateRaster");

            if (!amrex::FileExists(
                    fire_raster_name + "_H")) {
                throw std::runtime_error(
                    "ERF-Fire distributed checkpoint is missing FireStateRaster");
            }

            amrex::MultiFab checkpoint_raster;

            amrex::VisMF::Read(
                checkpoint_raster,
                fire_raster_name);

            ERFFireSpreadRuntime restored =
                checkpoint_version == 4
                ? collective_restore_erf_fire_checkpoint_v4(
                    ERFFireCheckpointV4Metadata{
                        ERFFireCheckpointState{
                            std::move(restore_state),
                            ERFFireCouplingMode::OneWay,
                            ERFFireWindMode::DirectReference,
                            amrex::Real(0),
                            amrex::Real(1),
                            amrex::Real(50)},
                        v4_spatial_fuel_schema_version,
                        static_cast<std::uint64_t>(
                            v4_spatial_fuel_fingerprint_fnv1a64)},
                    checkpoint_raster)
                : ERFFireSpreadRuntime::
                    collective_restore_from_checkpoint_raster(
                        std::move(restore_state),
                        checkpoint_raster);

            spread_runtime_ =
                std::make_unique<
                    ERFFireSpreadRuntime>(
                        std::move(restored));
        }

        step_index_ =
            inputs.level0_step;

        environment_snapshot_time_ =
            std::numeric_limits<double>::quiet_NaN();

        atmospheric_source_tendency_.reset();

        atmospheric_source_time_ =
            std::numeric_limits<double>::quiet_NaN();

    } catch (const std::exception& error) {
        amrex::Error(
            std::string(
                "failed to restore ERF-Fire checkpoint state: ")
            + error.what());
    }
}

namespace
{

enum class FireTerrainSourcePolicyMode
{
    NotApplicable,
    Level0,
    RegularFile
};

struct FireTerrainSourcePolicy
{
    FireTerrainSourcePolicyMode mode{
        FireTerrainSourcePolicyMode::NotApplicable};
    std::uint64_t fingerprint_fnv1a64{};
};

const char*
fire_terrain_source_policy_mode_token(
    FireTerrainSourcePolicyMode mode)
{
    if (mode == FireTerrainSourcePolicyMode::NotApplicable) {
        return "not_applicable";
    }
    if (mode == FireTerrainSourcePolicyMode::Level0) {
        return "level0";
    }
    if (mode == FireTerrainSourcePolicyMode::RegularFile) {
        return "regular_file";
    }

    throw std::logic_error(
        "unsupported ERF-Fire terrain-source policy mode");
}

FireTerrainSourcePolicyMode
read_fire_terrain_source_policy_mode(
    std::istream& stream)
{
    std::string token;

    if (!(stream >> token)) {
        throw std::runtime_error(
            "missing ERF-Fire terrain-source policy mode");
    }

    if (token == "not_applicable") {
        return FireTerrainSourcePolicyMode::NotApplicable;
    }
    if (token == "level0") {
        return FireTerrainSourcePolicyMode::Level0;
    }
    if (token == "regular_file") {
        return FireTerrainSourcePolicyMode::RegularFile;
    }

    throw std::runtime_error(
        "invalid ERF-Fire terrain-source policy mode");
}

void
expect_fire_terrain_source_policy_token(
    std::istream& stream,
    const char* expected)
{
    std::string token;

    if (!(stream >> token) || token != expected) {
        throw std::runtime_error(
            std::string(
                "invalid ERF-Fire terrain-source policy field; expected ")
            + expected);
    }
}

FireTerrainSourcePolicy
current_fire_terrain_source_policy(
    const SolverChoice& choices,
    const ERFFireTerrainSource* terrain_source)
{
    if (choices.mesh_type != MeshType::VariableDz
        || choices.terrain_type
            != TerrainType::StaticFittedMesh) {
        return {
            FireTerrainSourcePolicyMode::NotApplicable,
            0};
    }

    amrex::ParmParse pp("erf");

    // Match ProblemBase::init_terrain_surface() and terrain_source()
    // precedence. NetCDF/WPS terrain is consumed by Fire through the
    // authoritative level-0 terrain surface rather than ERFFireTerrainSource.
    std::string filename_nc;
    if (pp.query("terrain_file_name_nc", filename_nc)) {
        return {
            FireTerrainSourcePolicyMode::Level0,
            0};
    }

    std::string filename;
    if (!pp.query(
            "terrain_file_name",
            filename)) {
        return {
            FireTerrainSourcePolicyMode::Level0,
            0};
    }

    if (filename.empty()) {
        throw std::runtime_error(
            "erf.terrain_file_name must not be empty");
    }

    if (terrain_source == nullptr) {
        throw std::runtime_error(
            "regular ERF terrain source is not loaded while evaluating Fire checkpoint policy");
    }

    const auto fingerprint =
        terrain_source->source_fingerprint_fnv1a64();

    if (!fingerprint.has_value()) {
        throw std::runtime_error(
            "loaded regular ERF terrain source has no file fingerprint");
    }

    return {
        FireTerrainSourcePolicyMode::RegularFile,
        *fingerprint};
}

FireTerrainSourcePolicy
read_fire_terrain_source_policy(
    std::istream& stream)
{
    expect_fire_terrain_source_policy_token(
        stream,
        "ERF_FIRE_TERRAIN_SOURCE_POLICY");

    int version = 0;
    if (!(stream >> version)
        || version != 1) {
        throw std::runtime_error(
            "unsupported ERF-Fire terrain-source policy version");
    }

    expect_fire_terrain_source_policy_token(
        stream,
        "mode");

    FireTerrainSourcePolicy policy;
    policy.mode =
        read_fire_terrain_source_policy_mode(
            stream);

    if (policy.mode
        == FireTerrainSourcePolicyMode::RegularFile) {
        expect_fire_terrain_source_policy_token(
            stream,
            "fingerprint_fnv1a64");

        if (!(stream
              >> policy.fingerprint_fnv1a64)) {
            throw std::runtime_error(
                "invalid ERF-Fire terrain-source fingerprint");
        }
    }

    expect_fire_terrain_source_policy_token(
        stream,
        "END_ERF_FIRE_TERRAIN_SOURCE_POLICY");

    return policy;
}

bool
same_fire_fuel(
    const RothermelFuelParameters& lhs,
    const RothermelFuelParameters& rhs) noexcept
{
    return lhs.dead_1h_load_kg_m2 == rhs.dead_1h_load_kg_m2
        && lhs.dead_1h_sav_m_inv == rhs.dead_1h_sav_m_inv
        && lhs.fuel_bed_depth_m == rhs.fuel_bed_depth_m
        && lhs.dead_heat_content_j_kg == rhs.dead_heat_content_j_kg
        && lhs.particle_density_kg_m3 == rhs.particle_density_kg_m3
        && lhs.total_mineral_fraction == rhs.total_mineral_fraction
        && lhs.effective_mineral_fraction
            == rhs.effective_mineral_fraction
        && lhs.dead_moisture_of_extinction
            == rhs.dead_moisture_of_extinction;
}

} // namespace

void
write_fire_terrain_source_policy(
    std::ostream& stream,
    const SolverChoice& choices,
    const ERFFireTerrainSource* terrain_source)
{
    if (!stream.good()) {
        throw std::runtime_error(
            "ERF-Fire terrain-source policy stream is not writable");
    }

    const FireTerrainSourcePolicy policy =
        current_fire_terrain_source_policy(
            choices,
            terrain_source);

    stream
        << "ERF_FIRE_TERRAIN_SOURCE_POLICY 1\n"
        << "mode "
        << fire_terrain_source_policy_mode_token(
               policy.mode)
        << "\n";

    if (policy.mode
        == FireTerrainSourcePolicyMode::RegularFile) {
        stream
            << "fingerprint_fnv1a64 "
            << policy.fingerprint_fnv1a64
            << "\n";
    }

    stream
        << "END_ERF_FIRE_TERRAIN_SOURCE_POLICY\n";

    if (!stream.good()) {
        throw std::runtime_error(
            "failed while writing ERF-Fire terrain-source policy");
    }
}

void
validate_fire_terrain_source_restart_policy(
    const std::string& checkpoint_directory,
    const SolverChoice& choices,
    const ERFFireTerrainSource* terrain_source)
{
    const FireTerrainSourcePolicy current =
        current_fire_terrain_source_policy(
            choices,
            terrain_source);

    const std::string policy_name =
        checkpoint_directory
        + "/FireTerrainSourcePolicy";

    if (!amrex::FileExists(policy_name)) {
        if (current.mode
            == FireTerrainSourcePolicyMode::RegularFile) {
            throw std::runtime_error(
                "legacy ERF-Fire checkpoint cannot verify the current regular terrain source");
        }

        return;
    }

    std::ifstream stream(
        policy_name,
        std::ios::in | std::ios::binary);

    if (!stream.good()) {
        throw std::runtime_error(
            "unable to open ERF-Fire terrain-source policy "
            + policy_name);
    }

    const FireTerrainSourcePolicy checkpoint =
        read_fire_terrain_source_policy(
            stream);

    if (checkpoint.mode != current.mode) {
        throw std::runtime_error(
            "ERF-Fire checkpoint terrain-source mode does not match current inputs");
    }

    if (checkpoint.mode
            == FireTerrainSourcePolicyMode::RegularFile
        && checkpoint.fingerprint_fnv1a64
            != current.fingerprint_fnv1a64) {
        throw std::runtime_error(
            "ERF-Fire checkpoint terrain-source fingerprint does not match current terrain file");
    }
}

bool
same_fire_spread_config(
    const ERFFireSpreadConfig& lhs,
    const ERFFireSpreadConfig& rhs) noexcept
{
    const auto& lc = lhs.combustion_parameters;
    const auto& rc = rhs.combustion_parameters;
    const auto& lr = lhs.remesh_options;
    const auto& rr = rhs.remesh_options;
    const auto& lg = lhs.raster_geometry;
    const auto& rg = rhs.raster_geometry;

    return same_fire_fuel(lhs.fuel, rhs.fuel)
        && lhs.dead_fuel_moisture_fraction
            == rhs.dead_fuel_moisture_fraction
        && lc.dry_fuel_load_kg_m2
            == rc.dry_fuel_load_kg_m2
        && lc.sensible_heat_release_j_kg_dry
            == rc.sensible_heat_release_j_kg_dry
        && lc.fuel_moisture_fraction
            == rc.fuel_moisture_fraction
        && lc.burn_time_constant_s
            == rc.burn_time_constant_s
        && lc.combustion_water_yield_kg_per_kg_dry
            == rc.combustion_water_yield_kg_per_kg_dry
        && lhs.combustion_options.temporal_substeps
            == rhs.combustion_options.temporal_substeps
        && lr.min_edge_length_m == rr.min_edge_length_m
        && lr.max_edge_length_m == rr.max_edge_length_m
        && lr.max_chord_error_m == rr.max_chord_error_m
        && lg.nx == rg.nx
        && lg.ny == rg.ny
        && lg.xlo_m == rg.xlo_m
        && lg.ylo_m == rg.ylo_m
        && lg.dx_m == rg.dx_m
        && lg.dy_m == rg.dy_m
        && lhs.arrival_time_tolerance_s
            == rhs.arrival_time_tolerance_s;
}

void
validate_fire_checkpoint_policy(
    const ERFFireCheckpointState& checkpoint,
    const ERFFireRuntimeOptions& options)
{
    if (checkpoint.coupling_mode
        != options.coupling_mode) {
        throw std::runtime_error(
            "ERF-Fire checkpoint coupling mode does not match current inputs");
    }

    if (checkpoint.wind_mode
        != options.wind_mode) {
        throw std::runtime_error(
            "ERF-Fire checkpoint wind mode does not match current inputs");
    }

    if (options.wind_mode
        == ERFFireWindMode::DirectReference) {
        if (checkpoint.reference_height_agl_m
            != options.reference_height_agl_m) {
            throw std::runtime_error(
                "ERF-Fire checkpoint reference height does not match current inputs");
        }
    } else {
        if (checkpoint.wind_adjustment_factor
            != options.wind_adjustment_factor) {
            throw std::runtime_error(
                "ERF-Fire checkpoint wind adjustment factor does not match current inputs");
        }
    }

    if (options.coupling_mode
            == ERFFireCouplingMode::TwoWay
        && checkpoint.feedback_extinction_depth_m
            != options.feedback_extinction_depth_m) {
        throw std::runtime_error(
            "ERF-Fire checkpoint feedback extinction depth does not match current inputs");
    }
}

} // namespace ERFFire
