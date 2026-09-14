#include <ERF_FireCheckpoint.H>

#include <ERF_TerrainSource.H>

#include <AMReX.H>
#include <AMReX_ParmParse.H>

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

namespace ERFFire
{

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
    const ERFTerrainSource* terrain_source)
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
    // authoritative level-0 terrain surface rather than ERFTerrainSource.
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
    const ERFTerrainSource* terrain_source)
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
    const ERFTerrainSource* terrain_source)
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
