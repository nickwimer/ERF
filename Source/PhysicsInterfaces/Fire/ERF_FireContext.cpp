#include <ERF.H>
#include <ERF_FireContext.H>
#include <ERF_FireTerrainSource.H>

#include <AMReX_ParmParse.H>

#include <string>


#ifdef ERF_USE_FIRE

namespace ERFFire
{

ERFFireContext::ERFFireContext() = default;

ERFFireContext::~ERFFireContext() = default;

const ERFFireTerrainSource*
ERFFireContext::resolve_terrain_source() const
{
    amrex::ParmParse pp("erf");

    // Match ProblemBase::init_terrain_surface() precedence exactly.
    // NetCDF terrain remains authoritative at level 0 rather than
    // falling through to a lower-priority regular-text source.
    std::string filename_nc;
    if (pp.query("terrain_file_name_nc", filename_nc)) {
        return nullptr;
    }

    if (!regular_text_terrain_source_) {
        std::string filename;
        if (pp.query("terrain_file_name", filename)) {
            regular_text_terrain_source_ =
                std::make_unique<ERFFireTerrainSource>(
                    ERFFireTerrainSource::read_regular_text_file(
                        filename));
        }
    }

    // USGS and custom terrain use ERF's authoritative level-0
    // terrain surface rather than a retained regular-text source.
    return regular_text_terrain_source_.get();
}


void
ERFFireContext::configure_from_inputs(
    const ERFFireHostCapabilities& host)
{
    runtime_options_ = read_erf_fire_runtime_options(host);
}

} // namespace ERFFire

void
ERF::initialize_fire ()
{
    m_fire =
        std::make_unique<ERFFire::ERFFireContext>();

    ERFFire::ERFFireHostCapabilities fire_host{};
    fire_host.max_level = max_level;
    fire_host.variable_dz =
        solverChoice.mesh_type == MeshType::VariableDz;
    fire_host.static_fitted_mesh =
        solverChoice.terrain_type
            == TerrainType::StaticFittedMesh;
    fire_host.moist_no_condensation =
        solverChoice.moisture_type
            == MoistureType::MoistNoCondensation;
    fire_host.anelastic_level0 =
        solverChoice.anelastic[0] != 0;

    m_fire->configure_from_inputs(fire_host);
}

#endif
