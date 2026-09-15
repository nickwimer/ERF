#include <ERF.H>
#include <ERF_FireContext.H>
#include <ERF_FireTerrainSource.H>

#include <AMReX_ParmParse.H>

#include <string>

#include <limits>

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

bool
ERF::FireEnabled () const noexcept
{
    return m_fire && m_fire->runtime_options().enabled;
}

const ERFFire::FireFlatEnvironmentSampler*
ERF::FireEnvironmentSnapshot () const noexcept
{
    // Environment samplers are deliberately scoped to one Fire advance.
    // Retaining one would make the atmospheric snapshot lifetime implicit.
    return nullptr;
}

double
ERF::FireEnvironmentSnapshotTime () const noexcept
{
    return m_fire
        ? m_fire->environment_snapshot_time()
        : std::numeric_limits<double>::quiet_NaN();
}

double
ERF::FireEnvironmentReferenceHeightAGL () const noexcept
{
    return m_fire
        ? m_fire->environment_reference_height_agl_m()
        : std::numeric_limits<double>::quiet_NaN();
}

const ERFFire::ERFFireSpreadRuntime*
ERF::FireSpreadRuntime () const noexcept
{
    return m_fire
        ? m_fire->spread_runtime().get()
        : nullptr;
}

const amrex::MultiFab*
ERF::FireAtmosphericSourceTendency () const noexcept
{
    return m_fire
        ? m_fire->atmospheric_source_tendency().get()
        : nullptr;
}

double
ERF::FireAtmosphericSourceTime () const noexcept
{
    return m_fire
        ? m_fire->atmospheric_source_time()
        : std::numeric_limits<double>::quiet_NaN();
}

#endif
