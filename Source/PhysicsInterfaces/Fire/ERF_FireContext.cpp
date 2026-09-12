#include <ERF.H>
#include <ERF_FireContext.H>

#include <limits>

#ifdef ERF_USE_FIRE

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
