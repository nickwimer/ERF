#include <ERF.H>
#include <ERF_FireContext.H>

#include <AMReX.H>
#include <AMReX_MultiFab.H>

namespace ERFFire
{

void
ERFFireContext::add_atmospheric_source(
    const int level,
    amrex::MultiFab& conserved_source) const
{
    if (level != 0 || !atmospheric_source_tendency_) {
        return;
    }

    if (!runtime_options_.enabled
        || runtime_options_.coupling_mode
            != ERFFireCouplingMode::TwoWay) {
        amrex::Error(
            "ERF Fire atmospheric source exists outside two_way coupling");
    }

    if (conserved_source.nComp()
            != atmospheric_source_tendency_->nComp()
        || !amrex::match(
            conserved_source.boxArray(),
            atmospheric_source_tendency_->boxArray())) {
        amrex::Error(
            "ERF Fire atmospheric source layout does not match cc_src");
    }

    amrex::MultiFab::Add(
        conserved_source,
        *atmospheric_source_tendency_,
        0,
        0,
        conserved_source.nComp(),
        0);
}

} // namespace ERFFire

#ifdef ERF_USE_FIRE

void
ERF::add_fire_atmospheric_source (
    int level,
    amrex::MultiFab& conserved_source) const
{
    if (m_fire) {
        m_fire->add_atmospheric_source(
            level,
            conserved_source);
    }
}

#endif
