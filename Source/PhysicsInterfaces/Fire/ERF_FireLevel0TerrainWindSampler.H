#ifndef ERF_FIRE_LEVEL0_TERRAIN_WIND_SAMPLER_H_
#define ERF_FIRE_LEVEL0_TERRAIN_WIND_SAMPLER_H_

#include <ERF_FireLevel0Environment.H>

#include <AMReX_REAL.H>

#include <vector>

namespace ERFFire
{

/**
 * Collective sparse reference-wind sampler for static fitted level-0 terrain.
 *
 * The sampler retains read-only references to the already-fillpatched ERF
 * x/y velocity fields and authoritative nodal physical height field. It owns
 * no full-domain atmospheric snapshot.
 *
 * sample_points() constructs distributed full-height columns only for the
 * native staggered faces touched by the requested bilinear point stencils.
 * Each face column is paired with the two z_phys_nd nodal lines required to
 * form that face's local AGL cell-center coordinates. Vertical interpolation
 * is performed on the face-column owner and only the final O(point-count)
 * face values are reduced to the common ordered point samples.
 *
 * The source MultiFabs must remain valid and unchanged for this object's
 * lifetime. This is a t^n coupling view, not an owning snapshot.
 */
class ERFFireLevel0TerrainWindSampler
{
public:
    ERFFireLevel0TerrainWindSampler(
        const ERFFireLevel0EnvironmentInputs& inputs,
        amrex::Real reference_height_agl_m);

    [[nodiscard]] amrex::Real
    reference_height_agl_m() const noexcept
    {
        return reference_height_agl_m_;
    }

    [[nodiscard]] const FireFlatEnvironmentLayout2D&
    layout() const noexcept
    {
        return layout_;
    }

    [[nodiscard]] std::vector<FireEnvironmentSample>
    sample_points(
        const std::vector<FireVec2>& positions_m) const;

private:
    const amrex::MultiFab* x_velocity_{nullptr};
    const amrex::MultiFab* y_velocity_{nullptr};
    const amrex::MultiFab* z_phys_nd_{nullptr};

    FireFlatEnvironmentLayout2D layout_;
    amrex::Real reference_height_agl_m_{};

    int domain_ilo_{};
    int domain_jlo_{};
    int domain_klo_{};
    int domain_khi_{};
};

} // namespace ERFFire

#endif
