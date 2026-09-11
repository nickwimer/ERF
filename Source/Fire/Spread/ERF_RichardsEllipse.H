#ifndef ERF_RICHARDS_ELLIPSE_H
#define ERF_RICHARDS_ELLIPSE_H

#include <ERF_FireTypes.H>

namespace ERFFire
{

/**
 * Local Richards/FARSITE elliptical wavelet in physical SI units.
 *
 * The wavelet translates in heading_unit at center_translation_rate_mps while
 * expanding with semi-major rate b along heading_unit and semi-minor rate a
 * perpendicular to it. Finney (1998, revised 2004), equations 13-17, uses the
 * rear-focus fire-origin assumption to infer backing spread from the heading
 * rate and length-to-breadth ratio.
 *
 * effective_midflame_wind_mps is the FARSITE virtual/effective midflame wind
 * used only to determine ellipse shape. It is not an atmospheric sampling
 * height contract. A later directional-forcing adapter will obtain this value
 * from the resultant wind/slope effect.
 */
struct RichardsEllipse
{
    FireVec2 heading_unit{};
    amrex::Real heading_ros_mps{};
    amrex::Real effective_midflame_wind_mps{};

    amrex::Real unclamped_length_to_breadth{};
    amrex::Real length_to_breadth{};
    amrex::Real head_to_back_ratio{};
    amrex::Real eccentricity{};

    // Finney notation: a = semi-minor rate, b = semi-major rate,
    // c = center translation rate from the rear-focus ignition point.
    amrex::Real semi_minor_rate_mps{};
    amrex::Real semi_major_rate_mps{};
    amrex::Real center_translation_rate_mps{};

    [[nodiscard]] amrex::Real backing_ros_mps() const noexcept
    {
        return semi_major_rate_mps - center_translation_rate_mps;
    }

    [[nodiscard]] amrex::Real flank_ros_mps() const noexcept
    {
        return semi_minor_rate_mps;
    }
};

/**
 * FARSITE equation 13 (Finney 1998, revised 2004), before the documented
 * LB <= 8 truncation. U is effective midflame windspeed in m/s.
 */
[[nodiscard]] amrex::Real farsite_unclamped_length_to_breadth(
    amrex::Real effective_midflame_wind_mps);

/**
 * Construct the FARSITE single-ellipse wavelet from a heading spread rate.
 *
 * Uses Finney equations 13-17 and the rear-focus fire-origin assumption.
 * The supplied heading direction must already be a unit vector.
 */
[[nodiscard]] RichardsEllipse make_richards_ellipse(
    const FireVec2& heading_unit,
    amrex::Real heading_ros_mps,
    amrex::Real effective_midflame_wind_mps);

/**
 * Exact geometric normal speed of the translating/expanding ellipse.
 *
 * For outward unit normal n and heading unit vector e, mu=e dot n,
 *
 *   V_n = c mu + sqrt(b^2 mu^2 + a^2 (1-mu^2)).
 *
 * This is the support function of the local elliptical Huygens wavelet per
 * unit time. It equals the normal component of the Richards (1990) vertex
 * velocity. Tangential Richards motion only reparameterizes the perimeter and
 * is intentionally absent from the normal-speed interface used by the ERF-Fire
 * front propagator.
 */
[[nodiscard]] amrex::Real richards_normal_speed_mps(
    const RichardsEllipse& ellipse,
    const FireVec2& outward_normal_unit);

} // namespace ERFFire

#endif // ERF_RICHARDS_ELLIPSE_H
