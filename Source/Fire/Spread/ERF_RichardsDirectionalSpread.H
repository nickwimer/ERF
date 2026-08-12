#ifndef ERF_RICHARDS_DIRECTIONAL_SPREAD_H
#define ERF_RICHARDS_DIRECTIONAL_SPREAD_H

#include <ERF_RichardsEllipse.H>
#include <ERF_RothermelModel.H>

namespace ERFFire
{

/**
 * Resultant of the dimensionless Rothermel wind and slope forcing vectors.
 *
 * wind_push_unit is the direction the wind pushes the fire, not the
 * meteorological direction from which the wind originates. upslope_unit is
 * the direction of increasing elevation. Both directions must already be
 * expressed in the same orthonormal two-dimensional basis. ERF-Fire uses the
 * horizontal map-plane x/y basis: wind is the sampled horizontal push vector,
 * while upslope is the normalized horizontal terrain gradient. The Fire
 * perimeter, normals, burned area, and propagation rates remain map-plane
 * quantities; no terrain-surface arclength rescaling is applied here.
 *
 * Only the resultant vector is stored. Its magnitude, preferred-heading state,
 * and unit heading are derived accessors so they cannot drift out of
 * consistency with one another.
 */
struct RothermelWindSlopeVector
{
    FireVec2 resultant_vector{};

    [[nodiscard]] amrex::Real resultant_factor() const noexcept
    {
        return norm(resultant_vector);
    }

    [[nodiscard]] bool has_preferred_heading() const noexcept
    {
        return resultant_vector.x != amrex::Real(0.0)
            || resultant_vector.y != amrex::Real(0.0);
    }

    [[nodiscard]] FireVec2 heading_unit() const noexcept
    {
        const amrex::Real magnitude = resultant_factor();
        return magnitude > amrex::Real(0.0)
            ? resultant_vector / magnitude
            : FireVec2{1.0, 0.0};
    }
};

/**
 * Directional spread state obtained by coupling validated Rothermel scalar
 * behavior to the validated Richards/FARSITE ellipse geometry.
 *
 * The only stored physical spread geometry is ellipse. The forcing vector is
 * retained as an audit diagnostic; heading, heading ROS, and effective wind
 * are available from ellipse without duplicate storage.
 */
struct RichardsDirectionalSpread
{
    RothermelWindSlopeVector forcing{};
    RichardsEllipse ellipse{};
};

/**
 * Vector the Rothermel dimensionless wind and slope factors.
 *
 * The wind component points in wind_push_unit and the slope component points
 * upslope. If the components cancel within floating-point tolerance, the
 * stored resultant is set exactly to zero. has_preferred_heading() then returns
 * false and heading_unit() returns canonical +x; this arbitrary heading is
 * harmless because the corresponding effective wind is zero and the ellipse
 * is isotropic.
 */
[[nodiscard]] RothermelWindSlopeVector combine_rothermel_wind_slope_factors(
    amrex::Real wind_factor,
    const FireVec2& wind_push_unit,
    amrex::Real slope_factor,
    const FireVec2& upslope_unit);

/**
 * Build the directional Richards/FARSITE wavelet from a Rothermel result.
 *
 * The resultant factor magnitude sets the heading spread rate
 *
 *   R_head = R0 (1 + |Phi_w_vec + Phi_s_vec|).
 *
 * The effective midflame wind used only for ellipse shape is obtained by
 * inverting the same audited Rothermel wind-factor relation.
 */
[[nodiscard]] RichardsDirectionalSpread make_richards_directional_spread(
    const RothermelResult& behavior,
    const FireVec2& wind_push_unit,
    const FireVec2& upslope_unit);

} // namespace ERFFire

#endif // ERF_RICHARDS_DIRECTIONAL_SPREAD_H
