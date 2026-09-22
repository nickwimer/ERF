#ifndef ERF_FIRE_TERRAIN_METRIC_H_
#define ERF_FIRE_TERRAIN_METRIC_H_

#include <ERF_RichardsEllipse.H>

#include <cmath>
#include <limits>
#include <stdexcept>

namespace ERFFire
{

/**
 * Orthonormal local-surface coordinates for z = h(x,y), with gradient g.
 *
 * G = I + g g^T is the map-plane metric; S = sqrt(G) lifts map displacement
 * into surface coordinates and B = inverse(S) projects it back. For a local
 * surface wavelet K, its horizontal support is h_(B K)(n) = h_K(B n).
 * Transforming the normal as a covector is essential: dividing a directional
 * rate by sqrt(1 + (g dot n)^2) is not the general anisotropic transform.
 *
 * Finney (1998, revised 2004), RMRS-RP-4, pp. 5-6, requires propagation and
 * wind/slope angles to be calculated on the local surface, then mapped back.
 * The adapter lifts the horizontal wind AZIMUTH and preserves the supplied
 * model/midflame wind magnitude. It does not infer an atmospheric vertical
 * wind. Fuel loading and feedback continue to use horizontal map-plane area.
 */
class FireTerrainMetric
{
public:
    explicit FireTerrainMetric(const FireVec2& gradient)
    {
        if (!std::isfinite(gradient.x) || !std::isfinite(gradient.y)) {
            throw std::invalid_argument("Fire terrain gradient must be finite");
        }
        const amrex::Real slope = std::hypot(gradient.x, gradient.y);
        stretch_ = std::hypot(amrex::Real(1), slope);
        if (!std::isfinite(stretch_)) {
            throw std::overflow_error("Fire terrain metric is not representable");
        }
        if (slope > amrex::Real(0)) {
            upslope_ = gradient / slope;
        }
    }

    [[nodiscard]] FireVec2 surface_vector(const FireVec2& v) const
    {
        return transform(v, stretch_);
    }

    [[nodiscard]] FireVec2 map_vector(const FireVec2& v) const
    {
        return transform(v, amrex::Real(1) / stretch_);
    }

    /** Lift an azimuth, not its magnitude, into orthonormal surface axes. */
    [[nodiscard]] FireVec2 surface_unit_direction(const FireVec2& unit) const
    {
        require_unit(unit);
        if (stretch_ == amrex::Real(1)) { return unit; }
        const FireVec2 lifted = surface_vector(unit);
        const amrex::Real length = std::hypot(lifted.x, lifted.y);
        if (!std::isfinite(length) || !(length > amrex::Real(0))) {
            throw std::overflow_error("Fire surface direction is not representable");
        }
        return lifted / length;
    }

    /** Support of a surface ellipse in the requested MAP-plane normal. */
    [[nodiscard]] amrex::Real normal_speed_mps(
        const RichardsEllipse& surface_ellipse,
        const FireVec2& map_normal) const
    {
        require_unit(map_normal);
        if (stretch_ == amrex::Real(1)) {
            return richards_normal_speed_mps(surface_ellipse, map_normal);
        }
        const FireVec2 covector = map_vector(map_normal);
        const amrex::Real length = std::hypot(covector.x, covector.y);
        if (!std::isfinite(length) || !(length > amrex::Real(0))) {
            throw std::overflow_error("Fire surface normal is not representable");
        }
        return length * richards_normal_speed_mps(
            surface_ellipse, covector / length);
    }

private:
    static void require_unit(const FireVec2& v)
    {
        const amrex::Real length = std::hypot(v.x, v.y);
        const amrex::Real tolerance = amrex::Real(128)
            * std::numeric_limits<amrex::Real>::epsilon();
        if (!std::isfinite(length)
            || std::abs(length - amrex::Real(1)) > tolerance) {
            throw std::invalid_argument("Fire terrain direction must be a unit vector");
        }
    }

    [[nodiscard]] FireVec2 transform(const FireVec2& v, amrex::Real scale) const
    {
        if (!std::isfinite(v.x) || !std::isfinite(v.y)) {
            throw std::invalid_argument("Fire terrain vector must be finite");
        }
        if (stretch_ == amrex::Real(1)) { return v; }
        const FireVec2 contour{-upslope_.y, upslope_.x};
        const FireVec2 result = upslope_ * (scale * dot(upslope_, v))
            + contour * dot(contour, v);
        if (!std::isfinite(result.x) || !std::isfinite(result.y)) {
            throw std::overflow_error("Fire transformed vector is not representable");
        }
        return result;
    }

    FireVec2 upslope_{amrex::Real(1), amrex::Real(0)};
    amrex::Real stretch_{amrex::Real(1)};
};

} // namespace ERFFire

#endif
