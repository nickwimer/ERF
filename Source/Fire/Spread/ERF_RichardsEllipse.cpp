#include <ERF_RichardsEllipse.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace ERFFire
{
namespace
{

constexpr amrex::Real maximum_length_to_breadth = 8.0;

bool
finite (amrex::Real value) noexcept
{
    return std::isfinite(value);
}

void
require_unit_vector (const FireVec2& value, const char* description)
{
    if (!finite(value.x) || !finite(value.y)) {
        throw std::invalid_argument(
            std::string("Richards ") + description + " must be finite");
    }

    const amrex::Real length_squared = norm_squared(value);
    const amrex::Real tolerance =
        amrex::Real(128.0) * std::numeric_limits<amrex::Real>::epsilon();

    if (std::abs(length_squared - amrex::Real(1.0)) > tolerance) {
        throw std::invalid_argument(
            std::string("Richards ") + description + " must be a unit vector");
    }
}

} // namespace

amrex::Real
farsite_unclamped_length_to_breadth (
    amrex::Real effective_midflame_wind_mps)
{
    if (!finite(effective_midflame_wind_mps)
        || effective_midflame_wind_mps < amrex::Real(0.0)) {
        throw std::invalid_argument(
            "FARSITE effective midflame wind must be finite and non-negative");
    }

    // Finney (1998, revised 2004), equation 13. The -0.397 modification makes
    // LB exactly 1 at U=0; Finney documents U in m/s for this formulation.
    const amrex::Real length_to_breadth =
        amrex::Real(0.936)
            * std::exp(amrex::Real(0.2566) * effective_midflame_wind_mps)
        + amrex::Real(0.461)
            * std::exp(amrex::Real(-0.1548) * effective_midflame_wind_mps)
        - amrex::Real(0.397);

    if (!finite(length_to_breadth)) {
        throw std::overflow_error(
            "FARSITE length-to-breadth relation overflowed");
    }

    return length_to_breadth;
}

RichardsEllipse
make_richards_ellipse (
    const FireVec2& heading_unit,
    amrex::Real heading_ros_mps,
    amrex::Real effective_midflame_wind_mps)
{
    require_unit_vector(heading_unit, "heading direction");

    if (!finite(heading_ros_mps) || heading_ros_mps < amrex::Real(0.0)) {
        throw std::invalid_argument(
            "Richards heading spread rate must be finite and non-negative");
    }

    const amrex::Real unclamped_lb =
        farsite_unclamped_length_to_breadth(effective_midflame_wind_mps);
    const amrex::Real lb = std::clamp(
        unclamped_lb,
        amrex::Real(1.0),
        maximum_length_to_breadth);

    const amrex::Real root = std::sqrt(
        std::max(amrex::Real(0.0), lb * lb - amrex::Real(1.0)));

    // Finney equation 14. Algebraically, (LB-root)(LB+root)=1, so the
    // squared form avoids subtractive cancellation while remaining identical.
    const amrex::Real head_to_back = (lb + root) * (lb + root);
    const amrex::Real backing_ros = heading_ros_mps / head_to_back;

    // Finney equations 15-17.
    const amrex::Real a =
        amrex::Real(0.5) * (heading_ros_mps + backing_ros) / lb;
    const amrex::Real b =
        amrex::Real(0.5) * (heading_ros_mps + backing_ros);
    const amrex::Real c = b - backing_ros;

    const amrex::Real eccentricity = root / lb;

    return {
        heading_unit,
        heading_ros_mps,
        effective_midflame_wind_mps,
        unclamped_lb,
        lb,
        head_to_back,
        eccentricity,
        a,
        b,
        c
    };
}

amrex::Real
richards_normal_speed_mps (
    const RichardsEllipse& ellipse,
    const FireVec2& outward_normal_unit)
{
    require_unit_vector(ellipse.heading_unit, "ellipse heading direction");
    require_unit_vector(outward_normal_unit, "outward normal");

    if (ellipse.semi_minor_rate_mps == amrex::Real(0.0)
        && ellipse.semi_major_rate_mps == amrex::Real(0.0)
        && ellipse.center_translation_rate_mps == amrex::Real(0.0)) {
        return amrex::Real(0.0);
    }

    if (!finite(ellipse.semi_minor_rate_mps)
        || !finite(ellipse.semi_major_rate_mps)
        || !finite(ellipse.center_translation_rate_mps)
        || ellipse.semi_minor_rate_mps <= amrex::Real(0.0)
        || ellipse.semi_major_rate_mps <= amrex::Real(0.0)
        || ellipse.center_translation_rate_mps < amrex::Real(0.0)
        || ellipse.center_translation_rate_mps >= ellipse.semi_major_rate_mps) {
        throw std::invalid_argument("Richards ellipse rates are invalid");
    }

    const amrex::Real mu = std::clamp(
        dot(ellipse.heading_unit, outward_normal_unit),
        amrex::Real(-1.0),
        amrex::Real(1.0));
    const amrex::Real mu_squared = mu * mu;
    const amrex::Real transverse_squared =
        std::max(amrex::Real(0.0), amrex::Real(1.0) - mu_squared);

    const amrex::Real support = std::sqrt(
        ellipse.semi_major_rate_mps * ellipse.semi_major_rate_mps * mu_squared
        + ellipse.semi_minor_rate_mps * ellipse.semi_minor_rate_mps
            * transverse_squared);

    return ellipse.center_translation_rate_mps * mu + support;
}

} // namespace ERFFire
