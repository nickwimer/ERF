#include <ERF_RichardsDirectionalSpread.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace ERFFire
{
namespace
{

bool
finite (amrex::Real value) noexcept
{
    return std::isfinite(value);
}

void
require_finite_vector (const FireVec2& value, const char* description)
{
    if (!finite(value.x) || !finite(value.y)) {
        throw std::invalid_argument(
            std::string("Rothermel/Richards ") + description + " must be finite");
    }
}

void
require_unit_vector_if_active (
    const FireVec2& value,
    amrex::Real factor,
    const char* description)
{
    require_finite_vector(value, description);
    if (factor == amrex::Real(0.0)) {
        return;
    }

    const amrex::Real length_squared = norm_squared(value);
    const amrex::Real tolerance =
        amrex::Real(128.0) * std::numeric_limits<amrex::Real>::epsilon();

    if (std::abs(length_squared - amrex::Real(1.0)) > tolerance) {
        throw std::invalid_argument(
            std::string("Rothermel/Richards active ") + description
            + " must be a unit vector");
    }
}

} // namespace

RothermelWindSlopeVector
combine_rothermel_wind_slope_factors (
    amrex::Real wind_factor,
    const FireVec2& wind_push_unit,
    amrex::Real slope_factor,
    const FireVec2& upslope_unit)
{
    if (!finite(wind_factor) || wind_factor < amrex::Real(0.0)) {
        throw std::invalid_argument(
            "Rothermel/Richards wind factor must be finite and non-negative");
    }
    if (!finite(slope_factor) || slope_factor < amrex::Real(0.0)) {
        throw std::invalid_argument(
            "Rothermel/Richards slope factor must be finite and non-negative");
    }

    require_unit_vector_if_active(
        wind_push_unit, wind_factor, "wind-push direction");
    require_unit_vector_if_active(
        upslope_unit, slope_factor, "upslope direction");

    const FireVec2 resultant =
        wind_push_unit * wind_factor + upslope_unit * slope_factor;
    const amrex::Real magnitude = norm(resultant);

    const amrex::Real scale = std::max(
        amrex::Real(1.0), wind_factor + slope_factor);
    const amrex::Real cancellation_tolerance =
        amrex::Real(128.0)
        * std::numeric_limits<amrex::Real>::epsilon()
        * scale;

    if (magnitude <= cancellation_tolerance) {
        return {{0.0, 0.0}};
    }

    return {resultant};
}

RichardsDirectionalSpread
make_richards_directional_spread (
    const RothermelResult& behavior,
    const FireVec2& wind_push_unit,
    const FireVec2& upslope_unit)
{
    if (!finite(behavior.no_wind_no_slope_ros_mps)
        || behavior.no_wind_no_slope_ros_mps < amrex::Real(0.0)) {
        throw std::invalid_argument(
            "Rothermel/Richards no-wind no-slope ROS must be finite and non-negative");
    }

    const auto forcing = combine_rothermel_wind_slope_factors(
        behavior.wind_factor,
        wind_push_unit,
        behavior.slope_factor,
        upslope_unit);

    const amrex::Real resultant_factor = forcing.resultant_factor();
    const FireVec2 heading_unit = forcing.heading_unit();

    const amrex::Real effective_wind_mps =
        rothermel_model_wind_speed_for_factor_mps(
            behavior, resultant_factor);

    const amrex::Real heading_ros_mps =
        behavior.no_wind_no_slope_ros_mps
        * (amrex::Real(1.0) + resultant_factor);

    if (!finite(heading_ros_mps)) {
        throw std::invalid_argument(
            "Rothermel/Richards heading ROS is not finite");
    }

    return {
        forcing,
        make_richards_ellipse(
            heading_unit,
            heading_ros_mps,
            effective_wind_mps)
    };
}

} // namespace ERFFire
