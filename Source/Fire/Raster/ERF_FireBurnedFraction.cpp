#include <ERF_FireBurnedFraction.H>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace ERFFire
{
namespace
{

void
require_unit_fraction (
    amrex::Real value,
    const char* description)
{
    if (!std::isfinite(value)
        || value < amrex::Real(0.0)
        || value > amrex::Real(1.0)) {
        throw std::invalid_argument(
            std::string("Fire ") + description
            + " must be finite and lie in [0,1]");
    }
}

} // namespace

FireBurnedFractionUpdate
update_fire_burned_fraction (
    amrex::Real previous_burned_fraction,
    amrex::Real current_coverage_fraction)
{
    require_unit_fraction(
        previous_burned_fraction,
        "previous burned fraction");
    require_unit_fraction(
        current_coverage_fraction,
        "current coverage fraction");

    const amrex::Real burned_fraction = std::max(
        previous_burned_fraction,
        current_coverage_fraction);

    return {
        burned_fraction,
        burned_fraction - previous_burned_fraction
    };
}

} // namespace ERFFire
