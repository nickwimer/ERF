#include "ERF_FirePerimeterSweep.H"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ERFFire
{

FirePerimeter
interpolate_fire_perimeter_linear_sweep(
    const FirePerimeter& start_perimeter,
    const FirePerimeter& end_perimeter,
    amrex::Real alpha)
{
    if (start_perimeter.size() != end_perimeter.size()) {
        throw std::invalid_argument(
            "Fire linear perimeter sweep requires matching vertex counts");
    }

    if (!std::isfinite(alpha)
        || alpha < amrex::Real(0)
        || alpha > amrex::Real(1)) {
        throw std::invalid_argument(
            "Fire linear perimeter sweep alpha must be finite in [0,1]");
    }

    if (alpha == amrex::Real(0)) {
        return start_perimeter;
    }
    if (alpha == amrex::Real(1)) {
        return end_perimeter;
    }

    std::vector<FireVec2> vertices;
    vertices.reserve(start_perimeter.size());

    const auto& start = start_perimeter.vertices_m();
    const auto& end = end_perimeter.vertices_m();

    for (std::size_t i = 0; i < start.size(); ++i) {
        vertices.push_back(
            start[i] + (end[i] - start[i]) * alpha);
    }

    return FirePerimeter(std::move(vertices));
}

} // namespace ERFFire
