#include <ERF_FireCellCoverage.H>

#include <ERF_FireGeometry.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ERFFire
{
namespace
{

bool
finite (amrex::Real value) noexcept
{
    return std::isfinite(value);
}

struct ValidatedCell
{
    amrex::Real width_m;
    amrex::Real height_m;
    amrex::Real area_m2;
};

ValidatedCell
validate_cell (const FireCartesianCell2D& cell)
{
    if (!finite(cell.xlo_m)
        || !finite(cell.xhi_m)
        || !finite(cell.ylo_m)
        || !finite(cell.yhi_m)) {
        throw std::invalid_argument(
            "Fire raster cell bounds must be finite");
    }

    if (!(cell.xhi_m > cell.xlo_m)
        || !(cell.yhi_m > cell.ylo_m)) {
        throw std::invalid_argument(
            "Fire raster cell must have strictly positive width and height");
    }

    const amrex::Real width_m = cell.xhi_m - cell.xlo_m;
    const amrex::Real height_m = cell.yhi_m - cell.ylo_m;
    const amrex::Real area_m2 = width_m * height_m;

    if (!finite(width_m)
        || !finite(height_m)
        || !finite(area_m2)
        || area_m2 <= amrex::Real(0.0)) {
        throw std::overflow_error(
            "Fire raster cell dimensions or area are not finite");
    }

    return {width_m, height_m, area_m2};
}

enum class ClipBoundary
{
    xlo,
    xhi,
    ylo,
    yhi
};

bool
inside (
    const FireVec2& point,
    ClipBoundary boundary,
    amrex::Real value) noexcept
{
    switch (boundary) {
    case ClipBoundary::xlo:
        return point.x >= value;
    case ClipBoundary::xhi:
        return point.x <= value;
    case ClipBoundary::ylo:
        return point.y >= value;
    case ClipBoundary::yhi:
        return point.y <= value;
    }

    return false;
}

FireVec2
boundary_intersection (
    const FireVec2& first,
    const FireVec2& second,
    ClipBoundary boundary,
    amrex::Real value)
{
    if (boundary == ClipBoundary::xlo
        || boundary == ClipBoundary::xhi) {
        const amrex::Real dx = second.x - first.x;
        if (dx == amrex::Real(0.0)) {
            throw std::runtime_error(
                "Fire polygon clipping encountered an invalid vertical crossing");
        }

        const amrex::Real t = (value - first.x) / dx;
        return {
            value,
            first.y + t * (second.y - first.y)
        };
    }

    const amrex::Real dy = second.y - first.y;
    if (dy == amrex::Real(0.0)) {
        throw std::runtime_error(
            "Fire polygon clipping encountered an invalid horizontal crossing");
    }

    const amrex::Real t = (value - first.y) / dy;
    return {
        first.x + t * (second.x - first.x),
        value
    };
}

std::vector<FireVec2>
clip_against_boundary (
    const std::vector<FireVec2>& input,
    ClipBoundary boundary,
    amrex::Real value)
{
    if (input.empty()) {
        return {};
    }

    std::vector<FireVec2> output;
    output.reserve(input.size() + 2);

    FireVec2 previous = input.back();
    bool previous_inside = inside(previous, boundary, value);

    for (const FireVec2& current : input) {
        const bool current_inside = inside(current, boundary, value);

        if (current_inside) {
            if (!previous_inside) {
                output.push_back(
                    boundary_intersection(
                        previous, current, boundary, value));
            }
            output.push_back(current);
        } else if (previous_inside) {
            output.push_back(
                boundary_intersection(
                    previous, current, boundary, value));
        }

        previous = current;
        previous_inside = current_inside;
    }

    return output;
}

amrex::Real
intersection_area_local_m2 (
    const FirePerimeter& perimeter,
    const ValidatedCell& cell,
    amrex::Real origin_x_m,
    amrex::Real origin_y_m)
{
    std::vector<FireVec2> clipped;
    clipped.reserve(perimeter.size());

    for (const FireVec2& vertex : perimeter.vertices_m()) {
        clipped.push_back({
            vertex.x - origin_x_m,
            vertex.y - origin_y_m
        });
    }

    clipped = clip_against_boundary(
        clipped, ClipBoundary::xlo, amrex::Real(0.0));
    clipped = clip_against_boundary(
        clipped, ClipBoundary::xhi, cell.width_m);
    clipped = clip_against_boundary(
        clipped, ClipBoundary::ylo, amrex::Real(0.0));
    clipped = clip_against_boundary(
        clipped, ClipBoundary::yhi, cell.height_m);

    if (clipped.size() < 3) {
        return amrex::Real(0.0);
    }

    const amrex::Real signed_area_m2 =
        detail::signed_polygon_area_m2(clipped);

    if (!finite(signed_area_m2)) {
        throw std::overflow_error(
            "Fire polygon-cell intersection area is not finite");
    }

    const amrex::Real tolerance =
        amrex::Real(1024.0)
        * std::numeric_limits<amrex::Real>::epsilon();
    const amrex::Real area_tolerance_m2 =
        tolerance * cell.area_m2;

    // FirePerimeter is normalized CCW. Sequential clipping against convex
    // half-planes preserves the positive winding of the retained region.
    // A materially negative signed area therefore signals a clipping/topology
    // error and must not be hidden with std::abs().
    if (signed_area_m2 < -area_tolerance_m2) {
        throw std::runtime_error(
            "Fire polygon-cell clipping reversed signed area");
    }

    const amrex::Real area_m2 =
        std::max(signed_area_m2, amrex::Real(0.0));

    if (area_m2 > cell.area_m2 * (amrex::Real(1.0) + tolerance)) {
        throw std::runtime_error(
            "Fire polygon-cell intersection exceeds cell area");
    }

    return std::min(area_m2, cell.area_m2);
}

} // namespace

amrex::Real
fire_perimeter_cell_intersection_area_m2 (
    const FirePerimeter& perimeter,
    const FireCartesianCell2D& cell)
{
    const ValidatedCell validated = validate_cell(cell);
    return intersection_area_local_m2(
        perimeter,
        validated,
        cell.xlo_m,
        cell.ylo_m);
}

amrex::Real
fire_perimeter_cell_coverage_fraction (
    const FirePerimeter& perimeter,
    const FireCartesianCell2D& cell)
{
    const ValidatedCell validated = validate_cell(cell);
    const amrex::Real area_m2 = intersection_area_local_m2(
        perimeter,
        validated,
        cell.xlo_m,
        cell.ylo_m);

    const amrex::Real fraction = area_m2 / validated.area_m2;
    const amrex::Real tolerance =
        amrex::Real(1024.0)
        * std::numeric_limits<amrex::Real>::epsilon();

    if (!finite(fraction)
        || fraction < -tolerance
        || fraction > amrex::Real(1.0) + tolerance) {
        throw std::runtime_error(
            "Fire polygon-cell coverage fraction is outside [0,1]");
    }

    return std::clamp(
        fraction,
        amrex::Real(0.0),
        amrex::Real(1.0));
}

} // namespace ERFFire
