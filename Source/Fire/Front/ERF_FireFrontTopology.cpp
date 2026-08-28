#include <ERF_FireFrontTopology.H>
#include <ERF_FireGeometry.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ERFFire
{
namespace
{

struct BoundaryLocation
{
    std::size_t edge_index{};
    amrex::Real fraction{};
    FireVec2 point{};
};

BoundaryLocation
make_boundary_location (
    const FirePerimeter& perimeter,
    std::size_t edge_index,
    amrex::Real fraction)
{
    const std::size_t count = perimeter.size();

    if (edge_index >= count) {
        throw std::out_of_range(
            "Fire perimeter pinch edge index out of range");
    }
    if (!std::isfinite(fraction)
        || fraction < amrex::Real(0.0)
        || fraction > amrex::Real(1.0)) {
        throw std::invalid_argument(
            "Fire perimeter pinch fraction must lie in [0,1]");
    }

    const auto& vertices = perimeter.vertices_m();

    if (fraction == amrex::Real(1.0)) {
        const std::size_t next = (edge_index + 1) % count;
        return {
            next,
            amrex::Real(0.0),
            vertices[next]
        };
    }

    const FireVec2& start = vertices[edge_index];
    if (fraction == amrex::Real(0.0)) {
        return {
            edge_index,
            fraction,
            start
        };
    }

    const FireVec2& end =
        vertices[(edge_index + 1) % count];

    return {
        edge_index,
        fraction,
        start + fraction * (end - start)
    };
}

bool
same_location (
    const BoundaryLocation& first,
    const BoundaryLocation& second) noexcept
{
    return first.edge_index == second.edge_index
        && first.fraction == second.fraction;
}

void
append_if_distinct (
    std::vector<FireVec2>& vertices,
    const FireVec2& point)
{
    if (vertices.empty()
        || vertices.back().x != point.x
        || vertices.back().y != point.y) {
        vertices.push_back(point);
    }
}

std::vector<FireVec2>
boundary_path (
    const FirePerimeter& perimeter,
    const BoundaryLocation& start,
    const BoundaryLocation& end)
{
    const auto& vertices = perimeter.vertices_m();
    const std::size_t count = vertices.size();

    std::vector<FireVec2> path;
    path.reserve(count + 2);
    path.push_back(start.point);

    if (start.edge_index == end.edge_index
        && start.fraction < end.fraction) {
        append_if_distinct(path, end.point);
        return path;
    }

    std::size_t edge = start.edge_index;
    append_if_distinct(
        path,
        vertices[(edge + 1) % count]);
    edge = (edge + 1) % count;

    std::size_t traversed = 1;
    while (edge != end.edge_index) {
        if (traversed >= count) {
            throw std::logic_error(
                "Fire perimeter pinch traversal failed to close");
        }

        append_if_distinct(
            path,
            vertices[(edge + 1) % count]);
        edge = (edge + 1) % count;
        ++traversed;
    }

    append_if_distinct(path, end.point);
    return path;
}

void
require_area_partition (
    amrex::Real original_area_m2,
    amrex::Real first_signed_area_m2,
    amrex::Real second_signed_area_m2)
{
    const amrex::Real partition_area_m2 =
        first_signed_area_m2 + second_signed_area_m2;
    const amrex::Real scale = std::max({
        std::abs(original_area_m2),
        std::abs(first_signed_area_m2),
        std::abs(second_signed_area_m2),
        amrex::Real(1.0)
    });
    const amrex::Real tolerance =
        amrex::Real(4096.0)
        * std::numeric_limits<amrex::Real>::epsilon()
        * scale;

    if (!std::isfinite(partition_area_m2)
        || std::abs(partition_area_m2 - original_area_m2)
            > tolerance) {
        throw std::runtime_error(
            "Fire perimeter pinch split does not preserve signed area");
    }
}

} // namespace

FireFront
split_perimeter_at_pinch (
    const FirePerimeter& perimeter,
    const FirePerimeterPinch& pinch)
{
    const BoundaryLocation first =
        make_boundary_location(
            perimeter,
            pinch.first_edge_index,
            pinch.first_edge_fraction);
    const BoundaryLocation second =
        make_boundary_location(
            perimeter,
            pinch.second_edge_index,
            pinch.second_edge_fraction);

    if (same_location(first, second)) {
        throw std::invalid_argument(
            "Fire perimeter pinch locations must be distinct");
    }

    std::vector<FireVec2> first_path =
        boundary_path(perimeter, first, second);
    std::vector<FireVec2> second_path =
        boundary_path(perimeter, second, first);

    if (first_path.size() < 3 || second_path.size() < 3) {
        throw std::invalid_argument(
            "Fire perimeter pinch must create two nondegenerate loops");
    }

    const amrex::Real first_signed_area_m2 =
        detail::signed_polygon_area_m2(first_path);
    const amrex::Real second_signed_area_m2 =
        detail::signed_polygon_area_m2(second_path);

    if (!std::isfinite(first_signed_area_m2)
        || !std::isfinite(second_signed_area_m2)
        || first_signed_area_m2 == amrex::Real(0.0)
        || second_signed_area_m2 == amrex::Real(0.0)) {
        throw std::invalid_argument(
            "Fire perimeter pinch must create two finite nonzero-area loops");
    }

    if ((first_signed_area_m2 > amrex::Real(0.0))
        == (second_signed_area_m2 > amrex::Real(0.0))) {
        throw std::invalid_argument(
            "Fire perimeter pinch must create one outer loop and one hole");
    }

    require_area_partition(
        perimeter.signed_area_m2(),
        first_signed_area_m2,
        second_signed_area_m2);

    std::vector<FireFrontComponent> components;
    components.reserve(2);

    if (first_signed_area_m2 > amrex::Real(0.0)) {
        components.push_back({
            FireFrontRole::Outer,
            FirePerimeter(std::move(first_path))
        });
        components.push_back({
            FireFrontRole::Hole,
            FirePerimeter(std::move(second_path))
        });
    } else {
        components.push_back({
            FireFrontRole::Outer,
            FirePerimeter(std::move(second_path))
        });
        components.push_back({
            FireFrontRole::Hole,
            FirePerimeter(std::move(first_path))
        });
    }

    return FireFront(std::move(components));
}

FireFront
split_perimeter_at_pinch (
    const std::vector<FireVec2>& vertices_m,
    const FirePerimeterPinch& pinch)
{
    if (vertices_m.size() < 3) {
        throw std::invalid_argument(
            "Fire raw-event perimeter requires at least three vertices");
    }

    for (const FireVec2& vertex : vertices_m) {
        if (!std::isfinite(vertex.x)
            || !std::isfinite(vertex.y)) {
            throw std::invalid_argument(
                "Fire raw-event perimeter vertices must be finite");
        }
    }

    const std::size_t count = vertices_m.size();

    struct RawBoundaryLocation
    {
        std::size_t edge_index{};
        amrex::Real fraction{};
        FireVec2 point{};
    };

    const auto make_location =
        [&] (
            std::size_t edge_index,
            amrex::Real fraction)
        {
            if (edge_index >= count) {
                throw std::out_of_range(
                    "Fire raw-event pinch edge index out of range");
            }

            if (!std::isfinite(fraction)
                || fraction < amrex::Real(0.0)
                || fraction > amrex::Real(1.0)) {
                throw std::invalid_argument(
                    "Fire raw-event pinch fraction must lie in [0,1]");
            }

            if (fraction == amrex::Real(1.0)) {
                const std::size_t next =
                    (edge_index + 1) % count;
                return RawBoundaryLocation{
                    next,
                    amrex::Real(0.0),
                    vertices_m[next]
                };
            }

            const FireVec2& start =
                vertices_m[edge_index];

            if (fraction == amrex::Real(0.0)) {
                return RawBoundaryLocation{
                    edge_index,
                    fraction,
                    start
                };
            }

            const FireVec2& end =
                vertices_m[(edge_index + 1) % count];

            return RawBoundaryLocation{
                edge_index,
                fraction,
                start + fraction * (end - start)
            };
        };

    RawBoundaryLocation first =
        make_location(
            pinch.first_edge_index,
            pinch.first_edge_fraction);
    RawBoundaryLocation second =
        make_location(
            pinch.second_edge_index,
            pinch.second_edge_fraction);

    if (first.edge_index == second.edge_index
        && first.fraction == second.fraction) {
        throw std::invalid_argument(
            "Fire raw-event pinch locations must be distinct");
    }

    const auto edge_length =
        [&] (const RawBoundaryLocation& location)
        {
            return norm(
                vertices_m[
                    (location.edge_index + 1) % count]
                - vertices_m[location.edge_index]);
        };

    const amrex::Real geometric_scale =
        std::max({
            edge_length(first),
            edge_length(second),
            amrex::Real(1.0)
        });
    const amrex::Real contact_tolerance_m =
        amrex::Real(4096.0)
        * std::numeric_limits<amrex::Real>::epsilon()
        * geometric_scale;

    if (norm(first.point - second.point)
        > contact_tolerance_m) {
        throw std::invalid_argument(
            "Fire raw-event pinch locations do not coincide");
    }

    const FireVec2 contact{
        amrex::Real(0.5)
            * (first.point.x + second.point.x),
        amrex::Real(0.5)
            * (first.point.y + second.point.y)
    };

    first.point = contact;
    second.point = contact;

    const auto append_if_distinct =
        [] (
            std::vector<FireVec2>& path,
            const FireVec2& point)
        {
            if (path.empty()
                || path.back().x != point.x
                || path.back().y != point.y) {
                path.push_back(point);
            }
        };

    const auto boundary_path =
        [&] (
            const RawBoundaryLocation& start,
            const RawBoundaryLocation& end)
        {
            std::vector<FireVec2> path;
            path.reserve(count + 2);
            path.push_back(start.point);

            if (start.edge_index == end.edge_index
                && start.fraction < end.fraction) {
                append_if_distinct(
                    path,
                    end.point);
            } else {
                std::size_t vertex_index =
                    (start.edge_index + 1) % count;
                std::size_t traversed = 0;

                while (vertex_index
                       != end.edge_index) {
                    if (traversed >= count) {
                        throw std::logic_error(
                            "Fire raw-event pinch traversal failed to close");
                    }

                    append_if_distinct(
                        path,
                        vertices_m[vertex_index]);

                    vertex_index =
                        (vertex_index + 1) % count;
                    ++traversed;
                }

                if (end.fraction
                    != amrex::Real(0.0)) {
                    append_if_distinct(
                        path,
                        vertices_m[end.edge_index]);
                }

                append_if_distinct(
                    path,
                    end.point);
            }

            if (path.size() > 1
                && path.front().x == path.back().x
                && path.front().y == path.back().y) {
                path.pop_back();
            }

            return path;
        };

    std::vector<FireVec2> first_path =
        boundary_path(first, second);
    std::vector<FireVec2> second_path =
        boundary_path(second, first);

    if (first_path.size() < 3
        || second_path.size() < 3) {
        throw std::invalid_argument(
            "Fire raw-event pinch must create two nondegenerate loops");
    }

    const amrex::Real raw_signed_area_m2 =
        detail::signed_polygon_area_m2(vertices_m);
    const amrex::Real first_signed_area_m2 =
        detail::signed_polygon_area_m2(first_path);
    const amrex::Real second_signed_area_m2 =
        detail::signed_polygon_area_m2(second_path);

    if (!std::isfinite(raw_signed_area_m2)
        || !std::isfinite(first_signed_area_m2)
        || !std::isfinite(second_signed_area_m2)
        || raw_signed_area_m2 <= amrex::Real(0.0)
        || first_signed_area_m2 == amrex::Real(0.0)
        || second_signed_area_m2 == amrex::Real(0.0)) {
        throw std::invalid_argument(
            "Fire raw-event pinch produced invalid signed area");
    }

    if ((first_signed_area_m2 > amrex::Real(0.0))
        == (second_signed_area_m2 > amrex::Real(0.0))) {
        throw std::invalid_argument(
            "Fire raw-event pinch must create one outer loop and one hole");
    }

    require_area_partition(
        raw_signed_area_m2,
        first_signed_area_m2,
        second_signed_area_m2);

    std::vector<FireFrontComponent> components;
    components.reserve(2);

    if (first_signed_area_m2 > amrex::Real(0.0)) {
        components.push_back({
            FireFrontRole::Outer,
            FirePerimeter(std::move(first_path))
        });
        components.push_back({
            FireFrontRole::Hole,
            FirePerimeter(std::move(second_path))
        });
    } else {
        components.push_back({
            FireFrontRole::Outer,
            FirePerimeter(std::move(second_path))
        });
        components.push_back({
            FireFrontRole::Hole,
            FirePerimeter(std::move(first_path))
        });
    }

    return FireFront(std::move(components));
}

} // namespace ERFFire
