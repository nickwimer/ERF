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

struct RawPinchSplit
{
    std::vector<FireVec2> first_path;
    std::vector<FireVec2> second_path;
    amrex::Real raw_signed_area_m2{};
    amrex::Real first_signed_area_m2{};
    amrex::Real second_signed_area_m2{};
};

RawPinchSplit
split_raw_perimeter_at_pinch (
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
                append_if_distinct(path, end.point);
            } else {
                std::size_t vertex_index =
                    (start.edge_index + 1) % count;
                std::size_t traversed = 0;

                while (vertex_index != end.edge_index) {
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

                if (end.fraction != amrex::Real(0.0)) {
                    append_if_distinct(
                        path,
                        vertices_m[end.edge_index]);
                }

                append_if_distinct(path, end.point);
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

    const amrex::Real raw_signed_area_m2 =
        detail::signed_polygon_area_m2(vertices_m);
    const amrex::Real first_signed_area_m2 =
        detail::signed_polygon_area_m2(first_path);
    const amrex::Real second_signed_area_m2 =
        detail::signed_polygon_area_m2(second_path);

    if (!std::isfinite(raw_signed_area_m2)
        || !std::isfinite(first_signed_area_m2)
        || !std::isfinite(second_signed_area_m2)
        || raw_signed_area_m2 <= amrex::Real(0.0)) {
        throw std::invalid_argument(
            "Fire raw-event pinch produced invalid signed area");
    }

    const bool first_degenerate =
        first_path.size() < 3
        || first_signed_area_m2 == amrex::Real(0.0);
    const bool second_degenerate =
        second_path.size() < 3
        || second_signed_area_m2 == amrex::Real(0.0);

    if (first_degenerate && second_degenerate) {
        throw std::invalid_argument(
            "Fire raw-event pinch cannot collapse both boundary paths");
    }

    if (first_degenerate || second_degenerate) {
        const amrex::Real surviving_signed_area_m2 =
            first_degenerate
                ? second_signed_area_m2
                : first_signed_area_m2;

        if (!(surviving_signed_area_m2
              > amrex::Real(0.0))) {
            throw std::invalid_argument(
                "Fire raw-event degenerate pinch must preserve one positive-area loop");
        }
    }

    require_area_partition(
        raw_signed_area_m2,
        first_signed_area_m2,
        second_signed_area_m2);

    return {
        std::move(first_path),
        std::move(second_path),
        raw_signed_area_m2,
        first_signed_area_m2,
        second_signed_area_m2
    };
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
    RawPinchSplit split =
        split_raw_perimeter_at_pinch(
            vertices_m,
            pinch);

    const bool first_zero_area =
        split.first_signed_area_m2
        == amrex::Real(0.0);
    const bool second_zero_area =
        split.second_signed_area_m2
        == amrex::Real(0.0);

    if (first_zero_area || second_zero_area) {
        if (first_zero_area == second_zero_area) {
            throw std::logic_error(
                "Fire degenerate raw pinch has an invalid zero-area partition");
        }

        std::vector<FireVec2> survivor =
            first_zero_area
                ? std::move(split.second_path)
                : std::move(split.first_path);

        std::vector<FireFrontComponent> components;
        components.reserve(1);
        components.push_back({
            FireFrontRole::Outer,
            FirePerimeter(std::move(survivor))
        });

        return FireFront(
            std::move(components));
    }

    if ((split.first_signed_area_m2 > amrex::Real(0.0))
        == (split.second_signed_area_m2 > amrex::Real(0.0))) {
        throw std::invalid_argument(
            "Fire raw-event pinch must create one outer loop and one hole");
    }

    std::vector<FireFrontComponent> components;
    components.reserve(2);

    if (split.first_signed_area_m2 > amrex::Real(0.0)) {
        components.push_back({
            FireFrontRole::Outer,
            FirePerimeter(std::move(split.first_path))
        });
        components.push_back({
            FireFrontRole::Hole,
            FirePerimeter(std::move(split.second_path))
        });
    } else {
        components.push_back({
            FireFrontRole::Outer,
            FirePerimeter(std::move(split.second_path))
        });
        components.push_back({
            FireFrontRole::Hole,
            FirePerimeter(std::move(split.first_path))
        });
    }

    return FireFront(std::move(components));
}

std::vector<FirePerimeter>
split_hole_perimeter_at_pinch (
    const std::vector<FireVec2>& vertices_m,
    const FirePerimeterPinch& pinch)
{
    RawPinchSplit split =
        split_raw_perimeter_at_pinch(
            vertices_m,
            pinch);

    const bool first_zero_area =
        split.first_signed_area_m2
        == amrex::Real(0.0);
    const bool second_zero_area =
        split.second_signed_area_m2
        == amrex::Real(0.0);

    if (first_zero_area || second_zero_area) {
        if (first_zero_area == second_zero_area) {
            throw std::logic_error(
                "Fire degenerate Hole pinch has an invalid zero-area partition");
        }

        std::vector<FirePerimeter> holes;
        holes.reserve(1);
        holes.emplace_back(
            first_zero_area
                ? std::move(split.second_path)
                : std::move(split.first_path));

        return holes;
    }

    if (!(split.first_signed_area_m2 > amrex::Real(0.0))
        || !(split.second_signed_area_m2 > amrex::Real(0.0))) {
        throw std::invalid_argument(
            "Fire Hole pinch must create two positive-area Hole loops");
    }

    std::vector<FirePerimeter> holes;
    holes.reserve(2);
    holes.emplace_back(
        std::move(split.first_path));
    holes.emplace_back(
        std::move(split.second_path));

    return holes;
}



bool
perimeters_are_strictly_disjoint (
    const FirePerimeter& first,
    const FirePerimeter& second)
{
    const auto& first_vertices =
        first.vertices_m();
    const auto& second_vertices =
        second.vertices_m();

    amrex::Real geometric_scale =
        amrex::Real(1.0);

    const auto accumulate_scale =
        [&geometric_scale] (
            const std::vector<FireVec2>& vertices)
        {
            for (std::size_t index = 0;
                 index < vertices.size();
                 ++index) {
                const FireVec2 edge =
                    vertices[(index + 1) % vertices.size()]
                    - vertices[index];

                geometric_scale =
                    std::max(
                        geometric_scale,
                        norm(edge));
            }
        };

    accumulate_scale(first_vertices);
    accumulate_scale(second_vertices);

    const amrex::Real distance_tolerance_m =
        amrex::Real(8192.0)
        * std::numeric_limits<amrex::Real>::epsilon()
        * geometric_scale;

    const auto point_on_segment =
        [distance_tolerance_m] (
            const FireVec2& point,
            const FireVec2& start,
            const FireVec2& end)
        {
            const FireVec2 edge =
                end - start;
            const amrex::Real length_squared =
                norm_squared(edge);

            if (!(length_squared
                  > amrex::Real(0.0))) {
                return norm(point - start)
                    <= distance_tolerance_m;
            }

            const amrex::Real length =
                std::sqrt(length_squared);
            const amrex::Real parameter_tolerance =
                distance_tolerance_m / length;

            const amrex::Real parameter =
                dot(point - start, edge)
                / length_squared;

            if (parameter < -parameter_tolerance
                || parameter
                    > amrex::Real(1.0)
                        + parameter_tolerance) {
                return false;
            }

            const amrex::Real clamped_parameter =
                std::clamp(
                    parameter,
                    amrex::Real(0.0),
                    amrex::Real(1.0));

            const FireVec2 projected =
                start + clamped_parameter * edge;

            return norm(point - projected)
                <= distance_tolerance_m;
        };

    const auto orientation_sign =
        [distance_tolerance_m] (
            const FireVec2& first_point,
            const FireVec2& second_point,
            const FireVec2& third_point)
        {
            const FireVec2 first_edge =
                second_point - first_point;
            const FireVec2 second_edge =
                third_point - first_point;

            const amrex::Real value =
                detail::cross_2d(
                    first_edge,
                    second_edge);

            const amrex::Real area_tolerance =
                distance_tolerance_m
                * std::max({
                    norm(first_edge),
                    norm(second_edge),
                    amrex::Real(1.0)
                });

            if (value > area_tolerance) {
                return 1;
            }

            if (value < -area_tolerance) {
                return -1;
            }

            return 0;
        };

    const auto segments_intersect_or_touch =
        [&] (
            const FireVec2& first_start,
            const FireVec2& first_end,
            const FireVec2& second_start,
            const FireVec2& second_end)
        {
            const int first_orientation =
                orientation_sign(
                    first_start,
                    first_end,
                    second_start);
            const int second_orientation =
                orientation_sign(
                    first_start,
                    first_end,
                    second_end);
            const int third_orientation =
                orientation_sign(
                    second_start,
                    second_end,
                    first_start);
            const int fourth_orientation =
                orientation_sign(
                    second_start,
                    second_end,
                    first_end);

            if (first_orientation
                    * second_orientation < 0
                && third_orientation
                    * fourth_orientation < 0) {
                return true;
            }

            if (first_orientation == 0
                && point_on_segment(
                    second_start,
                    first_start,
                    first_end)) {
                return true;
            }

            if (second_orientation == 0
                && point_on_segment(
                    second_end,
                    first_start,
                    first_end)) {
                return true;
            }

            if (third_orientation == 0
                && point_on_segment(
                    first_start,
                    second_start,
                    second_end)) {
                return true;
            }

            if (fourth_orientation == 0
                && point_on_segment(
                    first_end,
                    second_start,
                    second_end)) {
                return true;
            }

            return false;
        };

    for (std::size_t first_index = 0;
         first_index < first_vertices.size();
         ++first_index) {
        const FireVec2& first_start =
            first_vertices[first_index];
        const FireVec2& first_end =
            first_vertices[
                (first_index + 1)
                % first_vertices.size()];

        for (std::size_t second_index = 0;
             second_index < second_vertices.size();
             ++second_index) {
            const FireVec2& second_start =
                second_vertices[second_index];
            const FireVec2& second_end =
                second_vertices[
                    (second_index + 1)
                    % second_vertices.size()];

            if (segments_intersect_or_touch(
                    first_start,
                    first_end,
                    second_start,
                    second_end)) {
                return false;
            }
        }
    }

    const auto point_inside_or_on =
        [&point_on_segment] (
            const FireVec2& point,
            const std::vector<FireVec2>& vertices)
        {
            bool inside = false;

            for (std::size_t index = 0;
                 index < vertices.size();
                 ++index) {
                const FireVec2& start =
                    vertices[index];
                const FireVec2& end =
                    vertices[
                        (index + 1)
                        % vertices.size()];

                if (point_on_segment(
                        point,
                        start,
                        end)) {
                    return true;
                }

                if ((start.y > point.y)
                    == (end.y > point.y)) {
                    continue;
                }

                const amrex::Real crossing_x =
                    start.x
                    + (point.y - start.y)
                        * (end.x - start.x)
                        / (end.y - start.y);

                if (crossing_x > point.x) {
                    inside = !inside;
                }
            }

            return inside;
        };

    if (point_inside_or_on(
            first_vertices.front(),
            second_vertices)) {
        return false;
    }

    if (point_inside_or_on(
            second_vertices.front(),
            first_vertices)) {
        return false;
    }

    return true;
}


} // namespace ERFFire
