#include <ERF_FirePerimeterRemesher.H>
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

amrex::Real
point_to_segment_distance_m (
    const FireVec2& point,
    const FireVec2& segment_start,
    const FireVec2& segment_end) noexcept
{
    const FireVec2 segment = segment_end - segment_start;
    const amrex::Real length_squared = norm_squared(segment);

    if (length_squared == amrex::Real(0.0)) {
        return norm(point - segment_start);
    }

    const amrex::Real projection = std::clamp(
        dot(point - segment_start, segment) / length_squared,
        amrex::Real(0.0),
        amrex::Real(1.0));

    return norm(point - (segment_start + projection * segment));
}

int
orientation_sign (
    const FireVec2& a,
    const FireVec2& b,
    const FireVec2& c) noexcept
{
    const amrex::Real value = detail::cross_2d(b - a, c - a);
    const amrex::Real scale =
        std::max({
            norm(b - a),
            norm(c - a),
            norm(c - b),
            amrex::Real(1.0)
        });
    const amrex::Real tolerance =
        amrex::Real(64.0)
        * std::numeric_limits<amrex::Real>::epsilon()
        * scale * scale;

    if (value > tolerance) {
        return 1;
    }
    if (value < -tolerance) {
        return -1;
    }
    return 0;
}

bool
point_on_segment (
    const FireVec2& point,
    const FireVec2& a,
    const FireVec2& b) noexcept
{
    if (orientation_sign(a, b, point) != 0) {
        return false;
    }

    const amrex::Real coordinate_scale = std::max({
        std::abs(a.x), std::abs(a.y),
        std::abs(b.x), std::abs(b.y),
        std::abs(point.x), std::abs(point.y),
        amrex::Real(1.0)
    });
    const amrex::Real tolerance =
        amrex::Real(64.0)
        * std::numeric_limits<amrex::Real>::epsilon()
        * coordinate_scale;

    return point.x >= std::min(a.x, b.x) - tolerance
        && point.x <= std::max(a.x, b.x) + tolerance
        && point.y >= std::min(a.y, b.y) - tolerance
        && point.y <= std::max(a.y, b.y) + tolerance;
}

bool
segments_intersect (
    const FireVec2& a,
    const FireVec2& b,
    const FireVec2& c,
    const FireVec2& d) noexcept
{
    const int ab_c = orientation_sign(a, b, c);
    const int ab_d = orientation_sign(a, b, d);
    const int cd_a = orientation_sign(c, d, a);
    const int cd_b = orientation_sign(c, d, b);

    if (ab_c != 0 && ab_d != 0 && cd_a != 0 && cd_b != 0) {
        return ab_c != ab_d && cd_a != cd_b;
    }

    return (ab_c == 0 && point_on_segment(c, a, b))
        || (ab_d == 0 && point_on_segment(d, a, b))
        || (cd_a == 0 && point_on_segment(a, c, d))
        || (cd_b == 0 && point_on_segment(b, c, d));
}

bool
candidate_chord_intersects_nonlocal_edge (
    const std::vector<FireVec2>& vertices_m,
    std::size_t remove_index) noexcept
{
    const std::size_t count = vertices_m.size();
    const std::size_t previous = (remove_index + count - 1) % count;
    const std::size_t next = (remove_index + 1) % count;

    const FireVec2& chord_start = vertices_m[previous];
    const FireVec2& chord_end = vertices_m[next];

    for (std::size_t edge = 0; edge < count; ++edge) {
        const std::size_t edge_next = (edge + 1) % count;

        // Ignore the two edges being replaced and the two surviving edges
        // adjacent to the new chord endpoints. Endpoint contact there is the
        // expected polygon connectivity, not a new self-intersection.
        if (edge == previous || edge_next == previous
            || edge == remove_index || edge_next == remove_index
            || edge == next || edge_next == next) {
            continue;
        }

        if (segments_intersect(
                chord_start,
                chord_end,
                vertices_m[edge],
                vertices_m[edge_next])) {
            return true;
        }
    }

    return false;
}

void
validate_options (const FirePerimeterRemeshOptions& options)
{
    if (!std::isfinite(options.min_edge_length_m)
        || options.min_edge_length_m <= amrex::Real(0.0)) {
        throw std::invalid_argument(
            "Fire remesh minimum edge length must be finite and positive");
    }

    if (!std::isfinite(options.max_edge_length_m)
        || options.max_edge_length_m <= amrex::Real(0.0)) {
        throw std::invalid_argument(
            "Fire remesh maximum edge length must be finite and positive");
    }

    if (options.min_edge_length_m
        > amrex::Real(0.5) * options.max_edge_length_m) {
        throw std::invalid_argument(
            "Fire remesh minimum edge length must not exceed half the maximum");
    }

    if (!std::isfinite(options.max_chord_error_m)
        || options.max_chord_error_m < amrex::Real(0.0)) {
        throw std::invalid_argument(
            "Fire remesh chord-error tolerance must be finite and non-negative");
    }
}

bool
can_remove_vertex (
    const std::vector<FireVec2>& vertices_m,
    std::size_t index,
    const FirePerimeterRemeshOptions& options) noexcept
{
    if (vertices_m.size() <= 3) {
        return false;
    }

    const std::size_t count = vertices_m.size();
    const std::size_t previous = (index + count - 1) % count;
    const std::size_t next = (index + 1) % count;

    const FireVec2& a = vertices_m[previous];
    const FireVec2& b = vertices_m[index];
    const FireVec2& c = vertices_m[next];

    const amrex::Real previous_edge_m = norm(b - a);
    const amrex::Real next_edge_m = norm(c - b);

    const bool short_edge =
        previous_edge_m < options.min_edge_length_m
        || next_edge_m < options.min_edge_length_m;

    const amrex::Real replacement_edge_m = norm(c - a);
    if (replacement_edge_m == amrex::Real(0.0)
        || replacement_edge_m > options.max_edge_length_m) {
        return false;
    }

    const FireVec2 incoming = b - a;
    const FireVec2 outgoing = c - b;

    const bool backtracking =
        dot(incoming, outgoing) < amrex::Real(0.0);

    const amrex::Real spike_width_m =
        std::min(
            point_to_segment_distance_m(a, b, c),
            point_to_segment_distance_m(c, a, b));

    const bool narrow_backtracking_spike =
        backtracking
        && spike_width_m <= options.max_chord_error_m;

    if (!short_edge && !narrow_backtracking_spike) {
        return false;
    }

    if (!narrow_backtracking_spike
        && point_to_segment_distance_m(b, a, c)
            > options.max_chord_error_m) {
        return false;
    }

    if (candidate_chord_intersects_nonlocal_edge(vertices_m, index)) {
        return false;
    }

    std::vector<FireVec2> candidate = vertices_m;
    candidate.erase(candidate.begin() + static_cast<std::ptrdiff_t>(index));
    return detail::signed_polygon_area_m2(candidate) > amrex::Real(0.0);
}

std::size_t
coarsen_vertices (
    std::vector<FireVec2>& vertices_m,
    const FirePerimeterRemeshOptions& options)
{
    std::size_t removed = 0;

    // Restart after each accepted local topology change.
    for (;;) {
        bool changed = false;

        for (std::size_t i = 0; i < vertices_m.size(); ++i) {
            if (can_remove_vertex(vertices_m, i, options)) {
                vertices_m.erase(
                    vertices_m.begin() + static_cast<std::ptrdiff_t>(i));
                ++removed;
                changed = true;
                break;
            }
        }

        if (!changed) {
            break;
        }
    }

    return removed;
}

std::size_t
refine_vertices (
    std::vector<FireVec2>& vertices_m,
    const FirePerimeterRemeshOptions& options)
{
    const std::vector<FireVec2> original = vertices_m;
    std::vector<FireVec2> refined;
    refined.reserve(original.size());

    std::size_t added = 0;

    for (std::size_t i = 0; i < original.size(); ++i) {
        const FireVec2& start = original[i];
        const FireVec2& end = original[(i + 1) % original.size()];
        const FireVec2 edge = end - start;
        const amrex::Real edge_length_m = norm(edge);

        const std::size_t segment_count = std::max<std::size_t>(
            1,
            static_cast<std::size_t>(
                std::ceil(edge_length_m / options.max_edge_length_m)));

        refined.push_back(start);

        for (std::size_t segment = 1; segment < segment_count; ++segment) {
            const amrex::Real fraction =
                static_cast<amrex::Real>(segment)
                / static_cast<amrex::Real>(segment_count);
            refined.push_back(start + fraction * edge);
            ++added;
        }
    }

    vertices_m = std::move(refined);
    return added;
}

} // namespace

FirePerimeterRemeshResult
remesh_perimeter (
    const FirePerimeter& perimeter,
    const FirePerimeterRemeshOptions& options)
{
    validate_options(options);

    std::vector<FireVec2> vertices_m = perimeter.vertices_m();
    FirePerimeterRemeshStats stats;

    stats.vertices_removed = coarsen_vertices(vertices_m, options);
    stats.vertices_added = refine_vertices(vertices_m, options);

    return {
        FirePerimeter(std::move(vertices_m)),
        stats
    };
}

FireFrontRemeshResult
remesh_front (
    const FireFront& front,
    const FirePerimeterRemeshOptions& options)
{
    std::vector<FireFrontComponent> components;
    components.reserve(
        front.components().size());

    FirePerimeterRemeshStats stats;

    for (const FireFrontComponent& component :
         front.components()) {
        FirePerimeterRemeshResult remeshed =
            remesh_perimeter(
                component.perimeter,
                options);

        stats.vertices_removed +=
            remeshed.stats.vertices_removed;
        stats.vertices_added +=
            remeshed.stats.vertices_added;

        components.push_back({
            component.role,
            std::move(remeshed.perimeter)
        });
    }

    return {
        FireFront(std::move(components)),
        stats
    };
}

} // namespace ERFFire
