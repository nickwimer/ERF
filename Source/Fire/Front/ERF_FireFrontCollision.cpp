#include <ERF_FireFrontCollision.H>

#include <ERF_FireGeometry.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ERFFire
{
namespace
{

struct BoundaryCoordinate
{
    std::size_t edge_index{};
    amrex::Real edge_fraction{};
};

struct CollisionCandidate
{
    FirePerimeterCollision collision;
};

struct QuadraticRootSet
{
    std::vector<amrex::Real> roots;
    bool identically_zero{};
};

amrex::Real
fraction_tolerance () noexcept
{
    return amrex::Real(8192.0)
        * std::numeric_limits<amrex::Real>::epsilon();
}

bool
finite (const FireVec2& point) noexcept
{
    return std::isfinite(point.x)
        && std::isfinite(point.y);
}

FireVec2
interpolate_vertex (
    const FirePerimeter& start,
    const std::vector<FireVec2>& end_vertices_m,
    std::size_t vertex_index,
    amrex::Real fraction) noexcept
{
    const FireVec2& initial =
        start.vertices_m()[vertex_index];
    const FireVec2& final =
        end_vertices_m[vertex_index];

    return initial + fraction * (final - initial);
}

bool
edges_are_adjacent (
    std::size_t first,
    std::size_t second,
    std::size_t count) noexcept
{
    return first == second
        || (first + 1) % count == second
        || (second + 1) % count == first;
}

void
add_unit_interval_root (
    std::vector<amrex::Real>& roots,
    amrex::Real root)
{
    if (!std::isfinite(root)) {
        return;
    }

    const amrex::Real tolerance =
        fraction_tolerance();

    if (root < -tolerance
        || root > amrex::Real(1.0) + tolerance) {
        return;
    }

    root = std::clamp(
        root,
        amrex::Real(0.0),
        amrex::Real(1.0));

    for (const amrex::Real existing : roots) {
        if (std::abs(existing - root) <= tolerance) {
            return;
        }
    }

    roots.push_back(root);
}

QuadraticRootSet
quadratic_roots_unit_interval (
    amrex::Real quadratic,
    amrex::Real linear,
    amrex::Real constant)
{
    QuadraticRootSet result;

    const amrex::Real coefficient_scale =
        std::max({
            std::abs(quadratic),
            std::abs(linear),
            std::abs(constant),
            amrex::Real(1.0)
        });
    const amrex::Real coefficient_tolerance =
        amrex::Real(256.0)
        * std::numeric_limits<amrex::Real>::epsilon()
        * coefficient_scale;

    if (std::abs(quadratic) <= coefficient_tolerance) {
        if (std::abs(linear) <= coefficient_tolerance) {
            result.identically_zero =
                std::abs(constant) <= coefficient_tolerance;
            return result;
        }

        add_unit_interval_root(
            result.roots,
            -constant / linear);
        return result;
    }

    const amrex::Real discriminant =
        linear * linear
        - amrex::Real(4.0) * quadratic * constant;
    const amrex::Real discriminant_scale =
        std::max({
            std::abs(linear * linear),
            std::abs(
                amrex::Real(4.0)
                * quadratic
                * constant),
            amrex::Real(1.0)
        });
    const amrex::Real discriminant_tolerance =
        amrex::Real(512.0)
        * std::numeric_limits<amrex::Real>::epsilon()
        * discriminant_scale;

    if (discriminant < -discriminant_tolerance) {
        return result;
    }

    const amrex::Real nonnegative_discriminant =
        std::max(discriminant, amrex::Real(0.0));
    const amrex::Real square_root =
        std::sqrt(nonnegative_discriminant);

    if (square_root == amrex::Real(0.0)) {
        add_unit_interval_root(
            result.roots,
            -linear
                / (amrex::Real(2.0) * quadratic));
        return result;
    }

    const amrex::Real q =
        -amrex::Real(0.5)
        * (linear + std::copysign(square_root, linear));

    if (q == amrex::Real(0.0)) {
        add_unit_interval_root(
            result.roots,
            -linear
                / (amrex::Real(2.0) * quadratic));
        return result;
    }

    add_unit_interval_root(
        result.roots,
        q / quadratic);
    add_unit_interval_root(
        result.roots,
        constant / q);

    std::sort(
        result.roots.begin(),
        result.roots.end());

    return result;
}

std::optional<amrex::Real>
moving_point_coincidence_fraction (
    const FireVec2& first_start,
    const FireVec2& first_end,
    const FireVec2& second_start,
    const FireVec2& second_end)
{
    const FireVec2 offset =
        first_start - second_start;
    const FireVec2 relative_motion =
        (first_end - first_start)
        - (second_end - second_start);

    const amrex::Real scale =
        std::max({
            norm(offset),
            norm(relative_motion),
            amrex::Real(1.0)
        });
    const amrex::Real distance_tolerance =
        amrex::Real(4096.0)
        * std::numeric_limits<amrex::Real>::epsilon()
        * scale;

    const bool use_x =
        std::abs(relative_motion.x)
        >= std::abs(relative_motion.y);
    const amrex::Real denominator =
        use_x
            ? relative_motion.x
            : relative_motion.y;

    if (std::abs(denominator) <= distance_tolerance) {
        return std::nullopt;
    }

    const amrex::Real numerator =
        use_x ? offset.x : offset.y;
    amrex::Real fraction =
        -numerator / denominator;

    const amrex::Real time_tolerance =
        fraction_tolerance();
    if (fraction < -time_tolerance
        || fraction
            > amrex::Real(1.0) + time_tolerance) {
        return std::nullopt;
    }

    fraction = std::clamp(
        fraction,
        amrex::Real(0.0),
        amrex::Real(1.0));

    const FireVec2 residual =
        offset + fraction * relative_motion;
    if (norm(residual) > distance_tolerance) {
        return std::nullopt;
    }

    return fraction;
}

BoundaryCoordinate
canonical_boundary_coordinate (
    std::size_t edge_index,
    amrex::Real edge_fraction,
    std::size_t count,
    amrex::Real parameter_tolerance)
{
    edge_fraction = std::clamp(
        edge_fraction,
        amrex::Real(0.0),
        amrex::Real(1.0));

    if (edge_fraction <= parameter_tolerance) {
        return {
            edge_index,
            amrex::Real(0.0)
        };
    }

    if (edge_fraction
        >= amrex::Real(1.0) - parameter_tolerance) {
        return {
            (edge_index + 1) % count,
            amrex::Real(0.0)
        };
    }

    return {
        edge_index,
        edge_fraction
    };
}

bool
boundary_coordinate_less (
    const BoundaryCoordinate& first,
    const BoundaryCoordinate& second) noexcept
{
    if (first.edge_index != second.edge_index) {
        return first.edge_index < second.edge_index;
    }

    return first.edge_fraction
        < second.edge_fraction;
}

bool
same_boundary_coordinate (
    const BoundaryCoordinate& first,
    const BoundaryCoordinate& second) noexcept
{
    return first.edge_index == second.edge_index
        && first.edge_fraction == second.edge_fraction;
}

bool
candidate_precedes (
    const CollisionCandidate& first,
    const CollisionCandidate& second) noexcept
{
    const amrex::Real tolerance =
        fraction_tolerance();

    if (first.collision.motion_fraction
        < second.collision.motion_fraction - tolerance) {
        return true;
    }

    if (first.collision.motion_fraction
        > second.collision.motion_fraction + tolerance) {
        return false;
    }

    const auto& first_pinch =
        first.collision.pinch;
    const auto& second_pinch =
        second.collision.pinch;

    if (first_pinch.first_edge_index
        != second_pinch.first_edge_index) {
        return first_pinch.first_edge_index
            < second_pinch.first_edge_index;
    }
    if (first_pinch.first_edge_fraction
        != second_pinch.first_edge_fraction) {
        return first_pinch.first_edge_fraction
            < second_pinch.first_edge_fraction;
    }
    if (first_pinch.second_edge_index
        != second_pinch.second_edge_index) {
        return first_pinch.second_edge_index
            < second_pinch.second_edge_index;
    }

    return first_pinch.second_edge_fraction
        < second_pinch.second_edge_fraction;
}

void
consider_candidate (
    std::optional<CollisionCandidate>& best,
    const std::optional<CollisionCandidate>& candidate)
{
    if (!candidate.has_value()) {
        return;
    }

    if (!best.has_value()
        || candidate_precedes(*candidate, *best)) {
        best = candidate;
    }
}

std::optional<CollisionCandidate>
locate_vertex_edge_contact (
    const FirePerimeter& start,
    const std::vector<FireVec2>& end_vertices_m,
    std::size_t vertex_index,
    std::size_t edge_index)
{
    const std::size_t count = start.size();
    const std::size_t edge_next =
        (edge_index + 1) % count;

    const FireVec2& point_start =
        start.vertices_m()[vertex_index];
    const FireVec2& point_end =
        end_vertices_m[vertex_index];
    const FireVec2& edge_start_initial =
        start.vertices_m()[edge_index];
    const FireVec2& edge_start_final =
        end_vertices_m[edge_index];
    const FireVec2& edge_end_initial =
        start.vertices_m()[edge_next];
    const FireVec2& edge_end_final =
        end_vertices_m[edge_next];

    const FireVec2 edge_initial =
        edge_end_initial - edge_start_initial;
    const FireVec2 edge_motion =
        (edge_end_final - edge_end_initial)
        - (edge_start_final - edge_start_initial);
    const FireVec2 offset_initial =
        point_start - edge_start_initial;
    const FireVec2 offset_motion =
        (point_end - point_start)
        - (edge_start_final - edge_start_initial);

    const amrex::Real constant =
        detail::cross_2d(
            edge_initial,
            offset_initial);
    const amrex::Real linear =
        detail::cross_2d(
            edge_initial,
            offset_motion)
        + detail::cross_2d(
            edge_motion,
            offset_initial);
    const amrex::Real quadratic =
        detail::cross_2d(
            edge_motion,
            offset_motion);

    QuadraticRootSet root_set =
        quadratic_roots_unit_interval(
            quadratic,
            linear,
            constant);

    if (root_set.identically_zero) {
        const auto first_endpoint_contact =
            moving_point_coincidence_fraction(
                point_start,
                point_end,
                edge_start_initial,
                edge_start_final);
        if (first_endpoint_contact.has_value()) {
            add_unit_interval_root(
                root_set.roots,
                *first_endpoint_contact);
        }

        const auto second_endpoint_contact =
            moving_point_coincidence_fraction(
                point_start,
                point_end,
                edge_end_initial,
                edge_end_final);
        if (second_endpoint_contact.has_value()) {
            add_unit_interval_root(
                root_set.roots,
                *second_endpoint_contact);
        }
    }

    std::sort(
        root_set.roots.begin(),
        root_set.roots.end());

    std::optional<CollisionCandidate> best;

    for (const amrex::Real motion_fraction
         : root_set.roots) {
        const FireVec2 point =
            interpolate_vertex(
                start,
                end_vertices_m,
                vertex_index,
                motion_fraction);
        const FireVec2 edge_start =
            interpolate_vertex(
                start,
                end_vertices_m,
                edge_index,
                motion_fraction);
        const FireVec2 edge_end =
            interpolate_vertex(
                start,
                end_vertices_m,
                edge_next,
                motion_fraction);
        const FireVec2 edge =
            edge_end - edge_start;
        const amrex::Real edge_length_squared =
            norm_squared(edge);

        if (!(edge_length_squared
              > amrex::Real(0.0))
            || !std::isfinite(edge_length_squared)) {
            continue;
        }

        const amrex::Real edge_length =
            std::sqrt(edge_length_squared);
        const amrex::Real geometric_scale =
            std::max({
                edge_length,
                norm(point_end - point_start),
                norm(
                    edge_start_final
                    - edge_start_initial),
                norm(
                    edge_end_final
                    - edge_end_initial),
                amrex::Real(1.0)
            });
        const amrex::Real distance_tolerance =
            amrex::Real(8192.0)
            * std::numeric_limits<amrex::Real>::epsilon()
            * geometric_scale;
        const amrex::Real parameter_tolerance =
            std::max(
                fraction_tolerance(),
                distance_tolerance / edge_length);

        amrex::Real edge_fraction =
            dot(point - edge_start, edge)
            / edge_length_squared;

        if (edge_fraction < -parameter_tolerance
            || edge_fraction
                > amrex::Real(1.0)
                    + parameter_tolerance) {
            continue;
        }

        edge_fraction = std::clamp(
            edge_fraction,
            amrex::Real(0.0),
            amrex::Real(1.0));

        const FireVec2 projected =
            edge_start + edge_fraction * edge;

        if (norm(point - projected)
            > distance_tolerance) {
            continue;
        }

        BoundaryCoordinate edge_coordinate =
            canonical_boundary_coordinate(
                edge_index,
                edge_fraction,
                count,
                parameter_tolerance);
        BoundaryCoordinate vertex_coordinate{
            vertex_index,
            amrex::Real(0.0)
        };

        if (same_boundary_coordinate(
                edge_coordinate,
                vertex_coordinate)) {
            continue;
        }

        if (boundary_coordinate_less(
                vertex_coordinate,
                edge_coordinate)) {
            std::swap(
                edge_coordinate,
                vertex_coordinate);
        }

        FirePerimeterCollision collision;
        collision.motion_fraction =
            motion_fraction;
        collision.pinch = {
            edge_coordinate.edge_index,
            edge_coordinate.edge_fraction,
            vertex_coordinate.edge_index,
            vertex_coordinate.edge_fraction
        };
        collision.contact_point_m = {
            amrex::Real(0.5)
                * (point.x + projected.x),
            amrex::Real(0.5)
                * (point.y + projected.y)
        };

        CollisionCandidate candidate{
            std::move(collision)
        };

        if (!best.has_value()
            || candidate_precedes(
                candidate,
                *best)) {
            best = std::move(candidate);
        }
    }

    return best;
}

} // namespace

std::optional<FirePerimeterCollision>
locate_first_perimeter_collision (
    const FirePerimeter& start,
    const std::vector<FireVec2>& end_vertices_m)
{
    if (end_vertices_m.size() != start.size()) {
        throw std::invalid_argument(
            "Fire collision endpoint must preserve perimeter vertex count");
    }

    for (const FireVec2& vertex : end_vertices_m) {
        if (!finite(vertex)) {
            throw std::invalid_argument(
                "Fire collision endpoint vertices must be finite");
        }
    }

    const std::size_t count =
        start.size();
    std::optional<CollisionCandidate> best;

    for (std::size_t first_edge = 0;
         first_edge < count;
         ++first_edge) {
        for (std::size_t second_edge =
                 first_edge + 1;
             second_edge < count;
             ++second_edge) {
            if (edges_are_adjacent(
                    first_edge,
                    second_edge,
                    count)) {
                continue;
            }

            const std::size_t first_next =
                (first_edge + 1) % count;
            const std::size_t second_next =
                (second_edge + 1) % count;

            consider_candidate(
                best,
                locate_vertex_edge_contact(
                    start,
                    end_vertices_m,
                    first_edge,
                    second_edge));
            consider_candidate(
                best,
                locate_vertex_edge_contact(
                    start,
                    end_vertices_m,
                    first_next,
                    second_edge));
            consider_candidate(
                best,
                locate_vertex_edge_contact(
                    start,
                    end_vertices_m,
                    second_edge,
                    first_edge));
            consider_candidate(
                best,
                locate_vertex_edge_contact(
                    start,
                    end_vertices_m,
                    second_next,
                    first_edge));
        }
    }

    if (!best.has_value()) {
        return std::nullopt;
    }

    return best->collision;
}

std::optional<amrex::Real>
locate_first_perimeter_area_collapse (
    const FirePerimeter& start,
    const std::vector<FireVec2>& end_vertices_m)
{
    if (end_vertices_m.size() != start.size()) {
        throw std::invalid_argument(
            "Fire area-collapse endpoint must preserve perimeter vertex count");
    }

    for (const FireVec2& vertex : end_vertices_m) {
        if (!finite(vertex)) {
            throw std::invalid_argument(
                "Fire area-collapse endpoint vertices must be finite");
        }
    }

    const auto& start_vertices_m =
        start.vertices_m();

    // Translate by a fixed origin before forming the area polynomial. Polygon
    // area is translation invariant, and this substantially reduces
    // cancellation for georeferenced coordinates such as the Palisades case.
    const FireVec2 origin =
        start_vertices_m.front();

    amrex::Real quadratic = amrex::Real(0.0);
    amrex::Real linear = amrex::Real(0.0);
    amrex::Real constant = amrex::Real(0.0);

    for (std::size_t i = 0;
         i < start_vertices_m.size();
         ++i) {
        const std::size_t j =
            (i + 1) % start_vertices_m.size();

        const FireVec2 initial_i =
            start_vertices_m[i] - origin;
        const FireVec2 initial_j =
            start_vertices_m[j] - origin;
        const FireVec2 motion_i =
            end_vertices_m[i] - start_vertices_m[i];
        const FireVec2 motion_j =
            end_vertices_m[j] - start_vertices_m[j];

        constant +=
            detail::cross_2d(
                initial_i,
                initial_j);
        linear +=
            detail::cross_2d(
                motion_i,
                initial_j)
            + detail::cross_2d(
                initial_i,
                motion_j);
        quadratic +=
            detail::cross_2d(
                motion_i,
                motion_j);
    }

    const QuadraticRootSet roots =
        quadratic_roots_unit_interval(
            quadratic,
            linear,
            constant);

    const amrex::Real tolerance =
        fraction_tolerance();

    for (const amrex::Real root : roots.roots) {
        if (root > tolerance) {
            return root;
        }
    }

    return std::nullopt;
}

} // namespace ERFFire
