#include <ERF_FirePerimeter.H>
#include <ERF_FireGeometry.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ERFFire
{
namespace
{

bool
finite (const FireVec2& value) noexcept
{
    return std::isfinite(value.x) && std::isfinite(value.y);
}

int
orientation_sign (
    const FireVec2& a,
    const FireVec2& b,
    const FireVec2& c) noexcept
{
    const FireVec2 ab = b - a;
    const FireVec2 ac = c - a;
    const amrex::Real value =
        detail::cross_2d(ab, ac);

    const amrex::Real scale = std::max({
        norm(ab),
        norm(ac),
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

    const amrex::Real scale =
        std::max(norm(b - a), amrex::Real(1.0));
    const amrex::Real tolerance =
        amrex::Real(64.0)
        * std::numeric_limits<amrex::Real>::epsilon()
        * scale;

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

    if (ab_c != 0 && ab_d != 0
        && cd_a != 0 && cd_b != 0) {
        return ab_c != ab_d && cd_a != cd_b;
    }

    return (ab_c == 0 && point_on_segment(c, a, b))
        || (ab_d == 0 && point_on_segment(d, a, b))
        || (cd_a == 0 && point_on_segment(a, c, d))
        || (cd_b == 0 && point_on_segment(b, c, d));
}

bool
self_intersects (
    const std::vector<FireVec2>& vertices) noexcept
{
    const std::size_t count = vertices.size();

    for (std::size_t first = 0; first < count; ++first) {
        const std::size_t first_next =
            (first + 1) % count;

        for (std::size_t second = first + 1;
             second < count;
             ++second) {
            const std::size_t second_next =
                (second + 1) % count;

            if (first_next == second
                || second_next == first) {
                continue;
            }

            if (segments_intersect(
                    vertices[first],
                    vertices[first_next],
                    vertices[second],
                    vertices[second_next])) {
                return true;
            }
        }
    }

    return false;
}

} // namespace

FirePerimeter::FirePerimeter (std::vector<FireVec2> vertices_m)
    : m_vertices_m(std::move(vertices_m))
{
    validate();

    if (!is_counter_clockwise()) {
        std::reverse(m_vertices_m.begin(), m_vertices_m.end());
    }
}

void
FirePerimeter::validate () const
{
    if (m_vertices_m.size() < 3) {
        throw std::invalid_argument(
            "FirePerimeter requires at least three vertices");
    }

    for (std::size_t i = 0; i < m_vertices_m.size(); ++i) {
        const auto& current = m_vertices_m[i];
        const auto& next = m_vertices_m[(i + 1) % m_vertices_m.size()];

        if (!finite(current)) {
            throw std::invalid_argument(
                "FirePerimeter vertices must be finite");
        }

        if (current.x == next.x && current.y == next.y) {
            throw std::invalid_argument(
                "FirePerimeter cannot contain a zero-length edge");
        }
    }

    if (signed_area_m2() == amrex::Real(0.0)) {
        throw std::invalid_argument(
            "FirePerimeter requires non-zero signed area");
    }

    if (self_intersects(m_vertices_m)) {
        throw std::invalid_argument(
            "FirePerimeter cannot self-intersect");
    }
}

amrex::Real
FirePerimeter::signed_area_m2 () const noexcept
{
    return detail::signed_polygon_area_m2(m_vertices_m);
}

amrex::Real
FirePerimeter::area_m2 () const noexcept
{
    return std::abs(signed_area_m2());
}

amrex::Real
FirePerimeter::perimeter_length_m () const noexcept
{
    amrex::Real length = 0.0;

    for (std::size_t i = 0; i < m_vertices_m.size(); ++i) {
        const auto& current = m_vertices_m[i];
        const auto& next = m_vertices_m[(i + 1) % m_vertices_m.size()];
        length += norm(next - current);
    }

    return length;
}

bool
FirePerimeter::is_counter_clockwise () const noexcept
{
    return signed_area_m2() > amrex::Real(0.0);
}

FireVec2
FirePerimeter::tangent_unit (std::size_t i) const
{
    if (i >= m_vertices_m.size()) {
        throw std::out_of_range("FirePerimeter vertex index out of range");
    }

    const auto& previous =
        m_vertices_m[(i + m_vertices_m.size() - 1) % m_vertices_m.size()];
    const auto& next = m_vertices_m[(i + 1) % m_vertices_m.size()];

    const FireVec2 chord = next - previous;
    const amrex::Real chord_length = norm(chord);

    if (chord_length == amrex::Real(0.0)) {
        throw std::runtime_error(
            "FirePerimeter centered chord has zero length");
    }

    return chord / chord_length;
}

FireVec2
FirePerimeter::outward_normal_unit (std::size_t i) const
{
    const FireVec2 tangent = tangent_unit(i);

    // The stored loop is counter-clockwise, so the outward normal is the
    // right-hand normal of the direction of traversal.
    return {tangent.y, -tangent.x};
}

} // namespace ERFFire
