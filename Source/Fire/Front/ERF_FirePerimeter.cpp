#include <ERF_FirePerimeter.H>

#include <algorithm>
#include <cmath>
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
}

amrex::Real
FirePerimeter::signed_area_m2 () const noexcept
{
    amrex::Real twice_area = 0.0;

    for (std::size_t i = 0; i < m_vertices_m.size(); ++i) {
        const auto& current = m_vertices_m[i];
        const auto& next = m_vertices_m[(i + 1) % m_vertices_m.size()];
        twice_area += current.x * next.y - next.x * current.y;
    }

    return amrex::Real(0.5) * twice_area;
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
