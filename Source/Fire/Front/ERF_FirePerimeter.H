#ifndef ERF_FIRE_PERIMETER_H
#define ERF_FIRE_PERIMETER_H

#include <ERF_FireTypes.H>

#include <cstddef>
#include <vector>

namespace ERFFire
{

/**
 * One simple closed outer fire perimeter in physical horizontal coordinates.
 *
 * The closing edge is implicit: vertices_m().back() connects to
 * vertices_m().front(s). The duplicated closing vertex must not be stored.
 *
 * A FirePerimeter normalizes its stored orientation to counter-clockwise.
 * Self-intersection detection and multiple loops/holes are deferred for now.
 */
class FirePerimeter
{
public:
    explicit FirePerimeter(std::vector<FireVec2> vertices_m);

    [[nodiscard]] const std::vector<FireVec2>& vertices_m() const noexcept
    {
        return m_vertices_m;
    }

    [[nodiscard]] std::size_t size() const noexcept
    {
        return m_vertices_m.size();
    }

    [[nodiscard]] amrex::Real signed_area_m2() const noexcept;
    [[nodiscard]] amrex::Real area_m2() const noexcept;
    [[nodiscard]] amrex::Real perimeter_length_m() const noexcept;
    [[nodiscard]] bool is_counter_clockwise() const noexcept;

    /**
     * Centered-chord unit tangent at vertex i.
     *
     * The tangent points in the counter-clockwise vertex-order direction.
     */
    [[nodiscard]] FireVec2 tangent_unit(std::size_t i) const;

    /**
     * Unit normal pointing outside the stored counter-clockwise perimeter.
     */
    [[nodiscard]] FireVec2 outward_normal_unit(std::size_t i) const;

private:
    void validate() const;

    std::vector<FireVec2> m_vertices_m;
};

} // namespace ERFFire

#endif // ERF_FIRE_PERIMETER_H
