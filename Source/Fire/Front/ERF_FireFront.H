#ifndef ERF_FIRE_FRONT_H
#define ERF_FIRE_FRONT_H

#include <ERF_FirePerimeter.H>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ERFFire
{

enum class FireFrontRole
{
    Outer,
    Hole
};

struct FireFrontComponent
{
    FireFrontRole role;
    FirePerimeter perimeter;
};

/**
 * Boundary-component representation of a two-dimensional burned region.
 *
 * Every component stores a simple counter-clockwise FirePerimeter.
 * Component role supplies the region semantics:
 *
 *   Outer contributes burned area.
 *   Hole subtracts unburned area.
 *
 * Keeping every FirePerimeter counter-clockwise preserves the existing
 * geometric orientation invariant. FireFrontRole determines which side of
 * a component contains available fuel.
 */
class FireFront
{
public:
    explicit FireFront(std::vector<FireFrontComponent> components)
        : m_components(std::move(components))
    {
        validate();
    }

    [[nodiscard]]
    const std::vector<FireFrontComponent>& components() const noexcept
    {
        return m_components;
    }

    [[nodiscard]] amrex::Real burned_area_m2() const noexcept
    {
        amrex::Real area_m2 = 0.0;

        for (const FireFrontComponent& component : m_components) {
            if (component.role == FireFrontRole::Outer) {
                area_m2 += component.perimeter.area_m2();
            } else {
                area_m2 -= component.perimeter.area_m2();
            }
        }

        return area_m2;
    }

    /**
     * Unit normal pointing from burned material toward available fuel.
     *
     * For an outer boundary this is the perimeter's geometric outward
     * normal. For a hole boundary the fuel lies inside the hole, so the
     * spread normal is the opposite direction.
     */
    [[nodiscard]] FireVec2 spread_normal_unit(
        std::size_t component_index,
        std::size_t vertex_index) const
    {
        if (component_index >= m_components.size()) {
            throw std::out_of_range(
                "FireFront component index out of range");
        }

        const FireFrontComponent& component =
            m_components[component_index];
        const FireVec2 geometric_outward =
            component.perimeter.outward_normal_unit(vertex_index);

        if (component.role == FireFrontRole::Outer) {
            return geometric_outward;
        }

        return {
            -geometric_outward.x,
            -geometric_outward.y
        };
    }

private:
    void validate() const
    {
        if (m_components.empty()) {
            throw std::invalid_argument(
                "FireFront requires at least one boundary component");
        }

        bool has_outer = false;
        amrex::Real area_m2 = 0.0;

        for (const FireFrontComponent& component : m_components) {
            const amrex::Real component_area_m2 =
                component.perimeter.area_m2();

            if (!std::isfinite(component_area_m2)) {
                throw std::overflow_error(
                    "FireFront component area is not finite");
            }

            switch (component.role) {
            case FireFrontRole::Outer:
                has_outer = true;
                area_m2 += component_area_m2;
                break;
            case FireFrontRole::Hole:
                area_m2 -= component_area_m2;
                break;
            default:
                throw std::invalid_argument(
                    "FireFront component has an invalid role");
            }

            if (!std::isfinite(area_m2)) {
                throw std::overflow_error(
                    "FireFront burned area is not finite");
            }
        }

        if (!has_outer) {
            throw std::invalid_argument(
                "FireFront requires at least one outer boundary");
        }

        if (!(area_m2 > amrex::Real(0.0))) {
            throw std::invalid_argument(
                "FireFront requires positive burned area");
        }
    }

    std::vector<FireFrontComponent> m_components;
};

} // namespace ERFFire

#endif // ERF_FIRE_FRONT_H
