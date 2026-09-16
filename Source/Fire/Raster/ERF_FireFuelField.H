#ifndef ERF_FIRE_FUEL_FIELD_H_
#define ERF_FIRE_FUEL_FIELD_H_

#include <ERF_FireFuelMoisture.H>
#include <ERF_FireRasterGeometry.H>
#include <ERF_RothermelModel.H>

#include <type_traits>

namespace ERFFire
{

/**
 * Uniform compatibility material. single_dead_class retains the exact supplied
 * parameters, including non-FM1 synthetic beds; it is not an Anderson-13 model.
 * Class-aware moisture storage alone does not enable multi-class spread physics.
 */
struct FireFuelProperties
{
    RothermelFuelParameters single_dead_class{};
    FireFuelMoisture moisture{};
};

static_assert(std::is_trivially_copyable_v<FireFuelProperties>);

/**
 * Immutable uniform material on a validated Fire raster geometry.
 *
 * This first storage mode is O(1): no full-domain arrays, MPI collectives,
 * device allocations, file reads, or catalog reconstruction. Only a uniform
 * single-dead-class bed can be constructed. Spatial materials and nonburnable
 * barriers are deliberately not exposed until their evolution is implemented.
 *
 * Host point queries use the shared represented-cell-boundary convention.
 * uniform_properties() may be copied by value into cell-local device kernels;
 * host references must not be captured as device pointers. No host/device
 * bitwise equivalence of the Rothermel arithmetic is implied by this view.
 */
class FireFuelField
{
public:
    FireFuelField(
        FireCartesianRasterGeometry2D geometry,
        RothermelFuelParameters single_dead_class,
        amrex::Real dead_moisture_kg_kg)
        : geometry_(geometry),
          properties_{
              single_dead_class,
              FireFuelMoisture::uniform_dead(dead_moisture_kg_kg)}
    {
        (void)detail::validate_fire_cartesian_raster_geometry(geometry_);
        // Reuse the existing physics validation rather than duplicating its
        // admissible parameter ranges or silently substituting FM1 constants.
        (void)evaluate_rothermel(
            properties_.single_dead_class,
            RothermelInputs{dead_moisture_kg_kg, amrex::Real(0), amrex::Real(0)});
    }

    [[nodiscard]] const FireCartesianRasterGeometry2D& geometry() const noexcept
    {
        return geometry_;
    }

    [[nodiscard]] const FireFuelProperties& uniform_properties() const noexcept
    {
        return properties_;
    }

    [[nodiscard]] const FireFuelProperties& cell(
        std::size_t i, std::size_t j) const
    {
        (void)detail::fire_cartesian_raster_flat_index(geometry_, i, j);
        return properties_;
    }

    [[nodiscard]] const FireFuelProperties& sample(
        amrex::Real x_m, amrex::Real y_m) const
    {
        (void)detail::fire_cartesian_raster_locate_cell(geometry_, x_m, y_m);
        return properties_;
    }

private:
    FireCartesianRasterGeometry2D geometry_{};
    FireFuelProperties properties_{};
};

} // namespace ERFFire

#endif
