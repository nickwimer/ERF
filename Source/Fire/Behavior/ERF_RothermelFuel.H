#ifndef ERF_ROTHERMEL_FUEL_H
#define ERF_ROTHERMEL_FUEL_H

#include <AMReX_REAL.H>

namespace ERFFire
{

/** One published fuel-particle class in an Anderson/Albini fuel model. */
struct RothermelFuelClassParameters
{
    amrex::Real dry_load_kg_m2{};
    amrex::Real sav_m_inv{};
};

/**
 * Original 13 fire-behavior fuel-model description in SI.
 *
 * Albini (1976), Appendix III, table 7 defines separate dead 1-h, 10-h,
 * 100-h, and live-foliage loading/SAV classes, plus fuel-bed depth and dead
 * moisture of extinction. Particle density, heat content, total mineral
 * fraction, and effective mineral fraction are common to all 13 models.
 *
 * A zero class loading means that particle class is absent; its SAV is then
 * stored as zero. This catalog is a data description only. It does not imply
 * that the existing single-dead-class Rothermel evaluator can evaluate models
 * 2-13.
 */
struct Anderson13FuelParameters
{
    RothermelFuelClassParameters dead_1h{};
    RothermelFuelClassParameters dead_10h{};
    RothermelFuelClassParameters dead_100h{};
    RothermelFuelClassParameters live_foliage{};
    amrex::Real fuel_bed_depth_m{};
    amrex::Real heat_content_j_kg{};
    amrex::Real particle_density_kg_m3{};
    amrex::Real total_mineral_fraction{};
    amrex::Real effective_mineral_fraction{};
    amrex::Real dead_moisture_of_extinction{};
};

/**
 * Return one of the original Anderson/Albini fire-behavior fuel models 1-13.
 *
 * model_number follows the published numbering. Values outside [1,13] are
 * rejected rather than mapped to a default model.
 */
[[nodiscard]] Anderson13FuelParameters
make_anderson13_fuel_parameters(int model_number);

/**
 * Single dead-fuel-class inputs for Rothermel implementation.
 *
 * This implementation intentionally supports only the homogeneous
 * dead fine-fuel path required by Anderson/Albini Fuel Model 1. This type is
 * not intended to imply that one scalar loading describes a general
 * multi-class Rothermel fuel bed.
 */
struct RothermelFuelParameters
{
    amrex::Real dead_1h_load_kg_m2{};
    amrex::Real dead_1h_sav_m_inv{};
    amrex::Real fuel_bed_depth_m{};
    amrex::Real dead_heat_content_j_kg{};
    amrex::Real particle_density_kg_m3{};
    amrex::Real total_mineral_fraction{};
    amrex::Real effective_mineral_fraction{};
    amrex::Real dead_moisture_of_extinction{};
};

/**
 * Original fire-behavior Fuel Model 1 (short grass), expressed in SI.
 *
 * Source values are the Albini (1976) original-13 model description and the
 * Anderson (1982) FM1 description. Constants common to the original models
 * are the published 8000 Btu/lb heat content, 32 lb/ft^3 particle density,
 * 5.55% total mineral content, and 1.00% effective mineral content.
 */
[[nodiscard]] RothermelFuelParameters make_fm1_fuel_parameters() noexcept;

} // namespace ERFFire

#endif // ERF_ROTHERMEL_FUEL_H
