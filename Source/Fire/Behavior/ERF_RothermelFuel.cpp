#include <ERF_RothermelFuel.H>

#include <stdexcept>

namespace ERFFire
{

Anderson13FuelParameters
make_anderson13_fuel_parameters (int model_number)
{
    Anderson13FuelParameters result{};
    if (!try_make_anderson13_fuel_parameters(
            model_number,
            result)) {
        throw std::invalid_argument(
            "Anderson/Albini fuel model number must lie in [1,13]");
    }
    return result;
}

RothermelFuelParameters
make_fm1_fuel_parameters () noexcept
{
    Anderson13FuelParameters fm1{};
    (void)try_make_anderson13_fuel_parameters(1, fm1);

    // Preserve the existing single-dead-class API as an exact projection of
    // model 1. Models 2-13 require the generalized multi-class evaluator and
    // must never be silently collapsed into this type.
    return {
        fm1.dead_1h.dry_load_kg_m2,
        fm1.dead_1h.sav_m_inv,
        fm1.fuel_bed_depth_m,
        fm1.heat_content_j_kg,
        fm1.particle_density_kg_m3,
        fm1.total_mineral_fraction,
        fm1.effective_mineral_fraction,
        fm1.dead_moisture_of_extinction
    };
}

} // namespace ERFFire
