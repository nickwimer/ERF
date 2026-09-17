#ifndef ERF_FIRE_FUEL_BARRIER_REMESHER_H_
#define ERF_FIRE_FUEL_BARRIER_REMESHER_H_

#include <ERF_FireFuelBarrierFront.H>
#include <ERF_FirePerimeterRemesher.H>

#include <stdexcept>

namespace ERFFire
{

/**
 * Remesh a FireFront while preserving the NonBurnable edge-exclusion
 * invariant.
 *
 * The input front must already avoid the interior of the NonBurnable cell
 * union. The ordinary geometry-only remesher is then applied unchanged and
 * the resulting front is validated collectively before it is returned.
 *
 * Straight-edge refinement cannot change the represented polygon, but
 * coarsening can replace two legal incident edges by a new chord. If that
 * chord enters NonBurnable material this routine rejects the remesh result
 * rather than committing an arbitrary barrier detour or silently changing
 * represented topology.
 *
 * Every rank must call this routine with the same replicated FireFront and
 * FireFuelRaster. Edge validation uses the same ordered sparse collective as
 * the barrier contact primitives.
 */
[[nodiscard]] inline FireFrontRemeshResult
collective_remesh_fire_front_avoiding_nonburnable(
    const FireFront& front,
    const FirePerimeterRemeshOptions& options,
    const FireFuelRaster& fuel_raster)
{
    collective_require_fire_front_edges_avoid_nonburnable(
        front,
        fuel_raster,
        "pre-remesh state");

    FireFrontRemeshResult result =
        remesh_front(front, options);

    collective_require_fire_front_edges_avoid_nonburnable(
        result.front,
        fuel_raster,
        "post-remesh state");

    return result;
}

} // namespace ERFFire

#endif // ERF_FIRE_FUEL_BARRIER_REMESHER_H_
