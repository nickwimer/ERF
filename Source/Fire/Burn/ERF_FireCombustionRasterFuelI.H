#ifndef ERF_FIRE_COMBUSTION_RASTER_FUEL_I_H_
#define ERF_FIRE_COMBUSTION_RASTER_FUEL_I_H_

#ifndef ERF_FIRE_COMBUSTION_RASTER_H_
#error "ERF_FireCombustionRasterFuelI.H must be included through ERF_FireCombustionRaster.H"
#endif

#include <ERF_FireFuelCombustion.H>
#include <ERF_FireFuelRaster.H>

#include <AMReX_Gpu.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParReduce.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParallelFor.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ERFFire
{
namespace detail
{

inline amrex::Real
fire_spatial_combustion_water_increment(
    amrex::Real previous,
    amrex::Real next)
{
    if (!std::isfinite(previous)
        || !std::isfinite(next)
        || previous < amrex::Real(0)
        || next < amrex::Real(0)) {
        throw std::overflow_error(
            "Fire spatial combustion water accounting is not finite and nonnegative");
    }

    const amrex::Real scale =
        std::max(
            amrex::Real(1),
            std::max(std::abs(previous), std::abs(next)));
    const amrex::Real tolerance =
        amrex::Real(1024)
        * std::numeric_limits<amrex::Real>::epsilon()
        * scale;
    const amrex::Real increment = next - previous;

    if (increment < -tolerance) {
        throw std::logic_error(
            "Fire spatial combustion water accounting decreased for immutable fuel");
    }

    return increment < amrex::Real(0)
        ? amrex::Real(0)
        : increment;
}

} // namespace detail

inline void
FireCombustionRaster::validate_spatial_fuel(
    const FireFuelRaster& fuel_raster,
    bool validate_state) const
{
    const auto& fuel_geometry = fuel_raster.geometry();
    if (geometry_.nx != fuel_geometry.nx
        || geometry_.ny != fuel_geometry.ny
        || geometry_.xlo_m != fuel_geometry.xlo_m
        || geometry_.ylo_m != fuel_geometry.ylo_m
        || geometry_.dx_m != fuel_geometry.dx_m
        || geometry_.dy_m != fuel_geometry.dy_m) {
        throw std::invalid_argument(
            "Fire spatial combustion fuel geometry must match combustion geometry");
    }

    const amrex::MultiFab& fuel_values =
        fuel_raster.distributed_values();
    if (fuel_values.boxArray() != states_mf_.boxArray()
        || fuel_values.DistributionMap()
            != states_mf_.DistributionMap()
        || fuel_values.nComp()
            != FireFuelRasterComponents::component_count
        || fuel_values.nGrow() != 0) {
        throw std::invalid_argument(
            "Fire spatial combustion fuel raster must be co-located with combustion state");
    }

    const auto state_arrays = states_mf_.const_arrays();
    const auto fuel_arrays = fuel_values.const_arrays();
    const FireCombustionParameters base = parameters_;
    const int check_state = validate_state ? 1 : 0;

    const auto failures =
        amrex::ParReduce(
            amrex::TypeList<
                amrex::ReduceOpMax,
                amrex::ReduceOpMax>{},
            amrex::TypeList<int, int>{},
            states_mf_,
            [=] AMREX_GPU_DEVICE (
                int box_no,
                int i,
                int j,
                int k) noexcept
                -> amrex::GpuTuple<int, int>
            {
                FireFuelRasterCell fuel_cell;
                if (!detail::try_decode_fire_fuel_raster_cell(
                        fuel_arrays[box_no],
                        i,
                        j,
                        fuel_cell)) {
                    return {1, 0};
                }

                FireCombustionParameters local{};
                if (try_make_fire_combustion_parameters_for_fuel_cell(
                        base,
                        fuel_cell,
                        local)
                    != FireFuelCombustionParameterStatus::success) {
                    return {1, 0};
                }

                if (check_state == 0) {
                    return {0, 0};
                }

                const auto values = state_arrays[box_no];
                const FireCombustionState state{
                    values(i, j, k, ignited_area_fraction_comp),
                    values(i, j, k, remaining_dry_fuel_comp),
                    values(i, j, k, consumed_dry_fuel_comp),
                    values(i, j, k, sensible_energy_comp),
                    values(i, j, k, water_released_comp)};

                return {
                    0,
                    detail::combustion_state_valid(state, local)
                        ? 0
                        : 1};
            });

    int material_failure = amrex::get<0>(failures);
    int state_failure = amrex::get<1>(failures);
    amrex::ParallelDescriptor::ReduceIntMax(material_failure);
    amrex::ParallelDescriptor::ReduceIntMax(state_failure);

    if (material_failure != 0) {
        throw std::invalid_argument(
            "Fire spatial combustion currently requires every cell to be FM1 with prescribed dead 1-h moisture; NonBurnable remains disabled until barrier dynamics are enforced");
    }
    if (state_failure != 0) {
        throw std::invalid_argument(
            "Fire combustion state is inconsistent with the supplied immutable spatial fuel raster");
    }
}

inline void
FireCombustionRaster::rewrite_water_for_base_parameters()
{
    const amrex::Real water_per_kg_dry =
        parameters_.fuel_moisture_fraction
        + parameters_.combustion_water_yield_kg_per_kg_dry;

    for (amrex::MFIter mfi(states_mf_);
         mfi.isValid();
         ++mfi) {
        const amrex::Box box = mfi.validbox();
        const auto values = states_mf_.array(mfi);

        amrex::ParallelFor(
            box,
            [=] AMREX_GPU_DEVICE (
                int i,
                int j,
                int k) noexcept
            {
                values(i, j, k, water_released_comp) =
                    values(i, j, k, consumed_dry_fuel_comp)
                    * water_per_kg_dry;
            });
    }
    amrex::Gpu::streamSynchronize();

    amrex::Real local_water =
        states_mf_.sum(water_released_comp, true)
        * geometry_.dx_m
        * geometry_.dy_m;
    amrex::ParallelDescriptor::ReduceRealSum(local_water);

    if (!std::isfinite(local_water)
        || local_water < amrex::Real(0)) {
        throw std::overflow_error(
            "Fire base combustion water accounting is not finite and nonnegative");
    }
    totals_.water_released_kg = local_water;
}

inline void
FireCombustionRaster::rewrite_water_for_spatial_fuel(
    const FireFuelRaster& fuel_raster)
{
    validate_spatial_fuel(fuel_raster, false);

    const amrex::MultiFab& fuel_values =
        fuel_raster.distributed_values();
    const FireCombustionParameters base = parameters_;

    for (amrex::MFIter mfi(states_mf_);
         mfi.isValid();
         ++mfi) {
        const amrex::Box box = mfi.validbox();
        const auto values = states_mf_.array(mfi);
        const auto fuel = fuel_values.const_array(mfi);

        amrex::ParallelFor(
            box,
            [=] AMREX_GPU_DEVICE (
                int i,
                int j,
                int k) noexcept
            {
                FireFuelRasterCell fuel_cell;
                FireCombustionParameters local{};
                const bool decoded =
                    detail::try_decode_fire_fuel_raster_cell(
                        fuel,
                        i,
                        j,
                        fuel_cell);
                const auto status =
                    decoded
                    ? try_make_fire_combustion_parameters_for_fuel_cell(
                        base,
                        fuel_cell,
                        local)
                    : FireFuelCombustionParameterStatus::unsupported_model;

                if (status
                    != FireFuelCombustionParameterStatus::success) {
                    values(i, j, k, water_released_comp) =
                        amrex::Real(-1);
                    return;
                }

                values(i, j, k, water_released_comp) =
                    values(i, j, k, consumed_dry_fuel_comp)
                    * (local.fuel_moisture_fraction
                       + local.combustion_water_yield_kg_per_kg_dry);
            });
    }
    amrex::Gpu::streamSynchronize();

    validate_spatial_fuel(fuel_raster, true);

    amrex::Real local_water =
        states_mf_.sum(water_released_comp, true)
        * geometry_.dx_m
        * geometry_.dy_m;
    amrex::ParallelDescriptor::ReduceRealSum(local_water);

    if (!std::isfinite(local_water)
        || local_water < amrex::Real(0)) {
        throw std::overflow_error(
            "Fire spatial combustion water accounting is not finite and nonnegative");
    }
    totals_.water_released_kg = local_water;
}

inline FireCombustionRasterTotals
FireCombustionRaster::initialize_from_burned_fraction(
    const FireBurnedFractionRaster& burned_fraction,
    const FireFuelRaster& fuel_raster)
{
    validate_spatial_fuel(fuel_raster, true);

    FireCombustionRaster next(*this);
    next.rewrite_water_for_base_parameters();
    (void)next.initialize_from_burned_fraction(burned_fraction);
    next.rewrite_water_for_spatial_fuel(fuel_raster);

    const FireCombustionRasterTotals result = next.totals_;
    *this = std::move(next);
    return result;
}

inline FireCombustionRasterAdvance
FireCombustionRaster::advance_from_linear_sweep(
    const FirePerimeter& start_perimeter,
    const FirePerimeter& end_perimeter,
    const FireBurnedFractionRaster& burned_before,
    const FireBurnedFractionRaster& burned_after,
    const FireFuelRaster& fuel_raster,
    amrex::Real dt_s)
{
    validate_spatial_fuel(fuel_raster, true);
    const amrex::Real previous_water = totals_.water_released_kg;

    FireCombustionRaster next(*this);
    next.rewrite_water_for_base_parameters();
    FireCombustionRasterAdvance result =
        next.advance_from_linear_sweep(
            start_perimeter,
            end_perimeter,
            burned_before,
            burned_after,
            dt_s);
    next.rewrite_water_for_spatial_fuel(fuel_raster);

    result.totals = next.totals_;
    result.water_released_increment_kg =
        detail::fire_spatial_combustion_water_increment(
            previous_water,
            next.totals_.water_released_kg);

    *this = std::move(next);
    return result;
}

inline FireCombustionRasterAdvance
FireCombustionRaster::advance_from_front_linear_sweep(
    const FireFront& start_front,
    const FireFront& end_front,
    const FireBurnedFractionRaster& burned_before,
    const FireBurnedFractionRaster& burned_after,
    const FireFuelRaster& fuel_raster,
    amrex::Real dt_s)
{
    validate_spatial_fuel(fuel_raster, true);
    const amrex::Real previous_water = totals_.water_released_kg;

    FireCombustionRaster next(*this);
    next.rewrite_water_for_base_parameters();
    FireCombustionRasterAdvance result =
        next.advance_from_front_linear_sweep(
            start_front,
            end_front,
            burned_before,
            burned_after,
            dt_s);
    next.rewrite_water_for_spatial_fuel(fuel_raster);

    result.totals = next.totals_;
    result.water_released_increment_kg =
        detail::fire_spatial_combustion_water_increment(
            previous_water,
            next.totals_.water_released_kg);

    *this = std::move(next);
    return result;
}

inline FireCombustionRasterAdvance
FireCombustionRaster::advance_from_front_topology_event_sweep(
    const FireFront& start_front,
    const std::vector<std::vector<FireVec2>>& event_vertices_m,
    const FireFront& event_front,
    const FireBurnedFractionRaster& burned_before,
    const FireBurnedFractionRaster& burned_after,
    const FireFuelRaster& fuel_raster,
    amrex::Real dt_s)
{
    validate_spatial_fuel(fuel_raster, true);
    const amrex::Real previous_water = totals_.water_released_kg;

    FireCombustionRaster next(*this);
    next.rewrite_water_for_base_parameters();
    FireCombustionRasterAdvance result =
        next.advance_from_front_topology_event_sweep(
            start_front,
            event_vertices_m,
            event_front,
            burned_before,
            burned_after,
            dt_s);
    next.rewrite_water_for_spatial_fuel(fuel_raster);

    result.totals = next.totals_;
    result.water_released_increment_kg =
        detail::fire_spatial_combustion_water_increment(
            previous_water,
            next.totals_.water_released_kg);

    *this = std::move(next);
    return result;
}

inline FireCombustionRasterAdvance
FireCombustionRaster::advance_from_topology_event_sweep(
    const FirePerimeter& start_perimeter,
    const std::vector<FireVec2>& event_vertices_m,
    const FireFront& event_front,
    const FireBurnedFractionRaster& burned_before,
    const FireBurnedFractionRaster& burned_after,
    const FireFuelRaster& fuel_raster,
    amrex::Real dt_s)
{
    validate_spatial_fuel(fuel_raster, true);
    const amrex::Real previous_water = totals_.water_released_kg;

    FireCombustionRaster next(*this);
    next.rewrite_water_for_base_parameters();
    FireCombustionRasterAdvance result =
        next.advance_from_topology_event_sweep(
            start_perimeter,
            event_vertices_m,
            event_front,
            burned_before,
            burned_after,
            dt_s);
    next.rewrite_water_for_spatial_fuel(fuel_raster);

    result.totals = next.totals_;
    result.water_released_increment_kg =
        detail::fire_spatial_combustion_water_increment(
            previous_water,
            next.totals_.water_released_kg);

    *this = std::move(next);
    return result;
}

} // namespace ERFFire

#endif // ERF_FIRE_COMBUSTION_RASTER_FUEL_I_H_
