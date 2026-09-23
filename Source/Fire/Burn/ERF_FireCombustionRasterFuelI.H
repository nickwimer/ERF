#ifndef ERF_FIRE_COMBUSTION_RASTER_FUEL_I_H_
#define ERF_FIRE_COMBUSTION_RASTER_FUEL_I_H_

#ifndef ERF_FIRE_COMBUSTION_RASTER_H_
#error "ERF_FireCombustionRasterFuelI.H must be included through ERF_FireCombustionRaster.H"
#endif

#include <ERF_FireFront.H>
#include <ERF_FireFuelCombustion.H>
#include <ERF_FireFuelRaster.H>
#include <ERF_FirePerimeterSweep.H>

#include <AMReX_Gpu.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParReduce.H>
#include <AMReX_ParallelDescriptor.H>

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

inline void
validate_spatial_fuel_impl(
    const FireCartesianRasterGeometry2D& geometry,
    const FireCombustionParameters& base,
    const amrex::MultiFab& states,
    const FireFuelRaster& fuel_raster,
    bool validate_state)
{
    const auto& fuel_geometry = fuel_raster.geometry();
    if (geometry.nx != fuel_geometry.nx
        || geometry.ny != fuel_geometry.ny
        || geometry.xlo_m != fuel_geometry.xlo_m
        || geometry.ylo_m != fuel_geometry.ylo_m
        || geometry.dx_m != fuel_geometry.dx_m
        || geometry.dy_m != fuel_geometry.dy_m) {
        throw std::invalid_argument(
            "Fire spatial combustion fuel geometry must match combustion geometry");
    }

    const amrex::MultiFab& fuel_values =
        fuel_raster.distributed_values();
    if (fuel_values.boxArray() != states.boxArray()
        || fuel_values.DistributionMap()
            != states.DistributionMap()
        || fuel_values.nComp()
            != FireFuelRasterComponents::component_count
        || fuel_values.nGrow() != 0) {
        throw std::invalid_argument(
            "Fire spatial combustion fuel raster must be co-located with combustion state");
    }

    const auto state_arrays = states.const_arrays();
    const auto fuel_arrays = fuel_values.const_arrays();
    const int check_state = validate_state ? 1 : 0;

    const auto failures =
        amrex::ParReduce(
            amrex::TypeList<
                amrex::ReduceOpMax,
                amrex::ReduceOpMax>{},
            amrex::TypeList<int, int>{},
            states,
            [=] AMREX_GPU_DEVICE (
                int box_no,
                int i,
                int j,
                int k) noexcept
                -> amrex::GpuTuple<int, int>
            {
                FireFuelRasterCell fuel_cell;
                if (!try_decode_fire_fuel_raster_cell(
                        fuel_arrays[box_no],
                        i,
                        j,
                        fuel_cell)) {
                    return {1, 0};
                }

                FireFuelCombustionAccounting accounting{};
                const auto status =
                    try_make_spatial_fire_combustion_accounting(
                        base,
                        fuel_cell,
                        accounting);

                if (status
                        != FireFuelCombustionAccountingStatus::success
                    && status
                        != FireFuelCombustionAccountingStatus::nonburnable) {
                    return {1, 0};
                }
                const FireCombustionParameters local =
                    accounting.parameters;

                if (check_state == 0) {
                    return {0, 0};
                }

                const auto values = state_arrays[box_no];
                const FireCombustionState state{
                    values(
                        i,
                        j,
                        k,
                        FireCombustionRaster::ignited_area_fraction_comp),
                    values(
                        i,
                        j,
                        k,
                        FireCombustionRaster::remaining_dry_fuel_comp),
                    values(
                        i,
                        j,
                        k,
                        FireCombustionRaster::consumed_dry_fuel_comp),
                    values(
                        i,
                        j,
                        k,
                        FireCombustionRaster::sensible_energy_comp),
                    values(
                        i,
                        j,
                        k,
                        FireCombustionRaster::water_released_comp)};

                if (status
                    == FireFuelCombustionAccountingStatus::nonburnable) {
                    const bool canonical_zero =
                        state.ignited_area_fraction == amrex::Real(0)
                        && state.remaining_dry_fuel_kg_m2 == amrex::Real(0)
                        && state.consumed_dry_fuel_kg_m2 == amrex::Real(0)
                        && state.sensible_energy_j_m2 == amrex::Real(0)
                        && state.water_released_kg_m2 == amrex::Real(0);
                    return {0, canonical_zero ? 0 : 1};
                }

                return {
                    0,
                    combustion_state_valid(state, local)
                        ? 0
                        : 1};
            });

    int material_failure = amrex::get<0>(failures);
    int state_failure = amrex::get<1>(failures);
    amrex::ParallelDescriptor::ReduceIntMax(material_failure);
    amrex::ParallelDescriptor::ReduceIntMax(state_failure);

    if (material_failure != 0) {
        throw std::invalid_argument(
            "Fire spatial combustion requires complete Anderson-13 material or NonBurnable cells");
    }
    if (state_failure != 0) {
        throw std::invalid_argument(
            "Fire combustion state is inconsistent with the supplied immutable spatial fuel raster");
    }
}

inline amrex::Real
rewrite_water_for_base_parameters_impl(
    const FireCombustionParameters& parameters,
    const FireCartesianRasterGeometry2D& geometry,
    amrex::MultiFab& states)
{
    const amrex::Real water_per_kg_dry =
        parameters.fuel_moisture_fraction
        + parameters.combustion_water_yield_kg_per_kg_dry;

    for (amrex::MFIter mfi(states);
         mfi.isValid();
         ++mfi) {
        const amrex::Box box = mfi.validbox();
        const auto values = states.array(mfi);

        amrex::ParallelFor(
            box,
            [=] AMREX_GPU_DEVICE (
                int i,
                int j,
                int k) noexcept
            {
                values(
                    i,
                    j,
                    k,
                    FireCombustionRaster::water_released_comp) =
                    values(
                        i,
                        j,
                        k,
                        FireCombustionRaster::consumed_dry_fuel_comp)
                    * water_per_kg_dry;
            });
    }
    amrex::Gpu::streamSynchronize();

    amrex::Real local_water =
        states.sum(
            FireCombustionRaster::water_released_comp,
            true)
        * geometry.dx_m
        * geometry.dy_m;
    amrex::ParallelDescriptor::ReduceRealSum(local_water);

    if (!std::isfinite(local_water)
        || local_water < amrex::Real(0)) {
        throw std::overflow_error(
            "Fire base combustion water accounting is not finite and nonnegative");
    }
    return local_water;
}

inline amrex::Real
rewrite_water_for_spatial_fuel_impl(
    const FireCombustionParameters& base,
    const FireCartesianRasterGeometry2D& geometry,
    amrex::MultiFab& states,
    const FireFuelRaster& fuel_raster)
{
    validate_spatial_fuel_impl(
        geometry,
        base,
        states,
        fuel_raster,
        false);

    const amrex::MultiFab& fuel_values =
        fuel_raster.distributed_values();

    for (amrex::MFIter mfi(states);
         mfi.isValid();
         ++mfi) {
        const amrex::Box box = mfi.validbox();
        const auto values = states.array(mfi);
        const auto fuel = fuel_values.const_array(mfi);

        amrex::ParallelFor(
            box,
            [=] AMREX_GPU_DEVICE (
                int i,
                int j,
                int k) noexcept
            {
                FireFuelRasterCell fuel_cell;
                FireFuelCombustionAccounting accounting{};
                const bool decoded =
                    try_decode_fire_fuel_raster_cell(
                        fuel,
                        i,
                        j,
                        fuel_cell);
                const auto status =
                    decoded
                    ? try_make_spatial_fire_combustion_accounting(
                        base,
                        fuel_cell,
                        accounting)
                    : FireFuelCombustionAccountingStatus::invalid_model;
                const FireCombustionParameters local =
                    accounting.parameters;

                if (status
                    == FireFuelCombustionAccountingStatus::nonburnable) {
                    values(
                        i,
                        j,
                        k,
                        FireCombustionRaster::water_released_comp) =
                        amrex::Real(0);
                    return;
                }

                if (status
                    != FireFuelCombustionAccountingStatus::success) {
                    values(
                        i,
                        j,
                        k,
                        FireCombustionRaster::water_released_comp) =
                        amrex::Real(-1);
                    return;
                }

                values(
                    i,
                    j,
                    k,
                    FireCombustionRaster::water_released_comp) =
                    values(
                        i,
                        j,
                        k,
                        FireCombustionRaster::consumed_dry_fuel_comp)
                    * (local.fuel_moisture_fraction
                       + local.combustion_water_yield_kg_per_kg_dry);
            });
    }
    amrex::Gpu::streamSynchronize();

    validate_spatial_fuel_impl(
        geometry,
        base,
        states,
        fuel_raster,
        true);

    amrex::Real local_water =
        states.sum(
            FireCombustionRaster::water_released_comp,
            true)
        * geometry.dx_m
        * geometry.dy_m;
    amrex::ParallelDescriptor::ReduceRealSum(local_water);

    if (!std::isfinite(local_water)
        || local_water < amrex::Real(0)) {
        throw std::overflow_error(
            "Fire spatial combustion water accounting is not finite and nonnegative");
    }
    return local_water;
}

} // namespace detail

inline void
FireCombustionRaster::validate_spatial_fuel(
    const FireFuelRaster& fuel_raster,
    bool validate_state) const
{
    detail::validate_spatial_fuel_impl(
        geometry_,
        parameters_,
        states_mf_,
        fuel_raster,
        validate_state);
}

inline void
FireCombustionRaster::rewrite_water_for_base_parameters()
{
    totals_.water_released_kg =
        detail::rewrite_water_for_base_parameters_impl(
            parameters_,
            geometry_,
            states_mf_);
}

inline void
FireCombustionRaster::rewrite_water_for_spatial_fuel(
    const FireFuelRaster& fuel_raster)
{
    totals_.water_released_kg =
        detail::rewrite_water_for_spatial_fuel_impl(
            parameters_,
            geometry_,
            states_mf_,
            fuel_raster);
}

inline FireCombustionRasterTotals
FireCombustionRaster::initialize_from_burned_fraction(
    const FireBurnedFractionRaster& burned_fraction,
    const FireFuelRaster& fuel_raster)
{
    validate_spatial_fuel(fuel_raster, false);
    return initialize_from_burned_fraction_impl(
        burned_fraction,
        &fuel_raster);
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
    return detail::advance_fire_combustion_vertex_sweep(
        *this,
        start_perimeter,
        end_perimeter.vertices_m(),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        burned_before,
        burned_after,
        &fuel_raster,
        dt_s);
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
    const FirePerimeter& representative_start =
        start_front.components().front().perimeter;
    const FirePerimeter& representative_end =
        end_front.components().front().perimeter;
    return detail::advance_fire_combustion_vertex_sweep(
        *this,
        representative_start,
        representative_end.vertices_m(),
        nullptr,
        &start_front,
        &end_front,
        nullptr,
        nullptr,
        burned_before,
        burned_after,
        &fuel_raster,
        dt_s);
}

inline FireCombustionRasterAdvance
FireCombustionRaster::advance_from_front_rk2_sweep(
    const FireFront& start_front,
    const std::vector<std::vector<FireVec2>>& midpoint_vertices_m,
    const FireFront& end_front,
    const FireBurnedFractionRaster& burned_before,
    const FireBurnedFractionRaster& burned_after,
    const FireFuelRaster& fuel_raster,
    amrex::Real dt_s)
{
    validate_spatial_fuel(fuel_raster, true);
    (void)interpolate_fire_front_rk2_sweep(
        start_front, midpoint_vertices_m, end_front, amrex::Real(0));
    const FirePerimeter& representative_start =
        start_front.components().front().perimeter;
    const FirePerimeter& representative_end =
        end_front.components().front().perimeter;
    return detail::advance_fire_combustion_vertex_sweep(
        *this,
        representative_start,
        representative_end.vertices_m(),
        nullptr,
        &start_front,
        &end_front,
        nullptr,
        nullptr,
        burned_before,
        burned_after,
        &fuel_raster,
        dt_s,
        &midpoint_vertices_m);
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
    (void)interpolate_fire_front_topology_event_sweep(
        start_front,
        event_vertices_m,
        amrex::Real(0.0));
    const FirePerimeter& representative_start =
        start_front.components().front().perimeter;
    return detail::advance_fire_combustion_vertex_sweep(
        *this,
        representative_start,
        event_vertices_m.front(),
        &event_front,
        nullptr,
        nullptr,
        &start_front,
        &event_vertices_m,
        burned_before,
        burned_after,
        &fuel_raster,
        dt_s);
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
    return detail::advance_fire_combustion_vertex_sweep(
        *this,
        start_perimeter,
        event_vertices_m,
        &event_front,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        burned_before,
        burned_after,
        &fuel_raster,
        dt_s);
}

} // namespace ERFFire

#endif // ERF_FIRE_COMBUSTION_RASTER_FUEL_I_H_
