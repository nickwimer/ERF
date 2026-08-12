#include "ERF_FireLevel0Environment.H"

#include <AMReX_Arena.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_IntVect.H>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ERFFire
{
namespace
{

void
require (bool condition, const char* message)
{
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

void
require_level0_layout (
    const amrex::MultiFab& mf,
    const amrex::Box& expected,
    const char* message)
{
    require(
        mf.nComp() == 1,
        "fire environment source fields must have exactly one component");
    require(mf.boxArray().ixType() == expected.ixType(), message);
    require(
        amrex::match(mf.boxArray(), amrex::BoxArray(expected)),
        message);
}

amrex::FArrayBox
replicated_host_copy (
    const amrex::MultiFab& src,
    const amrex::Box& box,
    int nghost)
{
    amrex::FArrayBox result(box, 1, amrex::The_Pinned_Arena());
    src.copyTo(result, 0, 0, 1, nghost);
    return result;
}

amrex::Real
flat_ground_height (
    const amrex::MultiFab& z_phys_nd,
    const amrex::Box& cell_domain)
{
    amrex::Box bottom =
        amrex::convert(cell_domain, amrex::IntVect(1, 1, 1));
    bottom.setRange(2, bottom.smallEnd(2));

    const amrex::FArrayBox surface =
        replicated_host_copy(z_phys_nd, bottom, 0);
    const auto surface_arr = surface.const_array();

    const int ilo = bottom.smallEnd(0);
    const int ihi = bottom.bigEnd(0);
    const int jlo = bottom.smallEnd(1);
    const int jhi = bottom.bigEnd(1);
    const int k = bottom.smallEnd(2);

    const amrex::Real ground = surface_arr(ilo, jlo, k);
    require(
        std::isfinite(ground),
        "fire flat-grid ground height must be finite");

    // Any horizontal variation, including a
    // roundoff-sized one, is terrain for coupling purposes and is deferred.
    for (int j = jlo; j <= jhi; ++j) {
        for (int i = ilo; i <= ihi; ++i) {
            const amrex::Real value = surface_arr(i, j, k);
            if (!std::isfinite(value) || value != ground) {
                throw std::invalid_argument(
                    "M7 fire environment requires an exactly flat level-0 physical surface");
            }
        }
    }

    return ground;
}

std::vector<amrex::Real>
cell_center_heights (
    const amrex::MultiFab& z_phys_cc,
    const amrex::Box& domain)
{
    amrex::Box column(domain);
    column.setRange(0, domain.smallEnd(0));
    column.setRange(1, domain.smallEnd(1));

    const amrex::FArrayBox column_fab =
        replicated_host_copy(z_phys_cc, column, 0);
    const auto column_arr = column_fab.const_array();

    const int i = column.smallEnd(0);
    const int j = column.smallEnd(1);
    const int klo = domain.smallEnd(2);
    const int khi = domain.bigEnd(2);

    std::vector<amrex::Real> result(
        static_cast<std::size_t>(domain.length(2)));
    for (int k = klo; k <= khi; ++k) {
        result[static_cast<std::size_t>(k - klo)] =
            column_arr(i, j, k);
    }
    return result;
}

void
require_horizontally_uniform_z_phys_cc_plane (
    const amrex::MultiFab& z_phys_cc,
    const amrex::Box& domain,
    int k,
    amrex::Real expected_height_m)
{
    amrex::Box plane(domain);
    plane.setRange(2, k);

    const amrex::FArrayBox plane_fab =
        replicated_host_copy(z_phys_cc, plane, 0);
    const auto plane_arr = plane_fab.const_array();

    for (int j = plane.smallEnd(1); j <= plane.bigEnd(1); ++j) {
        for (int i = plane.smallEnd(0); i <= plane.bigEnd(0); ++i) {
            const amrex::Real value = plane_arr(i, j, k);
            if (!std::isfinite(value) || value != expected_height_m) {
                throw std::invalid_argument(
                    "M7 fire environment requires horizontally uniform "
                    "z_phys_cc on sampled levels");
            }
        }
    }
}

amrex::Box
horizontal_velocity_plane_box (
    const amrex::Box& domain,
    const amrex::IntVect& staggering,
    int k)
{
    amrex::Box plane = amrex::convert(domain, staggering);
    plane.setRange(2, k);
    plane.grow(amrex::IntVect(1, 1, 0));
    return plane;
}

void
validate_scope_and_layout (
    const ERFFireLevel0EnvironmentInputs& inputs)
{
    require(
        inputs.configured_max_level == 0,
        "M7 fire environment supports only configurations with max_level = 0");

    require(
        inputs.mesh_type != MeshType::VariableDz,
        "M7 fire environment defers VariableDz / terrain-fitted sampling");

    require(
        inputs.terrain_type != TerrainType::EB,
        "M7 fire environment does not support EB terrain");
    require(
        inputs.terrain_type != TerrainType::ImmersedForcing,
        "M7 fire environment does not support immersed-forcing terrain");
    require(
        inputs.terrain_type != TerrainType::MovingFittedMesh,
        "M7 fire environment does not support moving fitted terrain");
    require(
        inputs.buildings_type == BuildingsType::None,
        "M7 fire environment does not support immersed buildings");

    const amrex::Box& domain = inputs.geometry.Domain();
    require(
        domain.cellCentered(),
        "fire environment requires a cell-centered ERF geometry domain");

    const amrex::Box expected_u =
        amrex::convert(domain, amrex::IntVect(1, 0, 0));
    const amrex::Box expected_v =
        amrex::convert(domain, amrex::IntVect(0, 1, 0));
    const amrex::Box expected_zcc = domain;
    const amrex::Box expected_znd =
        amrex::convert(domain, amrex::IntVect(1, 1, 1));

    require_level0_layout(
        inputs.x_velocity, expected_u,
        "fire x-velocity BoxArray does not cover the level-0 x-face domain");
    require_level0_layout(
        inputs.y_velocity, expected_v,
        "fire y-velocity BoxArray does not cover the level-0 y-face domain");
    require_level0_layout(
        inputs.z_phys_cc, expected_zcc,
        "fire z_phys_cc BoxArray does not cover the level-0 cell domain");
    require_level0_layout(
        inputs.z_phys_nd, expected_znd,
        "fire z_phys_nd BoxArray does not cover the level-0 nodal domain");

    const amrex::IntVect x_ng = inputs.x_velocity.nGrowVect();
    const amrex::IntVect y_ng = inputs.y_velocity.nGrowVect();
    require(
        x_ng[0] >= 1 && x_ng[1] >= 1 && x_ng[2] >= 1,
        "fire x-velocity snapshot requires at least one filled ghost cell");
    require(
        y_ng[0] >= 1 && y_ng[1] >= 1 && y_ng[2] >= 1,
        "fire y-velocity snapshot requires at least one filled ghost cell");
}

} // namespace


std::vector<amrex::Real>
erf_fire_level0_flat_vertical_faces_agl (
    const ERFFireLevel0EnvironmentInputs& inputs)
{
    validate_scope_and_layout(inputs);

    const amrex::Box& domain = inputs.geometry.Domain();
    const amrex::Real ground =
        flat_ground_height(inputs.z_phys_nd, domain);

    const amrex::Box nodal_domain =
        amrex::convert(
            domain,
            amrex::IntVect(1, 1, 1));
    const amrex::FArrayBox coordinates =
        replicated_host_copy(
            inputs.z_phys_nd,
            nodal_domain,
            0);
    const auto z = coordinates.const_array();

    const int ilo = nodal_domain.smallEnd(0);
    const int ihi = nodal_domain.bigEnd(0);
    const int jlo = nodal_domain.smallEnd(1);
    const int jhi = nodal_domain.bigEnd(1);
    const int klo = nodal_domain.smallEnd(2);
    const int khi = nodal_domain.bigEnd(2);

    std::vector<amrex::Real> result(
        static_cast<std::size_t>(
            khi - klo + 1),
        amrex::Real(0));

    for (int k = klo; k <= khi; ++k) {
        const amrex::Real plane_height =
            z(ilo, jlo, k);
        require(
            std::isfinite(plane_height),
            "fire level-0 nodal height must be finite");

        for (int j = jlo; j <= jhi; ++j) {
            for (int i = ilo; i <= ihi; ++i) {
                if (!std::isfinite(z(i, j, k))
                    || z(i, j, k) != plane_height) {
                    throw std::invalid_argument(
                        "M9 Fire feedback requires horizontally uniform z_phys_nd planes");
                }
            }
        }

        const amrex::Real agl =
            plane_height - ground;
        require(
            std::isfinite(agl),
            "fire level-0 AGL face height must be finite");

        const std::size_t index =
            static_cast<std::size_t>(k - klo);
        result[index] = agl;

        if (index == 0) {
            require(
                agl == amrex::Real(0),
                "fire level-0 first AGL face must be exactly zero");
        } else {
            require(
                result[index] > result[index - 1],
                "fire level-0 AGL faces must increase strictly");
        }
    }

    return result;
}

FireFlatEnvironmentSampler
freeze_erf_level0_environment (
    const ERFFireLevel0EnvironmentInputs& inputs,
    amrex::Real reference_height_agl_m)
{
    validate_scope_and_layout(inputs);

    if (!std::isfinite(reference_height_agl_m) ||
        reference_height_agl_m < amrex::Real(0)) {
        throw std::invalid_argument(
            "fire atmospheric reference height AGL must be finite and nonnegative");
    }

    const amrex::Box& domain = inputs.geometry.Domain();
    const amrex::Real ground =
        flat_ground_height(inputs.z_phys_nd, domain);

    const std::vector<amrex::Real> z_cell_center_m =
        cell_center_heights(inputs.z_phys_cc, domain);

    const amrex::Real target_height_m =
        ground + reference_height_agl_m;
    if (!std::isfinite(target_height_m)) {
        throw std::invalid_argument(
            "fire atmospheric reference height is not representable");
    }

    const FireVerticalLinearBracket bracket =
        fire_vertical_linear_bracket(
            z_cell_center_m, target_height_m);

    const int domain_klo = domain.smallEnd(2);
    const int lower_k =
        domain_klo + static_cast<int>(bracket.lower_k);
    const int upper_k =
        domain_klo + static_cast<int>(bracket.upper_k);

    require_horizontally_uniform_z_phys_cc_plane(
        inputs.z_phys_cc, domain, lower_k,
        z_cell_center_m[bracket.lower_k]);
    if (upper_k != lower_k) {
        require_horizontally_uniform_z_phys_cc_plane(
            inputs.z_phys_cc, domain, upper_k,
            z_cell_center_m[bracket.upper_k]);
    }

    const amrex::Box u_lower_box =
        horizontal_velocity_plane_box(
            domain, amrex::IntVect(1, 0, 0), lower_k);
    const amrex::Box u_upper_box =
        horizontal_velocity_plane_box(
            domain, amrex::IntVect(1, 0, 0), upper_k);
    const amrex::Box v_lower_box =
        horizontal_velocity_plane_box(
            domain, amrex::IntVect(0, 1, 0), lower_k);
    const amrex::Box v_upper_box =
        horizontal_velocity_plane_box(
            domain, amrex::IntVect(0, 1, 0), upper_k);

    const amrex::FArrayBox u_lower =
        replicated_host_copy(
            inputs.x_velocity, u_lower_box, 1);
    const amrex::FArrayBox u_upper =
        replicated_host_copy(
            inputs.x_velocity, u_upper_box, 1);
    const amrex::FArrayBox v_lower =
        replicated_host_copy(
            inputs.y_velocity, v_lower_box, 1);
    const amrex::FArrayBox v_upper =
        replicated_host_copy(
            inputs.y_velocity, v_upper_box, 1);

    const auto u_lower_arr = u_lower.const_array();
    const auto u_upper_arr = u_upper.const_array();
    const auto v_lower_arr = v_lower.const_array();
    const auto v_upper_arr = v_upper.const_array();

    const auto dx = inputs.geometry.CellSizeArray();
    const auto plo = inputs.geometry.ProbLoArray();

    const std::size_t nx =
        static_cast<std::size_t>(domain.length(0));
    const std::size_t ny =
        static_cast<std::size_t>(domain.length(1));

    FireFlatEnvironmentLayout2D layout(
        plo[0], plo[1], dx[0], dx[1], nx, ny);

    std::vector<amrex::Real> u_wind_mps(
        layout.u_storage_size());
    std::vector<amrex::Real> v_wind_mps(
        layout.v_storage_size());

    const int domain_ilo = domain.smallEnd(0);
    const int domain_jlo = domain.smallEnd(1);
    const int nx_int = static_cast<int>(nx);
    const int ny_int = static_cast<int>(ny);

    for (int j = -1; j <= ny_int; ++j) {
        for (int i = -1; i <= nx_int + 1; ++i) {
            const int src_i = domain_ilo + i;
            const int src_j = domain_jlo + j;
            const amrex::Real value =
                fire_vertical_linear_interpolate(
                    u_lower_arr(src_i, src_j, lower_k),
                    u_upper_arr(src_i, src_j, upper_k),
                    bracket);
            u_wind_mps[
                layout.u_storage_index(i, j)] = value;
        }
    }

    for (int j = -1; j <= ny_int + 1; ++j) {
        for (int i = -1; i <= nx_int; ++i) {
            const int src_i = domain_ilo + i;
            const int src_j = domain_jlo + j;
            const amrex::Real value =
                fire_vertical_linear_interpolate(
                    v_lower_arr(src_i, src_j, lower_k),
                    v_upper_arr(src_i, src_j, upper_k),
                    bracket);
            v_wind_mps[
                layout.v_storage_index(i, j)] = value;
        }
    }

    return FireFlatEnvironmentSampler(
        std::move(layout),
        reference_height_agl_m,
        std::move(u_wind_mps),
        std::move(v_wind_mps));
}

} // namespace ERFFire
