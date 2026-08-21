#include "ERF_FireLevel0Environment.H"

#include <ERF_TerrainSource.H>

#include <AMReX_Arena.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Extension.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Gpu.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>
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

    const amrex::BoxArray& boxes = mf.boxArray();
    require(boxes.ixType() == expected.ixType(), message);
    require(boxes.contains(expected), message);

    for (int index = 0; index < boxes.size(); ++index) {
        require(expected.contains(boxes[index]), message);
    }
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

amrex::MFInfo
flat_coordinate_mf_info()
{
    amrex::MFInfo info;
    info.SetArena(amrex::The_Pinned_Arena());
    return info;
}

std::vector<amrex::Real>
canonical_vertical_column(
    const amrex::MultiFab& source,
    amrex::Box column)
{
    column.setRange(0, column.smallEnd(0));
    column.setRange(1, column.smallEnd(1));

    amrex::BoxArray column_boxes(column);
    amrex::Vector<int> processor_map(
        1,
        amrex::ParallelDescriptor::IOProcessorNumber());
    const amrex::DistributionMapping column_dm(
        std::move(processor_map));

    amrex::MultiFab column_values(
        column_boxes,
        column_dm,
        1,
        0,
        flat_coordinate_mf_info());
    column_values.ParallelCopy(
        source,
        0,
        0,
        1,
        0,
        0);
    amrex::Gpu::streamSynchronize();

    const int klo = column.smallEnd(2);
    const int khi = column.bigEnd(2);
    std::vector<amrex::Real> result(
        static_cast<std::size_t>(khi - klo + 1),
        amrex::Real(0));

    if (amrex::ParallelDescriptor::IOProcessor()) {
        for (amrex::MFIter mfi(column_values);
             mfi.isValid();
             ++mfi) {
            const auto values =
                column_values.const_array(mfi);
            const int i =
                column.smallEnd(0);
            const int j =
                column.smallEnd(1);
            for (int k = klo; k <= khi; ++k) {
                result[
                    static_cast<std::size_t>(k - klo)] =
                        values(i, j, k);
            }
        }
    }

    amrex::ParallelDescriptor::Bcast(
        result.data(),
        result.size(),
        amrex::ParallelDescriptor::IOProcessorNumber());
    return result;
}

void
require_exact_uniform_plane(
    const amrex::MultiFab& source,
    const amrex::Box& plane,
    amrex::Real expected,
    const char* message)
{
    amrex::Real plane_min =
        source.min(
            plane,
            0,
            0,
            true);
    amrex::Real plane_max =
        source.max(
            plane,
            0,
            0,
            true);

    amrex::ParallelDescriptor::ReduceRealMin(
        plane_min);
    amrex::ParallelDescriptor::ReduceRealMax(
        plane_max);

    if (!std::isfinite(plane_min)
        || !std::isfinite(plane_max)
        || plane_min != expected
        || plane_max != expected) {
        throw std::invalid_argument(message);
    }
}

void
require_exact_uniform_profile(
    const amrex::MultiFab& source,
    const amrex::Box& expected_box,
    const std::vector<amrex::Real>& expected,
    const char* message)
{
    if (expected.size()
        != static_cast<std::size_t>(
            expected_box.length(2))) {
        throw std::logic_error(
            "fire flat-coordinate profile has the wrong vertical size");
    }
    if (expected.size()
        > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "fire flat-coordinate profile exceeds int count");
    }

    const int klo =
        expected_box.smallEnd(2);
    const int profile_size =
        static_cast<int>(expected.size());

    std::vector<amrex::Real> plane_min(
        expected.size(),
        std::numeric_limits<amrex::Real>::max());
    std::vector<amrex::Real> plane_max(
        expected.size(),
        std::numeric_limits<amrex::Real>::lowest());

    for (int offset = 0;
         offset < profile_size;
         ++offset) {
        amrex::Box plane(expected_box);
        plane.setRange(
            2,
            klo + offset);

        plane_min[
            static_cast<std::size_t>(offset)] =
                source.min(
                    plane,
                    0,
                    0,
                    true);
        plane_max[
            static_cast<std::size_t>(offset)] =
                source.max(
                    plane,
                    0,
                    0,
                    true);
    }

    amrex::ParallelDescriptor::ReduceRealMin(
        plane_min.data(),
        profile_size);
    amrex::ParallelDescriptor::ReduceRealMax(
        plane_max.data(),
        profile_size);

    for (int offset = 0;
         offset < profile_size;
         ++offset) {
        const std::size_t index =
            static_cast<std::size_t>(offset);
        if (!std::isfinite(plane_min[index])
            || !std::isfinite(plane_max[index])
            || plane_min[index] != expected[index]
            || plane_max[index] != expected[index]) {
            throw std::invalid_argument(message);
        }
    }
}

amrex::Real
flat_ground_height (
    const amrex::MultiFab& z_phys_nd,
    const amrex::Box& cell_domain)
{
    amrex::Box bottom =
        amrex::convert(cell_domain, amrex::IntVect(1, 1, 1));
    bottom.setRange(2, bottom.smallEnd(2));

    const std::vector<amrex::Real> canonical =
        canonical_vertical_column(
            z_phys_nd,
            bottom);
    const amrex::Real ground =
        canonical.front();
    require(
        std::isfinite(ground),
        "fire flat-grid ground height must be finite");

    // Any horizontal variation, including a
    // roundoff-sized one, is terrain for coupling purposes and is deferred.
    require_exact_uniform_plane(
        z_phys_nd,
        bottom,
        ground,
        "fire environment requires an exactly flat level-0 physical surface");

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

    return canonical_vertical_column(
        z_phys_cc,
        column);
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

    require_exact_uniform_plane(
        z_phys_cc,
        plane,
        expected_height_m,
        "fire environment requires horizontally uniform "
        "z_phys_cc on sampled levels");
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

struct FlatAxisStencil
{
    int lower{};
    amrex::Real upper_weight{};
};

FlatAxisStencil
make_flat_axis_stencil (
    amrex::Real coordinate_m,
    amrex::Real origin_m,
    amrex::Real spacing_m,
    amrex::Real native_offset_cells)
{
    const amrex::Real logical =
        (coordinate_m - origin_m) / spacing_m
        - native_offset_cells;
    const amrex::Real floored = std::floor(logical);

    if (floored
            < static_cast<amrex::Real>(
                  std::numeric_limits<int>::min())
        || floored
            > static_cast<amrex::Real>(
                  std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "fire distributed flat-wind stencil index overflow");
    }

    const int lower = static_cast<int>(floored);
    const amrex::Real upper_weight =
        logical - static_cast<amrex::Real>(lower);

    return {lower, upper_weight};
}

AMREX_NO_INLINE
amrex::Real
flat_bilinear (
    amrex::Real q00,
    amrex::Real q10,
    amrex::Real q01,
    amrex::Real q11,
    amrex::Real wx,
    amrex::Real wy)
{
    const amrex::Real lower =
        (amrex::Real(1) - wx) * q00 + wx * q10;
    const amrex::Real upper =
        (amrex::Real(1) - wx) * q01 + wx * q11;
    return (amrex::Real(1) - wy) * lower + wy * upper;
}

using SparseVelocityKey = std::tuple<int, int, int>;

amrex::MFInfo
sparse_velocity_mf_info()
{
    amrex::MFInfo info;
    info.SetArena(amrex::The_Pinned_Arena());
    return info;
}

int
checked_sparse_count(std::size_t count)
{
    if (count
        > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "fire distributed flat-wind sparse point count exceeds int");
    }
    return static_cast<int>(count);
}

std::map<SparseVelocityKey, amrex::Real>
sample_sparse_velocity_values(
    const amrex::MultiFab& source,
    std::vector<SparseVelocityKey> keys)
{
    std::sort(keys.begin(), keys.end());
    keys.erase(
        std::unique(keys.begin(), keys.end()),
        keys.end());

    if (keys.empty()) {
        return {};
    }

    const int key_count =
        checked_sparse_count(keys.size());

    amrex::BoxArray sparse_boxes(keys.size());
    const amrex::IndexType index_type =
        source.boxArray().ixType();
    for (int n = 0; n < key_count; ++n) {
        const auto [i, j, k] =
            keys[static_cast<std::size_t>(n)];
        const amrex::IntVect point(i, j, k);
        sparse_boxes.set(
            n,
            amrex::Box(point, point, index_type));
    }

    const int nprocs =
        std::max(
            1,
            amrex::ParallelDescriptor::NProcs());
    amrex::Vector<int> processor_map(keys.size());
    for (int n = 0; n < key_count; ++n) {
        processor_map[static_cast<std::size_t>(n)] =
            n % nprocs;
    }
    const amrex::DistributionMapping sparse_dm(
        std::move(processor_map));

    amrex::MultiFab sparse_values(
        sparse_boxes,
        sparse_dm,
        1,
        0,
        sparse_velocity_mf_info());

    sparse_values.ParallelCopy(
        source,
        0,
        0,
        1,
        1,
        0);
    amrex::Gpu::streamSynchronize();

    std::map<SparseVelocityKey, std::size_t>
        key_to_index;
    for (std::size_t n = 0; n < keys.size(); ++n) {
        key_to_index.emplace(keys[n], n);
    }

    std::vector<amrex::Real> reduced_values(
        keys.size(),
        amrex::Real(0));
    std::vector<int> reduced_counts(
        keys.size(),
        0);

    for (amrex::MFIter mfi(sparse_values);
         mfi.isValid();
         ++mfi) {
        const amrex::Box& box = mfi.validbox();
        const int i = box.smallEnd(0);
        const int j = box.smallEnd(1);
        const int k = box.smallEnd(2);
        const SparseVelocityKey key{i, j, k};
        const auto found = key_to_index.find(key);
        if (found == key_to_index.end()) {
            throw std::logic_error(
                "fire distributed flat-wind sparse lookup failed");
        }

        const auto values =
            sparse_values.const_array(mfi);
        reduced_values[found->second] =
            values(i, j, k);
        reduced_counts[found->second] = 1;
    }

    amrex::ParallelDescriptor::ReduceRealSum(
        reduced_values.data(),
        key_count);
    amrex::ParallelDescriptor::ReduceIntSum(
        reduced_counts.data(),
        key_count);

    std::map<SparseVelocityKey, amrex::Real> result;
    for (std::size_t n = 0; n < keys.size(); ++n) {
        if (reduced_counts[n] != 1) {
            throw std::runtime_error(
                "fire distributed flat-wind sparse value has invalid ownership count");
        }
        if (!std::isfinite(reduced_values[n])) {
            throw std::invalid_argument(
                "fire distributed flat-wind source value must be finite");
        }
        result.emplace(keys[n], reduced_values[n]);
    }
    return result;
}

void
append_sparse_stencil_keys(
    std::vector<SparseVelocityKey>& keys,
    int lower_i,
    int lower_j,
    int lower_k,
    int upper_k)
{
    for (int dj = 0; dj <= 1; ++dj) {
        for (int di = 0; di <= 1; ++di) {
            keys.emplace_back(
                lower_i + di,
                lower_j + dj,
                lower_k);
            keys.emplace_back(
                lower_i + di,
                lower_j + dj,
                upper_k);
        }
    }
}

void
validate_common_level0_coordinate_scope (
    const ERFFireLevel0EnvironmentInputs& inputs)
{
    require(
        inputs.configured_max_level == 0,
        "Fire level-0 coupling supports only configurations with max_level = 0");

    require(
        inputs.terrain_type != TerrainType::EB,
        "Fire level-0 coupling does not support EB terrain");
    require(
        inputs.terrain_type != TerrainType::ImmersedForcing,
        "Fire level-0 coupling does not support immersed-forcing terrain");
    require(
        inputs.terrain_type != TerrainType::MovingFittedMesh,
        "Fire level-0 coupling does not support moving fitted terrain");
    require(
        inputs.buildings_type == BuildingsType::None,
        "Fire level-0 coupling does not support immersed buildings");

    const amrex::Box& domain = inputs.geometry.Domain();
    require(
        domain.cellCentered(),
        "Fire level-0 coupling requires a cell-centered ERF geometry domain");

    const amrex::Box expected_zcc = domain;
    const amrex::Box expected_znd =
        amrex::convert(domain, amrex::IntVect(1, 1, 1));

    require_level0_layout(
        inputs.z_phys_cc, expected_zcc,
        "fire z_phys_cc BoxArray does not cover the level-0 cell domain");
    require_level0_layout(
        inputs.z_phys_nd, expected_znd,
        "fire z_phys_nd BoxArray does not cover the level-0 nodal domain");
}


void
validate_horizontal_velocity_layout_and_ghosts (
    const ERFFireLevel0EnvironmentInputs& inputs)
{
    const amrex::Box& domain = inputs.geometry.Domain();
    const amrex::Box expected_u =
        amrex::convert(domain, amrex::IntVect(1, 0, 0));
    const amrex::Box expected_v =
        amrex::convert(domain, amrex::IntVect(0, 1, 0));

    require_level0_layout(
        inputs.x_velocity, expected_u,
        "fire x-velocity BoxArray does not cover the level-0 x-face domain");
    require_level0_layout(
        inputs.y_velocity, expected_v,
        "fire y-velocity BoxArray does not cover the level-0 y-face domain");

    const amrex::IntVect x_ng = inputs.x_velocity.nGrowVect();
    const amrex::IntVect y_ng = inputs.y_velocity.nGrowVect();
    require(
        x_ng[0] >= 1 && x_ng[1] >= 1 && x_ng[2] >= 1,
        "fire x-velocity snapshot requires at least one filled ghost cell");
    require(
        y_ng[0] >= 1 && y_ng[1] >= 1 && y_ng[2] >= 1,
        "fire y-velocity snapshot requires at least one filled ghost cell");
}

void
validate_reference_height_agl (amrex::Real reference_height_agl_m)
{
    if (!std::isfinite(reference_height_agl_m)
        || reference_height_agl_m < amrex::Real(0)) {
        throw std::invalid_argument(
            "fire atmospheric reference height AGL must be finite and nonnegative");
    }
}

enum class HorizontalVelocityFace
{
    X,
    Y
};

amrex::Real
terrain_face_ground_height_m (
    const amrex::Array4<const amrex::Real>& z,
    int i,
    int j,
    int bottom_k,
    HorizontalVelocityFace face)
{
    amrex::Real value{};
    if (face == HorizontalVelocityFace::X) {
        value =
            amrex::Real(0.5)
            * (z(i, j, bottom_k)
               + z(i, j + 1, bottom_k));
    } else {
        value =
            amrex::Real(0.5)
            * (z(i, j, bottom_k)
               + z(i + 1, j, bottom_k));
    }

    require(
        std::isfinite(value),
        "fire terrain face ground height must be finite");
    return value;
}

amrex::Real
terrain_face_cell_center_height_m (
    const amrex::Array4<const amrex::Real>& z,
    int i,
    int j,
    int k,
    HorizontalVelocityFace face)
{
    amrex::Real value{};
    if (face == HorizontalVelocityFace::X) {
        value =
            amrex::Real(0.25)
            * (z(i, j, k)
               + z(i, j + 1, k)
               + z(i, j, k + 1)
               + z(i, j + 1, k + 1));
    } else {
        value =
            amrex::Real(0.25)
            * (z(i, j, k)
               + z(i + 1, j, k)
               + z(i, j, k + 1)
               + z(i + 1, j, k + 1));
    }

    require(
        std::isfinite(value),
        "fire terrain face cell-center height must be finite");
    return value;
}

amrex::Real
terrain_face_reference_wind_value (
    const amrex::Array4<const amrex::Real>& velocity,
    const amrex::Array4<const amrex::Real>& z,
    int i,
    int j,
    int domain_klo,
    int domain_khi,
    amrex::Real reference_height_agl_m,
    HorizontalVelocityFace face,
    std::vector<amrex::Real>& agl_cell_center_m)
{
    const amrex::Real ground =
        terrain_face_ground_height_m(
            z, i, j, domain_klo, face);

    const std::size_t nz =
        static_cast<std::size_t>(
            domain_khi - domain_klo + 1);
    if (agl_cell_center_m.size() != nz) {
        throw std::logic_error(
            "fire terrain AGL column scratch has wrong size");
    }

    for (int k = domain_klo; k <= domain_khi; ++k) {
        const amrex::Real physical_height =
            terrain_face_cell_center_height_m(
                z, i, j, k, face);
        const amrex::Real agl =
            physical_height - ground;

        require(
            std::isfinite(agl),
            "fire terrain face AGL cell-center height must be finite");

        agl_cell_center_m[
            static_cast<std::size_t>(k - domain_klo)] = agl;
    }

    const FireVerticalLinearBracket bracket =
        fire_vertical_linear_bracket(
            agl_cell_center_m,
            reference_height_agl_m);

    const int lower_k =
        domain_klo
        + static_cast<int>(bracket.lower_k);
    const int upper_k =
        domain_klo
        + static_cast<int>(bracket.upper_k);

    const amrex::Real lower_value =
        velocity(i, j, lower_k);
    const amrex::Real upper_value =
        velocity(i, j, upper_k);

    require(
        std::isfinite(lower_value)
        && std::isfinite(upper_value),
        "fire terrain reference-wind source values must be finite");

    if (lower_k == upper_k) {
        return lower_value;
    }

    const amrex::Real value =
        lower_value
        + bracket.upper_weight
            * (upper_value - lower_value);

    if (!std::isfinite(value)) {
        throw std::overflow_error(
            "fire terrain vertically interpolated wind is not finite");
    }
    return value;
}

void
validate_flat_scope_and_layout (
    const ERFFireLevel0EnvironmentInputs& inputs)
{
    validate_common_level0_coordinate_scope(inputs);

    require(
        inputs.mesh_type != MeshType::VariableDz,
        "flat Fire environment defers VariableDz / terrain-fitted wind sampling");

    validate_horizontal_velocity_layout_and_ghosts(inputs);
}


void
validate_terrain_reference_wind_scope_and_layout (
    const ERFFireLevel0EnvironmentInputs& inputs)
{
    validate_common_level0_coordinate_scope(inputs);

    require(
        inputs.mesh_type == MeshType::VariableDz,
        "terrain reference-wind snapshot requires VariableDz");
    require(
        inputs.terrain_type == TerrainType::StaticFittedMesh,
        "terrain reference-wind snapshot requires static fitted terrain");

    validate_horizontal_velocity_layout_and_ghosts(inputs);

    const amrex::IntVect z_ng =
        inputs.z_phys_nd.nGrowVect();
    require(
        z_ng[0] >= 1
        && z_ng[1] >= 1
        && z_ng[2] >= 1,
        "terrain reference-wind snapshot requires at least one z_phys_nd ghost cell in every direction");
}

void
validate_terrain_surface_scope_and_layout (
    const ERFFireLevel0EnvironmentInputs& inputs)
{
    validate_common_level0_coordinate_scope(inputs);

    if (inputs.mesh_type == MeshType::VariableDz) {
        require(
            inputs.terrain_type == TerrainType::StaticFittedMesh,
            "VariableDz Fire terrain extraction requires static fitted terrain");
    }
}

} // namespace


FireTerrainSurface
make_erf_level0_terrain_surface (
    const ERFFireLevel0EnvironmentInputs& inputs)
{
    validate_terrain_surface_scope_and_layout(inputs);

    const amrex::Box& domain = inputs.geometry.Domain();
    amrex::Box bottom =
        amrex::convert(
            domain,
            amrex::IntVect(1, 1, 1));
    bottom.setRange(2, bottom.smallEnd(2));

    const amrex::FArrayBox surface =
        replicated_host_copy(
            inputs.z_phys_nd,
            bottom,
            0);
    const auto z = surface.const_array();

    const auto prob_lo =
        inputs.geometry.ProbLoArray();
    const auto cell_size =
        inputs.geometry.CellSizeArray();

    const std::size_t nx =
        static_cast<std::size_t>(
            domain.length(0));
    const std::size_t ny =
        static_cast<std::size_t>(
            domain.length(1));

    FireCartesianRasterGeometry2D horizontal_geometry{
        nx,
        ny,
        prob_lo[0],
        prob_lo[1],
        cell_size[0],
        cell_size[1]};

    std::vector<amrex::Real> nodal_ground_height_m;
    nodal_ground_height_m.reserve(
        (nx + 1) * (ny + 1));

    const int ilo = bottom.smallEnd(0);
    const int jlo = bottom.smallEnd(1);
    const int k = bottom.smallEnd(2);

    for (std::size_t j = 0; j <= ny; ++j) {
        for (std::size_t i = 0; i <= nx; ++i) {
            nodal_ground_height_m.push_back(
                z(
                    ilo + static_cast<int>(i),
                    jlo + static_cast<int>(j),
                    k));
        }
    }

    return FireTerrainSurface(
        horizontal_geometry,
        std::move(nodal_ground_height_m));
}


FireTerrainSurface
make_erf_level0_terrain_surface_on_geometry (
    const ERFFireLevel0EnvironmentInputs& inputs,
    FireCartesianRasterGeometry2D target_geometry)
{
    validate_terrain_surface_scope_and_layout(inputs);

    return resample_fire_terrain_surface(
        make_erf_level0_terrain_surface(inputs),
        target_geometry);
}

FireTerrainSurface
make_erf_terrain_source_surface_on_geometry (
    const ERFTerrainSource& source,
    FireCartesianRasterGeometry2D target_geometry)
{
    (void)detail::validate_fire_cartesian_raster_geometry(
        target_geometry);

    std::vector<amrex::Real> nodal_ground_height_m;
    nodal_ground_height_m.reserve(
        (target_geometry.nx + 1)
        * (target_geometry.ny + 1));

    for (std::size_t j = 0;
         j <= target_geometry.ny;
         ++j) {
        const amrex::Real y =
            target_geometry.ylo_m
            + static_cast<amrex::Real>(j)
                * target_geometry.dy_m;

        for (std::size_t i = 0;
             i <= target_geometry.nx;
             ++i) {
            const amrex::Real x =
                target_geometry.xlo_m
                + static_cast<amrex::Real>(i)
                    * target_geometry.dx_m;

            nodal_ground_height_m.push_back(
                source.sample(x, y));
        }
    }

    return FireTerrainSurface(
        target_geometry,
        std::move(nodal_ground_height_m));
}


std::vector<amrex::Real>
erf_fire_level0_flat_vertical_faces_agl (
    const ERFFireLevel0EnvironmentInputs& inputs)
{
    validate_flat_scope_and_layout(inputs);

    const amrex::Box& domain =
        inputs.geometry.Domain();
    const amrex::Box nodal_domain =
        amrex::convert(
            domain,
            amrex::IntVect(1, 1, 1));
    const int klo = nodal_domain.smallEnd(2);
    const int khi = nodal_domain.bigEnd(2);

    const std::vector<amrex::Real>
        physical_face_height_m =
            canonical_vertical_column(
                inputs.z_phys_nd,
                nodal_domain);
    for (const amrex::Real value
         : physical_face_height_m) {
        require(
            std::isfinite(value),
            "fire level-0 nodal height must be finite");
    }

    require_exact_uniform_profile(
        inputs.z_phys_nd,
        nodal_domain,
        physical_face_height_m,
        "Fire feedback requires horizontally uniform z_phys_nd planes");

    const amrex::Real ground =
        physical_face_height_m.front();
    std::vector<amrex::Real> result(
        physical_face_height_m.size(),
        amrex::Real(0));

    for (int k = klo; k <= khi; ++k) {
        const amrex::Real agl =
            physical_face_height_m[
                static_cast<std::size_t>(k - klo)]
            - ground;
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

ERFFireLevel0FlatWindSampler::
ERFFireLevel0FlatWindSampler(
    const ERFFireLevel0EnvironmentInputs& inputs,
    amrex::Real reference_height_agl_m)
    : x_velocity_(&inputs.x_velocity),
      y_velocity_(&inputs.y_velocity),
      layout_(
          inputs.geometry.ProbLoArray()[0],
          inputs.geometry.ProbLoArray()[1],
          inputs.geometry.CellSizeArray()[0],
          inputs.geometry.CellSizeArray()[1],
          static_cast<std::size_t>(
              inputs.geometry.Domain().length(0)),
          static_cast<std::size_t>(
              inputs.geometry.Domain().length(1))),
      reference_height_agl_m_(reference_height_agl_m),
      domain_ilo_(inputs.geometry.Domain().smallEnd(0)),
      domain_jlo_(inputs.geometry.Domain().smallEnd(1))
{
    validate_flat_scope_and_layout(inputs);
    validate_reference_height_agl(
        reference_height_agl_m_);

    const amrex::Box& domain =
        inputs.geometry.Domain();
    const amrex::Real ground =
        flat_ground_height(
            inputs.z_phys_nd,
            domain);
    const std::vector<amrex::Real> z_cell_center_m =
        cell_center_heights(
            inputs.z_phys_cc,
            domain);

    const amrex::Real target_height_m =
        ground + reference_height_agl_m_;
    if (!std::isfinite(target_height_m)) {
        throw std::invalid_argument(
            "fire atmospheric reference height is not representable");
    }

    vertical_bracket_ =
        fire_vertical_linear_bracket(
            z_cell_center_m,
            target_height_m);

    const int domain_klo = domain.smallEnd(2);
    lower_k_ =
        domain_klo
        + static_cast<int>(
            vertical_bracket_.lower_k);
    upper_k_ =
        domain_klo
        + static_cast<int>(
            vertical_bracket_.upper_k);

    require_horizontally_uniform_z_phys_cc_plane(
        inputs.z_phys_cc,
        domain,
        lower_k_,
        z_cell_center_m[
            vertical_bracket_.lower_k]);
    if (upper_k_ != lower_k_) {
        require_horizontally_uniform_z_phys_cc_plane(
            inputs.z_phys_cc,
            domain,
            upper_k_,
            z_cell_center_m[
                vertical_bracket_.upper_k]);
    }
}

std::vector<FireEnvironmentSample>
ERFFireLevel0FlatWindSampler::sample_points(
    const std::vector<FireVec2>& positions_m) const
{
    if (positions_m.empty()) {
        return {};
    }

    struct PointStencils
    {
        FlatAxisStencil ux;
        FlatAxisStencil uy;
        FlatAxisStencil vx;
        FlatAxisStencil vy;
    };

    std::vector<PointStencils> stencils;
    stencils.reserve(positions_m.size());
    std::vector<SparseVelocityKey> u_keys;
    std::vector<SparseVelocityKey> v_keys;
    u_keys.reserve(positions_m.size() * 8);
    v_keys.reserve(positions_m.size() * 8);

    for (const FireVec2& point : positions_m) {
        if (!layout_.contains_physical_point(
                point.x,
                point.y)) {
            throw std::out_of_range(
                "fire distributed flat-wind sample point lies outside the physical level-0 domain");
        }

        PointStencils point_stencils{
            make_flat_axis_stencil(
                point.x,
                layout_.xlo_m(),
                layout_.dx_m(),
                amrex::Real(0)),
            make_flat_axis_stencil(
                point.y,
                layout_.ylo_m(),
                layout_.dy_m(),
                amrex::Real(0.5)),
            make_flat_axis_stencil(
                point.x,
                layout_.xlo_m(),
                layout_.dx_m(),
                amrex::Real(0.5)),
            make_flat_axis_stencil(
                point.y,
                layout_.ylo_m(),
                layout_.dy_m(),
                amrex::Real(0))};
        stencils.push_back(point_stencils);

        append_sparse_stencil_keys(
            u_keys,
            domain_ilo_ + point_stencils.ux.lower,
            domain_jlo_ + point_stencils.uy.lower,
            lower_k_,
            upper_k_);
        append_sparse_stencil_keys(
            v_keys,
            domain_ilo_ + point_stencils.vx.lower,
            domain_jlo_ + point_stencils.vy.lower,
            lower_k_,
            upper_k_);
    }

    const auto u_values =
        sample_sparse_velocity_values(
            *x_velocity_,
            std::move(u_keys));
    const auto v_values =
        sample_sparse_velocity_values(
            *y_velocity_,
            std::move(v_keys));

    const auto vertically_interpolated =
        [this](
            const auto& values,
            int i,
            int j) {
            return fire_vertical_linear_interpolate(
                values.at(
                    SparseVelocityKey{
                        i, j, lower_k_}),
                values.at(
                    SparseVelocityKey{
                        i, j, upper_k_}),
                vertical_bracket_);
        };

    std::vector<FireEnvironmentSample> result;
    result.reserve(positions_m.size());

    for (std::size_t n = 0;
         n < positions_m.size();
         ++n) {
        const PointStencils& s = stencils[n];

        const int ui =
            domain_ilo_ + s.ux.lower;
        const int uj =
            domain_jlo_ + s.uy.lower;
        const amrex::Real u =
            flat_bilinear(
                vertically_interpolated(
                    u_values, ui, uj),
                vertically_interpolated(
                    u_values, ui + 1, uj),
                vertically_interpolated(
                    u_values, ui, uj + 1),
                vertically_interpolated(
                    u_values, ui + 1, uj + 1),
                s.ux.upper_weight,
                s.uy.upper_weight);

        const int vi =
            domain_ilo_ + s.vx.lower;
        const int vj =
            domain_jlo_ + s.vy.lower;
        const amrex::Real v =
            flat_bilinear(
                vertically_interpolated(
                    v_values, vi, vj),
                vertically_interpolated(
                    v_values, vi + 1, vj),
                vertically_interpolated(
                    v_values, vi, vj + 1),
                vertically_interpolated(
                    v_values, vi + 1, vj + 1),
                s.vx.upper_weight,
                s.vy.upper_weight);

        if (!std::isfinite(u)
            || !std::isfinite(v)) {
            throw std::overflow_error(
                "fire distributed flat-wind interpolation produced non-finite wind");
        }
        result.push_back(
            FireEnvironmentSample{
                FireVec2{u, v}});
    }

    return result;
}

FireFlatEnvironmentSampler
freeze_erf_level0_environment (
    const ERFFireLevel0EnvironmentInputs& inputs,
    amrex::Real reference_height_agl_m)
{
    validate_flat_scope_and_layout(inputs);

    validate_reference_height_agl(
        reference_height_agl_m);

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


FireFlatEnvironmentSampler
freeze_erf_level0_terrain_reference_wind_environment (
    const ERFFireLevel0EnvironmentInputs& inputs,
    amrex::Real reference_height_agl_m)
{
    validate_terrain_reference_wind_scope_and_layout(inputs);
    validate_reference_height_agl(reference_height_agl_m);

    const amrex::Box& domain =
        inputs.geometry.Domain();

    amrex::Box u_box =
        amrex::convert(
            domain,
            amrex::IntVect(1, 0, 0));
    u_box.grow(amrex::IntVect(1, 1, 0));

    amrex::Box v_box =
        amrex::convert(
            domain,
            amrex::IntVect(0, 1, 0));
    v_box.grow(amrex::IntVect(1, 1, 0));

    amrex::Box z_box =
        amrex::convert(
            domain,
            amrex::IntVect(1, 1, 1));
    z_box.grow(amrex::IntVect(1, 1, 0));

    const amrex::FArrayBox u_host =
        replicated_host_copy(
            inputs.x_velocity,
            u_box,
            1);
    const amrex::FArrayBox v_host =
        replicated_host_copy(
            inputs.y_velocity,
            v_box,
            1);
    const amrex::FArrayBox z_host =
        replicated_host_copy(
            inputs.z_phys_nd,
            z_box,
            1);

    const auto u = u_host.const_array();
    const auto v = v_host.const_array();
    const auto z = z_host.const_array();

    const auto cell_size =
        inputs.geometry.CellSizeArray();
    const auto prob_lo =
        inputs.geometry.ProbLoArray();

    const std::size_t nx =
        static_cast<std::size_t>(
            domain.length(0));
    const std::size_t ny =
        static_cast<std::size_t>(
            domain.length(1));

    FireFlatEnvironmentLayout2D layout(
        prob_lo[0],
        prob_lo[1],
        cell_size[0],
        cell_size[1],
        nx,
        ny);

    std::vector<amrex::Real> u_wind_mps(
        layout.u_storage_size());
    std::vector<amrex::Real> v_wind_mps(
        layout.v_storage_size());

    const int domain_ilo =
        domain.smallEnd(0);
    const int domain_jlo =
        domain.smallEnd(1);
    const int domain_klo =
        domain.smallEnd(2);
    const int domain_khi =
        domain.bigEnd(2);

    const int nx_int =
        static_cast<int>(nx);
    const int ny_int =
        static_cast<int>(ny);

    std::vector<amrex::Real> agl_column(
        static_cast<std::size_t>(
            domain.length(2)));

    for (int j = -1; j <= ny_int; ++j) {
        for (int i = -1; i <= nx_int + 1; ++i) {
            const int src_i =
                domain_ilo + i;
            const int src_j =
                domain_jlo + j;

            u_wind_mps[
                layout.u_storage_index(i, j)] =
                terrain_face_reference_wind_value(
                    u,
                    z,
                    src_i,
                    src_j,
                    domain_klo,
                    domain_khi,
                    reference_height_agl_m,
                    HorizontalVelocityFace::X,
                    agl_column);
        }
    }

    for (int j = -1; j <= ny_int + 1; ++j) {
        for (int i = -1; i <= nx_int; ++i) {
            const int src_i =
                domain_ilo + i;
            const int src_j =
                domain_jlo + j;

            v_wind_mps[
                layout.v_storage_index(i, j)] =
                terrain_face_reference_wind_value(
                    v,
                    z,
                    src_i,
                    src_j,
                    domain_klo,
                    domain_khi,
                    reference_height_agl_m,
                    HorizontalVelocityFace::Y,
                    agl_column);
        }
    }

    return FireFlatEnvironmentSampler(
        std::move(layout),
        reference_height_agl_m,
        std::move(u_wind_mps),
        std::move(v_wind_mps));
}

} // namespace ERFFire
