#include "ERF_FireLevel0TerrainWindSampler.H"

#include <AMReX_Arena.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Gpu.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Periodicity.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ERFFire
{
namespace
{

void
require(bool condition, const char* message)
{
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

void
require_level0_layout(
    const amrex::MultiFab& mf,
    const amrex::Box& expected,
    const char* message)
{
    require(
        mf.nComp() == 1,
        "fire terrain wind source fields must have exactly one component");

    const amrex::BoxArray& boxes = mf.boxArray();
    require(boxes.ixType() == expected.ixType(), message);
    require(boxes.contains(expected), message);

    for (int index = 0; index < boxes.size(); ++index) {
        require(expected.contains(boxes[index]), message);
    }
}

void
validate_terrain_sampler_inputs(
    const ERFFireLevel0EnvironmentInputs& inputs,
    amrex::Real reference_height_agl_m)
{
    require(
        inputs.configured_max_level == 0,
        "Fire terrain wind sampling supports only configured max_level = 0");
    require(
        inputs.mesh_type == MeshType::VariableDz,
        "Fire terrain wind sampling requires VariableDz");
    require(
        inputs.terrain_type == TerrainType::StaticFittedMesh,
        "Fire terrain wind sampling requires static fitted terrain");
    require(
        inputs.buildings_type == BuildingsType::None,
        "Fire terrain wind sampling does not support immersed buildings");
    require(
        std::isfinite(reference_height_agl_m)
            && reference_height_agl_m >= amrex::Real(0),
        "fire atmospheric reference height AGL must be finite and nonnegative");

    const amrex::Box& domain = inputs.geometry.Domain();
    require(
        domain.cellCentered(),
        "Fire terrain wind sampling requires a cell-centered level-0 domain");

    const amrex::Box expected_u =
        amrex::convert(
            domain,
            amrex::IntVect(1, 0, 0));
    const amrex::Box expected_v =
        amrex::convert(
            domain,
            amrex::IntVect(0, 1, 0));
    const amrex::Box expected_zcc = domain;
    const amrex::Box expected_znd =
        amrex::convert(
            domain,
            amrex::IntVect(1, 1, 1));

    require_level0_layout(
        inputs.x_velocity,
        expected_u,
        "fire x-velocity BoxArray does not cover the level-0 x-face domain");
    require_level0_layout(
        inputs.y_velocity,
        expected_v,
        "fire y-velocity BoxArray does not cover the level-0 y-face domain");
    require_level0_layout(
        inputs.z_phys_cc,
        expected_zcc,
        "fire z_phys_cc BoxArray does not cover the level-0 cell domain");
    require_level0_layout(
        inputs.z_phys_nd,
        expected_znd,
        "fire z_phys_nd BoxArray does not cover the level-0 nodal domain");

    const amrex::IntVect x_ng =
        inputs.x_velocity.nGrowVect();
    const amrex::IntVect y_ng =
        inputs.y_velocity.nGrowVect();
    const amrex::IntVect z_ng =
        inputs.z_phys_nd.nGrowVect();

    require(
        x_ng[0] >= 1
            && x_ng[1] >= 1
            && x_ng[2] >= 1,
        "fire x-velocity terrain sampling requires at least one filled ghost cell");
    require(
        y_ng[0] >= 1
            && y_ng[1] >= 1
            && y_ng[2] >= 1,
        "fire y-velocity terrain sampling requires at least one filled ghost cell");
    require(
        z_ng[0] >= 1
            && z_ng[1] >= 1
            && z_ng[2] >= 1,
        "fire terrain wind sampling requires at least one z_phys_nd ghost cell in every direction");
}

struct AxisStencil
{
    int lower{};
    amrex::Real upper_weight{};
};

AxisStencil
make_axis_stencil(
    amrex::Real coordinate_m,
    amrex::Real origin_m,
    amrex::Real spacing_m,
    amrex::Real native_offset_cells)
{
    const amrex::Real logical =
        (coordinate_m - origin_m) / spacing_m
        - native_offset_cells;
    const amrex::Real floored =
        std::floor(logical);

    if (floored
            < static_cast<amrex::Real>(
                  std::numeric_limits<int>::min())
        || floored
            > static_cast<amrex::Real>(
                  std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "fire terrain wind stencil index overflow");
    }

    const int lower =
        static_cast<int>(floored);
    const amrex::Real upper_weight =
        logical
        - static_cast<amrex::Real>(lower);
    return {lower, upper_weight};
}

amrex::Real
represented_upper(
    amrex::Real lo,
    std::size_t n,
    amrex::Real spacing)
{
    return lo
        + static_cast<amrex::Real>(n) * spacing;
}

struct AxisLocation
{
    std::size_t index{};
    amrex::Real local{};
};

AxisLocation
locate_axis(
    amrex::Real coordinate,
    amrex::Real lo,
    std::size_t n,
    amrex::Real spacing,
    const char* message)
{
    const amrex::Real hi =
        represented_upper(lo, n, spacing);

    if (!std::isfinite(coordinate)
        || coordinate < lo
        || coordinate > hi) {
        throw std::out_of_range(message);
    }

    if (coordinate == hi) {
        return {
            n - 1,
            amrex::Real(1)};
    }

    const amrex::Real scaled =
        (coordinate - lo) / spacing;
    std::size_t index =
        static_cast<std::size_t>(
            std::floor(scaled));

    if (index >= n) {
        index = n - 1;
        return {
            index,
            amrex::Real(1)};
    }

    if (index > 0
        && coordinate
            == represented_upper(
                lo, index, spacing)) {
        return {
            index,
            amrex::Real(0)};
    }

    if (index + 1 < n
        && coordinate
            == represented_upper(
                lo, index + 1, spacing)) {
        return {
            index + 1,
            amrex::Real(0)};
    }

    return {
        index,
        scaled - static_cast<amrex::Real>(index)};
}

amrex::Real
bilinear(
    amrex::Real q00,
    amrex::Real q10,
    amrex::Real q01,
    amrex::Real q11,
    amrex::Real wx,
    amrex::Real wy)
{
    const amrex::Real lower =
        (amrex::Real(1) - wx) * q00
        + wx * q10;
    const amrex::Real upper =
        (amrex::Real(1) - wx) * q01
        + wx * q11;
    return
        (amrex::Real(1) - wy) * lower
        + wy * upper;
}

using FaceKey = std::pair<int, int>;

enum class HorizontalVelocityFace
{
    X,
    Y
};

amrex::MFInfo
column_mf_info()
{
    amrex::MFInfo info;
    info.SetArena(amrex::The_Pinned_Arena());
    return info;
}

int
checked_count(std::size_t count)
{
    if (count
        > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            "fire terrain sparse face count exceeds int");
    }
    return static_cast<int>(count);
}

enum class DistributedTerrainWindFailure : int
{
    none = 0,
    invalid_argument = 1,
    out_of_range = 2,
    overflow_error = 3,
    runtime_error = 4
};

[[noreturn]] void
throw_distributed_terrain_wind_failure(
    DistributedTerrainWindFailure failure,
    const std::string& local_error)
{
    const std::string message =
        local_error.empty()
        ? "distributed Fire terrain wind sampling failed on another MPI rank"
        : std::string("distributed Fire terrain wind sampling failed: ")
            + local_error;

    switch (failure) {
    case DistributedTerrainWindFailure::invalid_argument:
        throw std::invalid_argument(message);
    case DistributedTerrainWindFailure::out_of_range:
        throw std::out_of_range(message);
    case DistributedTerrainWindFailure::overflow_error:
        throw std::overflow_error(message);
    case DistributedTerrainWindFailure::runtime_error:
        throw std::runtime_error(message);
    case DistributedTerrainWindFailure::none:
        break;
    }

    throw std::runtime_error(
        "distributed Fire terrain wind sampling has an invalid error code");
}

template <typename Function>
void
run_distributed_terrain_wind_sampling(Function&& function)
{
    DistributedTerrainWindFailure local_failure =
        DistributedTerrainWindFailure::none;
    std::string local_error;

    try {
        function();
    } catch (const std::invalid_argument& error) {
        local_failure =
            DistributedTerrainWindFailure::invalid_argument;
        local_error = error.what();
    } catch (const std::out_of_range& error) {
        local_failure =
            DistributedTerrainWindFailure::out_of_range;
        local_error = error.what();
    } catch (const std::overflow_error& error) {
        local_failure =
            DistributedTerrainWindFailure::overflow_error;
        local_error = error.what();
    } catch (const std::exception& error) {
        local_failure =
            DistributedTerrainWindFailure::runtime_error;
        local_error = error.what();
    } catch (...) {
        local_failure =
            DistributedTerrainWindFailure::runtime_error;
        local_error = "unknown local terrain wind sampling error";
    }

    int failure = static_cast<int>(local_failure);
    amrex::ParallelDescriptor::ReduceIntMax(failure);
    if (failure
        != static_cast<int>(
            DistributedTerrainWindFailure::none)) {
        throw_distributed_terrain_wind_failure(
            static_cast<DistributedTerrainWindFailure>(
                failure),
            local_error);
    }
}

std::map<FaceKey, amrex::Real>
sample_face_reference_values(
    const amrex::MultiFab& velocity,
    const amrex::MultiFab& z_phys_nd,
    std::vector<FaceKey> face_keys,
    HorizontalVelocityFace face,
    int domain_klo,
    int domain_khi,
    amrex::Real reference_height_agl_m)
{
    std::sort(
        face_keys.begin(),
        face_keys.end());
    face_keys.erase(
        std::unique(
            face_keys.begin(),
            face_keys.end()),
        face_keys.end());

    if (face_keys.empty()) {
        return {};
    }

    const int face_count =
        checked_count(face_keys.size());

    amrex::BoxArray velocity_boxes(
        face_keys.size());
    amrex::BoxArray z_boxes(
        face_keys.size());

    const amrex::IndexType velocity_type =
        velocity.boxArray().ixType();
    const amrex::IndexType z_type =
        z_phys_nd.boxArray().ixType();

    for (int n = 0; n < face_count; ++n) {
        const auto [i, j] =
            face_keys[
                static_cast<std::size_t>(n)];

        velocity_boxes.set(
            n,
            amrex::Box(
                amrex::IntVect(
                    i, j, domain_klo),
                amrex::IntVect(
                    i, j, domain_khi),
                velocity_type));

        z_boxes.set(
            n,
            amrex::Box(
                amrex::IntVect(
                    i, j, domain_klo),
                amrex::IntVect(
                    i, j, domain_khi + 1),
                z_type));
    }

    const int nprocs =
        std::max(
            1,
            amrex::ParallelDescriptor::NProcs());
    amrex::Vector<int> processor_map(
        face_keys.size());
    for (int n = 0; n < face_count; ++n) {
        processor_map[
            static_cast<std::size_t>(n)] =
                n % nprocs;
    }
    const amrex::DistributionMapping dm(
        std::move(processor_map));

    amrex::MultiFab velocity_columns(
        velocity_boxes,
        dm,
        1,
        0,
        column_mf_info());
    amrex::MultiFab z_columns(
        z_boxes,
        dm,
        2,
        0,
        column_mf_info());

    velocity_columns.ParallelCopy(
        velocity,
        0,
        0,
        1,
        1,
        0);

    z_columns.ParallelCopy(
        z_phys_nd,
        0,
        0,
        1,
        1,
        0);

    const amrex::IntVect one_ghost(1);
    const amrex::IntVect no_ghost(0);
    const amrex::IntVect side_offset =
        face == HorizontalVelocityFace::X
        ? amrex::IntVect(0, -1, 0)
        : amrex::IntVect(-1, 0, 0);

    z_columns.ParallelCopy(
        z_phys_nd,
        0,
        1,
        1,
        one_ghost,
        no_ghost,
        side_offset,
        amrex::Periodicity::NonPeriodic());

    amrex::Gpu::streamSynchronize();

    std::map<FaceKey, std::size_t>
        key_to_index;
    for (std::size_t n = 0;
         n < face_keys.size();
         ++n) {
        key_to_index.emplace(
            face_keys[n],
            n);
    }

    std::vector<amrex::Real> reduced_values(
        face_keys.size(),
        amrex::Real(0));
    std::vector<int> reduced_counts(
        face_keys.size(),
        0);

    const std::size_t nz =
        static_cast<std::size_t>(
            domain_khi - domain_klo + 1);

    std::vector<amrex::Real>
        agl_cell_center_m(
            nz,
            amrex::Real(0));

    run_distributed_terrain_wind_sampling(
        [&] {
        for (amrex::MFIter mfi(velocity_columns);
             mfi.isValid();
             ++mfi) {
            const amrex::Box& box =
                mfi.validbox();
            const int i = box.smallEnd(0);
            const int j = box.smallEnd(1);

            const auto found =
                key_to_index.find(
                    FaceKey{i, j});
            if (found == key_to_index.end()) {
                throw std::logic_error(
                    "fire terrain sparse face lookup failed");
            }

            const auto velocity_values =
                velocity_columns.const_array(mfi);
            const auto z =
                z_columns.const_array(mfi);

            const amrex::Real ground =
                amrex::Real(0.5)
                * (z(i, j, domain_klo, 0)
                   + z(i, j, domain_klo, 1));
            require(
                std::isfinite(ground),
                "fire terrain face ground height must be finite");

            for (int k = domain_klo;
                 k <= domain_khi;
                 ++k) {
                const amrex::Real physical_height =
                    amrex::Real(0.25)
                    * (z(i, j, k, 0)
                       + z(i, j, k, 1)
                       + z(i, j, k + 1, 0)
                       + z(i, j, k + 1, 1));

                require(
                    std::isfinite(physical_height),
                    "fire terrain face cell-center height must be finite");

                const amrex::Real agl =
                    physical_height - ground;
                require(
                    std::isfinite(agl),
                    "fire terrain face AGL cell-center height must be finite");

                agl_cell_center_m[
                    static_cast<std::size_t>(
                        k - domain_klo)] =
                            agl;
            }

            const FireVerticalLinearBracket bracket =
                fire_vertical_linear_bracket(
                    agl_cell_center_m,
                    reference_height_agl_m);

            const int lower_k =
                domain_klo
                + static_cast<int>(
                    bracket.lower_k);
            const int upper_k =
                domain_klo
                + static_cast<int>(
                    bracket.upper_k);

            const amrex::Real lower_value =
                velocity_values(
                    i, j, lower_k);
            const amrex::Real upper_value =
                velocity_values(
                    i, j, upper_k);

            require(
                std::isfinite(lower_value)
                    && std::isfinite(upper_value),
                "fire terrain reference-wind source values must be finite");

            amrex::Real value = lower_value;
            if (lower_k != upper_k) {
                value =
                    lower_value
                    + bracket.upper_weight
                        * (upper_value
                           - lower_value);
            }

            if (!std::isfinite(value)) {
                throw std::overflow_error(
                    "fire terrain vertically interpolated wind is not finite");
            }

            reduced_values[
                found->second] = value;
            reduced_counts[
                found->second] = 1;
        }

        });

    amrex::ParallelDescriptor::ReduceRealSum(
        reduced_values.data(),
        face_count);
    amrex::ParallelDescriptor::ReduceIntSum(
        reduced_counts.data(),
        face_count);

    std::map<FaceKey, amrex::Real> result;
    for (std::size_t n = 0;
         n < face_keys.size();
         ++n) {
        if (reduced_counts[n] != 1) {
            throw std::runtime_error(
                "fire terrain sparse face has invalid ownership count");
        }
        if (!std::isfinite(
                reduced_values[n])) {
            throw std::runtime_error(
                "fire terrain sparse face produced non-finite wind");
        }
        result.emplace(
            face_keys[n],
            reduced_values[n]);
    }

    return result;
}

std::map<FaceKey, amrex::Real>
sample_bottom_nodal_values(
    const amrex::MultiFab& z_phys_nd,
    std::vector<FaceKey> node_keys,
    int bottom_k)
{
    std::sort(
        node_keys.begin(),
        node_keys.end());
    node_keys.erase(
        std::unique(
            node_keys.begin(),
            node_keys.end()),
        node_keys.end());

    if (node_keys.empty()) {
        return {};
    }

    const int node_count =
        checked_count(node_keys.size());
    amrex::BoxArray node_boxes(
        node_keys.size());
    const amrex::IndexType z_type =
        z_phys_nd.boxArray().ixType();

    for (int n = 0; n < node_count; ++n) {
        const auto [i, j] =
            node_keys[
                static_cast<std::size_t>(n)];
        const amrex::IntVect point(
            i, j, bottom_k);
        node_boxes.set(
            n,
            amrex::Box(
                point,
                point,
                z_type));
    }

    const int nprocs =
        std::max(
            1,
            amrex::ParallelDescriptor::NProcs());
    amrex::Vector<int> processor_map(
        node_keys.size());
    for (int n = 0; n < node_count; ++n) {
        processor_map[
            static_cast<std::size_t>(n)] =
                n % nprocs;
    }
    const amrex::DistributionMapping dm(
        std::move(processor_map));

    amrex::MultiFab node_values(
        node_boxes,
        dm,
        1,
        0,
        column_mf_info());
    node_values.ParallelCopy(
        z_phys_nd,
        0,
        0,
        1,
        0,
        0);
    amrex::Gpu::streamSynchronize();

    std::map<FaceKey, std::size_t>
        key_to_index;
    for (std::size_t n = 0;
         n < node_keys.size();
         ++n) {
        key_to_index.emplace(
            node_keys[n],
            n);
    }

    std::vector<amrex::Real> reduced_values(
        node_keys.size(),
        amrex::Real(0));
    std::vector<int> reduced_counts(
        node_keys.size(),
        0);

    run_distributed_terrain_wind_sampling(
        [&] {
        for (amrex::MFIter mfi(node_values);
             mfi.isValid();
             ++mfi) {
            const amrex::Box& box =
                mfi.validbox();
            const int i = box.smallEnd(0);
            const int j = box.smallEnd(1);
            const auto found =
                key_to_index.find(
                    FaceKey{i, j});
            if (found == key_to_index.end()) {
                throw std::logic_error(
                    "fire terrain sparse nodal lookup failed");
            }

            const amrex::Real value =
                node_values.const_array(mfi)(
                    i, j, bottom_k);
            require(
                std::isfinite(value),
                "fire terrain bottom nodal height must be finite");
            reduced_values[found->second] = value;
            reduced_counts[found->second] = 1;
        }
        });

    amrex::ParallelDescriptor::ReduceRealSum(
        reduced_values.data(),
        node_count);
    amrex::ParallelDescriptor::ReduceIntSum(
        reduced_counts.data(),
        node_count);

    std::map<FaceKey, amrex::Real> result;
    for (std::size_t n = 0;
         n < node_keys.size();
         ++n) {
        if (reduced_counts[n] != 1) {
            throw std::runtime_error(
                "fire terrain sparse node has invalid ownership count");
        }
        if (!std::isfinite(reduced_values[n])) {
            throw std::runtime_error(
                "fire terrain sparse node produced non-finite height");
        }
        result.emplace(
            node_keys[n],
            reduced_values[n]);
    }

    return result;
}

} // namespace

ERFFireLevel0TerrainWindSampler::
ERFFireLevel0TerrainWindSampler(
    const ERFFireLevel0EnvironmentInputs& inputs,
    amrex::Real reference_height_agl_m)
    : x_velocity_(&inputs.x_velocity),
      y_velocity_(&inputs.y_velocity),
      z_phys_nd_(&inputs.z_phys_nd),
      layout_(
          inputs.geometry.ProbLoArray()[0],
          inputs.geometry.ProbLoArray()[1],
          inputs.geometry.CellSizeArray()[0],
          inputs.geometry.CellSizeArray()[1],
          static_cast<std::size_t>(
              inputs.geometry.Domain().length(0)),
          static_cast<std::size_t>(
              inputs.geometry.Domain().length(1))),
      reference_height_agl_m_(
          reference_height_agl_m),
      domain_ilo_(
          inputs.geometry.Domain().smallEnd(0)),
      domain_jlo_(
          inputs.geometry.Domain().smallEnd(1)),
      domain_klo_(
          inputs.geometry.Domain().smallEnd(2)),
      domain_khi_(
          inputs.geometry.Domain().bigEnd(2))
{
    validate_terrain_sampler_inputs(
        inputs,
        reference_height_agl_m_);
}

std::vector<FireEnvironmentSample>
ERFFireLevel0TerrainWindSampler::sample_points(
    const std::vector<FireVec2>& positions_m) const
{
    if (positions_m.empty()) {
        return {};
    }

    struct PointStencils
    {
        AxisStencil ux;
        AxisStencil uy;
        AxisStencil vx;
        AxisStencil vy;
    };
    struct TerrainLocation
    {
        int i{};
        int j{};
        amrex::Real alpha{};
        amrex::Real beta{};
    };

    std::vector<PointStencils> stencils;
    stencils.reserve(
        positions_m.size());
    std::vector<TerrainLocation> terrain_locations;
    terrain_locations.reserve(
        positions_m.size());

    std::vector<FaceKey> u_faces;
    std::vector<FaceKey> v_faces;
    std::vector<FaceKey> ground_nodes;
    u_faces.reserve(
        positions_m.size() * 4);
    v_faces.reserve(
        positions_m.size() * 4);
    ground_nodes.reserve(
        positions_m.size() * 4);

    for (const FireVec2& point
         : positions_m) {
        if (!layout_.contains_physical_point(
                point.x,
                point.y)) {
            throw std::out_of_range(
                "fire terrain wind sample point lies outside the physical level-0 domain");
        }

        PointStencils s{
            make_axis_stencil(
                point.x,
                layout_.xlo_m(),
                layout_.dx_m(),
                amrex::Real(0)),
            make_axis_stencil(
                point.y,
                layout_.ylo_m(),
                layout_.dy_m(),
                amrex::Real(0.5)),
            make_axis_stencil(
                point.x,
                layout_.xlo_m(),
                layout_.dx_m(),
                amrex::Real(0.5)),
            make_axis_stencil(
                point.y,
                layout_.ylo_m(),
                layout_.dy_m(),
                amrex::Real(0))};

        const AxisLocation terrain_x =
            locate_axis(
                point.x,
                layout_.xlo_m(),
                layout_.nx(),
                layout_.dx_m(),
                "fire terrain x coordinate is outside represented domain");
        const AxisLocation terrain_y =
            locate_axis(
                point.y,
                layout_.ylo_m(),
                layout_.ny(),
                layout_.dy_m(),
                "fire terrain y coordinate is outside represented domain");
        const int terrain_i =
            domain_ilo_
            + static_cast<int>(terrain_x.index);
        const int terrain_j =
            domain_jlo_
            + static_cast<int>(terrain_y.index);

        stencils.push_back(s);
        terrain_locations.push_back(
            TerrainLocation{
                terrain_i,
                terrain_j,
                terrain_x.local,
                terrain_y.local});

        const int ui =
            domain_ilo_ + s.ux.lower;
        const int uj =
            domain_jlo_ + s.uy.lower;
        const int vi =
            domain_ilo_ + s.vx.lower;
        const int vj =
            domain_jlo_ + s.vy.lower;

        for (int dj = 0;
             dj <= 1;
             ++dj) {
            for (int di = 0;
                 di <= 1;
                 ++di) {
                u_faces.emplace_back(
                    ui + di,
                    uj + dj);
                v_faces.emplace_back(
                    vi + di,
                    vj + dj);
                ground_nodes.emplace_back(
                    terrain_i + di,
                    terrain_j + dj);
            }
        }
    }

    const auto u_values =
        sample_face_reference_values(
            *x_velocity_,
            *z_phys_nd_,
            std::move(u_faces),
            HorizontalVelocityFace::X,
            domain_klo_,
            domain_khi_,
            reference_height_agl_m_);

    const auto v_values =
        sample_face_reference_values(
            *y_velocity_,
            *z_phys_nd_,
            std::move(v_faces),
            HorizontalVelocityFace::Y,
            domain_klo_,
            domain_khi_,
            reference_height_agl_m_);
    const auto ground_values =
        sample_bottom_nodal_values(
            *z_phys_nd_,
            std::move(ground_nodes),
            domain_klo_);

    std::vector<FireEnvironmentSample> result;
    result.reserve(
        positions_m.size());

    for (std::size_t n = 0;
         n < positions_m.size();
         ++n) {
        const PointStencils& s =
            stencils[n];
        const TerrainLocation& terrain =
            terrain_locations[n];

        const int ui =
            domain_ilo_ + s.ux.lower;
        const int uj =
            domain_jlo_ + s.uy.lower;
        const int vi =
            domain_ilo_ + s.vx.lower;
        const int vj =
            domain_jlo_ + s.vy.lower;

        const amrex::Real u =
            bilinear(
                u_values.at(
                    FaceKey{ui, uj}),
                u_values.at(
                    FaceKey{ui + 1, uj}),
                u_values.at(
                    FaceKey{ui, uj + 1}),
                u_values.at(
                    FaceKey{ui + 1, uj + 1}),
                s.ux.upper_weight,
                s.uy.upper_weight);

        const amrex::Real v =
            bilinear(
                v_values.at(
                    FaceKey{vi, vj}),
                v_values.at(
                    FaceKey{vi + 1, vj}),
                v_values.at(
                    FaceKey{vi, vj + 1}),
                v_values.at(
                    FaceKey{vi + 1, vj + 1}),
                s.vx.upper_weight,
                s.vy.upper_weight);

        const amrex::Real h00 =
            ground_values.at(
                FaceKey{terrain.i, terrain.j});
        const amrex::Real h10 =
            ground_values.at(
                FaceKey{terrain.i + 1, terrain.j});
        const amrex::Real h01 =
            ground_values.at(
                FaceKey{terrain.i, terrain.j + 1});
        const amrex::Real h11 =
            ground_values.at(
                FaceKey{terrain.i + 1, terrain.j + 1});

        const amrex::Real dhdx =
            ((amrex::Real(1) - terrain.beta)
                 * (h10 - h00)
             + terrain.beta
                 * (h11 - h01))
            / layout_.dx_m();
        const amrex::Real dhdy =
            ((amrex::Real(1) - terrain.alpha)
                 * (h01 - h00)
             + terrain.alpha
                 * (h11 - h10))
            / layout_.dy_m();

        if (!std::isfinite(u)
            || !std::isfinite(v)) {
            throw std::overflow_error(
                "fire terrain wind interpolation produced non-finite wind");
        }
        if (!std::isfinite(dhdx)
            || !std::isfinite(dhdy)) {
            throw std::overflow_error(
                "fire terrain gradient is not finite");
        }

        result.push_back(
            FireEnvironmentSample{
                FireVec2{u, v},
                FireVec2{dhdx, dhdy}});
    }

    return result;
}

} // namespace ERFFire
