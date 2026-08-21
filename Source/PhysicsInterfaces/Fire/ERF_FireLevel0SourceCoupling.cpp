#include "ERF_FireLevel0SourceCoupling.H"

#include <ERF_EOS.H>
#include <ERF_IndexDefines.H>

#include <AMReX_Arena.H>
#include <AMReX_BoxArray.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_Math.H>
#include <AMReX_ParReduce.H>
#include <AMReX_Periodicity.H>
#include <AMReX_ParallelDescriptor.H>

#include <cmath>
#include <cstddef>
#include <limits>
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
require_level0_coverage(
    const amrex::BoxArray& boxes,
    const amrex::Box& expected,
    const char* message)
{
    require(boxes.ixType() == expected.ixType(), message);
    require(boxes.contains(expected), message);

    for (int index = 0; index < boxes.size(); ++index) {
        require(expected.contains(boxes[index]), message);
    }
}

FireCartesianRasterGeometry2D
level0_horizontal_geometry(
    const amrex::Geometry& erf_geometry)
{
    const amrex::Box& domain =
        erf_geometry.Domain();
    const auto prob_lo =
        erf_geometry.ProbLoArray();
    const auto cell_size =
        erf_geometry.CellSizeArray();

    return {
        static_cast<std::size_t>(domain.length(0)),
        static_cast<std::size_t>(domain.length(1)),
        prob_lo[0],
        prob_lo[1],
        cell_size[0],
        cell_size[1]};
}

bool
same_horizontal_geometry(
    const FireCartesianRasterGeometry2D& fire_geometry,
    const amrex::Geometry& erf_geometry) noexcept
{
    const auto level0 =
        level0_horizontal_geometry(erf_geometry);

    return fire_geometry.nx == level0.nx
        && fire_geometry.ny == level0.ny
        && fire_geometry.xlo_m == level0.xlo_m
        && fire_geometry.ylo_m == level0.ylo_m
        && fire_geometry.dx_m == level0.dx_m
        && fire_geometry.dy_m == level0.dy_m;
}

[[maybe_unused]]
std::vector<amrex::Real>
diagnose_level0_pressure_pa(
    const amrex::Geometry& geometry,
    const amrex::MultiFab& conserved_state_tn,
    MoistureType moisture_type)
{
    const amrex::Box& domain = geometry.Domain();

    require(
        domain.cellCentered(),
        "ERF Fire source coupling requires a cell-centered level-0 domain");
    require_level0_coverage(
        conserved_state_tn.boxArray(),
        domain,
        "ERF Fire source coupling conserved state does not cover level 0");
    require(
        moisture_type != MoistureType::None,
        "ERF Fire two-way source requires an allocated water-vapor state");
    require(
        conserved_state_tn.nComp() > RhoQ1_comp,
        "ERF Fire source coupling conserved state lacks RhoQ1");

    amrex::FArrayBox host_state(
        domain,
        conserved_state_tn.nComp(),
        amrex::The_Pinned_Arena());
    conserved_state_tn.copyTo(
        host_state,
        0,
        0,
        conserved_state_tn.nComp(),
        0);
    const auto state = host_state.const_array();

    const std::size_t nx =
        static_cast<std::size_t>(domain.length(0));
    const std::size_t ny =
        static_cast<std::size_t>(domain.length(1));
    const std::size_t nz =
        static_cast<std::size_t>(domain.length(2));

    std::vector<amrex::Real> pressure_pa(
        nx * ny * nz,
        amrex::Real(0));

    const int ilo = domain.smallEnd(0);
    const int jlo = domain.smallEnd(1);
    const int klo = domain.smallEnd(2);

    for (std::size_t k = 0; k < nz; ++k) {
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                const int ii = ilo + static_cast<int>(i);
                const int jj = jlo + static_cast<int>(j);
                const int kk = klo + static_cast<int>(k);

                const amrex::Real rho =
                    state(ii, jj, kk, Rho_comp);
                const amrex::Real rhotheta =
                    state(ii, jj, kk, RhoTheta_comp);
                const amrex::Real rhoqv =
                    state(ii, jj, kk, RhoQ1_comp);

                require(
                    std::isfinite(rho)
                        && rho > amrex::Real(0),
                    "ERF Fire source coupling dry density must be finite and positive");
                require(
                    std::isfinite(rhotheta)
                        && rhotheta > amrex::Real(0),
                    "ERF Fire source coupling rho theta must be finite and positive");
                require(
                    std::isfinite(rhoqv)
                        && rhoqv >= amrex::Real(0),
                    "ERF Fire source coupling rho qv must be finite and nonnegative");

                const amrex::Real qv = rhoqv / rho;
                const amrex::Real pressure =
                    getPgivenRTh(rhotheta, qv);

                if (!std::isfinite(pressure)
                    || !(pressure > amrex::Real(0))) {
                    throw std::overflow_error(
                        "ERF Fire source coupling diagnosed invalid pressure");
                }

                const std::size_t flat =
                    (k * ny + j) * nx + i;
                pressure_pa[flat] = pressure;
            }
        }
    }

    return pressure_pa;
}


void
validate_terrain_source_scope_and_layout(
    const ERFFireLevel0EnvironmentInputs& inputs,
    const amrex::MultiFab& detJ_cc)
{
    require(
        inputs.configured_max_level == 0,
        "ERF Fire terrain source supports only configured max_level = 0");
    require(
        inputs.mesh_type == MeshType::VariableDz,
        "ERF Fire terrain source requires VariableDz");
    require(
        inputs.terrain_type == TerrainType::StaticFittedMesh,
        "ERF Fire terrain source requires static fitted terrain");
    require(
        inputs.buildings_type == BuildingsType::None,
        "ERF Fire terrain source does not support immersed buildings");

    const amrex::Box& domain =
        inputs.geometry.Domain();
    require(
        domain.cellCentered(),
        "ERF Fire terrain source requires a cell-centered level-0 domain");

    const amrex::Box expected_znd =
        amrex::convert(
            domain,
            amrex::IntVect(1, 1, 1));

    require_level0_coverage(
        inputs.z_phys_cc.boxArray(),
        domain,
        "ERF Fire terrain source z_phys_cc does not cover level 0");
    require_level0_coverage(
        inputs.z_phys_nd.boxArray(),
        expected_znd,
        "ERF Fire terrain source z_phys_nd does not cover the level-0 nodal domain");
    require_level0_coverage(
        detJ_cc.boxArray(),
        domain,
        "ERF Fire terrain source detJ_cc does not cover level 0");
    require(
        inputs.z_phys_cc.nComp() >= 1
            && inputs.z_phys_nd.nComp() >= 1
            && detJ_cc.nComp() >= 1,
        "ERF Fire terrain source geometry fields require at least one component");
}

[[maybe_unused]]
amrex::Real
terrain_cell_vertical_face_height_m(
    const amrex::Array4<const amrex::Real>& z,
    int i,
    int j,
    int k)
{
    const amrex::Real value =
        amrex::Real(0.25)
        * (z(i, j, k)
           + z(i + 1, j, k)
           + z(i, j + 1, k)
           + z(i + 1, j + 1, k));

    require(
        std::isfinite(value),
        "ERF Fire terrain source vertical face height must be finite");
    return value;
}

constexpr int fire_state_rho_comp = 0;
constexpr int fire_state_rhotheta_comp = 1;
constexpr int fire_state_rhoqv_comp = 2;
constexpr int fire_state_component_count = 3;

constexpr int fire_source_rhotheta_comp = 0;
constexpr int fire_source_rhoqv_comp = 1;
constexpr int fire_source_component_count = 2;

enum class DistributedSourceFailure : int
{
    none = 0,
    invalid_argument = 1,
    overflow_error = 2,
    runtime_error = 3
};

amrex::MFInfo
fire_column_mf_info()
{
    amrex::MFInfo info;
    info.SetArena(amrex::The_Pinned_Arena());
    return info;
}

#ifdef AMREX_USE_GPU
constexpr int device_source_invalid_density = 1;
constexpr int device_source_invalid_rhotheta = 2;
constexpr int device_source_invalid_rhoqv = 3;
constexpr int device_source_invalid_source = 4;
constexpr int device_source_invalid_terrain_geometry = 5;
constexpr int device_source_invalid_detj = 6;
constexpr int device_source_invalid_pressure = 101;
constexpr int device_source_overflow_source = 102;
constexpr int device_source_overflow_weight = 103;
constexpr int device_source_overflow_volume = 104;
constexpr int device_source_overflow_normalization = 105;

std::vector<amrex::Real>
flat_normalized_layer_weights(
    const std::vector<amrex::Real>& faces,
    amrex::Real extinction_depth_m)
{
    require(
        faces.size() >= 2,
        "ERF Fire atmospheric source requires at least two vertical faces");

    for (const amrex::Real value : faces) {
        require(
            std::isfinite(value),
            "ERF Fire atmospheric source vertical faces must be finite");
    }

    const amrex::Real ground_tolerance =
        amrex::Real(64)
        * std::numeric_limits<amrex::Real>::epsilon();
    require(
        std::abs(faces.front()) <= ground_tolerance,
        "ERF Fire atmospheric source first AGL face must be zero");

    for (std::size_t k = 0; k + 1 < faces.size(); ++k) {
        require(
            faces[k + 1] > faces[k],
            "ERF Fire atmospheric source vertical faces must increase strictly");
    }

    require(
        std::isfinite(extinction_depth_m)
            && extinction_depth_m > amrex::Real(0),
        "ERF Fire atmospheric extinction depth must be finite and positive");

    std::vector<amrex::Real> weights(
        faces.size() - 1,
        amrex::Real(0));
    amrex::Real total = amrex::Real(0);

    for (std::size_t k = 0; k < weights.size(); ++k) {
        amrex::Real raw = amrex::Real(0);
        const auto status =
            try_erf_fire_atmospheric_source_raw_layer_weight(
                faces[k],
                faces[k + 1],
                extinction_depth_m,
                raw);
        if (status
            == ERFFireAtmosphericSourceStatus::overflow_error) {
            throw std::overflow_error(
                "ERF Fire atmospheric deposition weight is not finite and positive");
        }
        if (status
            != ERFFireAtmosphericSourceStatus::success) {
            throw std::invalid_argument(
                "ERF Fire atmospheric source layer geometry is invalid");
        }
        weights[k] = raw;
        total += raw;
    }

    if (!std::isfinite(total)
        || !(total > amrex::Real(0))) {
        throw std::overflow_error(
            "ERF Fire atmospheric deposition normalization is invalid");
    }

    amrex::Real normalized_sum = amrex::Real(0);
    for (std::size_t k = 0; k < weights.size(); ++k) {
        if (k + 1 == weights.size()) {
            weights[k] =
                std::max(
                    amrex::Real(0),
                    amrex::Real(1) - normalized_sum);
        } else {
            weights[k] /= total;
            normalized_sum += weights[k];
        }
    }

    return weights;
}
#endif

[[noreturn]] void
throw_distributed_source_failure(
    DistributedSourceFailure failure,
    const std::string& local_error,
    const char* operation)
{
    const std::string message =
        local_error.empty()
        ? std::string(operation) + " failed on another MPI rank"
        : std::string(operation) + " failed: " + local_error;

    switch (failure) {
    case DistributedSourceFailure::invalid_argument:
        throw std::invalid_argument(message);
    case DistributedSourceFailure::overflow_error:
        throw std::overflow_error(message);
    case DistributedSourceFailure::runtime_error:
        throw std::runtime_error(message);
    case DistributedSourceFailure::none:
        break;
    }

    throw std::runtime_error(
        std::string(operation)
        + " failed with an invalid distributed error code");
}

#ifdef AMREX_USE_GPU
void
throw_device_source_failure(
    int local_device_failure,
    int device_failure,
    const char* operation)
{
    if (device_failure == 0) {
        return;
    }

    DistributedSourceFailure failure =
        device_failure >= device_source_invalid_pressure
        ? DistributedSourceFailure::overflow_error
        : DistributedSourceFailure::invalid_argument;

    std::string local_error;
    if (local_device_failure == device_failure) {
        switch (device_failure) {
        case device_source_invalid_density:
            local_error =
                "ERF Fire source coupling dry density must be finite and positive";
            break;
        case device_source_invalid_rhotheta:
            local_error =
                "ERF Fire source coupling rho theta must be finite and positive";
            break;
        case device_source_invalid_rhoqv:
            local_error =
                "ERF Fire source coupling rho qv must be finite and nonnegative";
            break;
        case device_source_invalid_source:
            local_error =
                "ERF Fire atmospheric source received invalid layer inputs";
            break;
        case device_source_invalid_terrain_geometry:
            local_error =
                "ERF Fire terrain source vertical-face geometry is invalid";
            break;
        case device_source_invalid_detj:
            local_error =
                "ERF Fire terrain source detJ_cc must be finite and positive";
            break;
        case device_source_invalid_pressure:
            local_error =
                "ERF Fire source coupling diagnosed invalid pressure";
            break;
        case device_source_overflow_source:
            local_error =
                "ERF Fire atmospheric source produced invalid tendency";
            break;
        case device_source_overflow_weight:
            local_error =
                "ERF Fire atmospheric deposition weight is not finite and positive";
            break;
        case device_source_overflow_volume:
            local_error =
                "ERF Fire terrain source physical cell volume is invalid";
            break;
        case device_source_overflow_normalization:
            local_error =
                "ERF Fire atmospheric deposition normalization is invalid";
            break;
        default:
            failure =
                DistributedSourceFailure::runtime_error;
            local_error =
                "invalid device source failure code";
            break;
        }
    }

    throw_distributed_source_failure(
        failure,
        local_error,
        operation);
}
#endif

template <typename Function>
void
run_distributed_source_projection(
    Function&& function,
    const char* operation)
{
    DistributedSourceFailure local_failure =
        DistributedSourceFailure::none;
    std::string local_error;

    try {
        function();
    } catch (const std::invalid_argument& error) {
        local_failure =
            DistributedSourceFailure::invalid_argument;
        local_error = error.what();
    } catch (const std::overflow_error& error) {
        local_failure =
            DistributedSourceFailure::overflow_error;
        local_error = error.what();
    } catch (const std::exception& error) {
        local_failure =
            DistributedSourceFailure::runtime_error;
        local_error = error.what();
    } catch (...) {
        local_failure =
            DistributedSourceFailure::runtime_error;
        local_error = "unknown local Fire source-projection error";
    }

    int failure = static_cast<int>(local_failure);
    amrex::ParallelDescriptor::ReduceIntMax(failure);

    if (failure
        != static_cast<int>(
            DistributedSourceFailure::none)) {
        throw_distributed_source_failure(
            static_cast<DistributedSourceFailure>(failure),
            local_error,
            operation);
    }
}

void
validate_source_state_scope_and_layout(
    const amrex::Geometry& geometry,
    const amrex::MultiFab& conserved_state_tn,
    MoistureType moisture_type)
{
    const amrex::Box& domain = geometry.Domain();

    require(
        domain.cellCentered(),
        "ERF Fire source coupling requires a cell-centered level-0 domain");
    require_level0_coverage(
        conserved_state_tn.boxArray(),
        domain,
        "ERF Fire source coupling conserved state does not cover level 0");
    require(
        moisture_type != MoistureType::None,
        "ERF Fire two-way source requires an allocated water-vapor state");
    require(
        conserved_state_tn.nComp() > RhoQ1_comp,
        "ERF Fire source coupling conserved state lacks RhoQ1");
}

amrex::BoxArray
make_fire_column_box_array(
    const FireSurfaceFeedbackRaster& feedback,
    const amrex::Box& domain)
{
    amrex::BoxArray columns =
        feedback.surface_layout().box_array();

    columns.shift(
        amrex::IntVect(
            domain.smallEnd(0),
            domain.smallEnd(1),
            domain.smallEnd(2)));

    for (int index = 0;
         index < columns.size();
         ++index) {
        amrex::Box column = columns[index];
        column.setRange(
            2,
            domain.smallEnd(2),
            domain.length(2));
        columns.set(index, column);
    }

    require_level0_coverage(
        columns,
        domain,
        "ERF Fire distributed column layout does not cover level 0");

    return columns;
}

[[maybe_unused]]
amrex::MultiFab
make_fire_state_columns(
    const amrex::BoxArray& column_boxes,
    const amrex::DistributionMapping& column_dm,
    const amrex::Geometry& geometry,
    const amrex::MultiFab& conserved_state_tn,
    MoistureType moisture_type)
{
    validate_source_state_scope_and_layout(
        geometry,
        conserved_state_tn,
        moisture_type);

    amrex::MultiFab result(
        column_boxes,
        column_dm,
        fire_state_component_count,
        0,
        fire_column_mf_info());

    result.ParallelCopy(
        conserved_state_tn,
        Rho_comp,
        fire_state_rho_comp,
        1,
        0,
        0);
    result.ParallelCopy(
        conserved_state_tn,
        RhoTheta_comp,
        fire_state_rhotheta_comp,
        1,
        0,
        0);
    result.ParallelCopy(
        conserved_state_tn,
        RhoQ1_comp,
        fire_state_rhoqv_comp,
        1,
        0,
        0);

    amrex::Gpu::streamSynchronize();
    return result;
}

[[maybe_unused]]
amrex::MultiFab
make_fire_source_columns(
    const FireSurfaceFeedbackRaster& feedback,
    const amrex::BoxArray& column_boxes,
    const amrex::Box& domain)
{
    amrex::MultiFab result(
        column_boxes,
        feedback.surface_layout().distribution_map(),
        fire_source_component_count,
        0,
        fire_column_mf_info());
    result.setVal(amrex::Real(0));

    const amrex::IntVect offset(
        domain.smallEnd(0),
        domain.smallEnd(1),
        domain.smallEnd(2));
    const amrex::IntVect no_ghost(0);

    result.ParallelCopy(
        feedback.distributed_values(),
        FireSurfaceFeedbackRaster::sensible_energy_comp,
        fire_source_rhotheta_comp,
        1,
        no_ghost,
        no_ghost,
        offset,
        amrex::Periodicity::NonPeriodic());
    result.ParallelCopy(
        feedback.distributed_values(),
        FireSurfaceFeedbackRaster::water_released_comp,
        fire_source_rhoqv_comp,
        1,
        no_ghost,
        no_ghost,
        offset,
        amrex::Periodicity::NonPeriodic());

    amrex::Gpu::streamSynchronize();
    return result;
}

[[maybe_unused]]
amrex::MultiFab
make_fire_scalar_columns(
    const amrex::BoxArray& column_boxes,
    const amrex::DistributionMapping& column_dm,
    const amrex::MultiFab& source)
{
    amrex::MultiFab result(
        column_boxes,
        column_dm,
        1,
        0,
        fire_column_mf_info());
    result.ParallelCopy(
        source,
        0,
        0,
        1,
        0,
        0);
    amrex::Gpu::streamSynchronize();
    return result;
}

[[maybe_unused]]
amrex::MultiFab
make_fire_nodal_height_columns(
    const amrex::BoxArray& column_boxes,
    const amrex::DistributionMapping& column_dm,
    const amrex::MultiFab& z_phys_nd)
{
    const amrex::BoxArray nodal_columns =
        amrex::convert(
            column_boxes,
            amrex::IntVect(1, 1, 1));

    amrex::MultiFab result(
        nodal_columns,
        column_dm,
        1,
        0,
        fire_column_mf_info());
    result.ParallelCopy(
        z_phys_nd,
        0,
        0,
        1,
        0,
        0);
    amrex::Gpu::streamSynchronize();
    return result;
}

[[maybe_unused]]
amrex::Real
diagnose_fire_column_pressure_pa(
    const amrex::Array4<const amrex::Real>& state,
    int i,
    int j,
    int k)
{
    const amrex::Real rho =
        state(i, j, k, fire_state_rho_comp);
    const amrex::Real rhotheta =
        state(i, j, k, fire_state_rhotheta_comp);
    const amrex::Real rhoqv =
        state(i, j, k, fire_state_rhoqv_comp);

    require(
        std::isfinite(rho)
            && rho > amrex::Real(0),
        "ERF Fire source coupling dry density must be finite and positive");
    require(
        std::isfinite(rhotheta)
            && rhotheta > amrex::Real(0),
        "ERF Fire source coupling rho theta must be finite and positive");
    require(
        std::isfinite(rhoqv)
            && rhoqv >= amrex::Real(0),
        "ERF Fire source coupling rho qv must be finite and nonnegative");

    const amrex::Real qv = rhoqv / rho;
    const amrex::Real pressure =
        getPgivenRTh(rhotheta, qv);

    if (!std::isfinite(pressure)
        || !(pressure > amrex::Real(0))) {
        throw std::overflow_error(
            "ERF Fire source coupling diagnosed invalid pressure");
    }

    return pressure;
}

std::unique_ptr<amrex::MultiFab>
map_fire_columns_to_native_source(
    const amrex::MultiFab& source_columns,
    const amrex::MultiFab& conserved_state_tn)
{
    auto result =
        std::make_unique<amrex::MultiFab>(
            conserved_state_tn.boxArray(),
            conserved_state_tn.DistributionMap(),
            conserved_state_tn.nComp(),
            0);
    result->setVal(amrex::Real(0));

    result->ParallelCopy(
        source_columns,
        fire_source_rhotheta_comp,
        RhoTheta_comp,
        1,
        0,
        0);
    result->ParallelCopy(
        source_columns,
        fire_source_rhoqv_comp,
        RhoQ1_comp,
        1,
        0,
        0);

    amrex::Gpu::streamSynchronize();
    return result;
}

[[maybe_unused]]
std::unique_ptr<amrex::MultiFab>
map_native_source_values_to_multifab(
    const std::vector<amrex::Real>& host_values,
    const amrex::Geometry& geometry,
    const amrex::MultiFab& conserved_state_tn)
{
    const amrex::Box& domain =
        geometry.Domain();
    const std::size_t nx =
        static_cast<std::size_t>(
            domain.length(0));
    const std::size_t ny =
        static_cast<std::size_t>(
            domain.length(1));
    const std::size_t nz =
        static_cast<std::size_t>(
            domain.length(2));

    require(
        host_values.size()
            == nx * ny * nz * std::size_t(2),
        "ERF Fire native source host-value size mismatch");

    auto result =
        std::make_unique<amrex::MultiFab>(
            conserved_state_tn.boxArray(),
            conserved_state_tn.DistributionMap(),
            conserved_state_tn.nComp(),
            0);
    result->setVal(amrex::Real(0));

    amrex::Gpu::DeviceVector<amrex::Real>
        device_values(host_values.size());
    amrex::Gpu::copy(
        amrex::Gpu::hostToDevice,
        host_values.begin(),
        host_values.end(),
        device_values.begin());

    const amrex::Real* values =
        device_values.data();
    const int ilo =
        domain.smallEnd(0);
    const int jlo =
        domain.smallEnd(1);
    const int klo =
        domain.smallEnd(2);
    const std::size_t nx_device = nx;
    const std::size_t ny_device = ny;

    for (amrex::MFIter mfi(
             *result,
             amrex::TilingIfNotGPU());
         mfi.isValid();
         ++mfi) {
        const amrex::Box box =
            mfi.tilebox();
        const auto array =
            result->array(mfi);

        amrex::ParallelFor(
            box,
            [=] AMREX_GPU_DEVICE (
                int i, int j, int k) noexcept {
                const std::size_t ii =
                    static_cast<std::size_t>(
                        i - ilo);
                const std::size_t jj =
                    static_cast<std::size_t>(
                        j - jlo);
                const std::size_t kk =
                    static_cast<std::size_t>(
                        k - klo);
                const std::size_t flat =
                    (kk * ny_device + jj)
                    * nx_device + ii;

                array(i, j, k, RhoTheta_comp) =
                    values[2 * flat];
                array(i, j, k, RhoQ1_comp) =
                    values[2 * flat + 1];
            });
    }

    amrex::Gpu::streamSynchronize();
    return result;
}

[[maybe_unused]]
std::unique_ptr<amrex::MultiFab>
map_source_field_to_multifab(
    const ERFFireAtmosphericSourceField& source,
    const amrex::Geometry& geometry,
    const amrex::MultiFab& conserved_state_tn)
{
    const amrex::Box& domain =
        geometry.Domain();
    const std::size_t nx =
        static_cast<std::size_t>(
            domain.length(0));
    const std::size_t ny =
        static_cast<std::size_t>(
            domain.length(1));
    const std::size_t nz =
        static_cast<std::size_t>(
            domain.length(2));

    require(
        source.nx() == nx
            && source.ny() == ny
            && source.nz() == nz,
        "ERF Fire source field dimensions do not match level 0");

    std::vector<amrex::Real> host_values(
        source.cell_count() * std::size_t(2),
        amrex::Real(0));

    for (std::size_t k = 0; k < nz; ++k) {
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                const std::size_t flat =
                    (k * ny + j) * nx + i;
                const auto& cell =
                    source.cell(i, j, k);
                host_values[2 * flat] =
                    cell.rhotheta_tendency_kg_K_m3_s;
                host_values[2 * flat + 1] =
                    cell.rhoqv_tendency_kg_m3_s;
            }
        }
    }

    return map_native_source_values_to_multifab(
        host_values,
        geometry,
        conserved_state_tn);
}

} // namespace

std::unique_ptr<amrex::MultiFab>
make_erf_fire_level0_source_tendency(
    const FireSurfaceFeedbackRaster& feedback,
    const ERFFireLevel0EnvironmentInputs& environment_inputs,
    const amrex::MultiFab& conserved_state_tn,
    MoistureType moisture_type,
    amrex::Real dt_s,
    ERFFireAtmosphericSourceOptions options)
{
    if (!same_horizontal_geometry(
            feedback.geometry(),
            environment_inputs.geometry)) {
        const FireSurfaceFeedbackRaster level0_feedback =
            conservatively_regrid_fire_surface_feedback(
                feedback,
                level0_horizontal_geometry(
                    environment_inputs.geometry));

        return make_erf_fire_level0_source_tendency(
            level0_feedback,
            environment_inputs,
            conserved_state_tn,
            moisture_type,
            dt_s,
            options);
    }

    const std::vector<amrex::Real> vertical_faces_agl_m =
        erf_fire_level0_flat_vertical_faces_agl(
            environment_inputs);
    const amrex::Box& domain =
        environment_inputs.geometry.Domain();

    require(
        vertical_faces_agl_m.size()
            == static_cast<std::size_t>(
                   domain.length(2) + 1),
        "ERF Fire flat source vertical-face count does not match level 0");

    const amrex::BoxArray column_boxes =
        make_fire_column_box_array(
            feedback,
            domain);
    const auto& column_dm =
        feedback.surface_layout().distribution_map();

#ifdef AMREX_USE_GPU
    validate_source_state_scope_and_layout(
        environment_inputs.geometry,
        conserved_state_tn,
        moisture_type);
    require(
        std::isfinite(dt_s) && dt_s > amrex::Real(0),
        "ERF Fire atmospheric source dt must be finite and positive");

    const std::vector<amrex::Real> normalized_layer_weights =
        flat_normalized_layer_weights(
            vertical_faces_agl_m,
            options.extinction_depth_m);

    const amrex::Real horizontal_area_m2 =
        feedback.geometry().dx_m
        * feedback.geometry().dy_m;
    std::vector<amrex::Real> physical_cell_volume_m3(
        static_cast<std::size_t>(domain.length(2)),
        amrex::Real(0));
    for (std::size_t k = 0;
         k < physical_cell_volume_m3.size();
         ++k) {
        physical_cell_volume_m3[k] =
            horizontal_area_m2
            * (vertical_faces_agl_m[k + 1]
               - vertical_faces_agl_m[k]);
        require(
            std::isfinite(physical_cell_volume_m3[k])
                && physical_cell_volume_m3[k] > amrex::Real(0),
            "ERF Fire atmospheric source physical cell volume must be finite and positive");
    }

    amrex::MultiFab state_columns(
        column_boxes,
        column_dm,
        fire_state_component_count,
        0);
    state_columns.ParallelCopy(
        conserved_state_tn,
        Rho_comp,
        fire_state_rho_comp,
        1,
        0,
        0);
    state_columns.ParallelCopy(
        conserved_state_tn,
        RhoTheta_comp,
        fire_state_rhotheta_comp,
        1,
        0,
        0);
    state_columns.ParallelCopy(
        conserved_state_tn,
        RhoQ1_comp,
        fire_state_rhoqv_comp,
        1,
        0,
        0);

    const int klo = domain.smallEnd(2);
    amrex::BoxArray surface_boxes(column_boxes.boxList());
    for (int index = 0;
         index < surface_boxes.size();
         ++index) {
        amrex::Box surface_box = surface_boxes[index];
        surface_box.setRange(2, klo, 1);
        surface_boxes.set(index, surface_box);
    }

    amrex::MultiFab surface_release(
        surface_boxes,
        column_dm,
        fire_source_component_count,
        0);
    surface_release.setVal(amrex::Real(0));

    const amrex::IntVect offset(
        domain.smallEnd(0),
        domain.smallEnd(1),
        domain.smallEnd(2));
    const amrex::IntVect no_ghost(0);
    surface_release.ParallelCopy(
        feedback.distributed_values(),
        FireSurfaceFeedbackRaster::sensible_energy_comp,
        fire_source_rhotheta_comp,
        1,
        no_ghost,
        no_ghost,
        offset,
        amrex::Periodicity::NonPeriodic());
    surface_release.ParallelCopy(
        feedback.distributed_values(),
        FireSurfaceFeedbackRaster::water_released_comp,
        fire_source_rhoqv_comp,
        1,
        no_ghost,
        no_ghost,
        offset,
        amrex::Periodicity::NonPeriodic());
    amrex::Gpu::streamSynchronize();

    amrex::MultiFab source_columns(
        column_boxes,
        column_dm,
        fire_source_component_count,
        0);
    source_columns.setVal(amrex::Real(0));

    amrex::Gpu::DeviceVector<amrex::Real> device_weights(
        normalized_layer_weights.size());
    amrex::Gpu::DeviceVector<amrex::Real> device_volumes(
        physical_cell_volume_m3.size());
    amrex::Gpu::copy(
        amrex::Gpu::hostToDevice,
        normalized_layer_weights.begin(),
        normalized_layer_weights.end(),
        device_weights.begin());
    amrex::Gpu::copy(
        amrex::Gpu::hostToDevice,
        physical_cell_volume_m3.begin(),
        physical_cell_volume_m3.end(),
        device_volumes.begin());
    amrex::Gpu::streamSynchronize();

    const auto state_arrays =
        state_columns.const_arrays();
    const auto release_arrays =
        surface_release.const_arrays();
    const auto source_arrays =
        source_columns.arrays();
    const amrex::Real* weights =
        device_weights.data();
    const amrex::Real* volumes =
        device_volumes.data();
    const amrex::Real device_dt_s = dt_s;

    const int local_device_failure =
        amrex::ParReduce(
            amrex::TypeList<
                amrex::ReduceOpMax>{},
            amrex::TypeList<int>{},
            source_columns,
            [=] AMREX_GPU_DEVICE (
                int box_no,
                int i,
                int j,
                int k) noexcept
                -> amrex::GpuTuple<int>
            {
                const auto state =
                    state_arrays[box_no];
                const auto release =
                    release_arrays[box_no];
                const auto source =
                    source_arrays[box_no];

                const FireSurfaceFeedbackCell surface_feedback{
                    amrex::Real(0),
                    release(
                        i, j, klo,
                        fire_source_rhotheta_comp),
                    release(
                        i, j, klo,
                        fire_source_rhoqv_comp)};

                const amrex::Real rho =
                    state(
                        i, j, k,
                        fire_state_rho_comp);
                const amrex::Real rhotheta =
                    state(
                        i, j, k,
                        fire_state_rhotheta_comp);
                const amrex::Real rhoqv =
                    state(
                        i, j, k,
                        fire_state_rhoqv_comp);

                if (!amrex::Math::isfinite(rho)
                    || !(rho > amrex::Real(0))) {
                    return {
                        device_source_invalid_density};
                }
                if (!amrex::Math::isfinite(rhotheta)
                    || !(rhotheta > amrex::Real(0))) {
                    return {
                        device_source_invalid_rhotheta};
                }
                if (!amrex::Math::isfinite(rhoqv)
                    || rhoqv < amrex::Real(0)) {
                    return {
                        device_source_invalid_rhoqv};
                }

                const amrex::Real qv = rhoqv / rho;
                const amrex::Real pressure =
                    getPgivenRTh(rhotheta, qv);
                if (!amrex::Math::isfinite(pressure)
                    || !(pressure > amrex::Real(0))) {
                    return {
                        device_source_invalid_pressure};
                }

                ERFFireAtmosphericSourceCell cell{};
                const std::size_t local_k =
                    static_cast<std::size_t>(
                        k - klo);
                const auto status =
                    try_make_erf_fire_atmospheric_source_cell(
                        surface_feedback,
                        weights[local_k],
                        volumes[local_k],
                        pressure,
                        device_dt_s,
                        cell);

                if (status
                    == ERFFireAtmosphericSourceStatus::invalid_argument) {
                    return {
                        device_source_invalid_source};
                }
                if (status
                    == ERFFireAtmosphericSourceStatus::overflow_error) {
                    return {
                        device_source_overflow_source};
                }
                if (status
                    != ERFFireAtmosphericSourceStatus::success) {
                    return {
                        device_source_overflow_source};
                }

                source(
                    i, j, k,
                    fire_source_rhotheta_comp) =
                        cell.rhotheta_tendency_kg_K_m3_s;
                source(
                    i, j, k,
                    fire_source_rhoqv_comp) =
                        cell.rhoqv_tendency_kg_m3_s;

                return {0};
            });

    int device_failure =
        local_device_failure;
    amrex::ParallelDescriptor::ReduceIntMax(
        device_failure);

    throw_device_source_failure(
        local_device_failure,
        device_failure,
        "distributed ERF Fire flat source projection");

    return map_fire_columns_to_native_source(
        source_columns,
        conserved_state_tn);
#else
    amrex::MultiFab state_columns =
        make_fire_state_columns(
            column_boxes,
            column_dm,
            environment_inputs.geometry,
            conserved_state_tn,
            moisture_type);
    amrex::MultiFab source_columns =
        make_fire_source_columns(
            feedback,
            column_boxes,
            domain);

    const amrex::Real horizontal_area_m2 =
        feedback.geometry().dx_m
        * feedback.geometry().dy_m;
    std::vector<amrex::Real> physical_cell_volume_m3(
        static_cast<std::size_t>(domain.length(2)),
        amrex::Real(0));
    for (std::size_t k = 0;
         k < physical_cell_volume_m3.size();
         ++k) {
        physical_cell_volume_m3[k] =
            horizontal_area_m2
            * (vertical_faces_agl_m[k + 1]
               - vertical_faces_agl_m[k]);
    }

    run_distributed_source_projection(
        [&] {
            const int klo = domain.smallEnd(2);
            const int khi = domain.bigEnd(2);
            std::vector<amrex::Real> column_pressure_pa(
                static_cast<std::size_t>(domain.length(2)),
                amrex::Real(0));

            for (amrex::MFIter mfi(source_columns);
                 mfi.isValid(); ++mfi) {
                const amrex::Box& box = mfi.validbox();
                const auto state = state_columns.const_array(mfi);
                const auto source = source_columns.array(mfi);

                for (int j = box.smallEnd(1);
                     j <= box.bigEnd(1);
                     ++j) {
                    for (int i = box.smallEnd(0);
                         i <= box.bigEnd(0);
                         ++i) {
                        const FireSurfaceFeedbackCell surface_feedback{
                            amrex::Real(0),
                            source(
                                i, j, klo,
                                fire_source_rhotheta_comp),
                            source(
                                i, j, klo,
                                fire_source_rhoqv_comp)};

                        for (int k = klo; k <= khi; ++k) {
                            column_pressure_pa[
                                static_cast<std::size_t>(k - klo)] =
                                diagnose_fire_column_pressure_pa(
                                    state,
                                    i,
                                    j,
                                    k);
                        }

                        const auto column =
                            make_erf_fire_atmospheric_source_column(
                                surface_feedback,
                                vertical_faces_agl_m,
                                physical_cell_volume_m3,
                                column_pressure_pa,
                                dt_s,
                                options);

                        for (int k = klo; k <= khi; ++k) {
                            const auto& value =
                                column[
                                    static_cast<std::size_t>(k - klo)];
                            source(
                                i, j, k,
                                fire_source_rhotheta_comp) =
                                value.rhotheta_tendency_kg_K_m3_s;
                            source(
                                i, j, k,
                                fire_source_rhoqv_comp) =
                                value.rhoqv_tendency_kg_m3_s;
                        }
                    }
                }
            }
        },
        "distributed ERF Fire flat source projection");

    return map_fire_columns_to_native_source(
        source_columns,
        conserved_state_tn);
#endif
}

std::unique_ptr<amrex::MultiFab>
make_erf_fire_level0_terrain_source_tendency(
    const FireSurfaceFeedbackRaster& feedback,
    const ERFFireLevel0EnvironmentInputs& environment_inputs,
    const amrex::MultiFab& detJ_cc,
    const amrex::MultiFab& conserved_state_tn,
    MoistureType moisture_type,
    amrex::Real dt_s,
    ERFFireAtmosphericSourceOptions options)
{
    if (!same_horizontal_geometry(
            feedback.geometry(),
            environment_inputs.geometry)) {
        const FireSurfaceFeedbackRaster level0_feedback =
            conservatively_regrid_fire_surface_feedback(
                feedback,
                level0_horizontal_geometry(
                    environment_inputs.geometry));

        return make_erf_fire_level0_terrain_source_tendency(
            level0_feedback,
            environment_inputs,
            detJ_cc,
            conserved_state_tn,
            moisture_type,
            dt_s,
            options);
    }

    validate_terrain_source_scope_and_layout(
        environment_inputs,
        detJ_cc);

    const amrex::Box& domain =
        environment_inputs.geometry.Domain();
    const std::size_t nz =
        static_cast<std::size_t>(
            domain.length(2));

    const amrex::BoxArray column_boxes =
        make_fire_column_box_array(
            feedback,
            domain);
    const auto& column_dm =
        feedback.surface_layout().distribution_map();

#ifdef AMREX_USE_GPU
    validate_source_state_scope_and_layout(
        environment_inputs.geometry,
        conserved_state_tn,
        moisture_type);
    require(
        std::isfinite(dt_s) && dt_s > amrex::Real(0),
        "ERF Fire atmospheric source dt must be finite and positive");
    require(
        std::isfinite(options.extinction_depth_m)
            && options.extinction_depth_m > amrex::Real(0),
        "ERF Fire atmospheric extinction depth must be finite and positive");

    amrex::MultiFab state_columns(
        column_boxes,
        column_dm,
        fire_state_component_count,
        0);
    state_columns.ParallelCopy(
        conserved_state_tn,
        Rho_comp,
        fire_state_rho_comp,
        1,
        0,
        0);
    state_columns.ParallelCopy(
        conserved_state_tn,
        RhoTheta_comp,
        fire_state_rhotheta_comp,
        1,
        0,
        0);
    state_columns.ParallelCopy(
        conserved_state_tn,
        RhoQ1_comp,
        fire_state_rhoqv_comp,
        1,
        0,
        0);

    const amrex::BoxArray nodal_column_boxes =
        amrex::convert(
            column_boxes,
            amrex::IntVect(1, 1, 1));
    amrex::MultiFab z_columns(
        nodal_column_boxes,
        column_dm,
        1,
        0);
    z_columns.ParallelCopy(
        environment_inputs.z_phys_nd,
        0,
        0,
        1,
        0,
        0);

    amrex::MultiFab detJ_columns(
        column_boxes,
        column_dm,
        1,
        0);
    detJ_columns.ParallelCopy(
        detJ_cc,
        0,
        0,
        1,
        0,
        0);

    const int klo = domain.smallEnd(2);
    const int khi = domain.bigEnd(2);
    const int device_nz =
        static_cast<int>(nz);

    amrex::BoxArray surface_boxes(
        column_boxes.boxList());
    for (int index = 0;
         index < surface_boxes.size();
         ++index) {
        amrex::Box surface_box =
            surface_boxes[index];
        surface_box.setRange(
            2,
            klo,
            1);
        surface_boxes.set(
            index,
            surface_box);
    }

    amrex::MultiFab surface_release(
        surface_boxes,
        column_dm,
        fire_source_component_count,
        0);
    surface_release.setVal(amrex::Real(0));

    const amrex::IntVect offset(
        domain.smallEnd(0),
        domain.smallEnd(1),
        domain.smallEnd(2));
    const amrex::IntVect no_ghost(0);
    surface_release.ParallelCopy(
        feedback.distributed_values(),
        FireSurfaceFeedbackRaster::sensible_energy_comp,
        fire_source_rhotheta_comp,
        1,
        no_ghost,
        no_ghost,
        offset,
        amrex::Periodicity::NonPeriodic());
    surface_release.ParallelCopy(
        feedback.distributed_values(),
        FireSurfaceFeedbackRaster::water_released_comp,
        fire_source_rhoqv_comp,
        1,
        no_ghost,
        no_ghost,
        offset,
        amrex::Periodicity::NonPeriodic());

    amrex::MultiFab source_columns(
        column_boxes,
        column_dm,
        fire_source_component_count,
        0);
    source_columns.setVal(amrex::Real(0));

    amrex::Gpu::streamSynchronize();

    const auto state_arrays =
        state_columns.const_arrays();
    const auto z_arrays =
        z_columns.const_arrays();
    const auto detJ_arrays =
        detJ_columns.const_arrays();
    const auto release_arrays =
        surface_release.const_arrays();
    const auto source_arrays =
        source_columns.arrays();

    const auto cell_size =
        environment_inputs.geometry.CellSizeArray();
    const amrex::Real computational_volume_m3 =
        cell_size[0]
        * cell_size[1]
        * cell_size[2];
    require(
        std::isfinite(computational_volume_m3)
            && computational_volume_m3 > amrex::Real(0),
        "ERF Fire terrain source computational cell volume must be finite and positive");

    const amrex::Real device_computational_volume_m3 =
        computational_volume_m3;
    const amrex::Real device_extinction_depth_m =
        options.extinction_depth_m;
    const amrex::Real device_dt_s =
        dt_s;

    const int local_device_failure =
        amrex::ParReduce(
            amrex::TypeList<
                amrex::ReduceOpMax>{},
            amrex::TypeList<int>{},
            surface_release,
            [=] AMREX_GPU_DEVICE (
                int box_no,
                int i,
                int j,
                int k_surface) noexcept
                -> amrex::GpuTuple<int>
            {
                if (k_surface != klo) {
                    return {
                        device_source_invalid_terrain_geometry};
                }

                const auto state =
                    state_arrays[box_no];
                const auto z =
                    z_arrays[box_no];
                const auto detJ =
                    detJ_arrays[box_no];
                const auto release =
                    release_arrays[box_no];
                const auto source =
                    source_arrays[box_no];

                const FireSurfaceFeedbackCell surface_feedback{
                    amrex::Real(0),
                    release(
                        i, j, klo,
                        fire_source_rhotheta_comp),
                    release(
                        i, j, klo,
                        fire_source_rhoqv_comp)};

                const amrex::Real ground_height_m =
                    amrex::Real(0.25)
                    * (z(i, j, klo)
                       + z(i + 1, j, klo)
                       + z(i, j + 1, klo)
                       + z(i + 1, j + 1, klo));
                if (!amrex::Math::isfinite(
                        ground_height_m)) {
                    return {
                        device_source_invalid_terrain_geometry};
                }

                amrex::Real raw_weight_total =
                    amrex::Real(0);

                for (int local_k = 0;
                     local_k < device_nz;
                     ++local_k) {
                    const int k =
                        klo + local_k;

                    amrex::Real zlo_m =
                        amrex::Real(0);
                    if (local_k != 0) {
                        const amrex::Real physical_lo_m =
                            amrex::Real(0.25)
                            * (z(i, j, k)
                               + z(i + 1, j, k)
                               + z(i, j + 1, k)
                               + z(i + 1, j + 1, k));
                        zlo_m =
                            physical_lo_m
                            - ground_height_m;
                    }

                    const amrex::Real physical_hi_m =
                        amrex::Real(0.25)
                        * (z(i, j, k + 1)
                           + z(i + 1, j, k + 1)
                           + z(i, j + 1, k + 1)
                           + z(i + 1, j + 1, k + 1));
                    const amrex::Real zhi_m =
                        physical_hi_m
                        - ground_height_m;

                    if (!amrex::Math::isfinite(zlo_m)
                        || !amrex::Math::isfinite(zhi_m)
                        || !(zhi_m > zlo_m)) {
                        return {
                            device_source_invalid_terrain_geometry};
                    }

                    amrex::Real raw_weight =
                        amrex::Real(0);
                    const auto weight_status =
                        try_erf_fire_atmospheric_source_raw_layer_weight(
                            zlo_m,
                            zhi_m,
                            device_extinction_depth_m,
                            raw_weight);
                    if (weight_status
                        == ERFFireAtmosphericSourceStatus::invalid_argument) {
                        return {
                            device_source_invalid_terrain_geometry};
                    }
                    if (weight_status
                        != ERFFireAtmosphericSourceStatus::success) {
                        return {
                            device_source_overflow_weight};
                    }

                    raw_weight_total +=
                        raw_weight;
                    if (!amrex::Math::isfinite(
                            raw_weight_total)) {
                        return {
                            device_source_overflow_normalization};
                    }
                }

                if (!(raw_weight_total
                        > amrex::Real(0))) {
                    return {
                        device_source_overflow_normalization};
                }

                amrex::Real normalized_sum =
                    amrex::Real(0);

                for (int local_k = 0;
                     local_k < device_nz;
                     ++local_k) {
                    const int k =
                        klo + local_k;

                    amrex::Real zlo_m =
                        amrex::Real(0);
                    if (local_k != 0) {
                        const amrex::Real physical_lo_m =
                            amrex::Real(0.25)
                            * (z(i, j, k)
                               + z(i + 1, j, k)
                               + z(i, j + 1, k)
                               + z(i + 1, j + 1, k));
                        zlo_m =
                            physical_lo_m
                            - ground_height_m;
                    }

                    const amrex::Real physical_hi_m =
                        amrex::Real(0.25)
                        * (z(i, j, k + 1)
                           + z(i + 1, j, k + 1)
                           + z(i, j + 1, k + 1)
                           + z(i + 1, j + 1, k + 1));
                    const amrex::Real zhi_m =
                        physical_hi_m
                        - ground_height_m;

                    amrex::Real raw_weight =
                        amrex::Real(0);
                    const auto weight_status =
                        try_erf_fire_atmospheric_source_raw_layer_weight(
                            zlo_m,
                            zhi_m,
                            device_extinction_depth_m,
                            raw_weight);
                    if (weight_status
                        == ERFFireAtmosphericSourceStatus::invalid_argument) {
                        return {
                            device_source_invalid_terrain_geometry};
                    }
                    if (weight_status
                        != ERFFireAtmosphericSourceStatus::success) {
                        return {
                            device_source_overflow_weight};
                    }

                    amrex::Real normalized_weight =
                        amrex::Real(0);
                    if (k == khi) {
                        const amrex::Real remainder =
                            amrex::Real(1)
                            - normalized_sum;
                        normalized_weight =
                            remainder > amrex::Real(0)
                            ? remainder
                            : amrex::Real(0);
                    } else {
                        normalized_weight =
                            raw_weight
                            / raw_weight_total;
                        normalized_sum +=
                            normalized_weight;
                    }

                    if (!amrex::Math::isfinite(
                            normalized_weight)
                        || normalized_weight
                            < amrex::Real(0)) {
                        return {
                            device_source_overflow_normalization};
                    }

                    const amrex::Real jacobian =
                        detJ(i, j, k);
                    if (!amrex::Math::isfinite(
                            jacobian)
                        || !(jacobian
                             > amrex::Real(0))) {
                        return {
                            device_source_invalid_detj};
                    }

                    const amrex::Real volume_m3 =
                        device_computational_volume_m3
                        * jacobian;
                    if (!amrex::Math::isfinite(
                            volume_m3)
                        || !(volume_m3
                             > amrex::Real(0))) {
                        return {
                            device_source_overflow_volume};
                    }

                    const amrex::Real rho =
                        state(
                            i, j, k,
                            fire_state_rho_comp);
                    const amrex::Real rhotheta =
                        state(
                            i, j, k,
                            fire_state_rhotheta_comp);
                    const amrex::Real rhoqv =
                        state(
                            i, j, k,
                            fire_state_rhoqv_comp);

                    if (!amrex::Math::isfinite(rho)
                        || !(rho > amrex::Real(0))) {
                        return {
                            device_source_invalid_density};
                    }
                    if (!amrex::Math::isfinite(
                            rhotheta)
                        || !(rhotheta
                             > amrex::Real(0))) {
                        return {
                            device_source_invalid_rhotheta};
                    }
                    if (!amrex::Math::isfinite(
                            rhoqv)
                        || rhoqv < amrex::Real(0)) {
                        return {
                            device_source_invalid_rhoqv};
                    }

                    const amrex::Real qv =
                        rhoqv / rho;
                    const amrex::Real pressure =
                        getPgivenRTh(
                            rhotheta,
                            qv);
                    if (!amrex::Math::isfinite(
                            pressure)
                        || !(pressure
                             > amrex::Real(0))) {
                        return {
                            device_source_invalid_pressure};
                    }

                    ERFFireAtmosphericSourceCell cell{};
                    const auto source_status =
                        try_make_erf_fire_atmospheric_source_cell(
                            surface_feedback,
                            normalized_weight,
                            volume_m3,
                            pressure,
                            device_dt_s,
                            cell);

                    if (source_status
                        == ERFFireAtmosphericSourceStatus::invalid_argument) {
                        return {
                            device_source_invalid_source};
                    }
                    if (source_status
                        != ERFFireAtmosphericSourceStatus::success) {
                        return {
                            device_source_overflow_source};
                    }

                    source(
                        i, j, k,
                        fire_source_rhotheta_comp) =
                            cell.rhotheta_tendency_kg_K_m3_s;
                    source(
                        i, j, k,
                        fire_source_rhoqv_comp) =
                            cell.rhoqv_tendency_kg_m3_s;
                }

                return {0};
            });

    int device_failure =
        local_device_failure;
    amrex::ParallelDescriptor::ReduceIntMax(
        device_failure);

    throw_device_source_failure(
        local_device_failure,
        device_failure,
        "distributed ERF Fire terrain source projection");

    return map_fire_columns_to_native_source(
        source_columns,
        conserved_state_tn);
#else
    amrex::MultiFab state_columns =
        make_fire_state_columns(
            column_boxes,
            column_dm,
            environment_inputs.geometry,
            conserved_state_tn,
            moisture_type);
    amrex::MultiFab z_columns =
        make_fire_nodal_height_columns(
            column_boxes,
            column_dm,
            environment_inputs.z_phys_nd);
    amrex::MultiFab detJ_columns =
        make_fire_scalar_columns(
            column_boxes,
            column_dm,
            detJ_cc);
    amrex::MultiFab source_columns =
        make_fire_source_columns(
            feedback,
            column_boxes,
            domain);

    const auto cell_size =
        environment_inputs.geometry.CellSizeArray();
    const amrex::Real computational_volume_m3 =
        cell_size[0]
        * cell_size[1]
        * cell_size[2];

    require(
        std::isfinite(computational_volume_m3)
            && computational_volume_m3 > amrex::Real(0),
        "ERF Fire terrain source computational cell volume must be finite and positive");

    run_distributed_source_projection(
        [&] {
            const int klo = domain.smallEnd(2);
            const int khi = domain.bigEnd(2);
            std::vector<amrex::Real> vertical_faces_agl_m(
                nz + 1,
                amrex::Real(0));
            std::vector<amrex::Real> physical_cell_volume_m3(
                nz,
                amrex::Real(0));
            std::vector<amrex::Real> column_pressure_pa(
                nz,
                amrex::Real(0));

            for (amrex::MFIter mfi(source_columns);
                 mfi.isValid(); ++mfi) {
                const amrex::Box& box = mfi.validbox();
                const auto state = state_columns.const_array(mfi);
                const auto z = z_columns.const_array(mfi);
                const auto detJ = detJ_columns.const_array(mfi);
                const auto source = source_columns.array(mfi);

                for (int j = box.smallEnd(1);
                     j <= box.bigEnd(1);
                     ++j) {
                    for (int i = box.smallEnd(0);
                         i <= box.bigEnd(0);
                         ++i) {
                        const FireSurfaceFeedbackCell surface_feedback{
                            amrex::Real(0),
                            source(
                                i, j, klo,
                                fire_source_rhotheta_comp),
                            source(
                                i, j, klo,
                                fire_source_rhoqv_comp)};

                        const amrex::Real ground_height_m =
                            terrain_cell_vertical_face_height_m(
                                z,
                                i,
                                j,
                                klo);
                        vertical_faces_agl_m[0] =
                            amrex::Real(0);

                        for (std::size_t kf = 1;
                             kf <= nz;
                             ++kf) {
                            const amrex::Real physical_height_m =
                                terrain_cell_vertical_face_height_m(
                                    z,
                                    i,
                                    j,
                                    klo
                                        + static_cast<int>(kf));
                            const amrex::Real agl_m =
                                physical_height_m
                                - ground_height_m;
                            require(
                                std::isfinite(agl_m),
                                "ERF Fire terrain source AGL face height must be finite");
                            vertical_faces_agl_m[kf] =
                                agl_m;
                        }

                        for (int k = klo; k <= khi; ++k) {
                            const std::size_t local_k =
                                static_cast<std::size_t>(k - klo);
                            const amrex::Real jacobian =
                                detJ(i, j, k);
                            require(
                                std::isfinite(jacobian)
                                    && jacobian > amrex::Real(0),
                                "ERF Fire terrain source detJ_cc must be finite and positive");

                            const amrex::Real volume_m3 =
                                computational_volume_m3
                                * jacobian;
                            if (!std::isfinite(volume_m3)
                                || !(volume_m3 > amrex::Real(0))) {
                                throw std::overflow_error(
                                    "ERF Fire terrain source physical cell volume is invalid");
                            }

                            physical_cell_volume_m3[local_k] =
                                volume_m3;
                            column_pressure_pa[local_k] =
                                diagnose_fire_column_pressure_pa(
                                    state,
                                    i,
                                    j,
                                    k);
                        }

                        const auto column =
                            make_erf_fire_atmospheric_source_column(
                                surface_feedback,
                                vertical_faces_agl_m,
                                physical_cell_volume_m3,
                                column_pressure_pa,
                                dt_s,
                                options);

                        for (int k = klo; k <= khi; ++k) {
                            const auto& value =
                                column[
                                    static_cast<std::size_t>(k - klo)];
                            source(
                                i, j, k,
                                fire_source_rhotheta_comp) =
                                value.rhotheta_tendency_kg_K_m3_s;
                            source(
                                i, j, k,
                                fire_source_rhoqv_comp) =
                                value.rhoqv_tendency_kg_m3_s;
                        }
                    }
                }
            }
        },
        "distributed ERF Fire terrain source projection");

    return map_fire_columns_to_native_source(
        source_columns,
        conserved_state_tn);
#endif
}


} // namespace ERFFire
