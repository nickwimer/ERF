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

bool
same_horizontal_geometry(
    const FireCartesianRasterGeometry2D& fire_geometry,
    const amrex::Geometry& erf_geometry) noexcept
{
    const amrex::Box& domain = erf_geometry.Domain();
    const auto prob_lo = erf_geometry.ProbLoArray();
    const auto cell_size = erf_geometry.CellSizeArray();

    return fire_geometry.nx
            == static_cast<std::size_t>(domain.length(0))
        && fire_geometry.ny
            == static_cast<std::size_t>(domain.length(1))
        && fire_geometry.xlo_m == prob_lo[0]
        && fire_geometry.ylo_m == prob_lo[1]
        && fire_geometry.dx_m == cell_size[0]
        && fire_geometry.dy_m == cell_size[1];
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
    require(
        same_horizontal_geometry(
            feedback.geometry(),
            environment_inputs.geometry),
        "ERF Fire feedback raster does not match level-0 horizontal geometry");

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
    require(
        same_horizontal_geometry(
            feedback.geometry(),
            environment_inputs.geometry),
        "ERF Fire feedback raster does not match level-0 horizontal geometry");

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
}


} // namespace ERFFire
