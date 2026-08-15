#include "ERF_FireLevel0SourceCoupling.H"

#include <ERF_EOS.H>
#include <ERF_IndexDefines.H>

#include <AMReX_Arena.H>
#include <AMReX_BoxArray.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_MFIter.H>

#include <cmath>
#include <cstddef>
#include <stdexcept>
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

    const std::vector<amrex::Real> pressure_pa =
        diagnose_level0_pressure_pa(
            environment_inputs.geometry,
            conserved_state_tn,
            moisture_type);

    const ERFFireAtmosphericSourceField source =
        make_erf_fire_atmospheric_source_field(
            feedback,
            vertical_faces_agl_m,
            pressure_pa,
            dt_s,
            options);

    return map_source_field_to_multifab(
        source,
        environment_inputs.geometry,
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

    const std::vector<amrex::Real> pressure_pa =
        diagnose_level0_pressure_pa(
            environment_inputs.geometry,
            conserved_state_tn,
            moisture_type);

    const amrex::Box& domain =
        environment_inputs.geometry.Domain();
    const amrex::Box nodal_domain =
        amrex::convert(
            domain,
            amrex::IntVect(1, 1, 1));

    amrex::FArrayBox z_host(
        nodal_domain,
        1,
        amrex::The_Pinned_Arena());
    environment_inputs.z_phys_nd.copyTo(
        z_host,
        0,
        0,
        1,
        0);

    amrex::FArrayBox detJ_host(
        domain,
        1,
        amrex::The_Pinned_Arena());
    detJ_cc.copyTo(
        detJ_host,
        0,
        0,
        1,
        0);

    const auto z =
        z_host.const_array();
    const auto detJ =
        detJ_host.const_array();

    const std::size_t nx =
        static_cast<std::size_t>(
            domain.length(0));
    const std::size_t ny =
        static_cast<std::size_t>(
            domain.length(1));
    const std::size_t nz =
        static_cast<std::size_t>(
            domain.length(2));

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

    const int ilo =
        domain.smallEnd(0);
    const int jlo =
        domain.smallEnd(1);
    const int klo =
        domain.smallEnd(2);

    std::vector<amrex::Real> host_values(
        nx * ny * nz * std::size_t(2),
        amrex::Real(0));
    std::vector<amrex::Real> vertical_faces_agl_m(
        nz + 1,
        amrex::Real(0));
    std::vector<amrex::Real> physical_cell_volume_m3(
        nz,
        amrex::Real(0));
    std::vector<amrex::Real> column_pressure_pa(
        nz,
        amrex::Real(0));

    for (std::size_t j = 0; j < ny; ++j) {
        for (std::size_t i = 0; i < nx; ++i) {
            const int ii =
                ilo + static_cast<int>(i);
            const int jj =
                jlo + static_cast<int>(j);

            const amrex::Real ground_height_m =
                terrain_cell_vertical_face_height_m(
                    z,
                    ii,
                    jj,
                    klo);
            vertical_faces_agl_m[0] =
                amrex::Real(0);

            for (std::size_t kf = 1;
                 kf <= nz;
                 ++kf) {
                const amrex::Real physical_height_m =
                    terrain_cell_vertical_face_height_m(
                        z,
                        ii,
                        jj,
                        klo + static_cast<int>(kf));
                const amrex::Real agl_m =
                    physical_height_m
                    - ground_height_m;

                require(
                    std::isfinite(agl_m),
                    "ERF Fire terrain source AGL face height must be finite");
                vertical_faces_agl_m[kf] =
                    agl_m;
            }

            for (std::size_t k = 0; k < nz; ++k) {
                const int kk =
                    klo + static_cast<int>(k);
                const amrex::Real jacobian =
                    detJ(ii, jj, kk);

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

                physical_cell_volume_m3[k] =
                    volume_m3;

                const std::size_t flat =
                    (k * ny + j) * nx + i;
                column_pressure_pa[k] =
                    pressure_pa[flat];
            }

            const std::vector<ERFFireAtmosphericSourceCell>
                column =
                    make_erf_fire_atmospheric_source_column(
                        feedback.cell(i, j),
                        vertical_faces_agl_m,
                        physical_cell_volume_m3,
                        column_pressure_pa,
                        dt_s,
                        options);

            for (std::size_t k = 0; k < nz; ++k) {
                const std::size_t flat =
                    (k * ny + j) * nx + i;
                host_values[2 * flat] =
                    column[k]
                        .rhotheta_tendency_kg_K_m3_s;
                host_values[2 * flat + 1] =
                    column[k]
                        .rhoqv_tendency_kg_m3_s;
            }
        }
    }

    return map_native_source_values_to_multifab(
        host_values,
        environment_inputs.geometry,
        conserved_state_tn);
}


} // namespace ERFFire
