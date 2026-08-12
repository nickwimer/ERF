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
    require(
        conserved_state_tn.boxArray().ixType()
            == domain.ixType()
        && amrex::match(
            conserved_state_tn.boxArray(),
            amrex::BoxArray(domain)),
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

std::unique_ptr<amrex::MultiFab>
map_source_field_to_multifab(
    const ERFFireAtmosphericSourceField& source,
    const amrex::Geometry& geometry,
    const amrex::MultiFab& conserved_state_tn)
{
    const amrex::Box& domain = geometry.Domain();
    const std::size_t nx =
        static_cast<std::size_t>(domain.length(0));
    const std::size_t ny =
        static_cast<std::size_t>(domain.length(1));
    const std::size_t nz =
        static_cast<std::size_t>(domain.length(2));

    require(
        source.nx() == nx
            && source.ny() == ny
            && source.nz() == nz,
        "ERF Fire source field dimensions do not match level 0");

    auto result =
        std::make_unique<amrex::MultiFab>(
            conserved_state_tn.boxArray(),
            conserved_state_tn.DistributionMap(),
            conserved_state_tn.nComp(),
            0);
    result->setVal(amrex::Real(0));

    std::vector<amrex::Real> host_values(
        source.cell_count() * std::size_t(2),
        amrex::Real(0));

    for (std::size_t k = 0; k < nz; ++k) {
        for (std::size_t j = 0; j < ny; ++j) {
            for (std::size_t i = 0; i < nx; ++i) {
                const std::size_t flat =
                    (k * ny + j) * nx + i;
                const auto& cell = source.cell(i, j, k);
                host_values[2 * flat] =
                    cell.rhotheta_tendency_kg_K_m3_s;
                host_values[2 * flat + 1] =
                    cell.rhoqv_tendency_kg_m3_s;
            }
        }
    }

    amrex::Gpu::DeviceVector<amrex::Real>
        device_values(host_values.size());
    amrex::Gpu::copy(
        amrex::Gpu::hostToDevice,
        host_values.begin(),
        host_values.end(),
        device_values.begin());

    const amrex::Real* values =
        device_values.data();
    const int ilo = domain.smallEnd(0);
    const int jlo = domain.smallEnd(1);
    const int klo = domain.smallEnd(2);
    const std::size_t nx_device = nx;
    const std::size_t ny_device = ny;

    for (amrex::MFIter mfi(
             *result,
             amrex::TilingIfNotGPU());
         mfi.isValid();
         ++mfi) {
        const amrex::Box box = mfi.tilebox();
        const auto array = result->array(mfi);

        amrex::ParallelFor(
            box,
            [=] AMREX_GPU_DEVICE (
                int i, int j, int k) noexcept {
                const std::size_t ii =
                    static_cast<std::size_t>(i - ilo);
                const std::size_t jj =
                    static_cast<std::size_t>(j - jlo);
                const std::size_t kk =
                    static_cast<std::size_t>(k - klo);
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

} // namespace ERFFire
