#include <ERF_FireLevel0SourceCoupling.H>

#include <ERF_EOS.H>
#include <ERF_FireBurnedFractionRaster.H>
#include <ERF_FireCombustionRaster.H>
#include <ERF_FireSurfaceFeedback.H>
#include <ERF_IndexDefines.H>

#include <gtest/gtest.h>

#include <AMReX_Array.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Arena.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_RealBox.H>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{

using amrex::Array;
using amrex::Box;
using amrex::BoxArray;
using amrex::DistributionMapping;
using amrex::Geometry;
using amrex::IntVect;
using amrex::MFIter;
using amrex::MultiFab;
using amrex::Real;
using amrex::RealBox;
using ERFFire::ERFFireLevel0EnvironmentInputs;
using ERFFire::FireBurnedFractionRaster;
using ERFFire::FireCartesianRasterGeometry2D;
using ERFFire::FireCombustionParameters;
using ERFFire::FireCombustionRaster;
using ERFFire::FireCombustionRasterOptions;
using ERFFire::FirePerimeter;
using ERFFire::FireSurfaceFeedbackRaster;
using ERFFire::FireVec2;

FirePerimeter
rectangle(
    Real xlo,
    Real xhi,
    Real ylo,
    Real yhi)
{
    std::vector<FireVec2> vertices{
        {xlo, ylo},
        {xhi, ylo},
        {xhi, yhi},
        {xlo, yhi}
    };
    return FirePerimeter(std::move(vertices));
}

FireSurfaceFeedbackRaster
make_feedback(
    const FireCartesianRasterGeometry2D& geometry,
    Real dt_s)
{
    const FireCombustionParameters parameters{
        Real(2.0),
        Real(10.0),
        Real(0.25),
        Real(4.0),
        Real(0.5)};

    const FirePerimeter full =
        rectangle(
            geometry.xlo_m - geometry.dx_m,
            geometry.xlo_m
                + Real(geometry.nx) * geometry.dx_m,
            geometry.ylo_m,
            geometry.ylo_m
                + Real(geometry.ny) * geometry.dy_m);

    FireBurnedFractionRaster burned(geometry);
    (void)burned.update_from_perimeter(full);

    FireCombustionRaster before(
        geometry,
        parameters,
        FireCombustionRasterOptions{8});
    (void)before.initialize_from_burned_fraction(
        burned);

    FireCombustionRaster after = before;
    (void)after.advance_from_linear_sweep(
        full,
        full,
        burned,
        burned,
        dt_s);

    return ERFFire::make_fire_surface_feedback_increment(
        before,
        after);
}

struct CouplingFixture
{
    static constexpr int nx = 2;
    static constexpr int ny = 2;
    static constexpr int nz = 2;

    Box domain{
        IntVect(0, 0, 0),
        IntVect(nx - 1, ny - 1, nz - 1)};
    RealBox real_box{
        Real(0), Real(0), Real(100),
        Real(4), Real(6), Real(112)};
    Array<int, AMREX_SPACEDIM> periodic{{0, 0, 0}};
    Geometry geometry{
        domain,
        real_box,
        amrex::CoordSys::cartesian,
        periodic};
    BoxArray cell_ba{domain};
    DistributionMapping dm{cell_ba};

    MultiFab x_velocity{
        amrex::convert(
            cell_ba,
            IntVect(1, 0, 0)),
        dm, 1, 1};
    MultiFab y_velocity{
        amrex::convert(
            cell_ba,
            IntVect(0, 1, 0)),
        dm, 1, 1};
    MultiFab z_phys_cc{
        cell_ba, dm, 1, 0};
    MultiFab z_phys_nd{
        amrex::convert(
            cell_ba,
            IntVect(1, 1, 1)),
        dm, 1, 0};
    MultiFab conserved{
        cell_ba,
        dm,
        RhoQ2_comp + 1,
        0};

    explicit CouplingFixture(
        bool nonflat_nodal = false)
    {
        x_velocity.setVal(Real(1.0));
        y_velocity.setVal(Real(0.0));
        conserved.setVal(Real(0.0));

        const Real qv = Real(0.01);
        const Real rho = Real(1.0);
        const Real rhotheta =
            getRhoThetagivenP(p_0, qv);

        for (MFIter mfi(conserved); mfi.isValid(); ++mfi) {
            const Box box = mfi.tilebox();
            const auto array = conserved.array(mfi);
            amrex::ParallelFor(
                box,
                [=] AMREX_GPU_DEVICE (
                    int i, int j, int k) noexcept {
                    array(i, j, k, Rho_comp) = rho;
                    array(i, j, k, RhoTheta_comp) =
                        rhotheta;
                    array(i, j, k, RhoQ1_comp) =
                        rho * qv;
                });
        }

        for (MFIter mfi(z_phys_cc); mfi.isValid(); ++mfi) {
            const Box box = mfi.tilebox();
            const auto array = z_phys_cc.array(mfi);
            amrex::ParallelFor(
                box,
                [=] AMREX_GPU_DEVICE (
                    int i, int j, int k) noexcept {
                    const Real z =
                        k == 0
                        ? Real(102)
                        : Real(108);
                    array(i, j, k) = z;
                });
        }

        for (MFIter mfi(z_phys_nd); mfi.isValid(); ++mfi) {
            const Box box = mfi.tilebox();
            const auto array = z_phys_nd.array(mfi);
            amrex::ParallelFor(
                box,
                [=] AMREX_GPU_DEVICE (
                    int i, int j, int k) noexcept {
                    Real z =
                        k == 0
                        ? Real(100)
                        : (k == 1
                           ? Real(104)
                           : Real(112));
                    if (nonflat_nodal
                        && i == 1
                        && j == 1
                        && k == 1) {
                        z += Real(0.25);
                    }
                    array(i, j, k) = z;
                });
        }

        amrex::Gpu::streamSynchronize();
    }

    ERFFireLevel0EnvironmentInputs
    environment_inputs() const
    {
        return {
            geometry,
            x_velocity,
            y_velocity,
            z_phys_cc,
            z_phys_nd,
            MeshType::StretchedDz,
            TerrainType::StaticFittedMesh,
            BuildingsType::None,
            0};
    }

    FireCartesianRasterGeometry2D
    fire_geometry() const
    {
        return {
            std::size_t(nx),
            std::size_t(ny),
            Real(0),
            Real(0),
            Real(2),
            Real(3)};
    }
};

std::vector<Real>
component_values(
    const MultiFab& mf,
    int component)
{
    const Box domain = mf.boxArray().minimalBox();
    amrex::FArrayBox host(
        domain,
        1,
        amrex::The_Pinned_Arena());
    mf.copyTo(
        host,
        component,
        0,
        1,
        0);
    const auto array = host.const_array();

    std::vector<Real> result;
    for (int k = domain.smallEnd(2);
         k <= domain.bigEnd(2);
         ++k) {
        for (int j = domain.smallEnd(1);
             j <= domain.bigEnd(1);
             ++j) {
            for (int i = domain.smallEnd(0);
                 i <= domain.bigEnd(0);
                 ++i) {
                result.push_back(
                    array(i, j, k));
            }
        }
    }
    return result;
}

} // namespace

TEST(
    FireLevel0SourceCoupling,
    BuildsPositiveNativeThetaAndVaporTendencies)
{
    CouplingFixture fixture;
    const Real dt_s = Real(2.0);
    const auto feedback =
        make_feedback(
            fixture.fire_geometry(),
            dt_s);

    const auto source =
        ERFFire::make_erf_fire_level0_source_tendency(
            feedback,
            fixture.environment_inputs(),
            fixture.conserved,
            MoistureType::MoistNoCondensation,
            dt_s);

    ASSERT_NE(source, nullptr);

    const auto theta =
        component_values(
            *source,
            RhoTheta_comp);
    const auto qv =
        component_values(
            *source,
            RhoQ1_comp);

    for (Real value : theta) {
        EXPECT_GT(value, Real(0.0));
    }
    for (Real value : qv) {
        EXPECT_GT(value, Real(0.0));
    }
}

TEST(
    FireLevel0SourceCoupling,
    LeavesDryDensityAndAllOtherComponentsExactlyZero)
{
    CouplingFixture fixture;
    const Real dt_s = Real(1.0);
    const auto feedback =
        make_feedback(
            fixture.fire_geometry(),
            dt_s);

    const auto source =
        ERFFire::make_erf_fire_level0_source_tendency(
            feedback,
            fixture.environment_inputs(),
            fixture.conserved,
            MoistureType::MoistNoCondensation,
            dt_s);

    for (int component = 0;
         component < source->nComp();
         ++component) {
        if (component == RhoTheta_comp
            || component == RhoQ1_comp) {
            continue;
        }
        for (Real value
             : component_values(
                   *source,
                   component)) {
            EXPECT_EQ(value, Real(0.0));
        }
    }
}

TEST(
    FireLevel0SourceCoupling,
    RejectsDryAtmosphereForWaterFeedback)
{
    CouplingFixture fixture;
    const auto feedback =
        make_feedback(
            fixture.fire_geometry(),
            Real(1.0));

    EXPECT_THROW(
        (void)ERFFire::make_erf_fire_level0_source_tendency(
            feedback,
            fixture.environment_inputs(),
            fixture.conserved,
            MoistureType::None,
            Real(1.0)),
        std::invalid_argument);
}

TEST(
    FireLevel0SourceCoupling,
    RejectsHorizontalGeometryMismatchAndNonFlatColumn)
{
    CouplingFixture fixture;
    auto mismatched_geometry =
        fixture.fire_geometry();
    mismatched_geometry.dx_m += Real(0.5);

    const auto mismatched_feedback =
        make_feedback(
            mismatched_geometry,
            Real(1.0));

    EXPECT_THROW(
        (void)ERFFire::make_erf_fire_level0_source_tendency(
            mismatched_feedback,
            fixture.environment_inputs(),
            fixture.conserved,
            MoistureType::MoistNoCondensation,
            Real(1.0)),
        std::invalid_argument);

    CouplingFixture nonflat(true);
    const auto valid_feedback =
        make_feedback(
            nonflat.fire_geometry(),
            Real(1.0));

    EXPECT_THROW(
        (void)ERFFire::make_erf_fire_level0_source_tendency(
            valid_feedback,
            nonflat.environment_inputs(),
            nonflat.conserved,
            MoistureType::MoistNoCondensation,
            Real(1.0)),
        std::invalid_argument);
}
