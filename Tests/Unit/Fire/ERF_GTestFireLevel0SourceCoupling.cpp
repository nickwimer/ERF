#include <ERF_FireAtmosphericSource.H>
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
#include <AMReX_Gpu.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_RealBox.H>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
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
        initialize(nonflat_nodal);
    }

    void initialize(bool nonflat_nodal)
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

void
expect_source_components_near(
    const MultiFab& actual,
    const MultiFab& expected)
{
    for (int component :
         {RhoTheta_comp, RhoQ1_comp}) {
        const auto actual_values =
            component_values(
                actual,
                component);
        const auto expected_values =
            component_values(
                expected,
                component);

        ASSERT_EQ(
            actual_values.size(),
            expected_values.size());

        for (std::size_t i = 0;
             i < actual_values.size();
             ++i) {
            const Real tolerance =
                Real(2.0e-11)
                * std::max(
                    Real(1.0),
                    std::abs(expected_values[i]));

            EXPECT_NEAR(
                actual_values[i],
                expected_values[i],
                tolerance);
        }
    }
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
    FlatAdapterMatchesColumnOracle)
{
    CouplingFixture fixture;
    const Real dt_s = Real(2.0);
    const ERFFire::ERFFireAtmosphericSourceOptions options{
        Real(7.0)};
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
            dt_s,
            options);

    ASSERT_NE(source, nullptr);

    const Real qv = Real(0.01);
    const Real rhotheta =
        getRhoThetagivenP(p_0, qv);
    const Real pressure =
        getPgivenRTh(rhotheta, qv);
    const auto expected =
        ERFFire::make_erf_fire_atmospheric_source_column(
            feedback.cell(0, 0),
            std::vector<Real>{
                Real(0), Real(4), Real(12)},
            std::vector<Real>{
                Real(24), Real(48)},
            std::vector<Real>{
                pressure, pressure},
            dt_s,
            options);
    ASSERT_EQ(expected.size(), std::size_t(2));

    const auto theta =
        component_values(
            *source,
            RhoTheta_comp);
    const auto vapor =
        component_values(
            *source,
            RhoQ1_comp);
    ASSERT_EQ(theta.size(), std::size_t(8));
    ASSERT_EQ(vapor.size(), std::size_t(8));

    constexpr std::size_t horizontal_cells =
        std::size_t(CouplingFixture::nx)
        * std::size_t(CouplingFixture::ny);
    for (std::size_t k = 0; k < expected.size(); ++k) {
        const Real expected_theta =
            expected[k].rhotheta_tendency_kg_K_m3_s;
        const Real expected_vapor =
            expected[k].rhoqv_tendency_kg_m3_s;
        const Real theta_tolerance =
            Real(2.0e-13)
            * std::max(
                Real(1),
                std::abs(expected_theta));
        const Real vapor_tolerance =
            Real(2.0e-13)
            * std::max(
                Real(1),
                std::abs(expected_vapor));

        for (std::size_t horizontal = 0;
             horizontal < horizontal_cells;
             ++horizontal) {
            const std::size_t index =
                k * horizontal_cells + horizontal;
            EXPECT_NEAR(
                theta[index],
                expected_theta,
                theta_tolerance);
            EXPECT_NEAR(
                vapor[index],
                expected_vapor,
                vapor_tolerance);
        }
    }
}

TEST(
    FireLevel0SourceCoupling,
    FinerFireFeedbackMatchesOneToOneSourceProjection)
{
    CouplingFixture fixture;
    const Real dt_s = Real(2.0);

    const auto baseline_feedback =
        make_feedback(
            fixture.fire_geometry(),
            dt_s);

    const FireCartesianRasterGeometry2D fine_geometry{
        std::size_t(4),
        std::size_t(4),
        Real(0),
        Real(0),
        Real(1),
        Real(1.5)};

    const auto fine_feedback =
        make_feedback(
            fine_geometry,
            dt_s);

    const auto baseline_source =
        ERFFire::make_erf_fire_level0_source_tendency(
            baseline_feedback,
            fixture.environment_inputs(),
            fixture.conserved,
            MoistureType::MoistNoCondensation,
            dt_s);

    const auto fine_source =
        ERFFire::make_erf_fire_level0_source_tendency(
            fine_feedback,
            fixture.environment_inputs(),
            fixture.conserved,
            MoistureType::MoistNoCondensation,
            dt_s);

    ASSERT_NE(baseline_source, nullptr);
    ASSERT_NE(fine_source, nullptr);

    expect_source_components_near(
        *fine_source,
        *baseline_source);
}

TEST(
    FireLevel0SourceCoupling,
    CoarserFireFeedbackMatchesOneToOneSourceProjection)
{
    CouplingFixture fixture;
    const Real dt_s = Real(2.0);

    const auto baseline_feedback =
        make_feedback(
            fixture.fire_geometry(),
            dt_s);

    const FireCartesianRasterGeometry2D coarse_geometry{
        std::size_t(1),
        std::size_t(1),
        Real(0),
        Real(0),
        Real(4),
        Real(6)};

    const auto coarse_feedback =
        make_feedback(
            coarse_geometry,
            dt_s);

    const auto baseline_source =
        ERFFire::make_erf_fire_level0_source_tendency(
            baseline_feedback,
            fixture.environment_inputs(),
            fixture.conserved,
            MoistureType::MoistNoCondensation,
            dt_s);

    const auto coarse_source =
        ERFFire::make_erf_fire_level0_source_tendency(
            coarse_feedback,
            fixture.environment_inputs(),
            fixture.conserved,
            MoistureType::MoistNoCondensation,
            dt_s);

    ASSERT_NE(baseline_source, nullptr);
    ASSERT_NE(coarse_source, nullptr);

    expect_source_components_near(
        *coarse_source,
        *baseline_source);
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


namespace
{

constexpr int terrain_nx = 2;
constexpr int terrain_ny = 1;
constexpr int terrain_nz = 3;
constexpr Real terrain_top_m = Real(12.0);
constexpr Real terrain_dx_m = Real(2.0);
constexpr Real terrain_dy_m = Real(3.0);
constexpr Real terrain_dz_computational_m = Real(4.0);

Real
terrain_eta_face(int k)
{
    if (k == 0) {
        return Real(0.0);
    }
    if (k == 1) {
        return Real(0.20);
    }
    if (k == 2) {
        return Real(0.55);
    }
    if (k == 3) {
        return Real(1.0);
    }
    throw std::out_of_range(
        "terrain test eta face index");
}

Real
terrain_cell_ground_m(int i, int j = 0)
{
    return Real(1.0)
        + Real(0.5) * (Real(i) + Real(0.5))
        + Real(0.25) * (Real(j) + Real(0.5));
}

Real
terrain_agl_face_m(int i, int k)
{
    return terrain_eta_face(k)
        * (terrain_top_m - terrain_cell_ground_m(i));
}

Real
terrain_detj(int i, int k)
{
    return (
        terrain_agl_face_m(i, k + 1)
        - terrain_agl_face_m(i, k))
        / terrain_dz_computational_m;
}

Real
terrain_pressure_pa(int i, int k)
{
    return p_0
        * (Real(1.0)
           - Real(0.03) * Real(i)
           - Real(0.04) * Real(k));
}

Real
terrain_layer_weight(
    int i,
    int k,
    Real extinction_depth_m)
{
    const Real zlo =
        terrain_agl_face_m(i, k);
    const Real zhi =
        terrain_agl_face_m(i, k + 1);
    const Real ztop =
        terrain_agl_face_m(i, terrain_nz);

    return (
        std::exp(-zlo / extinction_depth_m)
        - std::exp(-zhi / extinction_depth_m))
        / (Real(1.0)
           - std::exp(-ztop / extinction_depth_m));
}

std::size_t
terrain_flat_index(int i, int k)
{
    return (
        static_cast<std::size_t>(k)
        * static_cast<std::size_t>(terrain_ny)
        * static_cast<std::size_t>(terrain_nx))
        + static_cast<std::size_t>(i);
}

Real
scaled_test_tolerance(Real value)
{
    return Real(4096)
        * std::numeric_limits<Real>::epsilon()
        * std::max(Real(1.0), std::abs(value));
}

struct TerrainCouplingFixture
{
    Box domain{
        IntVect(0, 0, 0),
        IntVect(
            terrain_nx - 1,
            terrain_ny - 1,
            terrain_nz - 1)};
    RealBox real_box{
        Real(0), Real(0), Real(0),
        Real(terrain_nx) * terrain_dx_m,
        Real(terrain_ny) * terrain_dy_m,
        Real(terrain_nz)
            * terrain_dz_computational_m};
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
    MultiFab detJ_cc{
        cell_ba, dm, 1, 0};
    MultiFab conserved{
        cell_ba,
        dm,
        RhoQ2_comp + 1,
        0};

    TerrainCouplingFixture()
    {
        initialize();
    }

    void initialize()
    {
        x_velocity.setVal(Real(1.0));
        y_velocity.setVal(Real(0.0));
        z_phys_cc.setVal(Real(-12345.0));
        conserved.setVal(Real(0.0));

        for (MFIter mfi(z_phys_nd); mfi.isValid(); ++mfi) {
            const Box box =
                mfi.tilebox();
            const auto array =
                z_phys_nd.array(mfi);

            amrex::ParallelFor(
                box,
                [=] AMREX_GPU_DEVICE (
                    int i, int j, int k) noexcept {
                    const Real eta =
                        k == 0
                        ? Real(0.0)
                        : (k == 1
                           ? Real(0.20)
                           : (k == 2
                              ? Real(0.55)
                              : Real(1.0)));
                    const Real ground =
                        Real(1.0)
                        + Real(0.5) * Real(i)
                        + Real(0.25) * Real(j);

                    array(i, j, k) =
                        ground
                        + eta
                            * (terrain_top_m - ground);
                });
        }

        for (MFIter mfi(detJ_cc); mfi.isValid(); ++mfi) {
            const Box box =
                mfi.tilebox();
            const auto array =
                detJ_cc.array(mfi);

            amrex::ParallelFor(
                box,
                [=] AMREX_GPU_DEVICE (
                    int i, int j, int k) noexcept {
                    const Real eta_lo =
                        k == 0
                        ? Real(0.0)
                        : (k == 1
                           ? Real(0.20)
                           : Real(0.55));
                    const Real eta_hi =
                        k == 0
                        ? Real(0.20)
                        : (k == 1
                           ? Real(0.55)
                           : Real(1.0));
                    const Real ground =
                        Real(1.0)
                        + Real(0.5)
                            * (Real(i) + Real(0.5))
                        + Real(0.25)
                            * (Real(j) + Real(0.5));

                    array(i, j, k) =
                        (eta_hi - eta_lo)
                        * (terrain_top_m - ground)
                        / terrain_dz_computational_m;
                });
        }

        const Real qv = Real(0.01);
        const Real rho = Real(1.0);

        for (MFIter mfi(conserved); mfi.isValid(); ++mfi) {
            const Box box =
                mfi.tilebox();
            const auto array =
                conserved.array(mfi);

            amrex::ParallelFor(
                box,
                [=] AMREX_GPU_DEVICE (
                    int i, int j, int k) noexcept {
                    const Real pressure =
                        p_0
                        * (Real(1.0)
                           - Real(0.03) * Real(i)
                           - Real(0.04) * Real(k));

                    array(i, j, k, Rho_comp) =
                        rho;
                    array(i, j, k, RhoTheta_comp) =
                        getRhoThetagivenP(
                            pressure,
                            qv);
                    array(i, j, k, RhoQ1_comp) =
                        rho * qv;
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
            MeshType::VariableDz,
            TerrainType::StaticFittedMesh,
            BuildingsType::None,
            0};
    }

    FireCartesianRasterGeometry2D
    fire_geometry() const
    {
        return {
            std::size_t(terrain_nx),
            std::size_t(terrain_ny),
            Real(0),
            Real(0),
            terrain_dx_m,
            terrain_dy_m};
    }

    void
    double_first_detj_cell()
    {
        for (MFIter mfi(detJ_cc); mfi.isValid(); ++mfi) {
            const Box box =
                mfi.tilebox();
            const auto array =
                detJ_cc.array(mfi);

            amrex::ParallelFor(
                box,
                [=] AMREX_GPU_DEVICE (
                    int i, int j, int k) noexcept {
                    if (i == 0 && j == 0 && k == 0) {
                        array(i, j, k) *= Real(2.0);
                    }
                });
        }
        amrex::Gpu::streamSynchronize();
    }

    void
    fold_second_vertical_face()
    {
        for (MFIter mfi(z_phys_nd); mfi.isValid(); ++mfi) {
            const Box box =
                mfi.tilebox();
            const auto array =
                z_phys_nd.array(mfi);

            amrex::ParallelFor(
                box,
                [=] AMREX_GPU_DEVICE (
                    int i, int j, int k) noexcept {
                    if (k == 2) {
                        array(i, j, k) =
                            array(i, j, 1)
                            - Real(0.5);
                    }
                });
        }
        amrex::Gpu::streamSynchronize();
    }
};

} // namespace


TEST(
    FireLevel0TerrainSourceCoupling,
    NodalAGLAndDetJMatchIndependentCellOracle)
{
    TerrainCouplingFixture fixture;
    const Real dt_s = Real(2.0);
    const Real extinction_depth_m = Real(5.0);
    const auto feedback =
        make_feedback(
            fixture.fire_geometry(),
            dt_s);

    const auto source =
        ERFFire::make_erf_fire_level0_terrain_source_tendency(
            feedback,
            fixture.environment_inputs(),
            fixture.detJ_cc,
            fixture.conserved,
            MoistureType::MoistNoCondensation,
            dt_s,
            ERFFire::ERFFireAtmosphericSourceOptions{
                extinction_depth_m});

    const auto theta =
        component_values(
            *source,
            RhoTheta_comp);
    const auto qv =
        component_values(
            *source,
            RhoQ1_comp);

    for (int k = 0; k < terrain_nz; ++k) {
        for (int i = 0; i < terrain_nx; ++i) {
            const std::size_t flat =
                terrain_flat_index(i, k);
            const auto release =
                feedback.cell(
                    static_cast<std::size_t>(i),
                    0);

            const Real weight =
                terrain_layer_weight(
                    i,
                    k,
                    extinction_depth_m);
            const Real volume_m3 =
                terrain_dx_m
                * terrain_dy_m
                * terrain_dz_computational_m
                * terrain_detj(i, k);
            const Real pressure =
                terrain_pressure_pa(i, k);
            const Real exner =
                std::pow(
                    pressure * ip_0,
                    RdoCp);

            const Real expected_qv =
                release.water_released_kg
                * weight
                / (volume_m3 * dt_s);
            const Real expected_theta =
                release.sensible_energy_j
                * weight
                / (volume_m3
                   * dt_s
                   * Cp_d
                   * exner);

            EXPECT_NEAR(
                qv[flat],
                expected_qv,
                scaled_test_tolerance(expected_qv));
            EXPECT_NEAR(
                theta[flat],
                expected_theta,
                scaled_test_tolerance(expected_theta));
        }
    }

    EXPECT_NE(
        terrain_layer_weight(
            0, 0, extinction_depth_m),
        terrain_layer_weight(
            1, 0, extinction_depth_m));
}

TEST(
    FireLevel0TerrainSourceCoupling,
    ConservesHeatAndWaterInEachTerrainColumn)
{
    TerrainCouplingFixture fixture;
    const Real dt_s = Real(3.0);
    const Real extinction_depth_m = Real(7.0);
    const auto feedback =
        make_feedback(
            fixture.fire_geometry(),
            dt_s);

    const auto source =
        ERFFire::make_erf_fire_level0_terrain_source_tendency(
            feedback,
            fixture.environment_inputs(),
            fixture.detJ_cc,
            fixture.conserved,
            MoistureType::MoistNoCondensation,
            dt_s,
            ERFFire::ERFFireAtmosphericSourceOptions{
                extinction_depth_m});

    const auto theta =
        component_values(
            *source,
            RhoTheta_comp);
    const auto qv =
        component_values(
            *source,
            RhoQ1_comp);

    for (int i = 0; i < terrain_nx; ++i) {
        Real recovered_energy_j = Real(0.0);
        Real recovered_water_kg = Real(0.0);

        for (int k = 0; k < terrain_nz; ++k) {
            const std::size_t flat =
                terrain_flat_index(i, k);
            const Real volume_m3 =
                terrain_dx_m
                * terrain_dy_m
                * terrain_dz_computational_m
                * terrain_detj(i, k);
            const Real exner =
                std::pow(
                    terrain_pressure_pa(i, k)
                        * ip_0,
                    RdoCp);

            recovered_energy_j +=
                theta[flat]
                * Cp_d
                * exner
                * volume_m3
                * dt_s;
            recovered_water_kg +=
                qv[flat]
                * volume_m3
                * dt_s;
        }

        const auto release =
            feedback.cell(
                static_cast<std::size_t>(i),
                0);

        EXPECT_NEAR(
            recovered_energy_j,
            release.sensible_energy_j,
            scaled_test_tolerance(
                release.sensible_energy_j));
        EXPECT_NEAR(
            recovered_water_kg,
            release.water_released_kg,
            scaled_test_tolerance(
                release.water_released_kg));
    }
}

TEST(
    FireLevel0TerrainSourceCoupling,
    DetJIsAuthoritativeForPhysicalVolume)
{
    TerrainCouplingFixture fixture;
    const Real dt_s = Real(2.0);
    const auto feedback =
        make_feedback(
            fixture.fire_geometry(),
            dt_s);

    const auto baseline =
        ERFFire::make_erf_fire_level0_terrain_source_tendency(
            feedback,
            fixture.environment_inputs(),
            fixture.detJ_cc,
            fixture.conserved,
            MoistureType::MoistNoCondensation,
            dt_s);

    const auto baseline_theta =
        component_values(
            *baseline,
            RhoTheta_comp);
    const auto baseline_qv =
        component_values(
            *baseline,
            RhoQ1_comp);

    fixture.double_first_detj_cell();

    const auto doubled_volume =
        ERFFire::make_erf_fire_level0_terrain_source_tendency(
            feedback,
            fixture.environment_inputs(),
            fixture.detJ_cc,
            fixture.conserved,
            MoistureType::MoistNoCondensation,
            dt_s);

    const auto doubled_theta =
        component_values(
            *doubled_volume,
            RhoTheta_comp);
    const auto doubled_qv =
        component_values(
            *doubled_volume,
            RhoQ1_comp);

    const std::size_t changed =
        terrain_flat_index(0, 0);

    EXPECT_NEAR(
        doubled_theta[changed],
        Real(0.5) * baseline_theta[changed],
        scaled_test_tolerance(
            baseline_theta[changed]));
    EXPECT_NEAR(
        doubled_qv[changed],
        Real(0.5) * baseline_qv[changed],
        scaled_test_tolerance(
            baseline_qv[changed]));

    for (std::size_t n = 0;
         n < baseline_qv.size();
         ++n) {
        if (n == changed) {
            continue;
        }
        EXPECT_EQ(
            doubled_theta[n],
            baseline_theta[n]);
        EXPECT_EQ(
            doubled_qv[n],
            baseline_qv[n]);
    }
}

TEST(
    FireLevel0TerrainSourceCoupling,
    LeavesDryDensityAndOtherComponentsExactlyZero)
{
    TerrainCouplingFixture fixture;
    const Real dt_s = Real(1.0);
    const auto feedback =
        make_feedback(
            fixture.fire_geometry(),
            dt_s);

    const auto source =
        ERFFire::make_erf_fire_level0_terrain_source_tendency(
            feedback,
            fixture.environment_inputs(),
            fixture.detJ_cc,
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
            EXPECT_EQ(
                value,
                Real(0.0));
        }
    }
}

TEST(
    FireLevel0TerrainSourceCoupling,
    RejectsUnsupportedScopeInvalidMetricAndFoldedColumn)
{
    TerrainCouplingFixture fixture;
    const auto feedback =
        make_feedback(
            fixture.fire_geometry(),
            Real(1.0));

    auto wrong_mesh =
        fixture.environment_inputs();
    wrong_mesh.mesh_type =
        MeshType::StretchedDz;

    EXPECT_THROW(
        (void)ERFFire::make_erf_fire_level0_terrain_source_tendency(
            feedback,
            wrong_mesh,
            fixture.detJ_cc,
            fixture.conserved,
            MoistureType::MoistNoCondensation,
            Real(1.0)),
        std::invalid_argument);

    fixture.detJ_cc.setVal(Real(0.0));
    EXPECT_THROW(
        (void)ERFFire::make_erf_fire_level0_terrain_source_tendency(
            feedback,
            fixture.environment_inputs(),
            fixture.detJ_cc,
            fixture.conserved,
            MoistureType::MoistNoCondensation,
            Real(1.0)),
        std::invalid_argument);

    TerrainCouplingFixture folded;
    folded.fold_second_vertical_face();

    EXPECT_THROW(
        (void)ERFFire::make_erf_fire_level0_terrain_source_tendency(
            feedback,
            folded.environment_inputs(),
            folded.detJ_cc,
            folded.conserved,
            MoistureType::MoistNoCondensation,
            Real(1.0)),
        std::invalid_argument);
}
