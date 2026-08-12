#include <ERF_FireLevel0Environment.H>
#include <ERF_FireTerrainSurface.H>

#include <gtest/gtest.h>

#include <AMReX_Array.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_RealBox.H>

#include <cmath>
#include <limits>
#include <stdexcept>
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
using ERFFire::FireCartesianRasterGeometry2D;
using ERFFire::FireTerrainSurface;

Real
scaled_tolerance(Real expected)
{
    return Real(512)
        * std::numeric_limits<Real>::epsilon()
        * std::max(Real(1), std::abs(expected));
}

void
expect_near_real(Real actual, Real expected)
{
    EXPECT_NEAR(
        actual,
        expected,
        scaled_tolerance(expected));
}

Real
plane_height(
    Real x,
    Real y)
{
    return Real(90)
        + Real(0.125) * x
        - Real(0.2) * y;
}

struct TerrainFixture
{
    static constexpr int nx = 3;
    static constexpr int ny = 2;
    static constexpr int nz = 2;

    static constexpr Real xlo = Real(10);
    static constexpr Real ylo = Real(-4);
    static constexpr Real dx = Real(2);
    static constexpr Real dy = Real(3);

    Box domain{
        IntVect(0, 0, 0),
        IntVect(nx - 1, ny - 1, nz - 1)};
    RealBox real_box{
        xlo, ylo, Real(0),
        xlo + Real(nx) * dx,
        ylo + Real(ny) * dy,
        Real(nz)};
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

    TerrainFixture()
    {
        x_velocity.setVal(Real(0));
        y_velocity.setVal(Real(0));

        for (MFIter mfi(z_phys_nd); mfi.isValid(); ++mfi) {
            const Box box = mfi.tilebox();
            const auto z = z_phys_nd.array(mfi);
            amrex::ParallelFor(
                box,
                [=] AMREX_GPU_DEVICE (
                    int i, int j, int k) noexcept {
                    const Real x =
                        xlo + Real(i) * dx;
                    const Real y =
                        ylo + Real(j) * dy;
                    const Real ground =
                        Real(90)
                        + Real(0.125) * x
                        - Real(0.2) * y;
                    z(i, j, k) =
                        ground
                        + Real(k) * Real(20);
                });
        }

        for (MFIter mfi(z_phys_cc); mfi.isValid(); ++mfi) {
            const Box box = mfi.tilebox();
            const auto z = z_phys_cc.array(mfi);
            amrex::ParallelFor(
                box,
                [=] AMREX_GPU_DEVICE (
                    int i, int j, int k) noexcept {
                    const Real x =
                        xlo + (Real(i) + Real(0.5)) * dx;
                    const Real y =
                        ylo + (Real(j) + Real(0.5)) * dy;
                    const Real ground =
                        Real(90)
                        + Real(0.125) * x
                        - Real(0.2) * y;
                    z(i, j, k) =
                        ground
                        + (Real(k) + Real(0.5))
                            * Real(20);
                });
        }

        amrex::Gpu::streamSynchronize();
    }

    ERFFireLevel0EnvironmentInputs
    inputs() const
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
};

} // namespace

TEST(
    FireTerrainSurface,
    PlanarSurfaceMatchesIndependentHeightAndGradientOracle)
{
    const FireCartesianRasterGeometry2D geometry{
        4, 3,
        Real(10), Real(-4),
        Real(2), Real(3)};

    std::vector<Real> nodal;
    nodal.reserve((geometry.nx + 1) * (geometry.ny + 1));

    for (std::size_t j = 0; j <= geometry.ny; ++j) {
        for (std::size_t i = 0; i <= geometry.nx; ++i) {
            const Real x =
                geometry.xlo_m + Real(i) * geometry.dx_m;
            const Real y =
                geometry.ylo_m + Real(j) * geometry.dy_m;
            nodal.push_back(plane_height(x, y));
        }
    }

    const FireTerrainSurface surface(
        geometry,
        nodal);

    for (const auto point
         : std::vector<ERFFire::FireVec2>{
             {Real(10), Real(-4)},
             {Real(13.25), Real(-0.7)},
             {Real(18), Real(5)}}) {
        expect_near_real(
            surface.ground_height_m(point.x, point.y),
            plane_height(point.x, point.y));

        const auto gradient =
            surface.terrain_gradient_m_per_m(
                point.x, point.y);
        expect_near_real(
            gradient.x,
            Real(0.125));
        expect_near_real(
            gradient.y,
            Real(-0.2));
    }
}

TEST(
    FireTerrainSurface,
    BilinearPatchMatchesIndependentAnalyticDerivatives)
{
    const FireCartesianRasterGeometry2D geometry{
        1, 1,
        Real(0), Real(0),
        Real(2), Real(4)};

    const FireTerrainSurface surface(
        geometry,
        std::vector<Real>{
            Real(10), Real(14),
            Real(18), Real(30)});

    const Real alpha = Real(0.25);
    const Real beta = Real(0.75);
    const Real x = alpha * geometry.dx_m;
    const Real y = beta * geometry.dy_m;

    const Real expected_height =
        (Real(1) - alpha) * (Real(1) - beta) * Real(10)
        + alpha * (Real(1) - beta) * Real(14)
        + (Real(1) - alpha) * beta * Real(18)
        + alpha * beta * Real(30);

    const Real expected_dhdx =
        ((Real(1) - beta) * (Real(14) - Real(10))
         + beta * (Real(30) - Real(18)))
        / geometry.dx_m;
    const Real expected_dhdy =
        ((Real(1) - alpha) * (Real(18) - Real(10))
         + alpha * (Real(30) - Real(14)))
        / geometry.dy_m;

    expect_near_real(
        surface.ground_height_m(x, y),
        expected_height);

    const auto gradient =
        surface.terrain_gradient_m_per_m(x, y);
    expect_near_real(gradient.x, expected_dhdx);
    expect_near_real(gradient.y, expected_dhdy);
}

TEST(
    FireTerrainSurface,
    InternalRepresentedBoundaryUsesPositiveIndexCell)
{
    const FireCartesianRasterGeometry2D geometry{
        3, 1,
        Real(0.3), Real(0),
        Real(0.2), Real(1)};

    // Cells 0 and 1 are flat. Cell 2 rises by two meters, so the
    // represented internal boundary between cells 1 and 2 has a
    // deliberately discontinuous one-sided dh/dx. This independently
    // exposes which containing-cell convention is used.
    const FireTerrainSurface surface(
        geometry,
        std::vector<Real>{
            Real(0), Real(0), Real(0), Real(2),
            Real(0), Real(0), Real(0), Real(2)});

    const Real boundary_x =
        geometry.xlo_m
        + Real(2) * geometry.dx_m;

    const auto gradient =
        surface.terrain_gradient_m_per_m(
            boundary_x,
            Real(0.5));

    EXPECT_EQ(
        surface.ground_height_m(
            boundary_x,
            Real(0.5)),
        Real(0));
    expect_near_real(
        gradient.x,
        Real(2) / geometry.dx_m);
    EXPECT_EQ(gradient.y, Real(0));
}

TEST(
    FireTerrainSurface,
    FlatSurfaceHasExactZeroGradient)
{
    const FireCartesianRasterGeometry2D geometry{
        2, 2,
        Real(0), Real(0),
        Real(1), Real(1)};

    const FireTerrainSurface surface(
        geometry,
        std::vector<Real>(9, Real(123.5)));

    const auto gradient =
        surface.terrain_gradient_m_per_m(
            Real(0.7), Real(1.2));

    EXPECT_EQ(gradient.x, Real(0));
    EXPECT_EQ(gradient.y, Real(0));
    EXPECT_EQ(
        surface.ground_height_m(
            Real(0.7), Real(1.2)),
        Real(123.5));
}

TEST(
    FireTerrainSurface,
    RejectsInvalidConstructionAndOutOfDomainSampling)
{
    const FireCartesianRasterGeometry2D geometry{
        2, 1,
        Real(0), Real(0),
        Real(1), Real(1)};

    EXPECT_THROW(
        (void)FireTerrainSurface(
            geometry,
            std::vector<Real>(5, Real(0))),
        std::invalid_argument);

    std::vector<Real> nonfinite(6, Real(0));
    nonfinite[2] =
        std::numeric_limits<Real>::quiet_NaN();

    EXPECT_THROW(
        (void)FireTerrainSurface(
            geometry,
            nonfinite),
        std::invalid_argument);

    const FireTerrainSurface surface(
        geometry,
        std::vector<Real>(6, Real(0)));

    EXPECT_FALSE(
        surface.contains_physical_point(
            Real(-0.01), Real(0.5)));
    EXPECT_THROW(
        (void)surface.ground_height_m(
            Real(-0.01), Real(0.5)),
        std::out_of_range);
    EXPECT_THROW(
        (void)surface.terrain_gradient_m_per_m(
            Real(2.01), Real(0.5)),
        std::out_of_range);
}

TEST(
    FireTerrainSurface,
    Level0VariableDzExtractionMatchesIndependentPlanarOracle)
{
    TerrainFixture fixture;

    const FireTerrainSurface surface =
        ERFFire::make_erf_level0_terrain_surface(
            fixture.inputs());

    EXPECT_EQ(surface.geometry().nx, 3U);
    EXPECT_EQ(surface.geometry().ny, 2U);
    EXPECT_EQ(surface.geometry().xlo_m, Real(10));
    EXPECT_EQ(surface.geometry().ylo_m, Real(-4));
    EXPECT_EQ(surface.geometry().dx_m, Real(2));
    EXPECT_EQ(surface.geometry().dy_m, Real(3));

    for (const auto point
         : std::vector<ERFFire::FireVec2>{
             {Real(10), Real(-4)},
             {Real(12.7), Real(-1.1)},
             {Real(16), Real(2)}}) {
        expect_near_real(
            surface.ground_height_m(point.x, point.y),
            plane_height(point.x, point.y));

        const auto gradient =
            surface.terrain_gradient_m_per_m(
                point.x, point.y);
        expect_near_real(
            gradient.x,
            Real(0.125));
        expect_near_real(
            gradient.y,
            Real(-0.2));
    }
}

TEST(
    FireTerrainSurface,
    Level0ExtractionRejectsUnsupportedTerrainScopes)
{
    TerrainFixture fixture;
    auto inputs = fixture.inputs();

    inputs.configured_max_level = 1;
    EXPECT_THROW(
        (void)ERFFire::make_erf_level0_terrain_surface(
            inputs),
        std::invalid_argument);

    inputs.configured_max_level = 0;
    inputs.terrain_type = TerrainType::MovingFittedMesh;
    EXPECT_THROW(
        (void)ERFFire::make_erf_level0_terrain_surface(
            inputs),
        std::invalid_argument);

    inputs.terrain_type = TerrainType::EB;
    EXPECT_THROW(
        (void)ERFFire::make_erf_level0_terrain_surface(
            inputs),
        std::invalid_argument);

    inputs.terrain_type = TerrainType::ImmersedForcing;
    EXPECT_THROW(
        (void)ERFFire::make_erf_level0_terrain_surface(
            inputs),
        std::invalid_argument);

    inputs.terrain_type = TerrainType::None;
    EXPECT_THROW(
        (void)ERFFire::make_erf_level0_terrain_surface(
            inputs),
        std::invalid_argument);

    inputs.terrain_type = TerrainType::StaticFittedMesh;
    inputs.buildings_type = BuildingsType::ImmersedForcing;
    EXPECT_THROW(
        (void)ERFFire::make_erf_level0_terrain_surface(
            inputs),
        std::invalid_argument);
}
