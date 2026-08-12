#include <gtest/gtest.h>

#include <ERF_FireLevel0Environment.H>

#include <AMReX_Arena.H>
#include <AMReX_Array.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_IntVect.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_RealBox.H>

#include <algorithm>
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
using amrex::FArrayBox;
using amrex::Geometry;
using amrex::GpuArray;
using amrex::IntVect;
using amrex::MFIter;
using amrex::MultiFab;
using amrex::Real;
using amrex::RealBox;
using ERFFire::ERFFireLevel0EnvironmentInputs;
using ERFFire::freeze_erf_level0_terrain_reference_wind_environment;

constexpr int nx = 4;
constexpr int ny = 3;
constexpr int nz = 4;

constexpr Real xlo = Real(10);
constexpr Real ylo = Real(-6);
constexpr Real dx = Real(2);
constexpr Real dy = Real(3);
constexpr Real fixed_top_z = Real(150);

AMREX_GPU_HOST_DEVICE
Real
terrain_ground (Real x, Real y)
{
    return Real(50)
        + Real(2) * x
        + y;
}

AMREX_GPU_HOST_DEVICE
Real
analytic_u (Real x, Real y, Real z)
{
    return Real(4)
        + Real(0.1) * x
        - Real(0.2) * y
        + Real(0.03) * z;
}

AMREX_GPU_HOST_DEVICE
Real
analytic_v (Real x, Real y, Real z)
{
    return Real(-2)
        + Real(0.05) * x
        + Real(0.15) * y
        - Real(0.02) * z;
}

Real
scaled_tolerance (Real expected)
{
    return Real(1024)
        * std::numeric_limits<Real>::epsilon()
        * std::max(Real(1), std::abs(expected));
}

void
expect_near_real (Real actual, Real expected)
{
    EXPECT_NEAR(
        actual,
        expected,
        scaled_tolerance(expected));
}

BoxArray
make_nodal_boxarray (const BoxArray& cell_ba)
{
    BoxArray result(cell_ba);
    result.surroundingNodes();
    return result;
}

std::vector<Real>
replicated_values (
    const MultiFab& mf,
    int nghost)
{
    Box box = mf.boxArray().minimalBox();
    box.grow(nghost);

    FArrayBox fab(
        box, 1, amrex::The_Pinned_Arena());
    mf.copyTo(fab, 0, 0, 1, nghost);
    const auto arr = fab.const_array();

    std::vector<Real> result;
    result.reserve(
        static_cast<std::size_t>(
            box.numPts()));

    for (int k = box.smallEnd(2);
         k <= box.bigEnd(2); ++k) {
        for (int j = box.smallEnd(1);
             j <= box.bigEnd(1); ++j) {
            for (int i = box.smallEnd(0);
                 i <= box.bigEnd(0); ++i) {
                result.push_back(
                    arr(i, j, k));
            }
        }
    }

    return result;
}

struct TerrainWindFixture
{
    Box domain{
        IntVect(0, 0, 0),
        IntVect(nx - 1, ny - 1, nz - 1)};
    RealBox real_box{
        xlo,
        ylo,
        Real(0),
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
        dm,
        1,
        1};
    MultiFab y_velocity{
        amrex::convert(
            cell_ba,
            IntVect(0, 1, 0)),
        dm,
        1,
        1};
    MultiFab z_phys_cc{
        cell_ba,
        dm,
        1,
        0};
    MultiFab z_phys_nd{
        make_nodal_boxarray(cell_ba),
        dm,
        1,
        1};

    TerrainWindFixture()
    {
        const GpuArray<Real, nz + 1> eta{{
            Real(0),
            Real(0.1),
            Real(0.3),
            Real(0.6),
            Real(1)}};

        for (MFIter mfi(z_phys_nd); mfi.isValid(); ++mfi) {
            const Box bx =
                z_phys_nd[mfi].box();
            const auto z =
                z_phys_nd.array(mfi);

            amrex::ParallelFor(
                bx,
                [=] AMREX_GPU_DEVICE (
                    int i, int j, int k) noexcept {
                    const int kk =
                        amrex::max(
                            0,
                            amrex::min(k, nz));
                    const Real x =
                        xlo + Real(i) * dx;
                    const Real y =
                        ylo + Real(j) * dy;
                    const Real ground =
                        terrain_ground(x, y);

                    z(i, j, k) =
                        ground
                        + eta[kk]
                            * (fixed_top_z - ground);
                });
        }

        for (MFIter mfi(z_phys_cc); mfi.isValid(); ++mfi) {
            const Box bx =
                z_phys_cc[mfi].box();
            const auto z =
                z_phys_cc.array(mfi);

            amrex::ParallelFor(
                bx,
                [=] AMREX_GPU_DEVICE (
                    int i, int j, int k) noexcept {
                    const Real eta_cc =
                        Real(0.5)
                        * (eta[k] + eta[k + 1]);
                    const Real x =
                        xlo
                        + (Real(i) + Real(0.5)) * dx;
                    const Real y =
                        ylo
                        + (Real(j) + Real(0.5)) * dy;
                    const Real ground =
                        terrain_ground(x, y);

                    z(i, j, k) =
                        ground
                        + eta_cc
                            * (fixed_top_z - ground);
                });
        }

        for (MFIter mfi(x_velocity); mfi.isValid(); ++mfi) {
            const Box bx =
                x_velocity[mfi].box();
            const auto u =
                x_velocity.array(mfi);

            amrex::ParallelFor(
                bx,
                [=] AMREX_GPU_DEVICE (
                    int i, int j, int k) noexcept {
                    const int kk =
                        amrex::max(
                            0,
                            amrex::min(k, nz - 1));
                    const Real eta_cc =
                        Real(0.5)
                        * (eta[kk] + eta[kk + 1]);
                    const Real x =
                        xlo + Real(i) * dx;
                    const Real y =
                        ylo
                        + (Real(j) + Real(0.5)) * dy;
                    const Real ground =
                        terrain_ground(x, y);
                    const Real z =
                        ground
                        + eta_cc
                            * (fixed_top_z - ground);

                    u(i, j, k) =
                        analytic_u(x, y, z);
                });
        }

        for (MFIter mfi(y_velocity); mfi.isValid(); ++mfi) {
            const Box bx =
                y_velocity[mfi].box();
            const auto v =
                y_velocity.array(mfi);

            amrex::ParallelFor(
                bx,
                [=] AMREX_GPU_DEVICE (
                    int i, int j, int k) noexcept {
                    const int kk =
                        amrex::max(
                            0,
                            amrex::min(k, nz - 1));
                    const Real eta_cc =
                        Real(0.5)
                        * (eta[kk] + eta[kk + 1]);
                    const Real x =
                        xlo
                        + (Real(i) + Real(0.5)) * dx;
                    const Real y =
                        ylo + Real(j) * dy;
                    const Real ground =
                        terrain_ground(x, y);
                    const Real z =
                        ground
                        + eta_cc
                            * (fixed_top_z - ground);

                    v(i, j, k) =
                        analytic_v(x, y, z);
                });
        }

        amrex::Gpu::streamSynchronize();
    }

    ERFFireLevel0EnvironmentInputs
    inputs () const
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
    FireTerrainReferenceWind,
    AffineWindMatchesIndependentFixedLocalAGLOracle)
{
    TerrainWindFixture fixture;
    const Real reference_height_agl_m =
        Real(16.5);

    const auto sampler =
        freeze_erf_level0_terrain_reference_wind_environment(
            fixture.inputs(),
            reference_height_agl_m);

    EXPECT_EQ(
        sampler.reference_height_agl_m(),
        reference_height_agl_m);

    for (const auto& point
         : std::vector<ERFFire::FireVec2>{
             {xlo, ylo},
             {Real(13.3), Real(-1.7)},
             {xlo + Real(nx) * dx,
              ylo + Real(ny) * dy}}) {
        const Real target_z =
            terrain_ground(point.x, point.y)
            + reference_height_agl_m;

        const auto sample =
            sampler.sample(
                point.x,
                point.y);

        expect_near_real(
            sample.horizontal_wind_mps.x,
            analytic_u(
                point.x,
                point.y,
                target_z));
        expect_near_real(
            sample.horizontal_wind_mps.y,
            analytic_v(
                point.x,
                point.y,
                target_z));
    }
}

TEST(
    FireTerrainReferenceWind,
    OneSnapshotUsesDifferentVerticalBracketsAcrossTerrain)
{
    TerrainWindFixture fixture;
    const Real reference_height_agl_m =
        Real(16.5);

    const ERFFire::FireVec2 low_ground_point{
        Real(10),
        Real(-4.5)};
    const ERFFire::FireVec2 high_ground_point{
        Real(18),
        Real(1.5)};

    const Real low_second_center_agl =
        Real(0.2)
        * (fixed_top_z
           - terrain_ground(
               low_ground_point.x,
               low_ground_point.y));
    const Real high_second_center_agl =
        Real(0.2)
        * (fixed_top_z
           - terrain_ground(
               high_ground_point.x,
               high_ground_point.y));

    ASSERT_GT(
        low_second_center_agl,
        reference_height_agl_m);
    ASSERT_LT(
        high_second_center_agl,
        reference_height_agl_m);

    const auto sampler =
        freeze_erf_level0_terrain_reference_wind_environment(
            fixture.inputs(),
            reference_height_agl_m);

    for (const auto& point
         : std::vector<ERFFire::FireVec2>{
             low_ground_point,
             high_ground_point}) {
        const Real target_z =
            terrain_ground(point.x, point.y)
            + reference_height_agl_m;
        const auto sample =
            sampler.sample(
                point.x,
                point.y);

        expect_near_real(
            sample.horizontal_wind_mps.x,
            analytic_u(
                point.x,
                point.y,
                target_z));
        expect_near_real(
            sample.horizontal_wind_mps.y,
            analytic_v(
                point.x,
                point.y,
                target_z));
    }
}

TEST(
    FireTerrainReferenceWind,
    NodalCoordinatesAreAuthoritativeOverZPhysCc)
{
    TerrainWindFixture fixture;
    fixture.z_phys_cc.setVal(Real(-12345));
    amrex::Gpu::streamSynchronize();

    const Real reference_height_agl_m =
        Real(16.5);
    const auto sampler =
        freeze_erf_level0_terrain_reference_wind_environment(
            fixture.inputs(),
            reference_height_agl_m);

    const ERFFire::FireVec2 point{
        Real(14.2),
        Real(-0.4)};
    const Real target_z =
        terrain_ground(point.x, point.y)
        + reference_height_agl_m;
    const auto sample =
        sampler.sample(
            point.x,
            point.y);

    expect_near_real(
        sample.horizontal_wind_mps.x,
        analytic_u(
            point.x,
            point.y,
            target_z));
    expect_near_real(
        sample.horizontal_wind_mps.y,
        analytic_v(
            point.x,
            point.y,
            target_z));
}

TEST(
    FireTerrainReferenceWind,
    SnapshotOwnsValuesAndDoesNotModifySources)
{
    TerrainWindFixture fixture;

    const auto x_before =
        replicated_values(
            fixture.x_velocity,
            1);
    const auto y_before =
        replicated_values(
            fixture.y_velocity,
            1);
    const auto zcc_before =
        replicated_values(
            fixture.z_phys_cc,
            0);
    const auto znd_before =
        replicated_values(
            fixture.z_phys_nd,
            1);

    const auto sampler =
        freeze_erf_level0_terrain_reference_wind_environment(
            fixture.inputs(),
            Real(16.5));

    EXPECT_EQ(
        replicated_values(
            fixture.x_velocity,
            1),
        x_before);
    EXPECT_EQ(
        replicated_values(
            fixture.y_velocity,
            1),
        y_before);
    EXPECT_EQ(
        replicated_values(
            fixture.z_phys_cc,
            0),
        zcc_before);
    EXPECT_EQ(
        replicated_values(
            fixture.z_phys_nd,
            1),
        znd_before);

    const auto before_mutation =
        sampler.sample(
            Real(13.1),
            Real(-1.2));

    fixture.x_velocity.setVal(Real(999));
    fixture.y_velocity.setVal(Real(-999));
    fixture.z_phys_cc.setVal(Real(777));
    fixture.z_phys_nd.setVal(Real(888));
    amrex::Gpu::streamSynchronize();

    const auto after_mutation =
        sampler.sample(
            Real(13.1),
            Real(-1.2));

    EXPECT_EQ(
        after_mutation.horizontal_wind_mps.x,
        before_mutation.horizontal_wind_mps.x);
    EXPECT_EQ(
        after_mutation.horizontal_wind_mps.y,
        before_mutation.horizontal_wind_mps.y);
}

TEST(
    FireTerrainReferenceWind,
    RejectsUnsupportedScopeAndMissingGhosts)
{
    TerrainWindFixture fixture;
    auto inputs = fixture.inputs();

    inputs.configured_max_level = 1;
    EXPECT_THROW(
        freeze_erf_level0_terrain_reference_wind_environment(
            inputs,
            Real(16.5)),
        std::invalid_argument);

    inputs.configured_max_level = 0;
    inputs.mesh_type =
        MeshType::StretchedDz;
    EXPECT_THROW(
        freeze_erf_level0_terrain_reference_wind_environment(
            inputs,
            Real(16.5)),
        std::invalid_argument);

    inputs.mesh_type =
        MeshType::VariableDz;
    inputs.terrain_type =
        TerrainType::None;
    EXPECT_THROW(
        freeze_erf_level0_terrain_reference_wind_environment(
            inputs,
            Real(16.5)),
        std::invalid_argument);

    inputs.terrain_type =
        TerrainType::MovingFittedMesh;
    EXPECT_THROW(
        freeze_erf_level0_terrain_reference_wind_environment(
            inputs,
            Real(16.5)),
        std::invalid_argument);

    inputs.terrain_type =
        TerrainType::StaticFittedMesh;
    inputs.buildings_type =
        BuildingsType::ImmersedForcing;
    EXPECT_THROW(
        freeze_erf_level0_terrain_reference_wind_environment(
            inputs,
            Real(16.5)),
        std::invalid_argument);

    const auto base =
        fixture.inputs();

    MultiFab x_without_ghost(
        fixture.x_velocity.boxArray(),
        fixture.x_velocity.DistributionMap(),
        1,
        0);
    x_without_ghost.setVal(Real(0));

    const ERFFireLevel0EnvironmentInputs no_x_ghost{
        base.geometry,
        x_without_ghost,
        base.y_velocity,
        base.z_phys_cc,
        base.z_phys_nd,
        base.mesh_type,
        base.terrain_type,
        base.buildings_type,
        base.configured_max_level};

    EXPECT_THROW(
        freeze_erf_level0_terrain_reference_wind_environment(
            no_x_ghost,
            Real(16.5)),
        std::invalid_argument);

    MultiFab z_without_vertical_ghost(
        fixture.z_phys_nd.boxArray(),
        fixture.z_phys_nd.DistributionMap(),
        1,
        IntVect(1, 1, 0));
    z_without_vertical_ghost.setVal(Real(0));

    const ERFFireLevel0EnvironmentInputs no_z_vertical_ghost{
        base.geometry,
        base.x_velocity,
        base.y_velocity,
        base.z_phys_cc,
        z_without_vertical_ghost,
        base.mesh_type,
        base.terrain_type,
        base.buildings_type,
        base.configured_max_level};

    EXPECT_THROW(
        freeze_erf_level0_terrain_reference_wind_environment(
            no_z_vertical_ghost,
            Real(16.5)),
        std::invalid_argument);
}

TEST(
    FireTerrainReferenceWind,
    RejectsInvalidOrOutOfColumnReferenceHeight)
{
    TerrainWindFixture fixture;

    EXPECT_THROW(
        freeze_erf_level0_terrain_reference_wind_environment(
            fixture.inputs(),
            Real(-1)),
        std::invalid_argument);

    EXPECT_THROW(
        freeze_erf_level0_terrain_reference_wind_environment(
            fixture.inputs(),
            std::numeric_limits<Real>::quiet_NaN()),
        std::invalid_argument);

    EXPECT_THROW(
        freeze_erf_level0_terrain_reference_wind_environment(
            fixture.inputs(),
            Real(1)),
        std::out_of_range);

    EXPECT_THROW(
        freeze_erf_level0_terrain_reference_wind_environment(
            fixture.inputs(),
            Real(100)),
        std::out_of_range);
}
