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
using ERFFire::freeze_erf_level0_environment;

constexpr int nx = 4;
constexpr int ny = 3;
constexpr int nz = 3;

constexpr Real xlo = Real(100);
constexpr Real ylo = Real(50);
constexpr Real dx = Real(2);
constexpr Real dy = Real(4);
constexpr Real ground_z = Real(100);

Real
analytic_u (Real x, Real y, Real z)
{
    return Real(2.0)
         + Real(0.125) * x
         - Real(0.25) * y
         + Real(0.05) * z;
}

Real
analytic_v (Real x, Real y, Real z)
{
    return Real(-1.0)
         + Real(0.35) * x
         + Real(0.15) * y
         - Real(0.025) * z;
}

Real
scaled_tolerance (Real expected)
{
    return Real(256)
         * std::numeric_limits<Real>::epsilon()
         * std::max(Real(1), std::abs(expected));
}

void
expect_near_real (Real actual, Real expected)
{
    EXPECT_NEAR(
        actual, expected, scaled_tolerance(expected));
}

BoxArray
make_nodal_boxarray (const BoxArray& cell_ba)
{
    BoxArray result(cell_ba);
    result.surroundingNodes();
    return result;
}

struct FlatAtmosphereFixture
{
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
        domain, real_box,
        amrex::CoordSys::cartesian, periodic};
    BoxArray cell_ba{domain};
    DistributionMapping dm{cell_ba};
    MultiFab x_velocity{
        amrex::convert(cell_ba, IntVect(1, 0, 0)),
        dm, 1, 1};
    MultiFab y_velocity{
        amrex::convert(cell_ba, IntVect(0, 1, 0)),
        dm, 1, 1};
    MultiFab z_phys_cc{
        cell_ba, dm, 1, 0};
    MultiFab z_phys_nd{
        make_nodal_boxarray(cell_ba), dm, 1, 0};

    explicit FlatAtmosphereFixture (
        bool nonflat_surface = false,
        bool nonflat_sampled_level = false)
    {
        const GpuArray<Real, nz> zcc{
            ground_z + Real(2),
            ground_z + Real(9),
            ground_z + Real(24)};
        const GpuArray<Real, nz + 1> znd{
            ground_z,
            ground_z + Real(4),
            ground_z + Real(14),
            ground_z + Real(34)};

        for (MFIter mfi(x_velocity); mfi.isValid(); ++mfi) {
            const Box bx = x_velocity[mfi].box();
            const auto arr = x_velocity.array(mfi);
            amrex::ParallelFor(
                bx,
                [=] AMREX_GPU_DEVICE (
                    int i, int j, int k) noexcept {
                    const int kk =
                        amrex::max(
                            0, amrex::min(k, nz - 1));
                    const Real x =
                        xlo + Real(i) * dx;
                    const Real y =
                        ylo
                        + (Real(j) + Real(0.5)) * dy;
                    arr(i, j, k) =
                        Real(2.0)
                        + Real(0.125) * x
                        - Real(0.25) * y
                        + Real(0.05) * zcc[kk];
                });
        }

        for (MFIter mfi(y_velocity); mfi.isValid(); ++mfi) {
            const Box bx = y_velocity[mfi].box();
            const auto arr = y_velocity.array(mfi);
            amrex::ParallelFor(
                bx,
                [=] AMREX_GPU_DEVICE (
                    int i, int j, int k) noexcept {
                    const int kk =
                        amrex::max(
                            0, amrex::min(k, nz - 1));
                    const Real x =
                        xlo
                        + (Real(i) + Real(0.5)) * dx;
                    const Real y =
                        ylo + Real(j) * dy;
                    arr(i, j, k) =
                        Real(-1.0)
                        + Real(0.35) * x
                        + Real(0.15) * y
                        - Real(0.025) * zcc[kk];
                });
        }

        for (MFIter mfi(z_phys_cc); mfi.isValid(); ++mfi) {
            const Box bx = z_phys_cc[mfi].box();
            const auto arr = z_phys_cc.array(mfi);
            amrex::ParallelFor(
                bx,
                [=] AMREX_GPU_DEVICE (
                    int i, int j, int k) noexcept {
                    Real value = zcc[k];
                    if (nonflat_sampled_level
                        && k == 1
                        && i == 1
                        && j == 1) {
                        value += Real(0.25);
                    }
                    arr(i, j, k) = value;
                });
        }

        for (MFIter mfi(z_phys_nd); mfi.isValid(); ++mfi) {
            const Box bx = z_phys_nd[mfi].box();
            const auto arr = z_phys_nd.array(mfi);
            amrex::ParallelFor(
                bx,
                [=] AMREX_GPU_DEVICE (
                    int i, int j, int k) noexcept {
                    Real value = znd[k];
                    if (nonflat_surface
                        && k == 0
                        && i == 1
                        && j == 1) {
                        value += Real(0.25);
                    }
                    arr(i, j, k) = value;
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
            MeshType::StretchedDz,
            TerrainType::StaticFittedMesh,
            BuildingsType::None,
            0};
    }
};

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
        static_cast<std::size_t>(box.numPts()));

    for (int k = box.smallEnd(2);
         k <= box.bigEnd(2); ++k) {
        for (int j = box.smallEnd(1);
             j <= box.bigEnd(1); ++j) {
            for (int i = box.smallEnd(0);
                 i <= box.bigEnd(0); ++i) {
                result.push_back(arr(i, j, k));
            }
        }
    }
    return result;
}

} // namespace


TEST(
    FireLevel0Environment,
    FlatVerticalFacesAreReportedInAGLCoordinates)
{
    FlatAtmosphereFixture fixture;

    const auto faces =
        ERFFire::erf_fire_level0_flat_vertical_faces_agl(
            fixture.inputs());

    ASSERT_EQ(faces.size(), 4U);
    EXPECT_EQ(faces[0], Real(0));
    EXPECT_EQ(faces[1], Real(4));
    EXPECT_EQ(faces[2], Real(14));
    EXPECT_EQ(faces[3], Real(34));
}

TEST(
    FireLevel0Environment,
    FlatStretchedSnapshotMatchesAnalyticNativeStaggeredWind)
{
    FlatAtmosphereFixture fixture;

    const Real reference_height_agl_m = Real(16);
    const Real target_z =
        ground_z + reference_height_agl_m;

    const auto sampler =
        freeze_erf_level0_environment(
            fixture.inputs(),
            reference_height_agl_m);

    EXPECT_EQ(
        sampler.reference_height_agl_m(),
        reference_height_agl_m);

    for (const auto& point
         : std::vector<ERFFire::FireVec2>{
             {xlo, ylo},
             {Real(103.3), Real(56.7)},
             {xlo + Real(nx) * dx,
              ylo + Real(ny) * dy}}) {
        const auto sample =
            sampler.sample(point.x, point.y);
        expect_near_real(
            sample.horizontal_wind_mps.x,
            analytic_u(point.x, point.y, target_z));
        expect_near_real(
            sample.horizontal_wind_mps.y,
            analytic_v(point.x, point.y, target_z));
    }
}

TEST(
    FireLevel0Environment,
    SnapshotReadDoesNotModifyAnySourceField)
{
    FlatAtmosphereFixture fixture;

    const auto x_before =
        replicated_values(fixture.x_velocity, 1);
    const auto y_before =
        replicated_values(fixture.y_velocity, 1);
    const auto zcc_before =
        replicated_values(fixture.z_phys_cc, 0);
    const auto znd_before =
        replicated_values(fixture.z_phys_nd, 0);

    const auto sampler =
        freeze_erf_level0_environment(
            fixture.inputs(), Real(16));
    const auto sample =
        sampler.sample(Real(103.1), Real(55.2));
    EXPECT_TRUE(
        std::isfinite(sample.horizontal_wind_mps.x));
    EXPECT_TRUE(
        std::isfinite(sample.horizontal_wind_mps.y));

    EXPECT_EQ(
        replicated_values(fixture.x_velocity, 1),
        x_before);
    EXPECT_EQ(
        replicated_values(fixture.y_velocity, 1),
        y_before);
    EXPECT_EQ(
        replicated_values(fixture.z_phys_cc, 0),
        zcc_before);
    EXPECT_EQ(
        replicated_values(fixture.z_phys_nd, 0),
        znd_before);

    const Real frozen_u = sample.horizontal_wind_mps.x;
    const Real frozen_v = sample.horizontal_wind_mps.y;

    fixture.x_velocity.setVal(Real(1234));
    fixture.y_velocity.setVal(Real(-4321));
    amrex::Gpu::streamSynchronize();

    const auto frozen_after_source_mutation =
        sampler.sample(Real(103.1), Real(55.2));
    EXPECT_EQ(
        frozen_after_source_mutation.horizontal_wind_mps.x,
        frozen_u);
    EXPECT_EQ(
        frozen_after_source_mutation.horizontal_wind_mps.y,
        frozen_v);
}

TEST(
    FireLevel0Environment,
    RejectsUnsupportedConfiguredLevelAndTerrainScopes)
{
    FlatAtmosphereFixture fixture;
    auto inputs = fixture.inputs();

    inputs.configured_max_level = 1;
    EXPECT_THROW(
        freeze_erf_level0_environment(
            inputs, Real(16)),
        std::invalid_argument);

    inputs.configured_max_level = 0;
    inputs.mesh_type = MeshType::VariableDz;
    EXPECT_THROW(
        freeze_erf_level0_environment(
            inputs, Real(16)),
        std::invalid_argument);

    inputs.mesh_type = MeshType::ConstantDz;
    inputs.terrain_type = TerrainType::EB;
    EXPECT_THROW(
        freeze_erf_level0_environment(
            inputs, Real(16)),
        std::invalid_argument);

    inputs.terrain_type =
        TerrainType::ImmersedForcing;
    EXPECT_THROW(
        freeze_erf_level0_environment(
            inputs, Real(16)),
        std::invalid_argument);

    inputs.terrain_type =
        TerrainType::MovingFittedMesh;
    EXPECT_THROW(
        freeze_erf_level0_environment(
            inputs, Real(16)),
        std::invalid_argument);

    inputs.terrain_type = TerrainType::None;
    inputs.buildings_type =
        BuildingsType::ImmersedForcing;
    EXPECT_THROW(
        freeze_erf_level0_environment(
            inputs, Real(16)),
        std::invalid_argument);
}

TEST(
    FireLevel0Environment,
    RejectsNonFlatPhysicalCoordinates)
{
    FlatAtmosphereFixture nonflat_surface(true);

    EXPECT_THROW(
        freeze_erf_level0_environment(
            nonflat_surface.inputs(), Real(16)),
        std::invalid_argument);

    FlatAtmosphereFixture nonflat_sampled_level(false, true);

    EXPECT_THROW(
        freeze_erf_level0_environment(
            nonflat_sampled_level.inputs(), Real(16)),
        std::invalid_argument);
}

TEST(
    FireLevel0Environment,
    RejectsMissingGhostsAndInvalidReferenceHeight)
{
    FlatAtmosphereFixture fixture;

    MultiFab x_without_ghost(
        fixture.x_velocity.boxArray(),
        fixture.x_velocity.DistributionMap(),
        1, 0);
    x_without_ghost.setVal(Real(0));

    const auto base = fixture.inputs();
    const ERFFireLevel0EnvironmentInputs no_ghost_inputs{
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
        freeze_erf_level0_environment(
            no_ghost_inputs, Real(16)),
        std::invalid_argument);

    Box expected_x =
        amrex::convert(fixture.domain, IntVect(1, 0, 0));
    Box left = expected_x;
    Box right = expected_x;
    left.setBig(0, expected_x.smallEnd(0) + 1);
    right.setSmall(0, expected_x.smallEnd(0) + 3);
    const Box holey_boxes[2]{left, right};
    BoxArray holey_x_ba(holey_boxes, 2);
    DistributionMapping holey_dm(holey_x_ba);
    MultiFab holey_x_velocity(holey_x_ba, holey_dm, 1, 1);
    holey_x_velocity.setVal(Real(0));

    const ERFFireLevel0EnvironmentInputs holey_layout_inputs{
        base.geometry,
        holey_x_velocity,
        base.y_velocity,
        base.z_phys_cc,
        base.z_phys_nd,
        base.mesh_type,
        base.terrain_type,
        base.buildings_type,
        base.configured_max_level};

    EXPECT_THROW(
        freeze_erf_level0_environment(
            holey_layout_inputs, Real(16)),
        std::invalid_argument);

    EXPECT_THROW(
        freeze_erf_level0_environment(
            fixture.inputs(), Real(-1)),
        std::invalid_argument);

    // The first horizontal-velocity center is 2 m AGL.
    EXPECT_THROW(
        freeze_erf_level0_environment(
            fixture.inputs(), Real(1)),
        std::out_of_range);

    EXPECT_THROW(
        freeze_erf_level0_environment(
            fixture.inputs(),
            std::numeric_limits<Real>::quiet_NaN()),
        std::invalid_argument);
}
