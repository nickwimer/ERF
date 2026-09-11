#include <AMReX_Array.H>
#include <AMReX_BCRec.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_IntVect.H>
#include <AMReX_RealBox.H>
#include <AMReX_REAL.H>

#include <ERF_PhysBCFunct.H>

#include <gtest/gtest.h>

#include <memory>

namespace {

using amrex::Box;
using amrex::FArrayBox;
using amrex::IntVect;
using amrex::Real;

using BCValueArray =
    amrex::Array<amrex::Array<Real, AMREX_SPACEDIM*2>,
                 AMREX_SPACEDIM + NBCVAR_max>;

void initialize_velocity_fabs (
    FArrayBox& u_fab,
    FArrayBox& v_fab,
    FArrayBox& w_fab,
    const Box& u_valid,
    const Box& v_valid,
    const Box& w_valid,
    const Real sentinel)
{
    const auto u = u_fab.array();
    const auto v = v_fab.array();
    const auto w = w_fab.array();

    amrex::ParallelFor(
        u_fab.box(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            u(i,j,k) = sentinel;
        });
    amrex::ParallelFor(
        v_fab.box(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            v(i,j,k) = sentinel;
        });
    amrex::ParallelFor(
        w_fab.box(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            w(i,j,k) = sentinel;
        });

    amrex::ParallelFor(
        u_valid,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            u(i,j,k) = Real(3.0);
        });
    amrex::ParallelFor(
        v_valid,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            v(i,j,k) = Real(2.0);
        });
    amrex::ParallelFor(
        w_valid,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            w(i,j,k) = Real(1.0);
        });

    amrex::ParallelFor(
        1,
        [=] AMREX_GPU_DEVICE (int) noexcept
        {
            u(0,1,0) = Real(5.0);
            u(0,2,0) = Real(-5.0);
            u(4,1,0) = Real(-5.0);
            u(4,2,0) = Real(5.0);
            v(1,0,0) = Real(4.0);
            v(2,0,0) = Real(-4.0);
            v(1,4,0) = Real(-4.0);
            v(2,4,0) = Real(4.0);
            v(3,4,0) = Real(-4.0);
        });

    amrex::Gpu::streamSynchronize();
}

void sample_velocity_ghosts (
    const FArrayBox& u_fab,
    const FArrayBox& v_fab,
    const FArrayBox& w_fab,
    amrex::Gpu::DeviceVector<Real>& observed_d)
{
    Real* observed = observed_d.data();
    const auto u = u_fab.const_array();
    const auto v = v_fab.const_array();
    const auto w = w_fab.const_array();

    amrex::ParallelFor(
        1,
        [=] AMREX_GPU_DEVICE (int) noexcept
        {
            observed[0]  = u(5,1,0);
            observed[1]  = u(6,2,0);
            observed[2]  = u(1,4,0);
            observed[3]  = u(2,5,0);
            observed[4]  = u(5,4,0);
            observed[5]  = v(4,1,0);
            observed[6]  = v(5,2,0);
            observed[7]  = v(1,5,0);
            observed[8]  = v(2,6,0);
            observed[9]  = v(4,5,0);
            observed[10] = w(4,1,0);
            observed[11] = w(5,2,0);
            observed[12] = w(1,4,0);
            observed[13] = w(2,5,0);
            observed[14] = w(4,4,0);
            observed[15] = u(-1,1,0);
            observed[16] = u(-2,2,0);
            observed[17] = u(1,-1,0);
            observed[18] = u(2,-2,0);
            observed[19] = u(-1,-1,0);
            observed[20] = v(-1,1,0);
            observed[21] = v(-2,2,0);
            observed[22] = v(1,-1,0);
            observed[23] = v(2,-2,0);
            observed[24] = v(-1,-1,0);
            observed[25] = w(-1,1,0);
            observed[26] = w(-2,2,0);
            observed[27] = w(1,-1,0);
            observed[28] = w(2,-2,0);
            observed[29] = w(-1,-1,0);
        });

    amrex::Gpu::streamSynchronize();
}

TEST(InflowOutflowVelocity, GhostsUseInflowValuesOrOutflowExtrapolation)
{
    const Box domain(
        IntVect(AMREX_D_DECL(0, 0, 0)),
        IntVect(AMREX_D_DECL(3, 3, 0)));

    amrex::RealBox real_box(
        {AMREX_D_DECL(0.0, 0.0, 0.0)},
        {AMREX_D_DECL(4.0, 4.0, 1.0)});
    int is_periodic[AMREX_SPACEDIM] = {AMREX_D_DECL(0, 0, 0)};
    amrex::Geometry geom(domain, &real_box, 0, is_periodic);

    amrex::Vector<amrex::BCRec> bcs(BCVars::NumTypes);
    for (auto& bc : bcs) {
        for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {
            bc.setLo(dir, ERFBCType::foextrap);
            bc.setHi(dir, ERFBCType::foextrap);
        }
    }

    bcs[BCVars::xvel_bc].setLo(0, ERFBCType::ext_dir_upwind);
    bcs[BCVars::xvel_bc].setLo(1, ERFBCType::ext_dir_upwind);
    bcs[BCVars::xvel_bc].setHi(0, ERFBCType::ext_dir_upwind);
    bcs[BCVars::xvel_bc].setHi(1, ERFBCType::ext_dir_upwind);
    bcs[BCVars::yvel_bc].setLo(0, ERFBCType::ext_dir_upwind);
    bcs[BCVars::yvel_bc].setLo(1, ERFBCType::ext_dir_upwind);
    bcs[BCVars::yvel_bc].setHi(0, ERFBCType::ext_dir_upwind);
    bcs[BCVars::yvel_bc].setHi(1, ERFBCType::ext_dir_upwind);
    bcs[BCVars::zvel_bc].setLo(0, ERFBCType::ext_dir_upwind);
    bcs[BCVars::zvel_bc].setLo(1, ERFBCType::ext_dir_upwind);
    bcs[BCVars::zvel_bc].setHi(0, ERFBCType::ext_dir_upwind);
    bcs[BCVars::zvel_bc].setHi(1, ERFBCType::ext_dir_upwind);

    amrex::Gpu::DeviceVector<amrex::BCRec> bcs_d(bcs.size());
    amrex::Gpu::copy(
        amrex::Gpu::hostToDevice,
        bcs.begin(), bcs.end(), bcs_d.begin());

    BCValueArray extdir{};
    BCValueArray neumann{};
    extdir[BCVars::xvel_bc][0] = Real(15.0);
    extdir[BCVars::xvel_bc][1] = Real(16.0);
    extdir[BCVars::xvel_bc][3] = Real(-15.0);
    extdir[BCVars::xvel_bc][4] = Real(-15.0);
    extdir[BCVars::yvel_bc][0] = Real(8.0);
    extdir[BCVars::yvel_bc][1] = Real(9.0);
    extdir[BCVars::yvel_bc][3] = Real(-8.0);
    extdir[BCVars::yvel_bc][4] = Real(-8.0);
    extdir[BCVars::zvel_bc][0] = Real(-7.0);
    extdir[BCVars::zvel_bc][1] = Real(-9.0);
    extdir[BCVars::zvel_bc][3] = Real(7.0);
    extdir[BCVars::zvel_bc][4] = Real(9.0);

    std::unique_ptr<amrex::MultiFab> z_phys_nd;
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> mapfac_lev(MapFacType::num);

    ERFPhysBCFunct_u physbc_u(
        0, geom, bcs, bcs_d, extdir, neumann,
        z_phys_nd, false, nullptr);
    ERFPhysBCFunct_v physbc_v(
        0, geom, bcs, bcs_d, extdir, neumann,
        z_phys_nd, false, nullptr);
    ERFPhysBCFunct_w physbc_w(
        0, geom, bcs, bcs_d, extdir, neumann,
        TerrainType::None, mapfac_lev, z_phys_nd, false, nullptr);

    const Box u_valid = amrex::surroundingNodes(domain, 0);
    const Box v_valid = amrex::surroundingNodes(domain, 1);
    const Box w_valid = domain;

    Box u_box = u_valid;
    Box v_box = v_valid;
    Box w_box = w_valid;
    u_box.grow(IntVect(AMREX_D_DECL(2, 2, 0)));
    v_box.grow(IntVect(AMREX_D_DECL(2, 2, 0)));
    w_box.grow(IntVect(AMREX_D_DECL(2, 2, 0)));

    FArrayBox u_fab(u_box, 1);
    FArrayBox v_fab(v_box, 1);
    FArrayBox w_fab(w_box, 1);

    const Real sentinel = Real(1.0e100);
    initialize_velocity_fabs(
        u_fab, v_fab, w_fab,
        u_valid, v_valid, w_valid,
        sentinel);

    physbc_u.impose_lateral_xvel_bcs(
        u_fab.array(), u_fab.const_array(), v_fab.const_array(),
        u_box, domain, BCVars::xvel_bc, 0.0);

    physbc_v.impose_lateral_yvel_bcs(
        v_fab.array(), u_fab.const_array(), v_fab.const_array(),
        v_box, domain, BCVars::yvel_bc, 0.0);

    amrex::Array4<const Real> empty_array;
    amrex::GpuArray<Real, AMREX_SPACEDIM> dx_inv{};

    physbc_w.impose_lateral_zvel_bcs(
        w_fab.array(), u_fab.const_array(), v_fab.const_array(),
        w_box, domain, empty_array, empty_array, empty_array, dx_inv,
        TerrainType::None, BCVars::zvel_bc, 0.0);

    amrex::Gpu::DeviceVector<Real> observed_d(30);
    sample_velocity_ghosts(
        u_fab, v_fab, w_fab, observed_d);

    amrex::Vector<Real> observed_h(30);
    amrex::Gpu::copy(
        amrex::Gpu::deviceToHost,
        observed_d.begin(), observed_d.end(), observed_h.begin());

    EXPECT_EQ(observed_h[0], Real(-15.0));
    EXPECT_EQ(observed_h[1], Real(5.0));
    EXPECT_EQ(observed_h[2], Real(-15.0));
    EXPECT_EQ(observed_h[3], Real(3.0));
    EXPECT_EQ(observed_h[4], Real(-15.0));

    EXPECT_EQ(observed_h[5], Real(-8.0));
    EXPECT_EQ(observed_h[6], Real(2.0));
    EXPECT_EQ(observed_h[7], Real(-8.0));
    EXPECT_EQ(observed_h[8], Real(4.0));
    EXPECT_EQ(observed_h[9], Real(-8.0));

    EXPECT_EQ(observed_h[10], Real(7.0));
    EXPECT_EQ(observed_h[11], Real(1.0));
    EXPECT_EQ(observed_h[12], Real(9.0));
    EXPECT_EQ(observed_h[13], Real(1.0));
    EXPECT_EQ(observed_h[14], Real(9.0));

    EXPECT_EQ(observed_h[15], Real(15.0));
    EXPECT_EQ(observed_h[16], Real(-5.0));
    EXPECT_EQ(observed_h[17], Real(16.0));
    EXPECT_EQ(observed_h[18], Real(3.0));
    EXPECT_EQ(observed_h[19], Real(16.0));

    EXPECT_EQ(observed_h[20], Real(8.0));
    EXPECT_EQ(observed_h[21], Real(2.0));
    EXPECT_EQ(observed_h[22], Real(9.0));
    EXPECT_EQ(observed_h[23], Real(-4.0));
    EXPECT_EQ(observed_h[24], Real(9.0));

    EXPECT_EQ(observed_h[25], Real(-7.0));
    EXPECT_EQ(observed_h[26], Real(1.0));
    EXPECT_EQ(observed_h[27], Real(-9.0));
    EXPECT_EQ(observed_h[28], Real(1.0));
    EXPECT_EQ(observed_h[29], Real(-9.0));
}

}
