#ifndef ERF_GTEST_FIRE_FUEL_BARRIER_SWEEP_I_H_
#define ERF_GTEST_FIRE_FUEL_BARRIER_SWEEP_I_H_

#include <ERF_FireFuelBarrierSweep.H>

#include <AMReX_ParallelDescriptor.H>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{

using SweepReal = amrex::Real;
using ERFFire::FireCartesianRasterGeometry2D;
using ERFFire::FireFront;
using ERFFire::FireFrontComponent;
using ERFFire::FireFrontRole;
using ERFFire::FireFuelModelId;
using ERFFire::FireFuelMoistureClass;
using ERFFire::FireFuelRaster;
using ERFFire::FireFuelRasterState;
using ERFFire::FirePerimeter;
using ERFFire::FireVec2;

FireFuelRaster
make_sweep_barrier_raster(
    const std::vector<std::pair<std::size_t, std::size_t>>& nonburnable)
{
    const FireCartesianRasterGeometry2D geometry{
        4,
        4,
        SweepReal(0),
        SweepReal(0),
        SweepReal(1),
        SweepReal(1)};

    FireFuelRasterState state;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        state.cells.resize(geometry.nx * geometry.ny);
        for (std::size_t j = 0; j < geometry.ny; ++j) {
            for (std::size_t i = 0; i < geometry.nx; ++i) {
                auto& cell = state.cells[j * geometry.nx + i];
                const bool blocked =
                    std::find(
                        nonburnable.begin(),
                        nonburnable.end(),
                        std::make_pair(i, j))
                    != nonburnable.end();

                if (blocked) {
                    cell.model_id = FireFuelModelId::NonBurnable;
                } else {
                    cell.model_id = FireFuelModelId::FM1;
                    cell.moisture.set(
                        FireFuelMoistureClass::Dead1h,
                        SweepReal(0.08));
                }
            }
        }
    }

    return FireFuelRaster::collective_from_io_rank_state(
        geometry,
        state);
}

std::vector<std::pair<std::size_t, std::size_t>>
sweep_full_height_barrier()
{
    return {
        {2U, 0U},
        {2U, 1U},
        {2U, 2U},
        {2U, 3U}};
}

FireFront
make_transient_sweep_front()
{
    return FireFront(
        std::vector<FireFrontComponent>{
            {
                FireFrontRole::Outer,
                FirePerimeter(
                    std::vector<FireVec2>{
                        {SweepReal(1.5), SweepReal(2.5)},
                        {SweepReal(3.5), SweepReal(2.5)},
                        {SweepReal(3.5), SweepReal(2.7)},
                        {SweepReal(1.5), SweepReal(2.7)}
                    })
            }
        });
}

ERFFire::NormalSpeedBatchFunction
transient_sweep_speed(int& calls)
{
    return [&calls](
        const std::vector<FireVec2>& positions,
        const std::vector<FireVec2>& normals,
        SweepReal) {
        ++calls;
        if (positions.size() != normals.size()) {
            throw std::logic_error(
                "barrier sweep test positions/normals size mismatch");
        }

        std::vector<SweepReal> speeds(
            positions.size(), SweepReal(0));

        // Keep the RK midpoint equal to the initial state. At the second
        // stage move only the lower vertices down by exactly 2 m. Their x
        // trajectories stay outside the blocked cell, and the final edges are
        // legal, but the lower front edge continuously sweeps through it.
        if (calls == 2) {
            for (std::size_t index = 0;
                 index < positions.size();
                 ++index) {
                if (positions[index].y < SweepReal(2.6)
                    && normals[index].y < SweepReal(0)) {
                    speeds[index] =
                        SweepReal(2) / (-normals[index].y);
                }
            }
        }

        return speeds;
    };
}

TEST(FireFuelBarrier, SweptEdgeHullCollapsesRoundoffThinMotion)
{
    const SweepReal epsilon =
        std::numeric_limits<SweepReal>::epsilon();
    const SweepReal offset =
        SweepReal(8) * epsilon;

    const auto hull =
        ERFFire::detail::fire_barrier_swept_edge_hull(
            {SweepReal(0), SweepReal(0)},
            {SweepReal(1), -offset},
            {SweepReal(1), offset},
            {SweepReal(2), SweepReal(0)});

    // Exact cross products retain a four-vertex diamond here, while
    // FirePerimeter's scaled orientation test correctly treats it as
    // roundoff-collinear.  The swept-edge hull must use the same resolution.
    EXPECT_TRUE(hull.empty());
}

TEST(FireFuelBarrier, SweptEdgesDetectTransientBarrierBetweenLegalStates)
{
    const FireFuelRaster raster =
        make_sweep_barrier_raster({{2U, 1U}});
    const std::vector<FireFrontRole> roles{FireFrontRole::Outer};
    const std::vector<std::vector<FireVec2>> start{
        {
            {SweepReal(1.5), SweepReal(2.5)},
            {SweepReal(3.5), SweepReal(2.5)},
            {SweepReal(3.5), SweepReal(3.5)},
            {SweepReal(1.5), SweepReal(3.5)}
        }};
    const std::vector<std::vector<FireVec2>> end{
        {
            {SweepReal(1.5), SweepReal(0.2)},
            {SweepReal(3.5), SweepReal(0.2)},
            {SweepReal(3.5), SweepReal(0.8)},
            {SweepReal(1.5), SweepReal(0.8)}
        }};

    EXPECT_TRUE(
        ERFFire::collective_nonburnable_raw_front_edge_contacts(
            roles,
            start,
            raster).empty());
    EXPECT_TRUE(
        ERFFire::collective_nonburnable_raw_front_edge_contacts(
            roles,
            end,
            raster).empty());

    std::vector<FireVec2> vertex_starts = start[0];
    std::vector<FireVec2> vertex_ends = end[0];
    const auto vertex_contacts =
        ERFFire::collective_first_nonburnable_contacts(
            raster,
            vertex_starts,
            vertex_ends);
    ASSERT_EQ(vertex_contacts.size(), 4U);
    for (const auto& contact : vertex_contacts) {
        EXPECT_FALSE(contact.has_value());
    }

    const auto contacts =
        ERFFire::collective_nonburnable_raw_front_swept_edge_contacts(
            roles,
            start,
            end,
            raster);

    ASSERT_EQ(contacts.size(), 2U);
    EXPECT_EQ(contacts[0].component_index, 0U);
    EXPECT_EQ(contacts[0].edge_index, 0U);
    EXPECT_EQ(contacts[0].role, FireFrontRole::Outer);
    EXPECT_EQ(contacts[0].cell.i, 2U);
    EXPECT_EQ(contacts[0].cell.j, 1U);
    EXPECT_EQ(contacts[0].overlap_area_m2, SweepReal(1));

    EXPECT_EQ(contacts[1].component_index, 0U);
    EXPECT_EQ(contacts[1].edge_index, 2U);
    EXPECT_EQ(contacts[1].cell.i, 2U);
    EXPECT_EQ(contacts[1].cell.j, 1U);
    EXPECT_EQ(contacts[1].overlap_area_m2, SweepReal(1));

    EXPECT_THROW(
        ERFFire::collective_require_raw_fire_front_swept_edges_avoid_nonburnable(
            roles,
            start,
            end,
            raster,
            "transient sweep test"),
        std::runtime_error);
}

TEST(FireFuelBarrier, SweptEdgesAllowOneSidedBoundaryTangency)
{
    const FireFuelRaster raster =
        make_sweep_barrier_raster(
            sweep_full_height_barrier());
    const std::vector<FireFrontRole> roles{FireFrontRole::Outer};
    const std::vector<std::vector<FireVec2>> start{
        {
            {SweepReal(0.5), SweepReal(0.5)},
            {SweepReal(2.0), SweepReal(0.5)},
            {SweepReal(2.0), SweepReal(1.5)},
            {SweepReal(0.5), SweepReal(1.5)}
        }};
    const std::vector<std::vector<FireVec2>> end{
        {
            {SweepReal(0.5), SweepReal(1.0)},
            {SweepReal(2.0), SweepReal(1.0)},
            {SweepReal(2.0), SweepReal(2.0)},
            {SweepReal(0.5), SweepReal(2.0)}
        }};

    EXPECT_TRUE(
        ERFFire::collective_nonburnable_raw_front_swept_edge_contacts(
            roles,
            start,
            end,
            raster).empty());
    EXPECT_NO_THROW(
        ERFFire::collective_require_raw_fire_front_swept_edges_avoid_nonburnable(
            roles,
            start,
            end,
            raster,
            "boundary tangency sweep"));
}

TEST(FireFuelBarrier, SweptEdgesRejectMismatchedVertexCorrespondence)
{
    const FireFuelRaster raster = make_sweep_barrier_raster({});
    const std::vector<FireFrontRole> roles{FireFrontRole::Outer};
    const std::vector<std::vector<FireVec2>> start{
        {
            {SweepReal(0.5), SweepReal(0.5)},
            {SweepReal(1.5), SweepReal(0.5)},
            {SweepReal(1.0), SweepReal(1.5)}
        }};
    const std::vector<std::vector<FireVec2>> end{
        {
            {SweepReal(0.5), SweepReal(0.5)},
            {SweepReal(1.5), SweepReal(0.5)},
            {SweepReal(1.5), SweepReal(1.5)},
            {SweepReal(0.5), SweepReal(1.5)}
        }};

    EXPECT_THROW(
        (void)ERFFire::collective_nonburnable_raw_front_swept_edge_contacts(
            roles,
            start,
            end,
            raster),
        std::invalid_argument);
}

TEST(FireFuelBarrier, SweptEdgesAllBurnableRemainEmpty)
{
    const FireFuelRaster raster = make_sweep_barrier_raster({});
    const FireFront start(
        std::vector<FireFrontComponent>{
            {
                FireFrontRole::Outer,
                FirePerimeter(
                    std::vector<FireVec2>{
                        {SweepReal(0.2), SweepReal(0.2)},
                        {SweepReal(3.8), SweepReal(0.2)},
                        {SweepReal(3.8), SweepReal(0.8)},
                        {SweepReal(0.2), SweepReal(0.8)}
                    })
            }
        });
    const FireFront end(
        std::vector<FireFrontComponent>{
            {
                FireFrontRole::Outer,
                FirePerimeter(
                    std::vector<FireVec2>{
                        {SweepReal(0.2), SweepReal(3.2)},
                        {SweepReal(3.8), SweepReal(3.2)},
                        {SweepReal(3.8), SweepReal(3.8)},
                        {SweepReal(0.2), SweepReal(3.8)}
                    })
            }
        });

    EXPECT_TRUE(
        ERFFire::collective_nonburnable_front_swept_edge_contacts(
            start,
            end,
            raster).empty());
}

TEST(FireFuelBarrier, FinalFixedTopologyContractRejectsTransientSweptEdge)
{
    const FireFuelRaster raster =
        make_sweep_barrier_raster({{2U, 1U}});
    const FireFront initial = make_transient_sweep_front();

    int calls = 0;
    EXPECT_THROW(
        (void)ERFFire::advance_front_rk2_batched_constrained_by_nonburnable(
            initial,
            SweepReal(0),
            SweepReal(1),
            transient_sweep_speed(calls),
            raster),
        std::runtime_error);
    EXPECT_EQ(calls, 2);
}

TEST(FireFuelBarrier, FinalTopologyContractRejectsTransientSweptEdge)
{
    const FireFuelRaster raster =
        make_sweep_barrier_raster({{2U, 1U}});
    const FireFront initial = make_transient_sweep_front();

    int calls = 0;
    EXPECT_THROW(
        (void)ERFFire::advance_front_rk2_batched_until_topology_event_constrained_by_nonburnable(
            initial,
            SweepReal(0),
            SweepReal(1),
            transient_sweep_speed(calls),
            raster),
        std::runtime_error);
    EXPECT_EQ(calls, 2);
}

} // namespace

#endif // ERF_GTEST_FIRE_FUEL_BARRIER_SWEEP_I_H_
