#ifndef ERF_GTEST_FIRE_FUEL_BARRIER_TOPOLOGY_I_H_
#define ERF_GTEST_FIRE_FUEL_BARRIER_TOPOLOGY_I_H_

#include <ERF_FireFuelBarrierTopology.H>
#include <ERF_FireFrontTopology.H>

#include <AMReX_ParallelDescriptor.H>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{

using BarrierReal = amrex::Real;
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
make_topology_barrier_raster(
    const std::vector<std::pair<std::size_t, std::size_t>>& nonburnable)
{
    const FireCartesianRasterGeometry2D geometry{
        4,
        4,
        BarrierReal(0),
        BarrierReal(0),
        BarrierReal(1),
        BarrierReal(1)};

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
                        BarrierReal(0.08));
                }
            }
        }
    }

    return FireFuelRaster::collective_from_io_rank_state(
        geometry,
        state);
}

FireFront
make_topology_no_event_front()
{
    return FireFront(
        std::vector<FireFrontComponent>{
            {
                FireFrontRole::Outer,
                FirePerimeter(
                    std::vector<FireVec2>{
                        {BarrierReal(0.5), BarrierReal(1.25)},
                        {BarrierReal(1.5), BarrierReal(1.25)},
                        {BarrierReal(1.5), BarrierReal(2.25)},
                        {BarrierReal(0.5), BarrierReal(2.25)}
                    })
            }
        });
}

FireFront
make_topology_self_contact_front()
{
    return FireFront(
        std::vector<FireFrontComponent>{
            {
                FireFrontRole::Outer,
                FirePerimeter(
                    std::vector<FireVec2>{
                        {BarrierReal(0.0), BarrierReal(0.0)},
                        {BarrierReal(3.0), BarrierReal(0.0)},
                        {BarrierReal(3.0), BarrierReal(0.6)},
                        {BarrierReal(0.9), BarrierReal(0.6)},
                        {BarrierReal(0.9), BarrierReal(2.4)},
                        {BarrierReal(2.1), BarrierReal(2.4)},
                        {BarrierReal(2.1), BarrierReal(0.66)},
                        {BarrierReal(2.4), BarrierReal(0.66)},
                        {BarrierReal(2.7), BarrierReal(0.66)},
                        {BarrierReal(2.7), BarrierReal(2.4)},
                        {BarrierReal(3.0), BarrierReal(2.4)},
                        {BarrierReal(3.0), BarrierReal(3.0)},
                        {BarrierReal(0.0), BarrierReal(3.0)}
                    })
            }
        });
}

ERFFire::NormalSpeedBatchFunction
topology_self_contact_speed()
{
    return [](
        const std::vector<FireVec2>& positions,
        const std::vector<FireVec2>& normals,
        BarrierReal) {
        if (positions.size() != normals.size()) {
            throw std::logic_error(
                "barrier topology test positions/normals size mismatch");
        }

        std::vector<BarrierReal> speeds(
            positions.size(), BarrierReal(0));
        for (std::size_t index = 0;
             index < positions.size();
             ++index) {
            if (positions[index].x == BarrierReal(2.4)
                && positions[index].y < BarrierReal(0.9)) {
                speeds[index] = BarrierReal(0.3);
            }
        }
        return speeds;
    };
}

void
expect_same_topology_vertices(
    const std::vector<std::vector<FireVec2>>& actual,
    const std::vector<std::vector<FireVec2>>& expected)
{
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t component = 0;
         component < expected.size();
         ++component) {
        ASSERT_EQ(actual[component].size(), expected[component].size());
        for (std::size_t vertex = 0;
             vertex < expected[component].size();
             ++vertex) {
            EXPECT_EQ(actual[component][vertex].x, expected[component][vertex].x);
            EXPECT_EQ(actual[component][vertex].y, expected[component][vertex].y);
        }
    }
}

TEST(FireFuelBarrier, TopologyAwareNoContactMatchesExistingBitwise)
{
    const FireFuelRaster raster = make_topology_barrier_raster({});
    const FireFront initial = make_topology_no_event_front();
    const auto speeds = [](
        const std::vector<FireVec2>& positions,
        const std::vector<FireVec2>& normals,
        BarrierReal) {
        if (positions.size() != normals.size()) {
            throw std::logic_error(
                "barrier topology test positions/normals size mismatch");
        }
        return std::vector<BarrierReal>(
            positions.size(), BarrierReal(0.1));
    };

    const auto reference =
        ERFFire::advance_front_rk2_batched_until_topology_event(
            initial,
            BarrierReal(1),
            BarrierReal(0.1),
            speeds);
    const auto constrained =
        ERFFire::advance_front_rk2_batched_until_topology_event_avoiding_nonburnable(
            initial,
            BarrierReal(1),
            BarrierReal(0.1),
            speeds,
            raster);

    EXPECT_FALSE(reference.topology_event.has_value());
    EXPECT_FALSE(constrained.topology_event.has_value());
    EXPECT_EQ(constrained.advanced_dt_s, reference.advanced_dt_s);
    expect_same_topology_vertices(
        constrained.terminal_vertices_m,
        reference.terminal_vertices_m);

    ASSERT_TRUE(reference.completed_front.has_value());
    ASSERT_TRUE(constrained.completed_front.has_value());
    const auto& expected =
        reference.completed_front->components()[0].perimeter.vertices_m();
    const auto& actual =
        constrained.completed_front->components()[0].perimeter.vertices_m();
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t vertex = 0;
         vertex < expected.size();
         ++vertex) {
        EXPECT_EQ(actual[vertex].x, expected[vertex].x);
        EXPECT_EQ(actual[vertex].y, expected[vertex].y);
    }
}

TEST(FireFuelBarrier, TopologyAwareSelfContactMatchesExistingEvent)
{
    const FireFuelRaster raster = make_topology_barrier_raster({});
    const FireFront initial = make_topology_self_contact_front();
    const auto speeds = topology_self_contact_speed();

    const auto reference =
        ERFFire::advance_front_rk2_batched_until_topology_event(
            initial,
            BarrierReal(4),
            BarrierReal(0.3),
            speeds);
    const auto constrained =
        ERFFire::advance_front_rk2_batched_until_topology_event_avoiding_nonburnable(
            initial,
            BarrierReal(4),
            BarrierReal(0.3),
            speeds,
            raster);

    ASSERT_TRUE(reference.topology_event.has_value());
    ASSERT_TRUE(constrained.topology_event.has_value());
    EXPECT_FALSE(reference.completed_front.has_value());
    EXPECT_FALSE(constrained.completed_front.has_value());
    EXPECT_EQ(constrained.advanced_dt_s, reference.advanced_dt_s);
    EXPECT_EQ(
        constrained.topology_event->component_index,
        reference.topology_event->component_index);
    EXPECT_EQ(
        constrained.topology_event->role,
        reference.topology_event->role);
    EXPECT_EQ(
        constrained.topology_event->type,
        reference.topology_event->type);
    EXPECT_EQ(
        constrained.topology_event->motion_fraction,
        reference.topology_event->motion_fraction);
    EXPECT_EQ(
        constrained.topology_event->collision.pinch.first_edge_index,
        reference.topology_event->collision.pinch.first_edge_index);
    EXPECT_EQ(
        constrained.topology_event->collision.pinch.first_edge_fraction,
        reference.topology_event->collision.pinch.first_edge_fraction);
    EXPECT_EQ(
        constrained.topology_event->collision.pinch.second_edge_index,
        reference.topology_event->collision.pinch.second_edge_index);
    EXPECT_EQ(
        constrained.topology_event->collision.pinch.second_edge_fraction,
        reference.topology_event->collision.pinch.second_edge_fraction);
    expect_same_topology_vertices(
        constrained.terminal_vertices_m,
        reference.terminal_vertices_m);
}

TEST(FireFuelBarrier, TopologyAwareAdvanceClipsAgainstBarrier)
{
    const FireFuelRaster raster =
        make_topology_barrier_raster({
            {2U, 0U}, {2U, 1U}, {2U, 2U}, {2U, 3U}});
    const FireFront initial = make_topology_no_event_front();

    const auto speeds = [](
        const std::vector<FireVec2>& positions,
        const std::vector<FireVec2>& normals,
        BarrierReal) {
        if (positions.size() != normals.size()) {
            throw std::logic_error(
                "barrier topology test positions/normals size mismatch");
        }
        std::vector<BarrierReal> result(
            positions.size(), BarrierReal(0));
        for (std::size_t index = 0;
             index < positions.size();
             ++index) {
            if (positions[index].x > BarrierReal(1)) {
                result[index] = BarrierReal(1);
            }
        }
        return result;
    };

    const auto constrained =
        ERFFire::advance_front_rk2_batched_until_topology_event_avoiding_nonburnable(
            initial,
            BarrierReal(0),
            BarrierReal(1),
            speeds,
            raster);

    EXPECT_FALSE(constrained.topology_event.has_value());
    ASSERT_TRUE(constrained.completed_front.has_value());
    const auto& vertices =
        constrained.completed_front->components()[0].perimeter.vertices_m();
    ASSERT_EQ(vertices.size(), 4U);
    EXPECT_EQ(vertices[1].x, BarrierReal(2));
    EXPECT_EQ(vertices[2].x, BarrierReal(2));
    EXPECT_TRUE(
        ERFFire::collective_nonburnable_front_edge_contacts(
            *constrained.completed_front,
            raster).empty());
}

TEST(FireFuelBarrier, RawTopologyEventEdgesRejectBarrierPenetration)
{
    const FireFuelRaster raster =
        make_topology_barrier_raster({{2U, 1U}});
    const std::vector<FireFrontRole> roles{FireFrontRole::Outer};
    const std::vector<std::vector<FireVec2>> raw_vertices{
        {
            {BarrierReal(1.5), BarrierReal(1.25)},
            {BarrierReal(3.5), BarrierReal(1.25)},
            {BarrierReal(3.5), BarrierReal(1.75)},
            {BarrierReal(1.5), BarrierReal(1.75)}
        }};

    EXPECT_THROW(
        ERFFire::collective_require_raw_fire_front_edges_avoid_nonburnable(
            roles,
            raw_vertices,
            raster,
            "raw topology test"),
        std::runtime_error);
}

TEST(FireFuelBarrier, ResolvedSelfContactMustRemainBarrierLegal)
{
    const FireFuelRaster all_burnable =
        make_topology_barrier_raster({});
    const FireFront initial = make_topology_self_contact_front();
    const auto result =
        ERFFire::advance_front_rk2_batched_until_topology_event_avoiding_nonburnable(
            initial,
            BarrierReal(4),
            BarrierReal(0.3),
            topology_self_contact_speed(),
            all_burnable);

    ASSERT_TRUE(result.topology_event.has_value());
    ASSERT_EQ(
        result.topology_event->type,
        ERFFire::FireFrontComponentTopologyEventType::SelfContact);
    ASSERT_EQ(result.terminal_vertices_m.size(), 1U);

    const FireFront resolved =
        ERFFire::split_perimeter_at_pinch(
            result.terminal_vertices_m[0],
            result.topology_event->collision.pinch);

    EXPECT_NO_THROW(
        ERFFire::collective_require_resolved_fire_topology_avoids_nonburnable(
            resolved,
            all_burnable));

    const FireFuelRaster blocked =
        make_topology_barrier_raster({{0U, 1U}});
    EXPECT_THROW(
        ERFFire::collective_require_resolved_fire_topology_avoids_nonburnable(
            resolved,
            blocked),
        std::runtime_error);
}

} // namespace

#endif // ERF_GTEST_FIRE_FUEL_BARRIER_TOPOLOGY_I_H_
