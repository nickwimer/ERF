#include <ERF_FirePerimeter.H>
#include <ERF_FireFrontPropagator.H>
#include <ERF_VectorPerimeterPropagator.H>

#include <gtest/gtest.h>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{

using amrex::Real;
using ERFFire::FirePerimeter;
using ERFFire::FireVec2;

FirePerimeter
test_perimeter()
{
    return FirePerimeter(
        std::vector<FireVec2>{
            {Real(0.0), Real(0.0)},
            {Real(2.0), Real(0.0)},
            {Real(2.0), Real(1.0)},
            {Real(0.0), Real(1.0)}});
}

Real
test_speed(
    const FireVec2& position,
    const FireVec2& normal,
    Real time_s)
{
    return Real(0.3)
        + Real(0.02) * position.x
        + Real(0.01) * position.y
        + Real(0.005) * normal.x
        - Real(0.004) * normal.y
        + Real(0.001) * time_s;
}

} // namespace

TEST(
    FirePropagationBatch,
    MatchesScalarRk2BitwiseAndBatchesExactlyTwoStages)
{
    const FirePerimeter initial = test_perimeter();
    const Real time_s = Real(3.0);
    const Real dt_s = Real(0.5);

    const FirePerimeter scalar =
        ERFFire::advance_perimeter_rk2(
            initial,
            time_s,
            dt_s,
            test_speed);

    int batch_calls = 0;
    std::vector<std::size_t> batch_sizes;
    const FirePerimeter batched =
        ERFFire::advance_perimeter_rk2_batched(
            initial,
            time_s,
            dt_s,
            [&batch_calls, &batch_sizes](
                const std::vector<FireVec2>& positions,
                const std::vector<FireVec2>& normals,
                Real stage_time_s) {
                ++batch_calls;
                batch_sizes.push_back(positions.size());
                if (positions.size() != normals.size()) {
                    throw std::logic_error(
                        "test batch positions/normals size mismatch");
                }

                std::vector<Real> speeds;
                speeds.reserve(positions.size());
                for (std::size_t i = 0; i < positions.size(); ++i) {
                    speeds.push_back(
                        test_speed(
                            positions[i],
                            normals[i],
                            stage_time_s));
                }
                return speeds;
            });

    EXPECT_EQ(batch_calls, 2);
    ASSERT_EQ(batch_sizes.size(), std::size_t(2));
    EXPECT_EQ(batch_sizes[0], initial.size());
    EXPECT_EQ(batch_sizes[1], initial.size());
    ASSERT_EQ(batched.size(), scalar.size());

    for (std::size_t i = 0; i < scalar.size(); ++i) {
        EXPECT_DOUBLE_EQ(
            batched.vertices_m()[i].x,
            scalar.vertices_m()[i].x);
        EXPECT_DOUBLE_EQ(
            batched.vertices_m()[i].y,
            scalar.vertices_m()[i].y);
    }
}

TEST(
    FirePropagationBatch,
    RejectsInvalidCallbackResults)
{
    const FirePerimeter initial = test_perimeter();

    EXPECT_THROW(
        (void)ERFFire::advance_perimeter_rk2_batched(
            initial,
            Real(0.0),
            Real(1.0),
            ERFFire::NormalSpeedBatchFunction{}),
        std::invalid_argument);

    EXPECT_THROW(
        (void)ERFFire::advance_perimeter_rk2_batched(
            initial,
            Real(0.0),
            Real(1.0),
            [](const std::vector<FireVec2>& positions,
               const std::vector<FireVec2>&,
               Real) {
                return std::vector<Real>(
                    positions.size() - 1,
                    Real(1.0));
            }),
        std::invalid_argument);

    EXPECT_THROW(
        (void)ERFFire::advance_perimeter_rk2_batched(
            initial,
            Real(0.0),
            Real(1.0),
            [](const std::vector<FireVec2>& positions,
               const std::vector<FireVec2>&,
               Real) {
                return std::vector<Real>(
                    positions.size(),
                    std::numeric_limits<Real>::quiet_NaN());
            }),
        std::invalid_argument);
}

TEST(
    FirePropagationBatch,
    StopsAtFirstTopologyEventAndReturnsSplitFront)
{
    const FirePerimeter initial(
        std::vector<FireVec2>{
            {Real(0.0), Real(0.0)},
            {Real(10.0), Real(0.0)},
            {Real(10.0), Real(2.0)},
            {Real(3.0), Real(2.0)},
            {Real(3.0), Real(8.0)},
            {Real(7.0), Real(8.0)},
            {Real(7.0), Real(2.2)},
            {Real(8.0), Real(2.2)},
            {Real(9.0), Real(2.2)},
            {Real(9.0), Real(8.0)},
            {Real(10.0), Real(8.0)},
            {Real(10.0), Real(10.0)},
            {Real(0.0), Real(10.0)}
        });

    int batch_calls = 0;

    const ERFFire::FireFrontAdvanceResult result =
        ERFFire::
            advance_perimeter_rk2_batched_until_topology_event(
                initial,
                Real(4.0),
                Real(0.3),
                [&batch_calls](
                    const std::vector<FireVec2>& positions,
                    const std::vector<FireVec2>& normals,
                    Real) {
                    ++batch_calls;

                    if (positions.size() != normals.size()) {
                        throw std::logic_error(
                            "test batch positions/normals size mismatch");
                    }

                    std::vector<Real> speeds(
                        positions.size(),
                        Real(0.0));

                    for (std::size_t i = 0;
                         i < positions.size();
                         ++i) {
                        if (positions[i].x == Real(8.0)
                            && positions[i].y < Real(3.0)) {
                            speeds[i] = Real(1.0);
                        }
                    }

                    return speeds;
                });

    EXPECT_EQ(batch_calls, 2);
    ASSERT_TRUE(result.topology_event.has_value());

    EXPECT_NEAR(
        static_cast<double>(result.advanced_dt_s),
        0.2,
        1.0e-12);
    EXPECT_NEAR(
        static_cast<double>(
            result.topology_event->motion_fraction),
        2.0 / 3.0,
        1.0e-12);

    EXPECT_EQ(
        result.topology_event->pinch.first_edge_index,
        2U);
    EXPECT_NEAR(
        static_cast<double>(
            result.topology_event
                ->pinch.first_edge_fraction),
        2.0 / 7.0,
        1.0e-12);
    EXPECT_EQ(
        result.topology_event->pinch.second_edge_index,
        7U);
    EXPECT_NEAR(
        static_cast<double>(
            result.topology_event
                ->pinch.second_edge_fraction),
        0.0,
        1.0e-12);

    ASSERT_EQ(
        result.terminal_vertices_m.size(),
        initial.size());
    EXPECT_NEAR(
        static_cast<double>(
            result.terminal_vertices_m[7].x),
        8.0,
        1.0e-12);
    EXPECT_NEAR(
        static_cast<double>(
            result.terminal_vertices_m[7].y),
        2.0,
        1.0e-12);

    ASSERT_EQ(result.front.components().size(), 2U);
    EXPECT_EQ(
        result.front.components()[0].role,
        ERFFire::FireFrontRole::Outer);
    EXPECT_EQ(
        result.front.components()[1].role,
        ERFFire::FireFrontRole::Hole);

    EXPECT_NEAR(
        static_cast<double>(
            result.front.components()[0]
                .perimeter.area_m2()),
        93.9,
        1.0e-12);
    EXPECT_NEAR(
        static_cast<double>(
            result.front.components()[1]
                .perimeter.area_m2()),
        24.1,
        1.0e-12);
    EXPECT_NEAR(
        static_cast<double>(
            result.front.burned_area_m2()),
        69.8,
        1.0e-12);
}

TEST(
    FirePropagationBatch,
    TopologyAwareNoEventMatchesLegacyBitwise)
{
    const FirePerimeter initial = test_perimeter();
    const Real time_s = Real(3.0);
    const Real dt_s = Real(0.5);

    const auto speed_batch =
        [](const std::vector<FireVec2>& positions,
           const std::vector<FireVec2>& normals,
           Real stage_time_s) {
            if (positions.size() != normals.size()) {
                throw std::logic_error(
                    "test batch positions/normals size mismatch");
            }

            std::vector<Real> speeds;
            speeds.reserve(positions.size());
            for (std::size_t i = 0; i < positions.size(); ++i) {
                speeds.push_back(
                    test_speed(
                        positions[i],
                        normals[i],
                        stage_time_s));
            }
            return speeds;
        };

    const FirePerimeter legacy =
        ERFFire::advance_perimeter_rk2_batched(
            initial,
            time_s,
            dt_s,
            speed_batch);

    int topology_batch_calls = 0;
    const ERFFire::FireFrontAdvanceResult result =
        ERFFire::
            advance_perimeter_rk2_batched_until_topology_event(
                initial,
                time_s,
                dt_s,
                [&topology_batch_calls, &speed_batch](
                    const std::vector<FireVec2>& positions,
                    const std::vector<FireVec2>& normals,
                    Real stage_time_s) {
                    ++topology_batch_calls;
                    return speed_batch(
                        positions,
                        normals,
                        stage_time_s);
                });

    EXPECT_EQ(topology_batch_calls, 2);
    EXPECT_FALSE(result.topology_event.has_value());
    EXPECT_DOUBLE_EQ(result.advanced_dt_s, dt_s);

    ASSERT_EQ(result.front.components().size(), 1U);
    EXPECT_EQ(
        result.front.components()[0].role,
        ERFFire::FireFrontRole::Outer);

    const FirePerimeter& topology_perimeter =
        result.front.components()[0].perimeter;

    ASSERT_EQ(result.terminal_vertices_m.size(), legacy.size());
    ASSERT_EQ(topology_perimeter.size(), legacy.size());

    for (std::size_t i = 0; i < legacy.size(); ++i) {
        EXPECT_DOUBLE_EQ(
            result.terminal_vertices_m[i].x,
            legacy.vertices_m()[i].x);
        EXPECT_DOUBLE_EQ(
            result.terminal_vertices_m[i].y,
            legacy.vertices_m()[i].y);
        EXPECT_DOUBLE_EQ(
            topology_perimeter.vertices_m()[i].x,
            legacy.vertices_m()[i].x);
        EXPECT_DOUBLE_EQ(
            topology_perimeter.vertices_m()[i].y,
            legacy.vertices_m()[i].y);
    }

    EXPECT_DOUBLE_EQ(
        result.front.burned_area_m2(),
        legacy.area_m2());
}

TEST(
    FirePropagationBatch,
    AdvancesOuterAndHoleWithRoleAwareNormalsInTwoBatches)
{
    const ERFFire::FireFront initial(
        std::vector<ERFFire::FireFrontComponent>{
            {
                ERFFire::FireFrontRole::Outer,
                FirePerimeter(
                    std::vector<FireVec2>{
                        {Real(-5.0), Real(-5.0)},
                        {Real( 5.0), Real(-5.0)},
                        {Real( 5.0), Real( 5.0)},
                        {Real(-5.0), Real( 5.0)}
                    })
            },
            {
                ERFFire::FireFrontRole::Hole,
                FirePerimeter(
                    std::vector<FireVec2>{
                        {Real(-1.0), Real(-1.0)},
                        {Real( 1.0), Real(-1.0)},
                        {Real( 1.0), Real( 1.0)},
                        {Real(-1.0), Real( 1.0)}
                    })
            }
        });

    const Real initial_outer_area =
        initial.components()[0]
            .perimeter.area_m2();
    const Real initial_hole_area =
        initial.components()[1]
            .perimeter.area_m2();
    const Real initial_burned_area =
        initial.burned_area_m2();

    std::vector<FireVec2> expected_initial_normals;
    expected_initial_normals.reserve(8);

    for (std::size_t component = 0;
         component < initial.components().size();
         ++component) {
        const auto& perimeter =
            initial.components()[component].perimeter;

        for (std::size_t vertex = 0;
             vertex < perimeter.size();
             ++vertex) {
            expected_initial_normals.push_back(
                initial.spread_normal_unit(
                    component,
                    vertex));
        }
    }

    int batch_calls = 0;

    const ERFFire::FireFront advanced =
        ERFFire::advance_front_rk2_batched(
            initial,
            Real(4.0),
            Real(0.1),
            [&batch_calls, &expected_initial_normals](
                const std::vector<FireVec2>& positions,
                const std::vector<FireVec2>& normals,
                Real) {
                ++batch_calls;

                EXPECT_EQ(positions.size(), 8U);
                EXPECT_EQ(normals.size(), 8U);

                if (batch_calls == 1) {
                    EXPECT_EQ(
                        normals.size(),
                        expected_initial_normals.size());

                    for (std::size_t i = 0;
                         i < normals.size();
                         ++i) {
                        EXPECT_NEAR(
                            static_cast<double>(
                                normals[i].x),
                            static_cast<double>(
                                expected_initial_normals[i].x),
                            1.0e-14);
                        EXPECT_NEAR(
                            static_cast<double>(
                                normals[i].y),
                            static_cast<double>(
                                expected_initial_normals[i].y),
                            1.0e-14);
                    }
                }

                return std::vector<Real>(
                    positions.size(),
                    Real(1.0));
            });

    EXPECT_EQ(batch_calls, 2);

    ASSERT_EQ(
        advanced.components().size(),
        2U);
    EXPECT_EQ(
        advanced.components()[0].role,
        ERFFire::FireFrontRole::Outer);
    EXPECT_EQ(
        advanced.components()[1].role,
        ERFFire::FireFrontRole::Hole);

    EXPECT_GT(
        advanced.components()[0]
            .perimeter.area_m2(),
        initial_outer_area);
    EXPECT_LT(
        advanced.components()[1]
            .perimeter.area_m2(),
        initial_hole_area);
    EXPECT_GT(
        advanced.burned_area_m2(),
        initial_burned_area);
}

TEST(
    FirePropagationBatch,
    StopsAtHoleAreaCollapseBeforeClockwiseMidpoint)
{
    const ERFFire::FireFront initial(
        std::vector<ERFFire::FireFrontComponent>{
            {
                ERFFire::FireFrontRole::Outer,
                FirePerimeter(
                    std::vector<FireVec2>{
                        {Real(-10.0), Real(-10.0)},
                        {Real( 10.0), Real(-10.0)},
                        {Real( 10.0), Real( 10.0)},
                        {Real(-10.0), Real( 10.0)}
                    })
            },
            {
                ERFFire::FireFrontRole::Hole,
                FirePerimeter(
                    std::vector<FireVec2>{
                        {Real(0.0), Real(0.0)},
                        {Real(4.0), Real(0.0)},
                        {Real(0.0), Real(4.0)}
                    })
            }
        });

    int batch_calls = 0;
    ERFFire::FireFrontTopologyAdvanceResult result;

    EXPECT_NO_THROW(
        result =
            ERFFire::
                advance_front_rk2_batched_until_topology_event(
                    initial,
                    Real(4.0),
                    Real(2.0),
                    [&batch_calls](
                        const std::vector<FireVec2>& positions,
                        const std::vector<FireVec2>& normals,
                        Real) {
                        ++batch_calls;

                        EXPECT_EQ(positions.size(), 7U);
                        EXPECT_EQ(normals.size(), 7U);

                        std::vector<Real> speeds(
                            positions.size(),
                            Real(0.0));
                        speeds[4] = Real(4.0);
                        return speeds;
                    }));

    EXPECT_EQ(batch_calls, 1);
    EXPECT_FALSE(
        result.completed_front.has_value());
    ASSERT_TRUE(
        result.topology_event.has_value());
    EXPECT_EQ(
        result.topology_event->component_index,
        1U);
    EXPECT_EQ(
        result.topology_event->role,
        ERFFire::FireFrontRole::Hole);
    EXPECT_EQ(
        result.topology_event->type,
        ERFFire::FireFrontComponentTopologyEventType::
            HoleExtinction);
    EXPECT_NEAR(
        static_cast<double>(
            result.topology_event
                ->motion_fraction),
        0.3535533905932738,
        1.0e-12);
    EXPECT_NEAR(
        static_cast<double>(
            result.advanced_dt_s),
        0.7071067811865476,
        1.0e-12);
}

TEST(
    FirePropagationBatch,
    StopsAtEarliestFrontComponentSelfContact)
{
    const FirePerimeter stationary_outer(
        std::vector<FireVec2>{
            {Real(-20.0), Real(-5.0)},
            {Real(-10.0), Real(-5.0)},
            {Real(-10.0), Real( 5.0)},
            {Real(-20.0), Real( 5.0)}
        });

    const FirePerimeter pinching_outer(
        std::vector<FireVec2>{
            {Real(0.0), Real(0.0)},
            {Real(10.0), Real(0.0)},
            {Real(10.0), Real(2.0)},
            {Real(3.0), Real(2.0)},
            {Real(3.0), Real(8.0)},
            {Real(7.0), Real(8.0)},
            {Real(7.0), Real(2.2)},
            {Real(8.0), Real(2.2)},
            {Real(9.0), Real(2.2)},
            {Real(9.0), Real(8.0)},
            {Real(10.0), Real(8.0)},
            {Real(10.0), Real(10.0)},
            {Real(0.0), Real(10.0)}
        });

    const ERFFire::FireFront initial(
        std::vector<ERFFire::FireFrontComponent>{
            {
                ERFFire::FireFrontRole::Outer,
                stationary_outer
            },
            {
                ERFFire::FireFrontRole::Outer,
                pinching_outer
            }
        });

    int batch_calls = 0;

    const auto result =
        ERFFire::
            advance_front_rk2_batched_until_topology_event(
                initial,
                Real(4.0),
                Real(0.3),
                [&batch_calls](
                    const std::vector<FireVec2>& positions,
                    const std::vector<FireVec2>& normals,
                    Real) {
                    ++batch_calls;

                    EXPECT_EQ(
                        positions.size(),
                        17U);
                    EXPECT_EQ(
                        normals.size(),
                        17U);

                    std::vector<Real> speeds(
                        positions.size(),
                        Real(0.0));

                    for (std::size_t i = 0;
                         i < positions.size();
                         ++i) {
                        if (positions[i].x == Real(8.0)
                            && positions[i].y < Real(3.0)) {
                            speeds[i] = Real(1.0);
                        }
                    }

                    return speeds;
                });

    EXPECT_EQ(batch_calls, 2);
    EXPECT_FALSE(
        result.completed_front.has_value());
    ASSERT_TRUE(
        result.topology_event.has_value());

    EXPECT_NEAR(
        static_cast<double>(
            result.advanced_dt_s),
        0.2,
        1.0e-12);

    EXPECT_EQ(
        result.topology_event
            ->component_index,
        1U);
    EXPECT_EQ(
        result.topology_event->role,
        ERFFire::FireFrontRole::Outer);
    EXPECT_NEAR(
        static_cast<double>(
            result.topology_event
                ->collision.motion_fraction),
        2.0 / 3.0,
        1.0e-12);

    ASSERT_EQ(
        result.terminal_vertices_m.size(),
        2U);
    ASSERT_EQ(
        result.terminal_vertices_m[0].size(),
        stationary_outer.size());
    ASSERT_EQ(
        result.terminal_vertices_m[1].size(),
        pinching_outer.size());

    for (std::size_t i = 0;
         i < stationary_outer.size();
         ++i) {
        EXPECT_DOUBLE_EQ(
            result.terminal_vertices_m[0][i].x,
            stationary_outer.vertices_m()[i].x);
        EXPECT_DOUBLE_EQ(
            result.terminal_vertices_m[0][i].y,
            stationary_outer.vertices_m()[i].y);
    }

    EXPECT_NEAR(
        static_cast<double>(
            result.terminal_vertices_m[1][7].x),
        8.0,
        1.0e-12);
    EXPECT_NEAR(
        static_cast<double>(
            result.terminal_vertices_m[1][7].y),
        2.0,
        1.0e-12);
}


TEST(
    FirePropagationBatch,
    ExactlyTiedComponentSelfContactsChooseFirstComponentDeterministically)
{
    const auto make_pinching_outer =
        [](Real x_offset) {
            return FirePerimeter(
                std::vector<FireVec2>{
                    {x_offset + Real(0.0), Real(0.0)},
                    {x_offset + Real(10.0), Real(0.0)},
                    {x_offset + Real(10.0), Real(2.0)},
                    {x_offset + Real(3.0), Real(2.0)},
                    {x_offset + Real(3.0), Real(8.0)},
                    {x_offset + Real(7.0), Real(8.0)},
                    {x_offset + Real(7.0), Real(2.2)},
                    {x_offset + Real(8.0), Real(2.2)},
                    {x_offset + Real(9.0), Real(2.2)},
                    {x_offset + Real(9.0), Real(8.0)},
                    {x_offset + Real(10.0), Real(8.0)},
                    {x_offset + Real(10.0), Real(10.0)},
                    {x_offset + Real(0.0), Real(10.0)}
                });
        };

    const FirePerimeter first =
        make_pinching_outer(Real(0.0));
    const FirePerimeter second =
        make_pinching_outer(Real(20.0));

    const ERFFire::FireFront initial(
        std::vector<ERFFire::FireFrontComponent>{
            {
                ERFFire::FireFrontRole::Outer,
                first
            },
            {
                ERFFire::FireFrontRole::Outer,
                second
            }
        });

    int batch_calls = 0;

    const auto result =
        ERFFire::
            advance_front_rk2_batched_until_topology_event(
                initial,
                Real(4.0),
                Real(0.3),
                [&batch_calls](
                    const std::vector<FireVec2>& positions,
                    const std::vector<FireVec2>& normals,
                    Real) {
                    ++batch_calls;

                    EXPECT_EQ(
                        positions.size(),
                        26U);
                    EXPECT_EQ(
                        normals.size(),
                        26U);

                    std::vector<Real> speeds(
                        positions.size(),
                        Real(0.0));

                    for (std::size_t i = 0;
                         i < positions.size();
                         ++i) {
                        const bool first_driver =
                            std::abs(
                                positions[i].x
                                - Real(8.0))
                                < Real(1.0e-12)
                            && positions[i].y
                                < Real(3.0);

                        const bool second_driver =
                            std::abs(
                                positions[i].x
                                - Real(28.0))
                                < Real(1.0e-12)
                            && positions[i].y
                                < Real(3.0);

                        if (first_driver
                            || second_driver) {
                            speeds[i] = Real(1.0);
                        }
                    }

                    return speeds;
                });

    EXPECT_EQ(batch_calls, 2);
    EXPECT_FALSE(
        result.completed_front.has_value());
    ASSERT_TRUE(
        result.topology_event.has_value());

    EXPECT_NEAR(
        static_cast<double>(
            result.advanced_dt_s),
        0.2,
        1.0e-12);

    EXPECT_EQ(
        result.topology_event
            ->component_index,
        0U);
    EXPECT_EQ(
        result.topology_event->role,
        ERFFire::FireFrontRole::Outer);
    EXPECT_EQ(
        result.topology_event->type,
        ERFFire::FireFrontComponentTopologyEventType::
            SelfContact);

    EXPECT_NEAR(
        static_cast<double>(
            result.topology_event
                ->collision.motion_fraction),
        2.0 / 3.0,
        1.0e-12);

    ASSERT_EQ(
        result.terminal_vertices_m.size(),
        2U);

    EXPECT_NEAR(
        static_cast<double>(
            result.terminal_vertices_m[0][7].y),
        2.0,
        1.0e-12);
    EXPECT_NEAR(
        static_cast<double>(
            result.terminal_vertices_m[1][7].y),
        2.0,
        1.0e-12);
}


TEST(
    FirePropagationBatch,
    StopsAtTopologyEventDuringMidpointMotion)
{
    const FirePerimeter initial(
        std::vector<FireVec2>{
            {Real(0.0), Real(0.0)},
            {Real(10.0), Real(0.0)},
            {Real(10.0), Real(2.0)},
            {Real(3.0), Real(2.0)},
            {Real(3.0), Real(8.0)},
            {Real(7.0), Real(8.0)},
            {Real(7.0), Real(2.2)},
            {Real(8.0), Real(2.2)},
            {Real(9.0), Real(2.2)},
            {Real(9.0), Real(8.0)},
            {Real(10.0), Real(8.0)},
            {Real(10.0), Real(10.0)},
            {Real(0.0), Real(10.0)}
        });

    int batch_calls = 0;

    const ERFFire::FireFrontAdvanceResult result =
        ERFFire::
            advance_perimeter_rk2_batched_until_topology_event(
                initial,
                Real(4.0),
                Real(0.6),
                [&batch_calls](
                    const std::vector<FireVec2>& positions,
                    const std::vector<FireVec2>& normals,
                    Real) {
                    ++batch_calls;

                    if (positions.size() != normals.size()) {
                        throw std::logic_error(
                            "test batch positions/normals size mismatch");
                    }

                    std::vector<Real> speeds(
                        positions.size(),
                        Real(0.0));

                    for (std::size_t i = 0;
                         i < positions.size();
                         ++i) {
                        if (positions[i].x == Real(8.0)
                            && positions[i].y < Real(3.0)) {
                            speeds[i] = Real(1.0);
                        }
                    }

                    return speeds;
                });

    EXPECT_EQ(batch_calls, 1);
    ASSERT_TRUE(result.topology_event.has_value());

    EXPECT_NEAR(
        static_cast<double>(result.advanced_dt_s),
        0.2,
        1.0e-12);
    EXPECT_NEAR(
        static_cast<double>(
            result.topology_event->motion_fraction),
        1.0 / 3.0,
        1.0e-12);

    EXPECT_EQ(
        result.topology_event->pinch.first_edge_index,
        2U);
    EXPECT_NEAR(
        static_cast<double>(
            result.topology_event
                ->pinch.first_edge_fraction),
        2.0 / 7.0,
        1.0e-12);
    EXPECT_EQ(
        result.topology_event->pinch.second_edge_index,
        7U);

    ASSERT_EQ(
        result.terminal_vertices_m.size(),
        initial.size());
    EXPECT_NEAR(
        static_cast<double>(
            result.terminal_vertices_m[7].x),
        8.0,
        1.0e-12);
    EXPECT_NEAR(
        static_cast<double>(
            result.terminal_vertices_m[7].y),
        2.0,
        1.0e-12);

    ASSERT_EQ(result.front.components().size(), 2U);
    EXPECT_EQ(
        result.front.components()[0].role,
        ERFFire::FireFrontRole::Outer);
    EXPECT_EQ(
        result.front.components()[1].role,
        ERFFire::FireFrontRole::Hole);
    EXPECT_NEAR(
        static_cast<double>(
            result.front.components()[0]
                .perimeter.area_m2()),
        93.9,
        1.0e-12);
    EXPECT_NEAR(
        static_cast<double>(
            result.front.components()[1]
                .perimeter.area_m2()),
        24.1,
        1.0e-12);
    EXPECT_NEAR(
        static_cast<double>(
            result.front.burned_area_m2()),
        69.8,
        1.0e-12);
}
