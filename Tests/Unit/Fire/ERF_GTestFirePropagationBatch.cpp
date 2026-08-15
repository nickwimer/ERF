#include <ERF_FirePerimeter.H>
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
