#include <ERF_FireFuelBarrier.H>
#include <ERF_FireFuelBarrierAdvance.H>
#include <ERF_FireTypes.H>

#include <AMReX_ParallelDescriptor.H>
#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef ERF_USE_FIRE
#error "erf_fire_unit_tests must compile with ERF_USE_FIRE"
#endif

namespace
{

using amrex::Real;
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

FireCartesianRasterGeometry2D
barrier_geometry()
{
    return {
        4,
        2,
        Real(0),
        Real(0),
        Real(1),
        Real(1)};
}

FireFuelRaster
make_barrier_raster(
    const std::vector<std::pair<std::size_t, std::size_t>>& nonburnable)
{
    const auto geometry = barrier_geometry();
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
                        Real(0.08));
                }
            }
        }
    }

    return FireFuelRaster::collective_from_io_rank_state(
        geometry,
        state);
}

FireFront
make_barrier_test_front(Real right_x)
{
    return FireFront(
        std::vector<FireFrontComponent>{
            {
                FireFrontRole::Outer,
                FirePerimeter(
                    std::vector<FireVec2>{
                        {Real(0.5), Real(0.75)},
                        {right_x, Real(0.75)},
                        {right_x, Real(1.25)},
                        {Real(0.5), Real(1.25)}
                    })
            }
        });
}

ERFFire::NormalSpeedBatchFunction
east_edge_speed(Real speed)
{
    return [speed](
        const std::vector<FireVec2>& positions,
        const std::vector<FireVec2>& normals,
        Real) {
        if (positions.size() != normals.size()) {
            throw std::logic_error(
                "barrier test positions/normals size mismatch");
        }
        std::vector<Real> result(
            positions.size(), Real(0));
        for (std::size_t index = 0;
             index < positions.size();
             ++index) {
            if (positions[index].x > Real(1)) {
                result[index] = speed;
            }
        }
        return result;
    };
}

TEST(FireTypes, Vec2StoresPhysicalComponents)
{
    const ERFFire::FireVec2 value{1.25, -2.5};

    EXPECT_DOUBLE_EQ(static_cast<double>(value.x), 1.25);
    EXPECT_DOUBLE_EQ(static_cast<double>(value.y), -2.5);
}

TEST(FireTypes, Vec2IsDependencyLightweight)
{
    static_assert(std::is_standard_layout_v<ERFFire::FireVec2>);
    static_assert(std::is_trivially_copyable_v<ERFFire::FireVec2>);

    SUCCEED();
}

TEST(FireFuelBarrier, OrderedSegmentsLocateFirstNonburnableEntry)
{
    const FireFuelRaster raster =
        make_barrier_raster({{2U, 0U}});

    const std::vector<FireVec2> starts{
        {Real(0.25), Real(0.5)},
        {Real(0.25), Real(1.5)},
        {Real(3.75), Real(0.5)}};
    const std::vector<FireVec2> ends{
        {Real(3.75), Real(0.5)},
        {Real(3.75), Real(1.5)},
        {Real(0.25), Real(0.5)}};

    const auto contacts =
        ERFFire::collective_first_nonburnable_contacts(
            raster,
            starts,
            ends);

    ASSERT_EQ(contacts.size(), starts.size());

    ASSERT_TRUE(contacts[0].has_value());
    EXPECT_EQ(contacts[0]->motion_fraction, Real(0.5));
    EXPECT_EQ(contacts[0]->position_m.x, Real(2));
    EXPECT_EQ(contacts[0]->position_m.y, Real(0.5));
    EXPECT_EQ(contacts[0]->cell.i, 2U);
    EXPECT_EQ(contacts[0]->cell.j, 0U);

    EXPECT_FALSE(contacts[1].has_value());

    ASSERT_TRUE(contacts[2].has_value());
    EXPECT_EQ(
        contacts[2]->motion_fraction,
        Real(0.75) / Real(3.5));
    EXPECT_EQ(contacts[2]->position_m.x, Real(3));
    EXPECT_EQ(contacts[2]->position_m.y, Real(0.5));
    EXPECT_EQ(contacts[2]->cell.i, 2U);
    EXPECT_EQ(contacts[2]->cell.j, 0U);
}

TEST(FireFuelBarrier, InternalFaceTangencyUsesUnionInterior)
{
    const std::vector<FireVec2> starts{
        {Real(2), Real(0.25)}};
    const std::vector<FireVec2> ends{
        {Real(2), Real(1.75)}};

    // One NonBurnable side and one burnable side is a contact surface, not
    // barrier interior, so a tangential segment may remain on that face.
    const FireFuelRaster one_sided =
        make_barrier_raster({
            {2U, 0U},
            {2U, 1U}});
    const auto one_sided_contacts =
        ERFFire::collective_first_nonburnable_contacts(
            one_sided,
            starts,
            ends);
    ASSERT_EQ(one_sided_contacts.size(), 1U);
    EXPECT_FALSE(one_sided_contacts[0].has_value());

    // Above y=1 both cells adjacent to x=2 are NonBurnable. The same grid
    // seam is then interior to the union and cannot become a tunneling path.
    const FireFuelRaster two_sided =
        make_barrier_raster({
            {2U, 0U},
            {1U, 1U},
            {2U, 1U}});
    const auto two_sided_contacts =
        ERFFire::collective_first_nonburnable_contacts(
            two_sided,
            starts,
            ends);
    ASSERT_EQ(two_sided_contacts.size(), 1U);
    ASSERT_TRUE(two_sided_contacts[0].has_value());
    EXPECT_EQ(two_sided_contacts[0]->motion_fraction, Real(0.5));
    EXPECT_EQ(two_sided_contacts[0]->position_m.x, Real(2));
    EXPECT_EQ(two_sided_contacts[0]->position_m.y, Real(1));
    EXPECT_EQ(two_sided_contacts[0]->cell.i, 1U);
    EXPECT_EQ(two_sided_contacts[0]->cell.j, 1U);
}

TEST(FireFuelBarrier, FaceStartUsesDirectionOfMotion)
{
    const FireFuelRaster raster =
        make_barrier_raster({{2U, 0U}});
    const std::vector<FireVec2> starts{
        {Real(2), Real(0.5)},
        {Real(2), Real(0.5)}};
    const std::vector<FireVec2> ends{
        {Real(1.25), Real(0.5)},
        {Real(2.75), Real(0.5)}};

    const auto contacts =
        ERFFire::collective_first_nonburnable_contacts(
            raster,
            starts,
            ends);

    ASSERT_EQ(contacts.size(), 2U);
    EXPECT_FALSE(contacts[0].has_value());
    ASSERT_TRUE(contacts[1].has_value());
    EXPECT_EQ(contacts[1]->motion_fraction, Real(0));
    EXPECT_EQ(contacts[1]->position_m.x, Real(2));
    EXPECT_EQ(contacts[1]->position_m.y, Real(0.5));
    EXPECT_EQ(contacts[1]->cell.i, 2U);
    EXPECT_EQ(contacts[1]->cell.j, 0U);
}

TEST(FireFuelBarrier, RejectsInvalidSegmentBatchesBeforeSampling)
{
    const FireFuelRaster raster =
        make_barrier_raster({{2U, 0U}});

    EXPECT_THROW(
        (void)ERFFire::collective_first_nonburnable_contacts(
            raster,
            std::vector<FireVec2>{{Real(0.5), Real(0.5)}},
            std::vector<FireVec2>{}),
        std::invalid_argument);

    EXPECT_THROW(
        (void)ERFFire::collective_first_nonburnable_contacts(
            raster,
            std::vector<FireVec2>{
                {
                    std::numeric_limits<Real>::quiet_NaN(),
                    Real(0.5)}},
            std::vector<FireVec2>{{Real(1.5), Real(0.5)}}),
        std::invalid_argument);

    EXPECT_THROW(
        (void)ERFFire::collective_first_nonburnable_contacts(
            raster,
            std::vector<FireVec2>{{Real(0.5), Real(0.5)}},
            std::vector<FireVec2>{{Real(4.5), Real(0.5)}}),
        std::out_of_range);
}

TEST(FireFuelBarrier, MotionClipStopsAtFirstContact)
{
    const FireFuelRaster raster =
        make_barrier_raster({{2U, 0U}});
    const std::vector<FireVec2> starts{
        {Real(0.25), Real(0.5)},
        {Real(0.25), Real(1.5)}};
    const std::vector<FireVec2> candidates{
        {Real(3.75), Real(0.5)},
        {Real(3.75), Real(1.5)}};

    const auto clipped =
        ERFFire::collective_clip_fire_fuel_motion_to_nonburnable(
            raster,
            starts,
            candidates);

    ASSERT_EQ(clipped.positions_m.size(), 2U);
    ASSERT_EQ(clipped.contacts.size(), 2U);
    ASSERT_TRUE(clipped.contacts[0].has_value());
    EXPECT_EQ(clipped.positions_m[0].x, Real(2));
    EXPECT_EQ(clipped.positions_m[0].y, Real(0.5));
    EXPECT_FALSE(clipped.contacts[1].has_value());
    EXPECT_EQ(clipped.positions_m[1].x, candidates[1].x);
    EXPECT_EQ(clipped.positions_m[1].y, candidates[1].y);
}

TEST(FireFuelBarrier, Rk2NoContactMatchesUnconstrainedBitwise)
{
    const FireFuelRaster raster = make_barrier_raster({});
    const FireFront initial = make_barrier_test_front(Real(1.5));

    const auto speeds =
        [](const std::vector<FireVec2>& positions,
           const std::vector<FireVec2>& normals,
           Real) {
            if (positions.size() != normals.size()) {
                throw std::logic_error(
                    "barrier test positions/normals size mismatch");
            }
            return std::vector<Real>(
                positions.size(), Real(0.1));
        };

    const FireFront reference =
        ERFFire::advance_front_rk2_batched(
            initial,
            Real(1),
            Real(0.1),
            speeds);
    const FireFront constrained =
        ERFFire::advance_front_rk2_batched_clipped_to_nonburnable(
            initial,
            Real(1),
            Real(0.1),
            speeds,
            raster);

    ASSERT_EQ(reference.components().size(), constrained.components().size());
    for (std::size_t component = 0;
         component < reference.components().size();
         ++component) {
        const auto& expected =
            reference.components()[component].perimeter.vertices_m();
        const auto& actual =
            constrained.components()[component].perimeter.vertices_m();
        ASSERT_EQ(expected.size(), actual.size());
        for (std::size_t vertex = 0;
             vertex < expected.size();
             ++vertex) {
            EXPECT_EQ(actual[vertex].x, expected[vertex].x);
            EXPECT_EQ(actual[vertex].y, expected[vertex].y);
        }
    }
}

TEST(FireFuelBarrier, Rk2ClipsFinalTrajectoriesAtBarrier)
{
    const FireFuelRaster raster =
        make_barrier_raster({
            {2U, 0U},
            {2U, 1U}});
    const FireFront initial =
        make_barrier_test_front(Real(1.5));

    const FireFront constrained =
        ERFFire::advance_front_rk2_batched_clipped_to_nonburnable(
            initial,
            Real(0),
            Real(1),
            east_edge_speed(Real(1)),
            raster);

    ASSERT_EQ(constrained.components().size(), 1U);
    const auto& vertices =
        constrained.components()[0].perimeter.vertices_m();
    ASSERT_EQ(vertices.size(), 4U);
    EXPECT_EQ(vertices[1].x, Real(2));
    EXPECT_EQ(vertices[2].x, Real(2));
    EXPECT_GE(vertices[1].y, Real(0));
    EXPECT_LE(vertices[2].y, Real(2));
}

TEST(FireFuelBarrier, Rk2ClipsMidpointAndSticksOnRepeatedContact)
{
    const FireFuelRaster raster =
        make_barrier_raster({
            {2U, 0U},
            {2U, 1U}});
    const FireFront initial =
        make_barrier_test_front(Real(1.5));

    int calls = 0;
    Real second_stage_max_x = -std::numeric_limits<Real>::infinity();
    const auto speeds =
        [&calls, &second_stage_max_x](
            const std::vector<FireVec2>& positions,
            const std::vector<FireVec2>& normals,
            Real) {
            ++calls;
            if (positions.size() != normals.size()) {
                throw std::logic_error(
                    "barrier test positions/normals size mismatch");
            }
            if (calls == 2) {
                for (const FireVec2& position : positions) {
                    second_stage_max_x =
                        std::max(second_stage_max_x, position.x);
                }
            }
            std::vector<Real> result(
                positions.size(), Real(0));
            for (std::size_t index = 0;
                 index < positions.size();
                 ++index) {
                if (positions[index].x > Real(1)) {
                    result[index] = Real(2);
                }
            }
            return result;
        };

    const FireFront contacted =
        ERFFire::advance_front_rk2_batched_clipped_to_nonburnable(
            initial,
            Real(0),
            Real(1),
            speeds,
            raster);

    EXPECT_EQ(calls, 2);
    EXPECT_EQ(second_stage_max_x, Real(2));
    const auto& contacted_vertices =
        contacted.components()[0].perimeter.vertices_m();
    EXPECT_EQ(contacted_vertices[1].x, Real(2));
    EXPECT_EQ(contacted_vertices[2].x, Real(2));

    const FireFront repeated =
        ERFFire::advance_front_rk2_batched_clipped_to_nonburnable(
            contacted,
            Real(1),
            Real(0.25),
            east_edge_speed(Real(1)),
            raster);
    const auto& repeated_vertices =
        repeated.components()[0].perimeter.vertices_m();
    EXPECT_EQ(repeated_vertices[1].x, Real(2));
    EXPECT_EQ(repeated_vertices[2].x, Real(2));
}

} // namespace
