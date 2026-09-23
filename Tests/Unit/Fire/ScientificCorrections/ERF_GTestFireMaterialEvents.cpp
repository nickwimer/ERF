#include <ERF_FireMaterialAdvance.H>
#include <ERF_FireCellArrival.H>

#include <AMReX_ParallelDescriptor.H>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace
{
using amrex::Real;
using namespace ERFFire;

FireFuelRasterCell fm1(Real moisture)
{
    FireFuelRasterCell c;
    c.model_id = FireFuelModelId::FM1;
    c.moisture.set(FireFuelMoistureClass::Dead1h, moisture);
    return c;
}

FireFuelRaster make_raster(Real receiving_moisture, bool barrier = false,
                           bool thin = false, bool reverse = false)
{
    const FireCartesianRasterGeometry2D g{64, 32, Real(0), Real(0), Real(0.25), Real(0.25)};
    FireFuelRasterState state;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        state.cells.assign(g.nx*g.ny, fm1(Real(0.08)));
        for (std::size_t j = 0; j < g.ny; ++j) {
            for (std::size_t i = 0; i < g.nx; ++i) {
                const bool receiving = reverse ? (i < 16 && (!thin || i == 15))
                                               : (i >= 16 && (!thin || i == 16));
                if (!receiving) { continue; }
                auto& c = state.cells[j*g.nx+i];
                c = fm1(receiving_moisture);
                if (barrier) { c.model_id = FireFuelModelId::NonBurnable; }
            }
        }
    }
    return FireFuelRaster::collective_from_io_rank_state(g, state);
}

FireFront planar_front(bool reverse = false)
{
    std::vector<FireVec2> v{{Real(1),Real(2)}, {Real(3.8),Real(2)},
        {Real(3.8),Real(3)}, {Real(3.8),Real(4)}, {Real(3.8),Real(5)},
        {Real(3.8),Real(6)}, {Real(1),Real(6)}, {Real(1),Real(4)}};
    if (reverse) {
        for (auto& p : v) { p.x = Real(8) - p.x; }
        std::reverse(v.begin(), v.end());
    }
    return FireFront(std::vector<FireFrontComponent>{{FireFrontRole::Outer, FirePerimeter(std::move(v))}});
}

Real central_x(const FireFront& f, bool reverse = false)
{
    Real result = reverse ? Real(16) : Real(0);
    for (const auto& p : f.components()[0].perimeter.vertices_m()) {
        if (p.y == Real(4)) { result = reverse ? std::min(result,p.x) : std::max(result,p.x); }
    }
    return result;
}

NormalSpeedBatchFunction manufactured_speed(std::vector<FireFuelRasterCell>& stage, bool reverse = false)
{
    return [&stage, reverse](const std::vector<FireVec2>& p,
                             const std::vector<FireVec2>& n, Real) {
        if (p.size() != stage.size() || p.size() != n.size()) {
            throw std::logic_error("test stage material mismatch");
        }
        std::vector<Real> speeds;
        for (std::size_t i=0; i<p.size(); ++i) {
            Real rate = Real(0);
            if (stage[i].model_id != FireFuelModelId::NonBurnable) {
                const Real m = stage[i].moisture.get(FireFuelMoistureClass::Dead1h);
                rate = m >= Real(0.12) ? Real(0) : (m > Real(0.08) ? Real(0.5) : Real(1));
            }
            speeds.push_back(rate * std::max(Real(0), (reverse ? -n[i].x : n[i].x)));
        }
        return speeds;
    };
}

FireFront run_segments(FireFront front, const FireFuelRaster& raster, Real end, bool reverse = false)
{
    std::vector<FireFuelRasterCell> stage;
    const auto speed = manufactured_speed(stage, reverse);
    Real time = Real(0);
    int count = 0;
    while (time < end) {
        if (++count > 1000) { throw std::runtime_error("test material loop failed to finish"); }
        auto step = advance_fire_front_material_segment(front, time, end-time, speed, raster, stage);
        if (!step.completed_front || step.topology_event) {
            throw std::runtime_error("unexpected test topology event");
        }
        const Real remainder = end-time;
        time = step.advanced_dt_s == remainder ? end : time + step.advanced_dt_s;
        front = std::move(*step.completed_front);
    }
    return front;
}
Real tolerance() { return std::max(Real(2.e-7), Real(128)*std::numeric_limits<Real>::epsilon()); }
}

TEST(FireScientificMaterial, WetHalfSpaceStopsAtInterfaceNotBeforeIt)
{
    for (bool reverse : {false,true}) {
        const auto raster = make_raster(Real(0.2), false, false, reverse);
        const auto result = run_segments(planar_front(reverse), raster, Real(1), reverse);
        EXPECT_NEAR(central_x(result, reverse), Real(4), tolerance());
    }
}

TEST(FireScientificMaterial, WetStripCannotHideBetweenStageSamples)
{
    for (bool reverse : {false,true}) {
        const auto raster = make_raster(Real(0.2), false, true, reverse);
        const auto result = run_segments(planar_front(reverse), raster, Real(2), reverse);
        EXPECT_NEAR(central_x(result, reverse), Real(4), tolerance());
    }
}

TEST(FireScientificMaterial, FastToSlowUsesPiecewiseTravelTime)
{
    for (bool reverse : {false,true}) {
        const auto raster = make_raster(Real(0.09), false, false, reverse);
        const auto result = run_segments(planar_front(reverse), raster, Real(1), reverse);
        // 0.2 s at 1 m/s, then 0.8 s at 0.5 m/s; independent of RK samples.
        EXPECT_NEAR(central_x(result, reverse), reverse ? Real(3.6) : Real(4.4), tolerance());
    }
}

TEST(FireScientificMaterial, ContactSegmentRetainsItsPhysicalClock)
{
    const auto raster = make_raster(Real(0.08), true);
    const auto start = planar_front();
    std::vector<FireFuelRasterCell> stage;
    const auto speed = manufactured_speed(stage);
    const auto step = advance_fire_front_material_segment(start, Real(0), Real(1), speed, raster, stage);
    ASSERT_TRUE(step.completed_front.has_value());
    EXPECT_FALSE(step.topology_event.has_value());
    EXPECT_NEAR(step.advanced_dt_s, Real(0.2), tolerance());
    EXPECT_NEAR(central_x(*step.completed_front), Real(4), tolerance());
    const FireCartesianCell2D cell{Real(3.9),Real(3.95),Real(3.9),Real(4.1)};
    const auto arrival = fire_cell_first_arrival_time_linear_sweep(
        start.components()[0].perimeter,
        step.completed_front->components()[0].perimeter,
        cell, Real(0), step.advanced_dt_s, tolerance());
    EXPECT_TRUE(arrival.arrived);
    EXPECT_NEAR(arrival.arrival_time_s, Real(0.1), Real(2)*tolerance());
    const auto end = run_segments(start, raster, Real(1));
    EXPECT_NEAR(central_x(end), Real(4), tolerance());
}

TEST(FireScientificMaterial, EntireTrajectoryDetectsAThinMoistureBarrier)
{
    const auto raster = make_raster(Real(0.2), false, true);
    const std::vector<FireVec2> a{{Real(3),Real(4)}};
    const std::vector<FireVec2> b{{Real(7),Real(4)}};
    const Real fraction = detail::first_fire_material_change(raster,a,b,{fm1(Real(0.08))});
    EXPECT_NEAR(fraction, Real(0.25), tolerance());
}

TEST(FireScientificMaterial, HomogeneousEdgeResolutionPreservesVerticesExactly)
{
    const auto raster = make_raster(Real(0.08));
    const auto original = planar_front();
    const auto result = resolve_fire_front_material_edges(original, raster);
    const auto& a = original.components()[0].perimeter.vertices_m();
    const auto& b = result.components()[0].perimeter.vertices_m();
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        EXPECT_EQ(a[i].x, b[i].x);
        EXPECT_EQ(a[i].y, b[i].y);
    }
}


TEST(FireScientificMaterial, Rk2DenseTrajectoryDetectsExcursionIntoThinPatch)
{
    const auto raster = make_raster(Real(0.2), false, true);
    const std::vector<FireVec2> start{{Real(3), Real(4)}};
    const std::vector<FireVec2> stage{{Real(6), Real(4)}};
    const std::vector<FireVec2> end{{Real(3), Real(4)}};
    const std::vector<FireFuelRasterCell> initial{
        fm1(Real(0.08))};

    EXPECT_EQ(
        detail::first_fire_material_change(
            raster, start, end, initial),
        Real(1));

    const Real fraction =
        detail::first_fire_material_change_rk2(
            raster, start, stage, end, initial);

    const Real expected =
        (Real(3) - std::sqrt(Real(3))) / Real(6);
    EXPECT_NEAR(fraction, expected, Real(3.e-4));
}
