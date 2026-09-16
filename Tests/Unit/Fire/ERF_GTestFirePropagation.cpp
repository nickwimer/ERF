#include <ERF_VectorPerimeterPropagator.H>
#include "ERF_FireTestUtils.H"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace
{

using ERFFire::FirePerimeter;
using ERFFire::FireVec2;
FirePerimeter
make_planar_strip (
    std::size_t short_side_segments,
    std::size_t long_side_segments,
    amrex::Real half_width_m,
    amrex::Real half_length_m)
{
    std::vector<FireVec2> vertices;
    vertices.reserve(
        2 * (short_side_segments + long_side_segments));

    for (std::size_t i = 0; i < short_side_segments; ++i) {
        const amrex::Real f =
            static_cast<amrex::Real>(i)
            / static_cast<amrex::Real>(short_side_segments);
        vertices.push_back({
            -half_width_m + amrex::Real(2.0) * half_width_m * f,
            -half_length_m
        });
    }

    for (std::size_t i = 0; i < long_side_segments; ++i) {
        const amrex::Real f =
            static_cast<amrex::Real>(i)
            / static_cast<amrex::Real>(long_side_segments);
        vertices.push_back({
            half_width_m,
            -half_length_m + amrex::Real(2.0) * half_length_m * f
        });
    }

    for (std::size_t i = 0; i < short_side_segments; ++i) {
        const amrex::Real f =
            static_cast<amrex::Real>(i)
            / static_cast<amrex::Real>(short_side_segments);
        vertices.push_back({
            half_width_m - amrex::Real(2.0) * half_width_m * f,
            half_length_m
        });
    }

    for (std::size_t i = 0; i < long_side_segments; ++i) {
        const amrex::Real f =
            static_cast<amrex::Real>(i)
            / static_cast<amrex::Real>(long_side_segments);
        vertices.push_back({
            -half_width_m,
            half_length_m - amrex::Real(2.0) * half_length_m * f
        });
    }

    return FirePerimeter(std::move(vertices));
}
TEST(FirePropagation, PrescribedPlanarStripAdvancesAtRequestedSpeed)
{
    constexpr std::size_t short_side_segments = 20;
    constexpr std::size_t long_side_segments = 200;
    constexpr amrex::Real initial_half_width_m = 5.0;
    constexpr amrex::Real half_length_m = 100.0;
    constexpr amrex::Real speed_mps = 0.1;
    constexpr amrex::Real dt_s = 1.0;
    constexpr int total_steps = 100;

    FirePerimeter perimeter = make_planar_strip(
        short_side_segments,
        long_side_segments,
        initial_half_width_m,
        half_length_m);

    const auto speed = [=] (
        const FireVec2&,
        const FireVec2&,
        amrex::Real) -> amrex::Real
    {
        return speed_mps;
    };

    amrex::Real time_s = 0.0;
    ERFFireTest::maybe_write_snapshot("planar", time_s, perimeter);

    for (int step = 1; step <= total_steps; ++step) {
        perimeter = ERFFire::advance_perimeter_rk2(
            perimeter, time_s, dt_s, speed);
        time_s += dt_s;

        if (step % 25 == 0) {
            ERFFireTest::maybe_write_snapshot("planar", time_s, perimeter);
        }
    }

    const std::size_t right_center =
        short_side_segments + long_side_segments / 2;
    const std::size_t left_center =
        2 * short_side_segments
        + long_side_segments
        + long_side_segments / 2;

    const amrex::Real expected_half_width_m =
        initial_half_width_m + speed_mps * time_s;

    EXPECT_NEAR(
        static_cast<double>(perimeter.vertices_m()[right_center].x),
        static_cast<double>(expected_half_width_m),
        1.0e-11);
    EXPECT_NEAR(
        static_cast<double>(perimeter.vertices_m()[left_center].x),
        static_cast<double>(-expected_half_width_m),
        1.0e-11);
    EXPECT_NEAR(
        static_cast<double>(perimeter.vertices_m()[right_center].y),
        0.0,
        1.0e-11);
    EXPECT_NEAR(
        static_cast<double>(perimeter.vertices_m()[left_center].y),
        0.0,
        1.0e-11);
}

TEST(FirePropagation, PrescribedCircleExpandsRadially)
{
    constexpr std::size_t vertex_count = 128;
    constexpr amrex::Real initial_radius_m = 10.0;
    constexpr amrex::Real speed_mps = 0.1;
    constexpr amrex::Real dt_s = 1.0;
    constexpr int total_steps = 100;

    FirePerimeter perimeter = ERFFireTest::make_circle(vertex_count, initial_radius_m);
    const amrex::Real initial_area_m2 = perimeter.area_m2();

    const auto speed = [=] (
        const FireVec2&,
        const FireVec2&,
        amrex::Real) -> amrex::Real
    {
        return speed_mps;
    };

    amrex::Real time_s = 0.0;
    ERFFireTest::maybe_write_snapshot("circle", time_s, perimeter);

    for (int step = 1; step <= total_steps; ++step) {
        perimeter = ERFFire::advance_perimeter_rk2(
            perimeter, time_s, dt_s, speed);
        time_s += dt_s;

        if (step % 25 == 0) {
            ERFFireTest::maybe_write_snapshot("circle", time_s, perimeter);
        }
    }

    const amrex::Real expected_radius_m =
        initial_radius_m + speed_mps * time_s;

    amrex::Real mean_radius_m = 0.0;
    for (const auto& vertex : perimeter.vertices_m()) {
        mean_radius_m += ERFFire::norm(vertex);
    }
    mean_radius_m /= static_cast<amrex::Real>(perimeter.size());

    amrex::Real max_radial_deviation_m = 0.0;
    for (const auto& vertex : perimeter.vertices_m()) {
        max_radial_deviation_m = std::max(
            max_radial_deviation_m,
            std::abs(ERFFire::norm(vertex) - mean_radius_m));
    }

    EXPECT_NEAR(
        static_cast<double>(mean_radius_m),
        static_cast<double>(expected_radius_m),
        1.0e-10);
    EXPECT_LT(
        static_cast<double>(max_radial_deviation_m),
        1.0e-10);

    const amrex::Real expected_area_ratio =
        (expected_radius_m * expected_radius_m)
        / (initial_radius_m * initial_radius_m);

    EXPECT_NEAR(
        static_cast<double>(perimeter.area_m2() / initial_area_m2),
        static_cast<double>(expected_area_ratio),
        1.0e-10);
}

TEST(FirePropagation, MidpointIntegratorShowsSecondOrderConvergence)
{
    constexpr amrex::Real initial_radius_m = 10.0;
    constexpr amrex::Real growth_rate_per_s = 0.02;
    constexpr amrex::Real final_time_s = 10.0;

    const auto run = [=] (amrex::Real dt_s) -> amrex::Real
    {
        FirePerimeter perimeter = ERFFireTest::make_circle(64, initial_radius_m);

        const auto speed = [=] (
            const FireVec2& position_m,
            const FireVec2&,
            amrex::Real) -> amrex::Real
        {
            return growth_rate_per_s * ERFFire::norm(position_m);
        };

        const int step_count =
            static_cast<int>(std::llround(final_time_s / dt_s));
        amrex::Real time_s = 0.0;

        for (int step = 0; step < step_count; ++step) {
            perimeter = ERFFire::advance_perimeter_rk2(
                perimeter, time_s, dt_s, speed);
            time_s += dt_s;
        }

        amrex::Real mean_radius_m = 0.0;
        for (const auto& vertex : perimeter.vertices_m()) {
            mean_radius_m += ERFFire::norm(vertex);
        }

        return mean_radius_m
            / static_cast<amrex::Real>(perimeter.size());
    };

    const amrex::Real exact_radius_m =
        initial_radius_m
        * std::exp(growth_rate_per_s * final_time_s);

    const amrex::Real coarse_error =
        std::abs(run(2.0) - exact_radius_m);
    const amrex::Real medium_error =
        std::abs(run(1.0) - exact_radius_m);
    const amrex::Real fine_error =
        std::abs(run(0.5) - exact_radius_m);

    EXPECT_GT(static_cast<double>(coarse_error / medium_error), 3.5);
    EXPECT_LT(static_cast<double>(coarse_error / medium_error), 4.5);
    EXPECT_GT(static_cast<double>(medium_error / fine_error), 3.5);
    EXPECT_LT(static_cast<double>(medium_error / fine_error), 4.5);
}

TEST(FirePropagation, RejectsInvalidIntegratorInputs)
{
    const FirePerimeter perimeter = ERFFireTest::make_circle(32, 10.0);

    const auto constant_speed = [] (
        const FireVec2&,
        const FireVec2&,
        amrex::Real) -> amrex::Real
    {
        return 0.1;
    };

    EXPECT_THROW(
        (void) ERFFire::advance_perimeter_rk2(
            perimeter, 0.0, -1.0, constant_speed),
        std::invalid_argument);

    const auto negative_speed = [] (
        const FireVec2&,
        const FireVec2&,
        amrex::Real) -> amrex::Real
    {
        return -0.1;
    };

    EXPECT_THROW(
        (void) ERFFire::advance_perimeter_rk2(
            perimeter, 0.0, 1.0, negative_speed),
        std::invalid_argument);

    const auto nonfinite_speed = [] (
        const FireVec2&,
        const FireVec2&,
        amrex::Real) -> amrex::Real
    {
        return std::numeric_limits<amrex::Real>::infinity();
    };

    EXPECT_THROW(
        (void) ERFFire::advance_perimeter_rk2(
            perimeter, 0.0, 1.0, nonfinite_speed),
        std::invalid_argument);
}

} // namespace
