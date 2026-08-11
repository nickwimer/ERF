#include <gtest/gtest.h>

#include <Environment/ERF_FireFlatEnvironmentSampler.H>

#include <AMReX_REAL.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{

using amrex::Real;
using ERFFire::FireFlatEnvironmentLayout2D;
using ERFFire::FireFlatEnvironmentSampler;
using ERFFire::fire_vertical_linear_bracket;
using ERFFire::fire_vertical_linear_interpolate;

Real
scaled_tolerance (Real expected)
{
    return Real(128) * std::numeric_limits<Real>::epsilon()
         * std::max(Real(1), std::abs(expected));
}

void
expect_near_real (Real actual, Real expected)
{
    EXPECT_NEAR(actual, expected, scaled_tolerance(expected));
}

Real
affine_u (Real x_m, Real y_m)
{
    return Real(2.25) + Real(0.125) * x_m - Real(0.375) * y_m;
}

Real
affine_v (Real x_m, Real y_m)
{
    return Real(-1.75) + Real(0.45) * x_m + Real(0.20) * y_m;
}

void
fill_affine_native_wind (
    const FireFlatEnvironmentLayout2D& layout,
    std::vector<Real>& u,
    std::vector<Real>& v)
{
    const int nx = static_cast<int>(layout.nx());
    const int ny = static_cast<int>(layout.ny());

    for (int j = -1; j <= ny; ++j) {
        for (int i = -1; i <= nx + 1; ++i) {
            const Real x = layout.xlo_m() + static_cast<Real>(i) * layout.dx_m();
            const Real y = layout.ylo_m() + (static_cast<Real>(j) + Real(0.5)) * layout.dy_m();
            u[layout.u_storage_index(i, j)] = affine_u(x, y);
        }
    }

    for (int j = -1; j <= ny + 1; ++j) {
        for (int i = -1; i <= nx; ++i) {
            const Real x = layout.xlo_m() + (static_cast<Real>(i) + Real(0.5)) * layout.dx_m();
            const Real y = layout.ylo_m() + static_cast<Real>(j) * layout.dy_m();
            v[layout.v_storage_index(i, j)] = affine_v(x, y);
        }
    }
}

} // namespace

TEST(FireEnvironmentSampler, VerticalBracketHandlesUniformAndStretchedLevels)
{
    const std::vector<Real> uniform_z{Real(5), Real(15), Real(25)};
    const auto uniform = fire_vertical_linear_bracket(uniform_z, Real(10));

    EXPECT_EQ(uniform.lower_k, 0U);
    EXPECT_EQ(uniform.upper_k, 1U);
    expect_near_real(uniform.upper_weight, Real(0.5));

    const std::vector<Real> stretched_z{Real(2), Real(7), Real(19), Real(40)};
    const auto stretched = fire_vertical_linear_bracket(stretched_z, Real(13));

    EXPECT_EQ(stretched.lower_k, 1U);
    EXPECT_EQ(stretched.upper_k, 2U);
    expect_near_real(stretched.upper_weight, Real(0.5));

    const auto exact = fire_vertical_linear_bracket(stretched_z, Real(19));
    EXPECT_EQ(exact.lower_k, 2U);
    EXPECT_EQ(exact.upper_k, 2U);
    EXPECT_EQ(exact.upper_weight, Real(0));
}

TEST(FireEnvironmentSampler, VerticalBracketRejectsInvalidCoordinates)
{
    EXPECT_THROW(
        fire_vertical_linear_bracket({}, Real(1)),
        std::invalid_argument);

    EXPECT_THROW(
        fire_vertical_linear_bracket(
            std::vector<Real>{Real(1), Real(1), Real(3)}, Real(2)),
        std::invalid_argument);

    EXPECT_THROW(
        fire_vertical_linear_bracket(
            std::vector<Real>{Real(1), std::numeric_limits<Real>::infinity()}, Real(1)),
        std::invalid_argument);

    EXPECT_THROW(
        fire_vertical_linear_bracket(
            std::vector<Real>{Real(2), Real(5), Real(9)}, Real(1)),
        std::out_of_range);

    EXPECT_THROW(
        fire_vertical_linear_bracket(
            std::vector<Real>{Real(2), Real(5), Real(9)}, Real(10)),
        std::out_of_range);
}

TEST(FireEnvironmentSampler, VerticalInterpolationReproducesAffineProfile)
{
    const std::vector<Real> z{Real(1), Real(3.5), Real(9), Real(20)};
    const Real target = Real(6.25);
    const auto bracket = fire_vertical_linear_bracket(z, target);

    const auto profile = [] (Real height) {
        return Real(-3.0) + Real(0.75) * height;
    };

    const Real sampled = fire_vertical_linear_interpolate(
        profile(z[bracket.lower_k]),
        profile(z[bracket.upper_k]),
        bracket);

    expect_near_real(sampled, profile(target));
}

TEST(FireEnvironmentSampler, FlatSamplerPreservesUniformWind)
{
    const FireFlatEnvironmentLayout2D layout(
        Real(100), Real(50), Real(2), Real(4), 4, 3);

    std::vector<Real> u(layout.u_storage_size(), Real(3.25));
    std::vector<Real> v(layout.v_storage_size(), Real(-1.5));

    const FireFlatEnvironmentSampler sampler(
        layout, Real(10), std::move(u), std::move(v));

    for (const auto& point : std::vector<std::pair<Real, Real>>{
             {Real(100), Real(50)},
             {Real(103.3), Real(56.7)},
             {Real(108), Real(62)}}) {
        const auto sample = sampler.sample(point.first, point.second);
        expect_near_real(sample.horizontal_wind_mps.x, Real(3.25));
        expect_near_real(sample.horizontal_wind_mps.y, Real(-1.5));
    }

    EXPECT_EQ(sampler.reference_height_agl_m(), Real(10));
}

TEST(FireEnvironmentSampler, FlatSamplerReproducesAffineNativeStaggeredWind)
{
    const FireFlatEnvironmentLayout2D layout(
        Real(100), Real(50), Real(2), Real(4), 4, 3);

    std::vector<Real> u(layout.u_storage_size());
    std::vector<Real> v(layout.v_storage_size());
    fill_affine_native_wind(layout, u, v);

    const FireFlatEnvironmentSampler sampler(
        layout, Real(12), std::move(u), std::move(v));

    for (const auto& point : std::vector<std::pair<Real, Real>>{
             {Real(103.3), Real(56.7)},
             {Real(106.25), Real(59.1)}}) {
        const auto sample = sampler.sample(point.first, point.second);
        expect_near_real(
            sample.horizontal_wind_mps.x,
            affine_u(point.first, point.second));
        expect_near_real(
            sample.horizontal_wind_mps.y,
            affine_v(point.first, point.second));
    }
}

TEST(FireEnvironmentSampler, FlatSamplerUsesFilledGhostsAtPhysicalBoundary)
{
    const FireFlatEnvironmentLayout2D layout(
        Real(100), Real(50), Real(2), Real(4), 4, 3);

    std::vector<Real> u(layout.u_storage_size());
    std::vector<Real> v(layout.v_storage_size());
    fill_affine_native_wind(layout, u, v);

    const FireFlatEnvironmentSampler sampler(
        layout, Real(8), std::move(u), std::move(v));

    for (const auto& point : std::vector<std::pair<Real, Real>>{
             {layout.xlo_m(), layout.ylo_m()},
             {layout.xhi_m(), layout.yhi_m()}}) {
        const auto sample = sampler.sample(point.first, point.second);
        expect_near_real(
            sample.horizontal_wind_mps.x,
            affine_u(point.first, point.second));
        expect_near_real(
            sample.horizontal_wind_mps.y,
            affine_v(point.first, point.second));
    }
}

TEST(FireEnvironmentSampler, FlatSamplerRejectsInvalidLayoutAndStorage)
{
    EXPECT_THROW(
        FireFlatEnvironmentLayout2D(Real(0), Real(0), Real(0), Real(1), 2, 2),
        std::invalid_argument);

    EXPECT_THROW(
        FireFlatEnvironmentLayout2D(Real(0), Real(0), Real(1), Real(1), 0, 2),
        std::invalid_argument);

    const Real collapsed_origin =
        std::ldexp(Real(1), std::numeric_limits<Real>::digits);
    EXPECT_THROW(
        FireFlatEnvironmentLayout2D(
            collapsed_origin, Real(0), Real(1), Real(1), 4, 2),
        std::invalid_argument);
    EXPECT_THROW(
        FireFlatEnvironmentLayout2D(
            Real(0), collapsed_origin, Real(1), Real(1), 2, 4),
        std::invalid_argument);

    const FireFlatEnvironmentLayout2D layout(
        Real(0), Real(0), Real(1), Real(1), 2, 2);

    std::vector<Real> u(layout.u_storage_size(), Real(0));
    std::vector<Real> v(layout.v_storage_size(), Real(0));

    auto short_u = u;
    short_u.pop_back();

    EXPECT_THROW(
        FireFlatEnvironmentSampler(layout, Real(10), std::move(short_u), v),
        std::invalid_argument);

    EXPECT_THROW(
        FireFlatEnvironmentSampler(layout, Real(-1), u, v),
        std::invalid_argument);

    auto nonfinite_v = v;
    nonfinite_v.front() = std::numeric_limits<Real>::quiet_NaN();
    EXPECT_THROW(
        FireFlatEnvironmentSampler(layout, Real(10), u, std::move(nonfinite_v)),
        std::invalid_argument);
}

TEST(FireEnvironmentSampler, FlatSamplerRejectsInvalidQuery)
{
    const FireFlatEnvironmentLayout2D layout(
        Real(0), Real(0), Real(1), Real(1), 3, 2);

    std::vector<Real> u(layout.u_storage_size(), Real(1));
    std::vector<Real> v(layout.v_storage_size(), Real(2));

    const FireFlatEnvironmentSampler sampler(
        layout, Real(10), std::move(u), std::move(v));

    EXPECT_THROW(sampler.sample(Real(-0.001), Real(1)), std::out_of_range);
    EXPECT_THROW(sampler.sample(Real(1), Real(2.001)), std::out_of_range);
    EXPECT_THROW(
        sampler.sample(std::numeric_limits<Real>::quiet_NaN(), Real(1)),
        std::out_of_range);
}
