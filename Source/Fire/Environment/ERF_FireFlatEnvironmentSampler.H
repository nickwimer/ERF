#ifndef ERF_FIRE_FLAT_ENVIRONMENT_SAMPLER_H_
#define ERF_FIRE_FLAT_ENVIRONMENT_SAMPLER_H_

#include "ERF_FireEnvironment.H"

#include <AMReX_REAL.H>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ERFFire
{

struct FireVerticalLinearBracket
{
    std::size_t lower_k{};
    std::size_t upper_k{};
    amrex::Real upper_weight{};
};

inline FireVerticalLinearBracket
fire_vertical_linear_bracket (
    const std::vector<amrex::Real>& z_cell_center_m,
    amrex::Real target_height_m)
{
    if (z_cell_center_m.empty()) {
        throw std::invalid_argument("fire vertical coordinates must not be empty");
    }
    if (!std::isfinite(target_height_m)) {
        throw std::invalid_argument("fire target height must be finite");
    }

    for (std::size_t k = 0; k < z_cell_center_m.size(); ++k) {
        if (!std::isfinite(z_cell_center_m[k])) {
            throw std::invalid_argument("fire vertical coordinates must be finite");
        }
        if (k > 0 && !(z_cell_center_m[k] > z_cell_center_m[k - 1])) {
            throw std::invalid_argument(
                "fire vertical cell-center coordinates must be strictly increasing");
        }
    }

    if (target_height_m < z_cell_center_m.front() ||
        target_height_m > z_cell_center_m.back()) {
        throw std::out_of_range(
            "fire reference height lies outside horizontal-velocity cell-center heights");
    }

    const auto upper =
        std::lower_bound(z_cell_center_m.begin(), z_cell_center_m.end(), target_height_m);

    if (upper != z_cell_center_m.end() && *upper == target_height_m) {
        const auto k = static_cast<std::size_t>(
            std::distance(z_cell_center_m.begin(), upper));
        return {k, k, amrex::Real(0)};
    }

    if (upper == z_cell_center_m.begin() || upper == z_cell_center_m.end()) {
        throw std::logic_error("fire vertical bracketing invariant violated");
    }

    const auto upper_k = static_cast<std::size_t>(
        std::distance(z_cell_center_m.begin(), upper));
    const auto lower_k = upper_k - 1;

    const amrex::Real zlo = z_cell_center_m[lower_k];
    const amrex::Real zhi = z_cell_center_m[upper_k];
    const amrex::Real weight = (target_height_m - zlo) / (zhi - zlo);

    return {lower_k, upper_k, weight};
}

inline amrex::Real
fire_vertical_linear_interpolate (
    amrex::Real lower_value,
    amrex::Real upper_value,
    const FireVerticalLinearBracket& bracket)
{
    if (!std::isfinite(lower_value) || !std::isfinite(upper_value)) {
        throw std::invalid_argument("fire vertical interpolation values must be finite");
    }
    if (!std::isfinite(bracket.upper_weight) ||
        bracket.upper_weight < amrex::Real(0) ||
        bracket.upper_weight > amrex::Real(1)) {
        throw std::invalid_argument("fire vertical interpolation weight must be in [0,1]");
    }

    if (bracket.lower_k == bracket.upper_k) {
        return lower_value;
    }
    if (bracket.lower_k > bracket.upper_k) {
        throw std::invalid_argument("fire vertical interpolation bracket is reversed");
    }

    return (amrex::Real(1) - bracket.upper_weight) * lower_value
         + bracket.upper_weight * upper_value;
}

class FireFlatEnvironmentLayout2D
{
public:
    static constexpr int ghost_width = 1;

    FireFlatEnvironmentLayout2D (
        amrex::Real xlo_m,
        amrex::Real ylo_m,
        amrex::Real dx_m,
        amrex::Real dy_m,
        std::size_t nx,
        std::size_t ny)
        : xlo_m_(xlo_m),
          ylo_m_(ylo_m),
          dx_m_(dx_m),
          dy_m_(dy_m),
          nx_(nx),
          ny_(ny)
    {
        if (!std::isfinite(xlo_m_) || !std::isfinite(ylo_m_) ||
            !std::isfinite(dx_m_) || !std::isfinite(dy_m_)) {
            throw std::invalid_argument("fire environment grid geometry must be finite");
        }
        if (!(dx_m_ > amrex::Real(0)) || !(dy_m_ > amrex::Real(0))) {
            throw std::invalid_argument("fire environment grid spacing must be positive");
        }
        if (nx_ == 0 || ny_ == 0) {
            throw std::invalid_argument("fire environment grid must contain at least one cell");
        }

        const auto int_limit =
            static_cast<std::size_t>(std::numeric_limits<int>::max() - 2);
        if (nx_ > int_limit || ny_ > int_limit) {
            throw std::overflow_error("fire environment grid dimensions exceed index range");
        }

        xhi_m_ = xlo_m_ + static_cast<amrex::Real>(nx_) * dx_m_;
        yhi_m_ = ylo_m_ + static_cast<amrex::Real>(ny_) * dy_m_;

        if (!std::isfinite(xhi_m_) || !std::isfinite(yhi_m_) ||
            !(xhi_m_ > xlo_m_) || !(yhi_m_ > ylo_m_)) {
            throw std::invalid_argument(
                "fire environment grid extents are not representable");
        }

        // The sampler uses both face and half-cell locations. Reject a
        // nominal spacing that collapses either nearest half-cell location
        // onto a physical boundary at the precision of amrex::Real.
        const amrex::Real half_dx = amrex::Real(0.5) * dx_m_;
        const amrex::Real half_dy = amrex::Real(0.5) * dy_m_;
        if (!(xlo_m_ + half_dx > xlo_m_) ||
            !(xhi_m_ - half_dx < xhi_m_) ||
            !(ylo_m_ + half_dy > ylo_m_) ||
            !(yhi_m_ - half_dy < yhi_m_)) {
            throw std::invalid_argument(
                "fire environment staggered coordinates are not representable");
        }

        u_storage_size_ = checked_product(nx_ + 3, ny_ + 2);
        v_storage_size_ = checked_product(nx_ + 2, ny_ + 3);
    }

    amrex::Real xlo_m () const noexcept { return xlo_m_; }
    amrex::Real ylo_m () const noexcept { return ylo_m_; }
    amrex::Real xhi_m () const noexcept { return xhi_m_; }
    amrex::Real yhi_m () const noexcept { return yhi_m_; }
    amrex::Real dx_m () const noexcept { return dx_m_; }
    amrex::Real dy_m () const noexcept { return dy_m_; }
    std::size_t nx () const noexcept { return nx_; }
    std::size_t ny () const noexcept { return ny_; }

    std::size_t u_storage_size () const noexcept { return u_storage_size_; }
    std::size_t v_storage_size () const noexcept { return v_storage_size_; }

    // Snapshot indices are normalized to zero-based level-0 cell indices.
    // With one retained ghost layer:
    //   u: i in [-1,nx+1], j in [-1,ny]
    //   v: i in [-1,nx],   j in [-1,ny+1]
    std::size_t u_storage_index (int i, int j) const
    {
        const int nx = static_cast<int>(nx_);
        const int ny = static_cast<int>(ny_);
        if (i < -1 || i > nx + 1 || j < -1 || j > ny) {
            throw std::out_of_range("u-wind index lies outside one-ghost snapshot");
        }

        const std::size_t stride = nx_ + 3;
        return static_cast<std::size_t>(j + 1) * stride
             + static_cast<std::size_t>(i + 1);
    }

    std::size_t v_storage_index (int i, int j) const
    {
        const int nx = static_cast<int>(nx_);
        const int ny = static_cast<int>(ny_);
        if (i < -1 || i > nx || j < -1 || j > ny + 1) {
            throw std::out_of_range("v-wind index lies outside one-ghost snapshot");
        }

        const std::size_t stride = nx_ + 2;
        return static_cast<std::size_t>(j + 1) * stride
             + static_cast<std::size_t>(i + 1);
    }

    bool contains_physical_point (amrex::Real x_m, amrex::Real y_m) const noexcept
    {
        return std::isfinite(x_m) && std::isfinite(y_m) &&
               x_m >= xlo_m_ && x_m <= xhi_m_ &&
               y_m >= ylo_m_ && y_m <= yhi_m_;
    }

private:
    static std::size_t checked_product (std::size_t a, std::size_t b)
    {
        if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
            throw std::overflow_error("fire environment storage size overflow");
        }
        return a * b;
    }

    amrex::Real xlo_m_{};
    amrex::Real ylo_m_{};
    amrex::Real xhi_m_{};
    amrex::Real yhi_m_{};
    amrex::Real dx_m_{};
    amrex::Real dy_m_{};
    std::size_t nx_{};
    std::size_t ny_{};
    std::size_t u_storage_size_{};
    std::size_t v_storage_size_{};
};

class FireFlatEnvironmentSampler
{
public:
    FireFlatEnvironmentSampler (
        FireFlatEnvironmentLayout2D layout,
        amrex::Real reference_height_agl_m,
        std::vector<amrex::Real> u_wind_mps,
        std::vector<amrex::Real> v_wind_mps)
        : layout_(std::move(layout)),
          reference_height_agl_m_(reference_height_agl_m),
          u_wind_mps_(std::move(u_wind_mps)),
          v_wind_mps_(std::move(v_wind_mps))
    {
        if (!std::isfinite(reference_height_agl_m_) ||
            reference_height_agl_m_ < amrex::Real(0)) {
            throw std::invalid_argument(
                "fire atmospheric reference height AGL must be finite and nonnegative");
        }
        if (u_wind_mps_.size() != layout_.u_storage_size() ||
            v_wind_mps_.size() != layout_.v_storage_size()) {
            throw std::invalid_argument(
                "fire environment snapshot storage does not match staggered layout");
        }
        if (!all_finite(u_wind_mps_) || !all_finite(v_wind_mps_)) {
            throw std::invalid_argument("fire environment wind snapshot must be finite");
        }
    }

    const FireFlatEnvironmentLayout2D& layout () const noexcept { return layout_; }

    amrex::Real reference_height_agl_m () const noexcept
    {
        return reference_height_agl_m_;
    }

    FireEnvironmentSample sample (amrex::Real x_m, amrex::Real y_m) const
    {
        if (!layout_.contains_physical_point(x_m, y_m)) {
            throw std::out_of_range(
                "fire environment sample point lies outside the physical level-0 domain");
        }

        // Interpolate each component directly on its native horizontal
        // staggering; do not first destagger to cell centers.
        const AxisStencil ux = make_axis_stencil(
            x_m, layout_.xlo_m(), layout_.dx_m(), amrex::Real(0));
        const AxisStencil uy = make_axis_stencil(
            y_m, layout_.ylo_m(), layout_.dy_m(), amrex::Real(0.5));

        const AxisStencil vx = make_axis_stencil(
            x_m, layout_.xlo_m(), layout_.dx_m(), amrex::Real(0.5));
        const AxisStencil vy = make_axis_stencil(
            y_m, layout_.ylo_m(), layout_.dy_m(), amrex::Real(0));

        const amrex::Real u = bilinear(
            u_value(ux.lower,     uy.lower),
            u_value(ux.lower + 1, uy.lower),
            u_value(ux.lower,     uy.lower + 1),
            u_value(ux.lower + 1, uy.lower + 1),
            ux.upper_weight,
            uy.upper_weight);

        const amrex::Real v = bilinear(
            v_value(vx.lower,     vy.lower),
            v_value(vx.lower + 1, vy.lower),
            v_value(vx.lower,     vy.lower + 1),
            v_value(vx.lower + 1, vy.lower + 1),
            vx.upper_weight,
            vy.upper_weight);

        return FireEnvironmentSample{FireVec2{u, v}};
    }

private:
    struct AxisStencil
    {
        int lower{};
        amrex::Real upper_weight{};
    };

    static bool all_finite (const std::vector<amrex::Real>& values)
    {
        return std::all_of(
            values.begin(), values.end(),
            [] (amrex::Real value) { return std::isfinite(value); });
    }

    static AxisStencil make_axis_stencil (
        amrex::Real coordinate_m,
        amrex::Real origin_m,
        amrex::Real spacing_m,
        amrex::Real native_offset_cells)
    {
        const amrex::Real logical =
            (coordinate_m - origin_m) / spacing_m - native_offset_cells;
        const amrex::Real floored = std::floor(logical);

        if (floored < static_cast<amrex::Real>(std::numeric_limits<int>::min()) ||
            floored > static_cast<amrex::Real>(std::numeric_limits<int>::max())) {
            throw std::overflow_error("fire interpolation stencil index overflow");
        }

        const int lower = static_cast<int>(floored);
        const amrex::Real upper_weight =
            logical - static_cast<amrex::Real>(lower);

        return {lower, upper_weight};
    }

    static amrex::Real bilinear (
        amrex::Real q00,
        amrex::Real q10,
        amrex::Real q01,
        amrex::Real q11,
        amrex::Real wx,
        amrex::Real wy)
    {
        const amrex::Real lower =
            (amrex::Real(1) - wx) * q00 + wx * q10;
        const amrex::Real upper =
            (amrex::Real(1) - wx) * q01 + wx * q11;
        return (amrex::Real(1) - wy) * lower + wy * upper;
    }

    amrex::Real u_value (int i, int j) const
    {
        return u_wind_mps_[layout_.u_storage_index(i, j)];
    }

    amrex::Real v_value (int i, int j) const
    {
        return v_wind_mps_[layout_.v_storage_index(i, j)];
    }

    FireFlatEnvironmentLayout2D layout_;
    amrex::Real reference_height_agl_m_{};
    std::vector<amrex::Real> u_wind_mps_;
    std::vector<amrex::Real> v_wind_mps_;
};

} // namespace ERFFire

#endif
