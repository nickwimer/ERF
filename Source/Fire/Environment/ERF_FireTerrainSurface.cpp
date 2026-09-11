#include "ERF_FireTerrainSurface.H"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ERFFire
{
namespace
{

void
require(bool condition, const char* message)
{
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

std::size_t
checked_nodal_count(
    const FireCartesianRasterGeometry2D& geometry)
{
    (void)detail::validate_fire_cartesian_raster_geometry(
        geometry);

    require(
        geometry.nx < std::numeric_limits<std::size_t>::max()
        && geometry.ny < std::numeric_limits<std::size_t>::max(),
        "fire terrain nodal dimensions overflow size_t");

    const std::size_t nx_nodes = geometry.nx + 1;
    const std::size_t ny_nodes = geometry.ny + 1;

    if (nx_nodes
        > std::numeric_limits<std::size_t>::max()
            / ny_nodes) {
        throw std::overflow_error(
            "fire terrain nodal count overflows size_t");
    }

    return nx_nodes * ny_nodes;
}

amrex::Real
represented_upper(
    amrex::Real lo,
    std::size_t n,
    amrex::Real spacing)
{
    return lo
        + static_cast<amrex::Real>(n) * spacing;
}

bool
physical_coordinate_equal(
    amrex::Real a,
    amrex::Real b) noexcept
{
    const amrex::Real scale =
        std::max(
            amrex::Real(1),
            std::max(std::abs(a), std::abs(b)));

    return std::abs(a - b)
        <= amrex::Real(4096)
            * std::numeric_limits<amrex::Real>::epsilon()
            * scale;
}

bool
same_physical_domain(
    const FireCartesianRasterGeometry2D& a,
    const FireCartesianRasterGeometry2D& b) noexcept
{
    return physical_coordinate_equal(a.xlo_m, b.xlo_m)
        && physical_coordinate_equal(a.ylo_m, b.ylo_m)
        && physical_coordinate_equal(
            represented_upper(a.xlo_m, a.nx, a.dx_m),
            represented_upper(b.xlo_m, b.nx, b.dx_m))
        && physical_coordinate_equal(
            represented_upper(a.ylo_m, a.ny, a.dy_m),
            represented_upper(b.ylo_m, b.ny, b.dy_m));
}

bool
same_geometry(
    const FireCartesianRasterGeometry2D& a,
    const FireCartesianRasterGeometry2D& b) noexcept
{
    return a.nx == b.nx
        && a.ny == b.ny
        && a.xlo_m == b.xlo_m
        && a.ylo_m == b.ylo_m
        && a.dx_m == b.dx_m
        && a.dy_m == b.dy_m;
}

struct AxisLocation
{
    std::size_t index{};
    amrex::Real local{};
};

AxisLocation
locate_axis(
    amrex::Real coordinate,
    amrex::Real lo,
    std::size_t n,
    amrex::Real spacing,
    const char* message)
{
    const amrex::Real hi =
        represented_upper(lo, n, spacing);

    if (!std::isfinite(coordinate)
        || coordinate < lo
        || coordinate > hi) {
        throw std::out_of_range(message);
    }

    if (coordinate == hi) {
        return {
            n - 1,
            amrex::Real(1)};
    }

    const amrex::Real scaled =
        (coordinate - lo) / spacing;

    std::size_t index =
        static_cast<std::size_t>(
            std::floor(scaled));

    if (index >= n) {
        index = n - 1;
        return {
            index,
            amrex::Real(1)};
    }

    // Recover the documented positive-index convention when a coordinate
    // is exactly one of the represented grid boundaries but the division
    // above rounds just below or just above its mathematical integer.
    if (index > 0
        && coordinate
            == represented_upper(
                lo, index, spacing)) {
        return {
            index,
            amrex::Real(0)};
    }

    if (index + 1 < n
        && coordinate
            == represented_upper(
                lo, index + 1, spacing)) {
        return {
            index + 1,
            amrex::Real(0)};
    }

    return {
        index,
        scaled - static_cast<amrex::Real>(index)};
}

} // namespace

FireTerrainSurface::FireTerrainSurface(
    FireCartesianRasterGeometry2D horizontal_geometry,
    std::vector<amrex::Real> nodal_ground_height_m)
    : geometry_(horizontal_geometry),
      nodal_ground_height_m_(
          std::move(nodal_ground_height_m))
{
    const std::size_t expected =
        checked_nodal_count(geometry_);

    require(
        nodal_ground_height_m_.size() == expected,
        "fire terrain nodal height count does not match horizontal geometry");

    for (amrex::Real height : nodal_ground_height_m_) {
        require(
            std::isfinite(height),
            "fire terrain nodal heights must be finite");
    }
}

bool
FireTerrainSurface::contains_physical_point(
    amrex::Real x_m,
    amrex::Real y_m) const noexcept
{
    if (!std::isfinite(x_m)
        || !std::isfinite(y_m)) {
        return false;
    }

    const amrex::Real xhi =
        represented_upper(
            geometry_.xlo_m,
            geometry_.nx,
            geometry_.dx_m);
    const amrex::Real yhi =
        represented_upper(
            geometry_.ylo_m,
            geometry_.ny,
            geometry_.dy_m);

    return x_m >= geometry_.xlo_m
        && x_m <= xhi
        && y_m >= geometry_.ylo_m
        && y_m <= yhi;
}

std::size_t
FireTerrainSurface::nodal_index(
    std::size_t i,
    std::size_t j) const
{
    if (i > geometry_.nx
        || j > geometry_.ny) {
        throw std::out_of_range(
            "fire terrain nodal index out of range");
    }

    return j * (geometry_.nx + 1) + i;
}

amrex::Real
FireTerrainSurface::nodal_height_m(
    std::size_t i,
    std::size_t j) const
{
    return nodal_ground_height_m_[
        nodal_index(i, j)];
}

FireTerrainSurface::HorizontalLocation
FireTerrainSurface::locate(
    amrex::Real x_m,
    amrex::Real y_m) const
{
    const AxisLocation x =
        locate_axis(
            x_m,
            geometry_.xlo_m,
            geometry_.nx,
            geometry_.dx_m,
            "fire terrain x coordinate is outside represented domain");
    const AxisLocation y =
        locate_axis(
            y_m,
            geometry_.ylo_m,
            geometry_.ny,
            geometry_.dy_m,
            "fire terrain y coordinate is outside represented domain");

    return {
        x.index,
        y.index,
        x.local,
        y.local};
}

amrex::Real
FireTerrainSurface::ground_height_m(
    amrex::Real x_m,
    amrex::Real y_m) const
{
    const HorizontalLocation location =
        locate(x_m, y_m);

    const amrex::Real h00 =
        nodal_height_m(location.i, location.j);
    const amrex::Real h10 =
        nodal_height_m(location.i + 1, location.j);
    const amrex::Real h01 =
        nodal_height_m(location.i, location.j + 1);
    const amrex::Real h11 =
        nodal_height_m(location.i + 1, location.j + 1);

    const amrex::Real hx0 =
        h00 + location.alpha * (h10 - h00);
    const amrex::Real hx1 =
        h01 + location.alpha * (h11 - h01);

    return
        hx0 + location.beta * (hx1 - hx0);
}

FireVec2
FireTerrainSurface::terrain_gradient_m_per_m(
    amrex::Real x_m,
    amrex::Real y_m) const
{
    const HorizontalLocation location =
        locate(x_m, y_m);

    const amrex::Real h00 =
        nodal_height_m(location.i, location.j);
    const amrex::Real h10 =
        nodal_height_m(location.i + 1, location.j);
    const amrex::Real h01 =
        nodal_height_m(location.i, location.j + 1);
    const amrex::Real h11 =
        nodal_height_m(location.i + 1, location.j + 1);

    const amrex::Real dhdx =
        ((amrex::Real(1) - location.beta)
             * (h10 - h00)
         + location.beta
             * (h11 - h01))
        / geometry_.dx_m;

    const amrex::Real dhdy =
        ((amrex::Real(1) - location.alpha)
             * (h01 - h00)
         + location.alpha
             * (h11 - h10))
        / geometry_.dy_m;

    if (!std::isfinite(dhdx)
        || !std::isfinite(dhdy)) {
        throw std::overflow_error(
            "fire terrain gradient is not finite");
    }

    return {dhdx, dhdy};
}

FireTerrainSurface
resample_fire_terrain_surface(
    const FireTerrainSurface& source,
    FireCartesianRasterGeometry2D target_geometry)
{
    const std::size_t target_nodal_count =
        checked_nodal_count(target_geometry);

    require(
        same_physical_domain(
            source.geometry(),
            target_geometry),
        "fire terrain resampling requires identical physical horizontal domains");

    if (same_geometry(
            source.geometry(),
            target_geometry)) {
        return source;
    }

    std::vector<amrex::Real> nodal_ground_height_m;
    nodal_ground_height_m.reserve(
        target_nodal_count);

    for (std::size_t j = 0;
         j <= target_geometry.ny;
         ++j) {
        const amrex::Real y =
            target_geometry.ylo_m
            + static_cast<amrex::Real>(j)
                * target_geometry.dy_m;

        for (std::size_t i = 0;
             i <= target_geometry.nx;
             ++i) {
            const amrex::Real x =
                target_geometry.xlo_m
                + static_cast<amrex::Real>(i)
                    * target_geometry.dx_m;

            nodal_ground_height_m.push_back(
                source.ground_height_m(x, y));
        }
    }

    return FireTerrainSurface(
        target_geometry,
        std::move(nodal_ground_height_m));
}

} // namespace ERFFire
