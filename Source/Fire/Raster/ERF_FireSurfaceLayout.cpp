#include <ERF_FireSurfaceLayout.H>

#include <AMReX_IntVect.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Vector.H>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ERFFire
{
namespace
{

int
checked_int_extent(std::size_t cells, const char* axis)
{
    if (cells
        > static_cast<std::size_t>(
            std::numeric_limits<int>::max())) {
        throw std::overflow_error(
            std::string("Fire surface ") + axis
            + " extent exceeds AMReX integer index range");
    }
    return static_cast<int>(cells);
}

std::pair<int, int>
target_box_partition(int nx, int ny)
{
    const long long cell_count =
        static_cast<long long>(nx)
        * static_cast<long long>(ny);
    const int target_boxes =
        static_cast<int>(
            std::min<long long>(
                std::max(1, amrex::ParallelDescriptor::NProcs()),
                cell_count));

    int px = 1;
    int py = 1;

    while (static_cast<long long>(px) * py < target_boxes) {
        const double cells_per_box_x =
            static_cast<double>(nx) / px;
        const double cells_per_box_y =
            static_cast<double>(ny) / py;

        if (((cells_per_box_x >= cells_per_box_y) && px < nx)
            || py >= ny) {
            ++px;
        } else if (py < ny) {
            ++py;
        } else {
            break;
        }
    }

    return {px, py};
}

int
ceil_div(int numerator, int denominator) noexcept
{
    return numerator / denominator
        + ((numerator % denominator) != 0 ? 1 : 0);
}

} // namespace

FireSurfaceLayout::FireSurfaceLayout(
    const FireCartesianRasterGeometry2D& geometry)
    : geometry_(geometry)
{
    (void)detail::validate_fire_cartesian_raster_geometry(
        geometry_);

    const int nx = checked_int_extent(geometry_.nx, "x");
    const int ny = checked_int_extent(geometry_.ny, "y");

    cell_domain_ = amrex::Box(
        amrex::IntVect(0, 0, 0),
        amrex::IntVect(nx - 1, ny - 1, 0));

    const auto [px, py] = target_box_partition(nx, ny);
    const int max_box_x = ceil_div(nx, px);
    const int max_box_y = ceil_div(ny, py);

    box_array_ = amrex::BoxArray(cell_domain_);
    box_array_.maxSize(
        amrex::IntVect(max_box_x, max_box_y, 1));
    const int nprocs =
        std::max(1, amrex::ParallelDescriptor::NProcs());
    amrex::Vector<int> processor_map(
        static_cast<std::size_t>(box_array_.size()));
    for (int box = 0; box < box_array_.size(); ++box) {
        processor_map[static_cast<std::size_t>(box)] =
            box % nprocs;
    }
    distribution_map_ =
        amrex::DistributionMapping(std::move(processor_map));
}

} // namespace ERFFire
