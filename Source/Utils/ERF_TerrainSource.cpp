#include "ERF_TerrainSource.H"

#include <AMReX_Gpu.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace
{

void
require(
    bool condition,
    const char* message)
{
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

template <typename T>
T
read_single_value(
    std::istream& input,
    int line_number,
    const std::string& filename)
{
    std::string line;
    if (!std::getline(input, line)) {
        amrex::Abort(
            "Unable to read line "
            + std::to_string(line_number)
            + " from terrain file "
            + filename);
    }

    std::stringstream parser(line);
    T value{};
    parser >> value;

    if (parser.fail()) {
        amrex::Abort(
            "Failed to parse line "
            + std::to_string(line_number)
            + " from terrain file "
            + filename);
    }

    std::string extra;
    if (parser >> extra) {
        amrex::Abort(
            "Terrain file "
            + filename
            + " must contain exactly one value per line; extra data on line "
            + std::to_string(line_number));
    }

    return value;
}

std::uint64_t
fingerprint_file_bytes(
    const std::string& filename)
{
    std::ifstream stream(
        filename,
        std::ios::in | std::ios::binary);

    if (!stream.good()) {
        throw std::runtime_error(
            "unable to open terrain source for fingerprinting: "
            + filename);
    }

    constexpr std::uint64_t offset_basis =
        UINT64_C(14695981039346656037);
    constexpr std::uint64_t prime =
        UINT64_C(1099511628211);

    std::uint64_t hash = offset_basis;
    char buffer[8192];

    while (stream) {
        stream.read(buffer, sizeof(buffer));
        const std::streamsize count =
            stream.gcount();

        for (std::streamsize index = 0;
             index < count;
             ++index) {
            hash ^=
                static_cast<unsigned char>(
                    buffer[index]);
            hash *= prime;
        }
    }

    if (!stream.eof()) {
        throw std::runtime_error(
            "failed while fingerprinting terrain source: "
            + filename);
    }

    return hash;
}

std::size_t
checked_value_count(
    std::size_t nx,
    std::size_t ny)
{
    if (nx > std::numeric_limits<std::size_t>::max() / ny) {
        throw std::overflow_error(
            "terrain source dimensions overflow represented value count");
    }
    return nx * ny;
}

bool
coordinate_in_closed_interval(
    amrex::Real value,
    amrex::Real lo,
    amrex::Real hi) noexcept
{
    const amrex::Real scale =
        std::max(
            amrex::Real(1),
            std::max(
                std::abs(value),
                std::max(std::abs(lo), std::abs(hi))));

    const amrex::Real tolerance =
        amrex::Real(4096)
        * std::numeric_limits<amrex::Real>::epsilon()
        * scale;

    return value >= lo - tolerance
        && value <= hi + tolerance;
}

struct AxisBracket
{
    std::size_t lower{};
    amrex::Real upper_weight{};
};

AxisBracket
axis_bracket(
    const std::vector<amrex::Real>& coordinates,
    amrex::Real query)
{
    if (query <= coordinates.front()) {
        return {0U, amrex::Real(0)};
    }

    if (query >= coordinates.back()) {
        return {
            coordinates.size() - 2U,
            amrex::Real(1)};
    }

    const auto upper =
        std::upper_bound(
            coordinates.begin(),
            coordinates.end(),
            query);

    const std::size_t upper_index =
        static_cast<std::size_t>(
            std::distance(
                coordinates.begin(),
                upper));

    const std::size_t lower_index =
        upper_index - 1U;

    const amrex::Real lo =
        coordinates[lower_index];
    const amrex::Real hi =
        coordinates[upper_index];

    return {
        lower_index,
        (query - lo) / (hi - lo)};
}

} // namespace

ERFTerrainSource::ERFTerrainSource(
    std::vector<amrex::Real> x_m,
    std::vector<amrex::Real> y_m,
    std::vector<amrex::Real> elevation_m)
    : x_m_(std::move(x_m)),
      y_m_(std::move(y_m)),
      elevation_m_(std::move(elevation_m))
{
    require(
        x_m_.size() >= 2U,
        "terrain source requires at least two x coordinates");
    require(
        y_m_.size() >= 2U,
        "terrain source requires at least two y coordinates");

    require(
        x_m_.size()
            <= static_cast<std::size_t>(
                std::numeric_limits<int>::max())
        && y_m_.size()
            <= static_cast<std::size_t>(
                std::numeric_limits<int>::max()),
        "terrain source dimensions exceed supported integer indexing");

    require(
        elevation_m_.size()
            == checked_value_count(
                x_m_.size(),
                y_m_.size()),
        "terrain source elevation count does not match nx*ny");

    for (const amrex::Real value : x_m_) {
        require(
            std::isfinite(value),
            "terrain source x coordinates must be finite");
    }

    for (const amrex::Real value : y_m_) {
        require(
            std::isfinite(value),
            "terrain source y coordinates must be finite");
    }

    for (const amrex::Real value : elevation_m_) {
        require(
            std::isfinite(value),
            "terrain source elevations must be finite");
    }

    for (std::size_t i = 1U; i < x_m_.size(); ++i) {
        require(
            x_m_[i] > x_m_[i - 1U],
            "terrain source x coordinates must be strictly increasing");
    }

    for (std::size_t j = 1U; j < y_m_.size(); ++j) {
        require(
            y_m_[j] > y_m_[j - 1U],
            "terrain source y coordinates must be strictly increasing");
    }
}

ERFTerrainSource
ERFTerrainSource::read_regular_text_file(
    const std::string& filename)
{
    int nx = 0;
    int ny = 0;
    std::vector<amrex::Real> x_m;
    std::vector<amrex::Real> y_m;
    std::vector<amrex::Real> elevation_m;
    unsigned long long source_fingerprint_wire = 0;

    if (amrex::ParallelDescriptor::IOProcessor()) {
        source_fingerprint_wire =
            static_cast<unsigned long long>(
                fingerprint_file_bytes(filename));
        amrex::Print()
            << "Reading terrain source file: "
            << filename
            << '\n';

        std::ifstream file(filename);
        if (!file.is_open()) {
            amrex::Abort(
                "Error: Could not open terrain file "
                + filename);
        }

        if (file.peek()
            == std::ifstream::traits_type::eof()) {
            amrex::Abort(
                "Error: Terrain file "
                + filename
                + " is empty");
        }

        int line_number = 1;
        nx =
            read_single_value<int>(
                file,
                line_number++,
                filename);
        ny =
            read_single_value<int>(
                file,
                line_number++,
                filename);

        if (nx < 2 || ny < 2) {
            amrex::Abort(
                "Terrain source file "
                + filename
                + " requires nx >= 2 and ny >= 2");
        }

        x_m.resize(
            static_cast<std::size_t>(nx));
        y_m.resize(
            static_cast<std::size_t>(ny));
        elevation_m.resize(
            static_cast<std::size_t>(nx)
            * static_cast<std::size_t>(ny));

        for (int i = 0; i < nx; ++i) {
            x_m[static_cast<std::size_t>(i)] =
                read_single_value<amrex::Real>(
                    file,
                    line_number++,
                    filename);
        }

        for (int j = 0; j < ny; ++j) {
            y_m[static_cast<std::size_t>(j)] =
                read_single_value<amrex::Real>(
                    file,
                    line_number++,
                    filename);
        }

        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny; ++j) {
                elevation_m[
                    static_cast<std::size_t>(i)
                        * static_cast<std::size_t>(ny)
                    + static_cast<std::size_t>(j)] =
                    read_single_value<amrex::Real>(
                        file,
                        line_number++,
                        filename);
            }
        }
    }

    const int root =
        amrex::ParallelDescriptor::IOProcessorNumber();

    amrex::ParallelDescriptor::Bcast(
        &nx,
        1,
        root);
    amrex::ParallelDescriptor::Bcast(
        &ny,
        1,
        root);
    amrex::ParallelDescriptor::Bcast(
        &source_fingerprint_wire,
        1,
        root);

    if (nx < 2 || ny < 2) {
        amrex::Abort(
            "broadcast terrain source dimensions are invalid");
    }

    x_m.resize(
        static_cast<std::size_t>(nx));
    y_m.resize(
        static_cast<std::size_t>(ny));
    elevation_m.resize(
        static_cast<std::size_t>(nx)
        * static_cast<std::size_t>(ny));

    amrex::ParallelDescriptor::Bcast(
        x_m.data(),
        nx,
        root);
    amrex::ParallelDescriptor::Bcast(
        y_m.data(),
        ny,
        root);
    amrex::ParallelDescriptor::Bcast(
        elevation_m.data(),
        nx * ny,
        root);

    ERFTerrainSource result(
        std::move(x_m),
        std::move(y_m),
        std::move(elevation_m));

    result.source_fingerprint_fnv1a64_ =
        static_cast<std::uint64_t>(
            source_fingerprint_wire);

    return result;
}

bool
ERFTerrainSource::contains(
    amrex::Real x_m,
    amrex::Real y_m) const noexcept
{
    return coordinate_in_closed_interval(
               x_m,
               x_m_.front(),
               x_m_.back())
        && coordinate_in_closed_interval(
               y_m,
               y_m_.front(),
               y_m_.back());
}

amrex::Real
ERFTerrainSource::sample(
    amrex::Real x_m,
    amrex::Real y_m) const
{
    require(
        std::isfinite(x_m)
            && std::isfinite(y_m),
        "terrain source sample coordinates must be finite");

    require(
        contains(x_m, y_m),
        "terrain source sample lies outside represented domain");

    x_m =
        std::min(
            std::max(x_m, x_m_.front()),
            x_m_.back());
    y_m =
        std::min(
            std::max(y_m, y_m_.front()),
            y_m_.back());

    const AxisBracket xb =
        axis_bracket(
            x_m_,
            x_m);
    const AxisBracket yb =
        axis_bracket(
            y_m_,
            y_m);

    const std::size_t ny =
        y_m_.size();

    const auto z =
        [this, ny](
            std::size_t i,
            std::size_t j) {
            return elevation_m_[i * ny + j];
        };

    const amrex::Real z00 =
        z(xb.lower, yb.lower);
    const amrex::Real z10 =
        z(xb.lower + 1U, yb.lower);
    const amrex::Real z01 =
        z(xb.lower, yb.lower + 1U);
    const amrex::Real z11 =
        z(xb.lower + 1U, yb.lower + 1U);

    const amrex::Real lower =
        z00
        + xb.upper_weight
            * (z10 - z00);
    const amrex::Real upper =
        z01
        + xb.upper_weight
            * (z11 - z01);

    return lower
        + yb.upper_weight
            * (upper - lower);
}

void
ERFTerrainSource::fill_nodal_surface(
    const amrex::Geometry& geometry,
    amrex::FArrayBox& terrain_fab) const
{
    const auto prob_lo =
        geometry.ProbLoArray();

    amrex::Gpu::DeviceVector<amrex::Real>
        d_x(x_m_.size());
    amrex::Gpu::DeviceVector<amrex::Real>
        d_y(y_m_.size());
    amrex::Gpu::DeviceVector<amrex::Real>
        d_z(elevation_m_.size());

    amrex::Gpu::copy(
        amrex::Gpu::hostToDevice,
        x_m_.begin(),
        x_m_.end(),
        d_x.begin());
    amrex::Gpu::copy(
        amrex::Gpu::hostToDevice,
        y_m_.begin(),
        y_m_.end(),
        d_y.begin());
    amrex::Gpu::copy(
        amrex::Gpu::hostToDevice,
        elevation_m_.begin(),
        elevation_m_.end(),
        d_z.begin());

    const amrex::Real* x_values =
        d_x.data();
    const amrex::Real* y_values =
        d_y.data();
    const amrex::Real* z_values =
        d_z.data();

    const int nx =
        static_cast<int>(x_m_.size());
    const int ny =
        static_cast<int>(y_m_.size());

    const auto dx =
        geometry.CellSizeArray();
    const amrex::Box& domain =
        geometry.Domain();

    const int ilo =
        domain.smallEnd(0);
    const int jlo =
        domain.smallEnd(1);
    const int klo =
        domain.smallEnd(2);
    const int ihi =
        domain.bigEnd(0) + 1;
    const int jhi =
        domain.bigEnd(1) + 1;

    const amrex::Box surface_box =
        terrain_fab.box();
    const auto terrain =
        terrain_fab.array();

    amrex::ParallelFor(
        surface_box,
        [=] AMREX_GPU_DEVICE(
            int i,
            int j,
            int)
        {
            const int ii =
                amrex::min(
                    amrex::max(i, ilo),
                    ihi);
            const int jj =
                amrex::min(
                    amrex::max(j, jlo),
                    jhi);

            const amrex::Real x =
                prob_lo[0]
                + static_cast<amrex::Real>(
                      ii - ilo)
                    * dx[0];
            const amrex::Real y =
                prob_lo[1]
                + static_cast<amrex::Real>(
                      jj - jlo)
                    * dx[1];

            int ix = 0;
            amrex::Real tx =
                amrex::Real(0);

            if (x <= x_values[0]) {
                ix = 0;
                tx = amrex::Real(0);
            } else if (x >= x_values[nx - 1]) {
                ix = nx - 2;
                tx = amrex::Real(1);
            } else {
                int lo = 0;
                int hi = nx - 1;
                while (hi - lo > 1) {
                    const int mid =
                        lo + (hi - lo) / 2;
                    if (x_values[mid] <= x) {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                ix = lo;
                tx =
                    (x - x_values[ix])
                    / (x_values[ix + 1]
                       - x_values[ix]);
            }

            int jy = 0;
            amrex::Real ty =
                amrex::Real(0);

            if (y <= y_values[0]) {
                jy = 0;
                ty = amrex::Real(0);
            } else if (y >= y_values[ny - 1]) {
                jy = ny - 2;
                ty = amrex::Real(1);
            } else {
                int lo = 0;
                int hi = ny - 1;
                while (hi - lo > 1) {
                    const int mid =
                        lo + (hi - lo) / 2;
                    if (y_values[mid] <= y) {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                jy = lo;
                ty =
                    (y - y_values[jy])
                    / (y_values[jy + 1]
                       - y_values[jy]);
            }

            const int index00 =
                ix * ny + jy;
            const int index10 =
                (ix + 1) * ny + jy;
            const int index01 =
                ix * ny + (jy + 1);
            const int index11 =
                (ix + 1) * ny + (jy + 1);

            const amrex::Real lower =
                z_values[index00]
                + tx
                    * (z_values[index10]
                       - z_values[index00]);
            const amrex::Real upper =
                z_values[index01]
                + tx
                    * (z_values[index11]
                       - z_values[index01]);

            terrain(i, j, klo) =
                lower
                + ty * (upper - lower);
        });

    amrex::Gpu::streamSynchronize();
}
