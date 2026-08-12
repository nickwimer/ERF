#include "ERF_FireSpreadOutput.H"

#include <ERF_FireSpreadRuntime.H>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ERFFire
{
namespace
{

std::string
indexed_name(const char* stem, int step_index)
{
    std::ostringstream name;
    name << stem
         << "_"
         << std::setw(6)
         << std::setfill('0')
         << step_index
         << ".csv";
    return name.str();
}

std::ofstream
open_output(const std::filesystem::path& path)
{
    std::ofstream stream(path, std::ios::out | std::ios::trunc);
    if (!stream.is_open()) {
        throw std::runtime_error(
            "unable to open ERF-Fire visualization output " + path.string());
    }
    stream << std::setprecision(17);
    return stream;
}

} // namespace

void
write_erf_fire_spread_snapshot(
    const ERFFireSpreadRuntime& runtime,
    const std::string& output_dir,
    int step_index)
{
    if (output_dir.empty()) {
        throw std::invalid_argument(
            "fire output directory must not be empty");
    }
    if (step_index < 0) {
        throw std::invalid_argument(
            "fire output step index must be nonnegative");
    }

    const std::filesystem::path directory(output_dir);
    std::filesystem::create_directories(directory);

    const std::string perimeter_name =
        indexed_name("perimeter", step_index);
    const std::string raster_name =
        indexed_name("raster", step_index);

    {
        auto stream = open_output(directory / perimeter_name);
        stream << "time_s,vertex_index,x_m,y_m\n";

        const auto& vertices = runtime.perimeter().vertices_m();
        for (std::size_t i = 0; i < vertices.size(); ++i) {
            stream
                << runtime.current_time_s() << ","
                << i << ","
                << vertices[i].x << ","
                << vertices[i].y << "\n";
        }
    }

    {
        auto stream = open_output(directory / raster_name);
        stream
            << "time_s,i,j,xlo_m,xhi_m,ylo_m,yhi_m,"
            << "burned_fraction,has_arrived,first_arrival_time_s\n";

        const auto& burned = runtime.burned_fraction_raster();
        const auto& arrival = runtime.first_arrival_raster();
        const auto& geometry = burned.geometry();

        for (std::size_t j = 0; j < geometry.ny; ++j) {
            for (std::size_t i = 0; i < geometry.nx; ++i) {
                const auto cell = burned.cell_bounds(i, j);
                const bool arrived = arrival.has_arrived(i, j);

                stream
                    << runtime.current_time_s() << ","
                    << i << ","
                    << j << ","
                    << cell.xlo_m << ","
                    << cell.xhi_m << ","
                    << cell.ylo_m << ","
                    << cell.yhi_m << ","
                    << burned.burned_fraction(i, j) << ","
                    << (arrived ? 1 : 0) << ",";

                if (arrived) {
                    stream
                        << arrival.first_arrival_time_s(i, j);
                }
                stream << "\n";
            }
        }
    }

    const std::filesystem::path summary_path =
        directory / "summary.csv";
    std::ofstream summary;
    if (step_index == 0) {
        summary.open(
            summary_path,
            std::ios::out | std::ios::trunc);
    } else {
        summary.open(
            summary_path,
            std::ios::out | std::ios::app);
    }

    if (!summary.is_open()) {
        throw std::runtime_error(
            "unable to open ERF-Fire visualization manifest "
            + summary_path.string());
    }

    summary << std::setprecision(17);
    if (step_index == 0) {
        summary
            << "step,time_s,vertex_count,burned_area_m2,"
            << "arrived_cell_count,perimeter_file,raster_file\n";
    }

    summary
        << step_index << ","
        << runtime.current_time_s() << ","
        << runtime.perimeter().size() << ","
        << runtime.burned_fraction_raster().burned_area_m2() << ","
        << runtime.first_arrival_raster().arrived_cell_count() << ","
        << perimeter_name << ","
        << raster_name << "\n";
}

} // namespace ERFFire
