#ifndef ERF_FIRE_TEST_UTILS_H
#define ERF_FIRE_TEST_UTILS_H

#include <ERF_FirePerimeter.H>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ERFFireTest
{

inline constexpr amrex::Real pi =
    amrex::Real(3.141592653589793238462643383279502884L);

inline ERFFire::FirePerimeter
make_circle (std::size_t vertex_count, amrex::Real radius_m)
{
    std::vector<ERFFire::FireVec2> vertices;
    vertices.reserve(vertex_count);

    for (std::size_t i = 0; i < vertex_count; ++i) {
        const amrex::Real theta =
            amrex::Real(2.0) * pi
            * static_cast<amrex::Real>(i)
            / static_cast<amrex::Real>(vertex_count);

        vertices.push_back({
            radius_m * std::cos(theta),
            radius_m * std::sin(theta)
        });
    }

    return ERFFire::FirePerimeter(std::move(vertices));
}

inline void
write_snapshot (
    const std::string& filename,
    const ERFFire::FirePerimeter& perimeter)
{
    const char* output_dir = std::getenv("ERF_FIRE_TEST_OUTPUT_DIR");
    if (output_dir == nullptr || output_dir[0] == '\0') {
        return;
    }

    const std::filesystem::path directory(output_dir);
    std::filesystem::create_directories(directory);

    std::ofstream stream(directory / filename);
    if (!stream.good()) {
        throw std::runtime_error(
            "Unable to open ERF-Fire test snapshot: " +
            (directory / filename).string());
    }

    stream << "vertex,x_m,y_m\n";
    stream << std::setprecision(17);
    for (std::size_t i = 0; i < perimeter.size(); ++i) {
        const auto& vertex = perimeter.vertices_m()[i];
        stream << i << ',' << vertex.x << ',' << vertex.y << '\n';
    }
}

inline void
maybe_write_snapshot (
    const std::string& case_name,
    amrex::Real time_s,
    const ERFFire::FirePerimeter& perimeter)
{
    const char* output_dir = std::getenv("ERF_FIRE_TEST_OUTPUT_DIR");
    if (output_dir == nullptr || output_dir[0] == '\0') {
        return;
    }

    std::ostringstream filename;
    filename << case_name
             << "_t"
             << std::setw(3)
             << std::setfill('0')
             << static_cast<long long>(std::llround(time_s))
             << ".csv";

    write_snapshot(filename.str(), perimeter);
}

} // namespace ERFFireTest

#endif // ERF_FIRE_TEST_UTILS_H
