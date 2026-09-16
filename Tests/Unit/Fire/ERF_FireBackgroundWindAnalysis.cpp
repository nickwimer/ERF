#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

void
require(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<std::string>
split_csv_line(const std::string& line)
{
    std::vector<std::string> fields;
    std::string field;
    std::stringstream stream(line);
    while (std::getline(stream, field, ',')) {
        fields.push_back(field);
    }
    return fields;
}

std::string
normalized_column_name(std::string value)
{
    std::string result;
    result.reserve(value.size());
    for (char ch : value) {
        if (ch != '_' && ch != ' ') {
            result.push_back(
                static_cast<char>(
                    std::tolower(
                        static_cast<unsigned char>(ch))));
        }
    }
    return result;
}

struct GeometryMetrics
{
    std::size_t vertices{};
    amrex::Real area_m2{};
    amrex::Real perimeter_m{};
    amrex::Real centroid_x_m{};
    amrex::Real centroid_y_m{};
    amrex::Real head_distance_m{};
    amrex::Real back_distance_m{};
    amrex::Real north_distance_m{};
    amrex::Real south_distance_m{};
    amrex::Real x_span_m{};
    amrex::Real y_span_m{};
};

GeometryMetrics
read_geometry(
    const std::string& path,
    amrex::Real ignition_x_m,
    amrex::Real ignition_y_m)
{
    std::ifstream input(path);
    require(
        input.good(),
        "Could not open Fire perimeter CSV " + path);

    std::string line;
    require(
        static_cast<bool>(std::getline(input, line)),
        "Fire perimeter CSV is empty " + path);

    const auto header = split_csv_line(line);
    int x_col = -1;
    int y_col = -1;

    for (std::size_t n = 0; n < header.size(); ++n) {
        const std::string name =
            normalized_column_name(header[n]);
        if (name == "xm" || name == "x") {
            x_col = static_cast<int>(n);
        }
        if (name == "ym" || name == "y") {
            y_col = static_cast<int>(n);
        }
    }

    require(
        x_col >= 0 && y_col >= 0,
        "Perimeter CSV must contain x_m/y_m columns");

    std::vector<amrex::Real> xs;
    std::vector<amrex::Real> ys;

    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }

        const auto fields = split_csv_line(line);
        const int required_col = std::max(x_col, y_col);
        require(
            static_cast<int>(fields.size()) > required_col,
            "Perimeter CSV row is missing coordinates");

        const amrex::Real x =
            static_cast<amrex::Real>(
                std::stod(fields[static_cast<std::size_t>(x_col)]));
        const amrex::Real y =
            static_cast<amrex::Real>(
                std::stod(fields[static_cast<std::size_t>(y_col)]));
        require(
            std::isfinite(x) && std::isfinite(y),
            "Perimeter CSV contains non-finite coordinates");

        xs.push_back(x);
        ys.push_back(y);
    }

    require(
        xs.size() >= 3 && xs.size() == ys.size(),
        "Perimeter CSV requires at least three coordinate pairs");

    if (xs.front() == xs.back() && ys.front() == ys.back()) {
        xs.pop_back();
        ys.pop_back();
    }

    require(
        xs.size() >= 3,
        "Perimeter CSV collapsed below three unique vertices");

    amrex::Real twice_area = amrex::Real(0.0);
    amrex::Real centroid_x_numerator = amrex::Real(0.0);
    amrex::Real centroid_y_numerator = amrex::Real(0.0);
    amrex::Real perimeter = amrex::Real(0.0);

    for (std::size_t n = 0; n < xs.size(); ++n) {
        const std::size_t next = (n + 1) % xs.size();
        const amrex::Real cross =
            xs[n] * ys[next] - xs[next] * ys[n];
        twice_area += cross;
        centroid_x_numerator +=
            (xs[n] + xs[next]) * cross;
        centroid_y_numerator +=
            (ys[n] + ys[next]) * cross;
        perimeter +=
            std::hypot(xs[next] - xs[n], ys[next] - ys[n]);
    }

    require(
        std::abs(twice_area) > amrex::Real(0.0),
        "Perimeter polygon has zero signed area");

    const auto [xmin_it, xmax_it] =
        std::minmax_element(xs.begin(), xs.end());
    const auto [ymin_it, ymax_it] =
        std::minmax_element(ys.begin(), ys.end());

    GeometryMetrics result;
    result.vertices = xs.size();
    result.area_m2 =
        amrex::Real(0.5) * std::abs(twice_area);
    result.perimeter_m = perimeter;
    result.centroid_x_m =
        centroid_x_numerator
        / (amrex::Real(3.0) * twice_area);
    result.centroid_y_m =
        centroid_y_numerator
        / (amrex::Real(3.0) * twice_area);
    result.head_distance_m = *xmax_it - ignition_x_m;
    result.back_distance_m = ignition_x_m - *xmin_it;
    result.north_distance_m = *ymax_it - ignition_y_m;
    result.south_distance_m = ignition_y_m - *ymin_it;
    result.x_span_m = *xmax_it - *xmin_it;
    result.y_span_m = *ymax_it - *ymin_it;

    require(
        std::isfinite(result.area_m2)
            && std::isfinite(result.perimeter_m)
            && std::isfinite(result.centroid_x_m)
            && std::isfinite(result.centroid_y_m)
            && std::isfinite(result.head_distance_m)
            && std::isfinite(result.back_distance_m)
            && std::isfinite(result.north_distance_m)
            && std::isfinite(result.south_distance_m),
        "Background-wind geometry metrics are non-finite");

    return result;
}

} // namespace

int
main(int argc, char** argv)
{
    amrex::Initialize(argc, argv);
    int result = 0;

    try {
        amrex::ParmParse pp("analysis");

        std::string one_way_perimeter;
        std::string two_way_perimeter;
        amrex::Real ignition_x_m{};
        amrex::Real ignition_y_m{};

        pp.get("one_way_perimeter", one_way_perimeter);
        pp.get("two_way_perimeter", two_way_perimeter);
        pp.get("ignition_x_m", ignition_x_m);
        pp.get("ignition_y_m", ignition_y_m);

        const GeometryMetrics one =
            read_geometry(
                one_way_perimeter,
                ignition_x_m,
                ignition_y_m);
        const GeometryMetrics two =
            read_geometry(
                two_way_perimeter,
                ignition_x_m,
                ignition_y_m);

        require(
            one.vertices == two.vertices,
            "One-way/two-way final perimeter vertex counts differ");

        const amrex::Real one_head_back_asymmetry_m =
            one.head_distance_m - one.back_distance_m;
        const amrex::Real one_centroid_shift_m =
            one.centroid_x_m - ignition_x_m;

        const amrex::Real delta_area_m2 =
            two.area_m2 - one.area_m2;
        const amrex::Real delta_head_m =
            two.head_distance_m - one.head_distance_m;
        const amrex::Real back_suppression_m =
            one.back_distance_m - two.back_distance_m;
        const amrex::Real delta_centroid_x_m =
            two.centroid_x_m - one.centroid_x_m;

        require(
            one_head_back_asymmetry_m >= amrex::Real(0.10),
            "One-way background wind does not produce sufficient head/back asymmetry");
        require(
            one_centroid_shift_m >= amrex::Real(0.05),
            "One-way background wind does not shift the Fire centroid downwind");
        require(
            delta_area_m2 >= amrex::Real(0.30),
            "Two-way feedback does not increase burned perimeter area sufficiently");
        require(
            delta_head_m >= amrex::Real(0.020),
            "Two-way feedback does not advance the downwind head sufficiently");
        require(
            back_suppression_m >= amrex::Real(0.0025),
            "Two-way feedback does not reduce backing spread sufficiently");
        require(
            delta_centroid_x_m >= amrex::Real(0.010),
            "Two-way feedback does not shift the Fire centroid farther downwind");

        amrex::Print()
            << std::setprecision(17)
            << "BACKGROUND_WIND_METRICS"
            << " one_area_m2=" << one.area_m2
            << " two_area_m2=" << two.area_m2
            << " delta_area_m2=" << delta_area_m2
            << " one_head_distance_m=" << one.head_distance_m
            << " two_head_distance_m=" << two.head_distance_m
            << " delta_head_distance_m=" << delta_head_m
            << " one_back_distance_m=" << one.back_distance_m
            << " two_back_distance_m=" << two.back_distance_m
            << " back_suppression_m=" << back_suppression_m
            << " one_centroid_x_m=" << one.centroid_x_m
            << " two_centroid_x_m=" << two.centroid_x_m
            << " delta_centroid_x_m=" << delta_centroid_x_m
            << " one_head_back_asymmetry_m="
            << one_head_back_asymmetry_m
            << " one_x_span_m=" << one.x_span_m
            << " two_x_span_m=" << two.x_span_m
            << " one_y_span_m=" << one.y_span_m
            << " two_y_span_m=" << two.y_span_m
            << "\n";
    } catch (const std::exception& error) {
        amrex::Print()
            << "Background-wind analysis error: "
            << error.what()
            << "\n";
        result = 2;
    }

    amrex::Finalize();
    return result;
}
