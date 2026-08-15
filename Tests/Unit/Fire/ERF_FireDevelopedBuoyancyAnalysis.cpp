#include "ERF_FireFlatEnvironmentSampler.H"

#include <AMReX.H>
#include <AMReX_Arena.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <array>
#include <cctype>
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

constexpr amrex::Real source_H_m = amrex::Real(25.0);
constexpr amrex::Real response_threshold_mps = amrex::Real(0.01);

void
require(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

bool
has_variable(
    const amrex::Vector<std::string>& names,
    const std::string& name)
{
    return std::find(names.begin(), names.end(), name) != names.end();
}

amrex::Real
scaled_tolerance(amrex::Real value)
{
    return amrex::Real(256.0)
        * std::numeric_limits<amrex::Real>::epsilon()
        * std::max(amrex::Real(1.0), std::abs(value));
}

struct BuoyancyMetrics
{
    amrex::Real time_s{};
    amrex::Real top_m{};
    amrex::Real max_delta_theta_K{};
    amrex::Real max_delta_w_mps{};
    amrex::Real peak_delta_w_z_m{};
    amrex::Real positive_w_volume_integral_m4ps{};
    amrex::Real delta_w_l2_volume_integral_m5ps2{};
    amrex::Real w_response_z50_m{};
    amrex::Real w_response_z95_m{};
    amrex::Real w2_response_z50_m{};
    amrex::Real w2_response_z95_m{};
    amrex::Real positive_w_fraction_above_source_z95{};
    amrex::Real w2_fraction_above_source_z95{};
    amrex::Real max_delta_w_above_source_z95_mps{};
    amrex::Real w_threshold_top_m{};
    amrex::Real w_threshold_0p05_top_m{-amrex::Real(1.0)};
    amrex::Real w_threshold_0p10_top_m{-amrex::Real(1.0)};
    amrex::Real w_threshold_0p25_top_m{-amrex::Real(1.0)};
    amrex::Real w_threshold_0p50_top_m{-amrex::Real(1.0)};
    amrex::Real positive_theta_volume_integral_K_m3{};
};

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

struct PerimeterGeometryMetrics
{
    std::size_t vertices{};
    amrex::Real area_m2{};
    amrex::Real perimeter_m{};
    amrex::Real centroid_x_m{};
    amrex::Real centroid_y_m{};
    amrex::Real mean_radius_m{};
};

std::vector<std::array<amrex::Real, 2>>
read_perimeter_points(const std::string& path)
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

    std::vector<std::array<amrex::Real, 2>> points;
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
        points.push_back({x, y});
    }

    require(
        points.size() >= 3,
        "Perimeter CSV requires at least three vertices");

    if (points.front() == points.back()) {
        points.pop_back();
    }

    require(
        points.size() >= 3,
        "Perimeter CSV collapsed below three unique vertices");

    return points;
}

PerimeterGeometryMetrics
read_perimeter_geometry(const std::string& path)
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

    const auto points = read_perimeter_points(path);

    amrex::Real twice_area = amrex::Real(0.0);
    amrex::Real centroid_x_numerator = amrex::Real(0.0);
    amrex::Real centroid_y_numerator = amrex::Real(0.0);
    amrex::Real perimeter = amrex::Real(0.0);

    for (std::size_t n = 0; n < points.size(); ++n) {
        const auto& p0 = points[n];
        const auto& p1 = points[(n + 1) % points.size()];
        const amrex::Real cross =
            p0[0] * p1[1] - p1[0] * p0[1];
        twice_area += cross;
        centroid_x_numerator += (p0[0] + p1[0]) * cross;
        centroid_y_numerator += (p0[1] + p1[1]) * cross;

        const amrex::Real dx = p1[0] - p0[0];
        const amrex::Real dy = p1[1] - p0[1];
        perimeter += std::hypot(dx, dy);
    }

    require(
        std::abs(twice_area) > amrex::Real(0.0),
        "Perimeter polygon has zero signed area");

    const amrex::Real centroid_x =
        centroid_x_numerator / (amrex::Real(3.0) * twice_area);
    const amrex::Real centroid_y =
        centroid_y_numerator / (amrex::Real(3.0) * twice_area);

    amrex::Real radius_sum = amrex::Real(0.0);
    for (const auto& point : points) {
        radius_sum +=
            std::hypot(
                point[0] - centroid_x,
                point[1] - centroid_y);
    }

    PerimeterGeometryMetrics result;
    result.vertices = points.size();
    result.area_m2 = amrex::Real(0.5) * std::abs(twice_area);
    result.perimeter_m = perimeter;
    result.centroid_x_m = centroid_x;
    result.centroid_y_m = centroid_y;
    result.mean_radius_m =
        radius_sum / static_cast<amrex::Real>(points.size());
    return result;
}

struct RadialWindMetrics
{
    std::size_t samples{};
    amrex::Real reference_height_m{};
    amrex::Real one_mean_radial_mps{};
    amrex::Real two_mean_radial_mps{};
    amrex::Real delta_mean_radial_mps{};
    amrex::Real delta_rms_radial_mps{};
    amrex::Real delta_min_radial_mps{
        std::numeric_limits<amrex::Real>::infinity()};
    amrex::Real delta_max_radial_mps{
        -std::numeric_limits<amrex::Real>::infinity()};
    amrex::Real inward_fraction{};
    amrex::Real delta_rms_tangential_mps{};
};

amrex::Real
sample_cell_centered_trilinear(
    const amrex::Array4<const amrex::Real>& values,
    const amrex::Box& domain,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& prob_lo,
    const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& dx,
    amrex::Real x,
    amrex::Real y,
    amrex::Real z)
{
    const std::array<amrex::Real, 3> position{x, y, z};
    std::array<int, 3> lower{};
    std::array<amrex::Real, 3> upper_weight{};

    for (int dir = 0; dir < 3; ++dir) {
        const amrex::Real logical =
            (position[static_cast<std::size_t>(dir)] - prob_lo[dir])
                / dx[dir]
            - amrex::Real(0.5)
            + amrex::Real(domain.smallEnd(dir));

        const int lo_index =
            static_cast<int>(std::floor(logical));
        const amrex::Real weight =
            logical - amrex::Real(lo_index);

        require(
            lo_index >= domain.smallEnd(dir)
                && lo_index + 1 <= domain.bigEnd(dir),
            "10-m perimeter sample lies outside plotfile cell-center support");

        lower[static_cast<std::size_t>(dir)] = lo_index;
        upper_weight[static_cast<std::size_t>(dir)] = weight;
    }

    amrex::Real result = amrex::Real(0.0);
    for (int kk = 0; kk <= 1; ++kk) {
        const amrex::Real wz =
            kk == 0
                ? amrex::Real(1.0) - upper_weight[2]
                : upper_weight[2];
        for (int jj = 0; jj <= 1; ++jj) {
            const amrex::Real wy =
                jj == 0
                    ? amrex::Real(1.0) - upper_weight[1]
                    : upper_weight[1];
            for (int ii = 0; ii <= 1; ++ii) {
                const amrex::Real wx =
                    ii == 0
                        ? amrex::Real(1.0) - upper_weight[0]
                        : upper_weight[0];
                result +=
                    wx * wy * wz
                    * values(
                        lower[0] + ii,
                        lower[1] + jj,
                        lower[2] + kk);
            }
        }
    }
    return result;
}

RadialWindMetrics
analyze_radial_wind(
    const std::string& one_way_plot,
    const std::string& two_way_plot,
    const std::string& reference_perimeter,
    amrex::Real reference_height_m)
{
    amrex::PlotFileData one_way(one_way_plot);
    amrex::PlotFileData two_way(two_way_plot);

    require(
        one_way.finestLevel() == 0 && two_way.finestLevel() == 0,
        "Radial-wind analysis requires level-0-only plotfiles");
    require(
        one_way.spaceDim() == 3 && two_way.spaceDim() == 3,
        "Radial-wind analysis requires 3-D plotfiles");

    const auto& one_names = one_way.varNames();
    const auto& two_names = two_way.varNames();
    for (const std::string& name : {"x_velocity", "y_velocity"}) {
        require(
            has_variable(one_names, name),
            "One-way plotfile is missing " + name);
        require(
            has_variable(two_names, name),
            "Two-way plotfile is missing " + name);
    }

    const auto domain = one_way.probDomain(0);
    require(
        domain == two_way.probDomain(0),
        "Matched plotfile domains differ");

    const auto one_dx_vector = one_way.cellSize(0);
    const auto two_dx_vector = two_way.cellSize(0);
    const auto one_lo_vector = one_way.probLo();
    const auto two_lo_vector = two_way.probLo();

    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx{};
    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo{};
    for (int dir = 0; dir < 3; ++dir) {
        require(
            std::abs(one_dx_vector[dir] - two_dx_vector[dir])
                <= scaled_tolerance(one_dx_vector[dir]),
            "Matched plotfile cell sizes differ");
        require(
            std::abs(one_lo_vector[dir] - two_lo_vector[dir])
                <= scaled_tolerance(one_lo_vector[dir]),
            "Matched plotfile lower bounds differ");
        dx[dir] = one_dx_vector[dir];
        prob_lo[dir] = one_lo_vector[dir];
    }

    two_way.syncDistributionMap(one_way);

    auto one_u = one_way.get(0, "x_velocity");
    auto one_v = one_way.get(0, "y_velocity");
    auto two_u = two_way.get(0, "x_velocity");
    auto two_v = two_way.get(0, "y_velocity");

    amrex::FArrayBox one_u_host(
        domain, 1, amrex::The_Pinned_Arena());
    amrex::FArrayBox one_v_host(
        domain, 1, amrex::The_Pinned_Arena());
    amrex::FArrayBox two_u_host(
        domain, 1, amrex::The_Pinned_Arena());
    amrex::FArrayBox two_v_host(
        domain, 1, amrex::The_Pinned_Arena());

    one_u.copyTo(one_u_host, 0, 0, 1, 0);
    one_v.copyTo(one_v_host, 0, 0, 1, 0);
    two_u.copyTo(two_u_host, 0, 0, 1, 0);
    two_v.copyTo(two_v_host, 0, 0, 1, 0);

    const auto one_u_arr = one_u_host.const_array();
    const auto one_v_arr = one_v_host.const_array();
    const auto two_u_arr = two_u_host.const_array();
    const auto two_v_arr = two_v_host.const_array();

    const auto points = read_perimeter_points(reference_perimeter);
    const auto geometry = read_perimeter_geometry(reference_perimeter);

    RadialWindMetrics metrics;
    metrics.samples = points.size();
    metrics.reference_height_m = reference_height_m;

    amrex::Real radial_square_sum = amrex::Real(0.0);
    amrex::Real tangential_square_sum = amrex::Real(0.0);
    std::size_t inward_count = 0;

    for (const auto& point : points) {
        const amrex::Real rx =
            point[0] - geometry.centroid_x_m;
        const amrex::Real ry =
            point[1] - geometry.centroid_y_m;
        const amrex::Real radius = std::hypot(rx, ry);
        require(
            radius > amrex::Real(0.0),
            "Perimeter point coincides with its centroid");

        const amrex::Real nx = rx / radius;
        const amrex::Real ny = ry / radius;
        const amrex::Real tx = -ny;
        const amrex::Real ty = nx;

        const amrex::Real one_u_value =
            sample_cell_centered_trilinear(
                one_u_arr, domain, prob_lo, dx,
                point[0], point[1], reference_height_m);
        const amrex::Real one_v_value =
            sample_cell_centered_trilinear(
                one_v_arr, domain, prob_lo, dx,
                point[0], point[1], reference_height_m);
        const amrex::Real two_u_value =
            sample_cell_centered_trilinear(
                two_u_arr, domain, prob_lo, dx,
                point[0], point[1], reference_height_m);
        const amrex::Real two_v_value =
            sample_cell_centered_trilinear(
                two_v_arr, domain, prob_lo, dx,
                point[0], point[1], reference_height_m);

        const amrex::Real one_radial =
            one_u_value * nx + one_v_value * ny;
        const amrex::Real two_radial =
            two_u_value * nx + two_v_value * ny;
        const amrex::Real delta_radial =
            two_radial - one_radial;

        const amrex::Real one_tangential =
            one_u_value * tx + one_v_value * ty;
        const amrex::Real two_tangential =
            two_u_value * tx + two_v_value * ty;
        const amrex::Real delta_tangential =
            two_tangential - one_tangential;

        require(
            std::isfinite(one_radial)
                && std::isfinite(two_radial)
                && std::isfinite(delta_radial)
                && std::isfinite(delta_tangential),
            "Reconstructed perimeter wind is non-finite");

        metrics.one_mean_radial_mps += one_radial;
        metrics.two_mean_radial_mps += two_radial;
        metrics.delta_mean_radial_mps += delta_radial;
        radial_square_sum += delta_radial * delta_radial;
        tangential_square_sum +=
            delta_tangential * delta_tangential;
        metrics.delta_min_radial_mps =
            std::min(metrics.delta_min_radial_mps, delta_radial);
        metrics.delta_max_radial_mps =
            std::max(metrics.delta_max_radial_mps, delta_radial);
        if (delta_radial < amrex::Real(0.0)) {
            ++inward_count;
        }
    }

    const amrex::Real inverse_n =
        amrex::Real(1.0)
        / static_cast<amrex::Real>(metrics.samples);
    metrics.one_mean_radial_mps *= inverse_n;
    metrics.two_mean_radial_mps *= inverse_n;
    metrics.delta_mean_radial_mps *= inverse_n;
    metrics.delta_rms_radial_mps =
        std::sqrt(radial_square_sum * inverse_n);
    metrics.delta_rms_tangential_mps =
        std::sqrt(tangential_square_sum * inverse_n);
    metrics.inward_fraction =
        static_cast<amrex::Real>(inward_count) * inverse_n;

    return metrics;
}

void
print_radial_wind_metrics(
    const char* phase,
    const RadialWindMetrics& metrics)
{
    amrex::Print()
        << std::setprecision(17)
        << "RADIAL_WIND_METRICS"
        << " phase=" << phase
        << " samples=" << metrics.samples
        << " reference_height_m=" << metrics.reference_height_m
        << " one_mean_radial_mps="
        << metrics.one_mean_radial_mps
        << " two_mean_radial_mps="
        << metrics.two_mean_radial_mps
        << " delta_mean_radial_mps="
        << metrics.delta_mean_radial_mps
        << " delta_rms_radial_mps="
        << metrics.delta_rms_radial_mps
        << " delta_min_radial_mps="
        << metrics.delta_min_radial_mps
        << " delta_max_radial_mps="
        << metrics.delta_max_radial_mps
        << " inward_fraction="
        << metrics.inward_fraction
        << " delta_rms_tangential_mps="
        << metrics.delta_rms_tangential_mps
        << "\n";
}

ERFFire::FireFlatEnvironmentSampler
make_exact_flat_fire_sampler(
    const std::string& regular_plot_path,
    const std::string& u_plot_path,
    const std::string& v_plot_path,
    amrex::Real reference_height_m)
{
    amrex::PlotFileData regular_plot(regular_plot_path);
    amrex::PlotFileData u_plot(u_plot_path);
    amrex::PlotFileData v_plot(v_plot_path);

    require(
        regular_plot.finestLevel() == 0
            && u_plot.finestLevel() == 0
            && v_plot.finestLevel() == 0,
        "Exact Fire sampler requires level-0-only plotfiles");
    require(
        regular_plot.spaceDim() == 3
            && u_plot.spaceDim() == 3
            && v_plot.spaceDim() == 3,
        "Exact Fire sampler requires 3-D plotfiles");

    require(
        has_variable(u_plot.varNames(), "x_velocity_stag"),
        "Exact Fire sampler staggered-u plotfile is missing x_velocity_stag");
    require(
        has_variable(v_plot.varNames(), "y_velocity_stag"),
        "Exact Fire sampler staggered-v plotfile is missing y_velocity_stag");

    const auto cell_domain = regular_plot.probDomain(0);
    const auto expected_u =
        amrex::convert(cell_domain, amrex::IntVect(1, 0, 0));
    const auto expected_v =
        amrex::convert(cell_domain, amrex::IntVect(0, 1, 0));

    auto u_mf = u_plot.get(0, "x_velocity_stag");
    auto v_mf = v_plot.get(0, "y_velocity_stag");

    const amrex::Box u_domain = u_mf.boxArray().minimalBox();
    const amrex::Box v_domain = v_mf.boxArray().minimalBox();

    require(
        u_domain == expected_u,
        "Staggered-u plotfile layout does not match the ERF x-face domain");
    require(
        v_domain == expected_v,
        "Staggered-v plotfile layout does not match the ERF y-face domain");

    const auto dx = regular_plot.cellSize(0);
    const auto prob_lo = regular_plot.probLo();
    const auto u_dx = u_plot.cellSize(0);
    const auto v_dx = v_plot.cellSize(0);
    const auto u_lo = u_plot.probLo();
    const auto v_lo = v_plot.probLo();

    for (int dir = 0; dir < 3; ++dir) {
        require(
            std::abs(dx[dir] - u_dx[dir])
                    <= scaled_tolerance(dx[dir])
                && std::abs(dx[dir] - v_dx[dir])
                    <= scaled_tolerance(dx[dir]),
            "Regular and staggered plotfile cell sizes differ");
        require(
            std::abs(prob_lo[dir] - u_lo[dir])
                    <= scaled_tolerance(prob_lo[dir])
                && std::abs(prob_lo[dir] - v_lo[dir])
                    <= scaled_tolerance(prob_lo[dir]),
            "Regular and staggered plotfile lower bounds differ");
    }

    amrex::FArrayBox u_host(
        u_domain, 1, amrex::The_Pinned_Arena());
    amrex::FArrayBox v_host(
        v_domain, 1, amrex::The_Pinned_Arena());

    u_mf.copyTo(u_host, 0, 0, 1, 0);
    v_mf.copyTo(v_host, 0, 0, 1, 0);

    const auto u = u_host.const_array();
    const auto v = v_host.const_array();

    const std::size_t nx =
        static_cast<std::size_t>(cell_domain.length(0));
    const std::size_t ny =
        static_cast<std::size_t>(cell_domain.length(1));
    const std::size_t nz =
        static_cast<std::size_t>(cell_domain.length(2));

    ERFFire::FireFlatEnvironmentLayout2D layout(
        prob_lo[0],
        prob_lo[1],
        dx[0],
        dx[1],
        nx,
        ny);

    std::vector<amrex::Real> z_cell_center_m(nz);
    for (std::size_t n = 0; n < nz; ++n) {
        z_cell_center_m[n] =
            prob_lo[2]
            + (static_cast<amrex::Real>(n) + amrex::Real(0.5))
                * dx[2];
    }

    const ERFFire::FireVerticalLinearBracket bracket =
        ERFFire::fire_vertical_linear_bracket(
            z_cell_center_m,
            reference_height_m);

    const int k0 = cell_domain.smallEnd(2);
    const int lower_k =
        k0 + static_cast<int>(bracket.lower_k);
    const int upper_k =
        k0 + static_cast<int>(bracket.upper_k);

    std::vector<amrex::Real> u_snapshot(
        layout.u_storage_size(),
        amrex::Real(0.0));
    std::vector<amrex::Real> v_snapshot(
        layout.v_storage_size(),
        amrex::Real(0.0));

    const int u_ilo = u_domain.smallEnd(0);
    const int u_jlo = u_domain.smallEnd(1);
    for (std::size_t j = 0; j < ny; ++j) {
        for (std::size_t i = 0; i <= nx; ++i) {
            const int src_i =
                u_ilo + static_cast<int>(i);
            const int src_j =
                u_jlo + static_cast<int>(j);
            const amrex::Real value =
                ERFFire::fire_vertical_linear_interpolate(
                    u(src_i, src_j, lower_k),
                    u(src_i, src_j, upper_k),
                    bracket);
            u_snapshot[
                layout.u_storage_index(
                    static_cast<int>(i),
                    static_cast<int>(j))] = value;
        }
    }

    const int v_ilo = v_domain.smallEnd(0);
    const int v_jlo = v_domain.smallEnd(1);
    for (std::size_t j = 0; j <= ny; ++j) {
        for (std::size_t i = 0; i < nx; ++i) {
            const int src_i =
                v_ilo + static_cast<int>(i);
            const int src_j =
                v_jlo + static_cast<int>(j);
            const amrex::Real value =
                ERFFire::fire_vertical_linear_interpolate(
                    v(src_i, src_j, lower_k),
                    v(src_i, src_j, upper_k),
                    bracket);
            v_snapshot[
                layout.v_storage_index(
                    static_cast<int>(i),
                    static_cast<int>(j))] = value;
        }
    }

    return ERFFire::FireFlatEnvironmentSampler(
        std::move(layout),
        reference_height_m,
        std::move(u_snapshot),
        std::move(v_snapshot));
}

RadialWindMetrics
analyze_exact_fire_sampler_radial_wind(
    const std::string& one_regular_plot,
    const std::string& one_u_plot,
    const std::string& one_v_plot,
    const std::string& two_regular_plot,
    const std::string& two_u_plot,
    const std::string& two_v_plot,
    const std::string& reference_perimeter,
    amrex::Real reference_height_m)
{
    const auto one_sampler =
        make_exact_flat_fire_sampler(
            one_regular_plot,
            one_u_plot,
            one_v_plot,
            reference_height_m);
    const auto two_sampler =
        make_exact_flat_fire_sampler(
            two_regular_plot,
            two_u_plot,
            two_v_plot,
            reference_height_m);

    const auto points =
        read_perimeter_points(reference_perimeter);
    const auto geometry =
        read_perimeter_geometry(reference_perimeter);

    const auto& layout = one_sampler.layout();
    require(
        layout.xlo_m() == two_sampler.layout().xlo_m()
            && layout.ylo_m() == two_sampler.layout().ylo_m()
            && layout.xhi_m() == two_sampler.layout().xhi_m()
            && layout.yhi_m() == two_sampler.layout().yhi_m()
            && layout.dx_m() == two_sampler.layout().dx_m()
            && layout.dy_m() == two_sampler.layout().dy_m(),
        "One-way/two-way exact Fire sampler layouts differ");

    RadialWindMetrics metrics;
    metrics.samples = points.size();
    metrics.reference_height_m = reference_height_m;

    amrex::Real radial_square_sum = amrex::Real(0.0);
    amrex::Real tangential_square_sum = amrex::Real(0.0);
    std::size_t inward_count = 0;

    for (const auto& point : points) {
        require(
            point[0] >= layout.xlo_m() + amrex::Real(2.0) * layout.dx_m()
                && point[0]
                    <= layout.xhi_m() - amrex::Real(2.0) * layout.dx_m()
                && point[1]
                    >= layout.ylo_m() + amrex::Real(2.0) * layout.dy_m()
                && point[1]
                    <= layout.yhi_m() - amrex::Real(2.0) * layout.dy_m(),
            "Reference perimeter is too close to the domain boundary");

        const amrex::Real rx =
            point[0] - geometry.centroid_x_m;
        const amrex::Real ry =
            point[1] - geometry.centroid_y_m;
        const amrex::Real radius = std::hypot(rx, ry);
        require(
            radius > amrex::Real(0.0),
            "Perimeter point coincides with its centroid");

        const amrex::Real nx = rx / radius;
        const amrex::Real ny = ry / radius;
        const amrex::Real tx = -ny;
        const amrex::Real ty = nx;

        const auto one_sample =
            one_sampler.sample(point[0], point[1]);
        const auto two_sample =
            two_sampler.sample(point[0], point[1]);

        const amrex::Real one_radial =
            one_sample.horizontal_wind_mps.x * nx
            + one_sample.horizontal_wind_mps.y * ny;
        const amrex::Real two_radial =
            two_sample.horizontal_wind_mps.x * nx
            + two_sample.horizontal_wind_mps.y * ny;
        const amrex::Real delta_radial =
            two_radial - one_radial;

        const amrex::Real one_tangential =
            one_sample.horizontal_wind_mps.x * tx
            + one_sample.horizontal_wind_mps.y * ty;
        const amrex::Real two_tangential =
            two_sample.horizontal_wind_mps.x * tx
            + two_sample.horizontal_wind_mps.y * ty;
        const amrex::Real delta_tangential =
            two_tangential - one_tangential;

        require(
            std::isfinite(one_radial)
                && std::isfinite(two_radial)
                && std::isfinite(delta_radial)
                && std::isfinite(delta_tangential),
            "Exact Fire sampler produced non-finite perimeter wind");

        metrics.one_mean_radial_mps += one_radial;
        metrics.two_mean_radial_mps += two_radial;
        metrics.delta_mean_radial_mps += delta_radial;
        radial_square_sum += delta_radial * delta_radial;
        tangential_square_sum +=
            delta_tangential * delta_tangential;
        metrics.delta_min_radial_mps =
            std::min(metrics.delta_min_radial_mps, delta_radial);
        metrics.delta_max_radial_mps =
            std::max(metrics.delta_max_radial_mps, delta_radial);
        if (delta_radial < amrex::Real(0.0)) {
            ++inward_count;
        }
    }

    const amrex::Real inverse_n =
        amrex::Real(1.0)
        / static_cast<amrex::Real>(metrics.samples);
    metrics.one_mean_radial_mps *= inverse_n;
    metrics.two_mean_radial_mps *= inverse_n;
    metrics.delta_mean_radial_mps *= inverse_n;
    metrics.delta_rms_radial_mps =
        std::sqrt(radial_square_sum * inverse_n);
    metrics.delta_rms_tangential_mps =
        std::sqrt(tangential_square_sum * inverse_n);
    metrics.inward_fraction =
        static_cast<amrex::Real>(inward_count) * inverse_n;

    return metrics;
}

void
print_exact_fire_sampler_metrics(
    const char* phase,
    const RadialWindMetrics& metrics,
    const RadialWindMetrics& cell_centered_metrics)
{
    amrex::Print()
        << std::setprecision(17)
        << "EXACT_FIRE_SAMPLER_METRICS"
        << " phase=" << phase
        << " samples=" << metrics.samples
        << " reference_height_m=" << metrics.reference_height_m
        << " one_mean_radial_mps="
        << metrics.one_mean_radial_mps
        << " two_mean_radial_mps="
        << metrics.two_mean_radial_mps
        << " delta_mean_radial_mps="
        << metrics.delta_mean_radial_mps
        << " delta_rms_radial_mps="
        << metrics.delta_rms_radial_mps
        << " delta_min_radial_mps="
        << metrics.delta_min_radial_mps
        << " delta_max_radial_mps="
        << metrics.delta_max_radial_mps
        << " inward_fraction="
        << metrics.inward_fraction
        << " delta_rms_tangential_mps="
        << metrics.delta_rms_tangential_mps
        << " cell_center_delta_mean_radial_mps="
        << cell_centered_metrics.delta_mean_radial_mps
        << " exact_minus_cell_center_delta_mean_mps="
        << metrics.delta_mean_radial_mps
            - cell_centered_metrics.delta_mean_radial_mps
        << "\n";
}

amrex::Real
source_z95(amrex::Real top_m)
{
    const amrex::Real normalization =
        amrex::Real(1.0) - std::exp(-top_m / source_H_m);
    return
        -source_H_m
        * std::log(
            amrex::Real(1.0)
            - amrex::Real(0.95) * normalization);
}

BuoyancyMetrics
analyze_pair(
    const std::string& one_way_path,
    const std::string& two_way_path)
{
    amrex::PlotFileData one_way(one_way_path);
    amrex::PlotFileData two_way(two_way_path);

    require(
        one_way.finestLevel() == 0 && two_way.finestLevel() == 0,
        "Analyzer requires level-0-only plotfiles");
    require(
        one_way.spaceDim() == 3 && two_way.spaceDim() == 3,
        "Analyzer requires 3-D plotfiles");

    const amrex::Real time_scale =
        std::max(
            amrex::Real(1.0),
            std::max(std::abs(one_way.time()), std::abs(two_way.time())));
    require(
        std::abs(one_way.time() - two_way.time())
            <= scaled_tolerance(time_scale),
        "Matched plotfiles are at different times");

    const auto& one_names = one_way.varNames();
    const auto& two_names = two_way.varNames();
    for (const std::string& name : {"theta", "z_velocity"}) {
        require(
            has_variable(one_names, name),
            "One-way plotfile is missing " + name);
        require(
            has_variable(two_names, name),
            "Two-way plotfile is missing " + name);
    }

    const auto domain = one_way.probDomain(0);
    require(
        domain == two_way.probDomain(0),
        "Matched plotfile domains differ");

    const auto dx = one_way.cellSize(0);
    const auto two_dx = two_way.cellSize(0);
    const auto lo = one_way.probLo();
    const auto two_lo = two_way.probLo();
    const auto hi = one_way.probHi();
    const auto two_hi = two_way.probHi();

    for (int dir = 0; dir < 3; ++dir) {
        require(
            std::abs(dx[dir] - two_dx[dir])
                <= scaled_tolerance(dx[dir]),
            "Matched plotfile cell sizes differ");
        require(
            std::abs(lo[dir] - two_lo[dir])
                <= scaled_tolerance(lo[dir]),
            "Matched plotfile lower bounds differ");
        require(
            std::abs(hi[dir] - two_hi[dir])
                <= scaled_tolerance(hi[dir]),
            "Matched plotfile upper bounds differ");
    }

    two_way.syncDistributionMap(one_way);

    auto one_theta = one_way.get(0, "theta");
    auto two_theta = two_way.get(0, "theta");
    auto one_w = one_way.get(0, "z_velocity");
    auto two_w = two_way.get(0, "z_velocity");

    const int klo = domain.smallEnd(2);
    const int nz = domain.length(2);
    const amrex::Real dz = dx[2];
    const amrex::Real cell_volume = dx[0] * dx[1] * dx[2];

    std::vector<amrex::Real> positive_w_by_k(
        static_cast<std::size_t>(nz),
        amrex::Real(0.0));
    std::vector<amrex::Real> w2_by_k(
        static_cast<std::size_t>(nz),
        amrex::Real(0.0));
    std::vector<amrex::Real> layer_max_w(
        static_cast<std::size_t>(nz),
        -std::numeric_limits<amrex::Real>::infinity());

    BuoyancyMetrics metrics;
    metrics.time_s = one_way.time();
    metrics.top_m = hi[2] - lo[2];
    metrics.max_delta_theta_K =
        -std::numeric_limits<amrex::Real>::infinity();
    metrics.max_delta_w_mps =
        -std::numeric_limits<amrex::Real>::infinity();
    metrics.peak_delta_w_z_m =
        std::numeric_limits<amrex::Real>::quiet_NaN();
    metrics.max_delta_w_above_source_z95_mps =
        -std::numeric_limits<amrex::Real>::infinity();
    metrics.w_threshold_top_m =
        std::numeric_limits<amrex::Real>::quiet_NaN();

    const amrex::Real source_z95_m = source_z95(metrics.top_m);

    for (amrex::MFIter mfi(one_theta); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.validbox();

        const auto one_theta_arr = one_theta.const_array(mfi);
        const auto two_theta_arr = two_theta.const_array(mfi);
        const auto one_w_arr = one_w.const_array(mfi);
        const auto two_w_arr = two_w.const_array(mfi);

        for (int k = box.smallEnd(2); k <= box.bigEnd(2); ++k) {
            const amrex::Real z_m =
                lo[2]
                + (amrex::Real(k - klo) + amrex::Real(0.5)) * dz;
            const std::size_t n =
                static_cast<std::size_t>(k - klo);

            for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
                for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                    const amrex::Real delta_theta =
                        two_theta_arr(i, j, k)
                        - one_theta_arr(i, j, k);
                    const amrex::Real delta_w =
                        two_w_arr(i, j, k)
                        - one_w_arr(i, j, k);

                    require(
                        std::isfinite(delta_theta)
                            && std::isfinite(delta_w),
                        "Response contains a non-finite difference");

                    metrics.max_delta_theta_K =
                        std::max(
                            metrics.max_delta_theta_K,
                            delta_theta);

                    if (delta_w > metrics.max_delta_w_mps) {
                        metrics.max_delta_w_mps = delta_w;
                        metrics.peak_delta_w_z_m = z_m;
                    }

                    layer_max_w[n] =
                        std::max(layer_max_w[n], delta_w);

                    const amrex::Real w2_contribution =
                        delta_w * delta_w * cell_volume;
                    metrics.delta_w_l2_volume_integral_m5ps2 +=
                        w2_contribution;
                    w2_by_k[n] += w2_contribution;

                    if (delta_w > amrex::Real(0.0)) {
                        const amrex::Real contribution =
                            delta_w * cell_volume;
                        positive_w_by_k[n] += contribution;
                        metrics.positive_w_volume_integral_m4ps +=
                            contribution;
                    }

                    if (z_m >= source_z95_m) {
                        metrics.max_delta_w_above_source_z95_mps =
                            std::max(
                                metrics.max_delta_w_above_source_z95_mps,
                                delta_w);
                    }

                    if (delta_theta > amrex::Real(0.0)) {
                        metrics.positive_theta_volume_integral_K_m3 +=
                            delta_theta * cell_volume;
                    }
                }
            }
        }
    }

    require(
        metrics.max_delta_theta_K > amrex::Real(0.0),
        "Response has no positive theta perturbation");
    require(
        metrics.max_delta_w_mps > amrex::Real(0.0),
        "Response has no upward velocity perturbation");
    require(
        metrics.positive_w_volume_integral_m4ps
            > amrex::Real(0.0),
        "Response has no positive bulk upward motion");
    require(
        metrics.delta_w_l2_volume_integral_m5ps2
            > amrex::Real(0.0),
        "Response has zero vertical-velocity L2 content");
    require(
        metrics.positive_theta_volume_integral_K_m3
            > amrex::Real(0.0),
        "Response has no positive bulk theta perturbation");

    auto response_height =
        [&](amrex::Real quantile)
    {
        const amrex::Real target =
            quantile
            * metrics.positive_w_volume_integral_m4ps;
        amrex::Real cumulative = amrex::Real(0.0);

        for (int n = 0; n < nz; ++n) {
            const amrex::Real layer =
                positive_w_by_k[static_cast<std::size_t>(n)];
            const amrex::Real next = cumulative + layer;

            if (next >= target && layer > amrex::Real(0.0)) {
                const amrex::Real within_layer =
                    std::clamp(
                        (target - cumulative) / layer,
                        amrex::Real(0.0),
                        amrex::Real(1.0));
                return
                    lo[2] + (amrex::Real(n) + within_layer) * dz;
            }
            cumulative = next;
        }
        return hi[2];
    };

    metrics.w_response_z50_m =
        response_height(amrex::Real(0.50));
    metrics.w_response_z95_m =
        response_height(amrex::Real(0.95));

    auto w2_response_height =
        [&](amrex::Real quantile)
    {
        const amrex::Real target =
            quantile * metrics.delta_w_l2_volume_integral_m5ps2;
        amrex::Real cumulative = amrex::Real(0.0);

        for (int n = 0; n < nz; ++n) {
            const amrex::Real layer =
                w2_by_k[static_cast<std::size_t>(n)];
            const amrex::Real next = cumulative + layer;

            if (next >= target && layer > amrex::Real(0.0)) {
                const amrex::Real within_layer =
                    std::clamp(
                        (target - cumulative) / layer,
                        amrex::Real(0.0),
                        amrex::Real(1.0));
                return lo[2] + (amrex::Real(n) + within_layer) * dz;
            }
            cumulative = next;
        }
        return hi[2];
    };

    metrics.w2_response_z50_m =
        w2_response_height(amrex::Real(0.50));
    metrics.w2_response_z95_m =
        w2_response_height(amrex::Real(0.95));

    amrex::Real positive_w_above_source = amrex::Real(0.0);
    amrex::Real w2_above_source = amrex::Real(0.0);
    for (int n = 0; n < nz; ++n) {
        const amrex::Real z_center =
            lo[2] + (amrex::Real(n) + amrex::Real(0.5)) * dz;
        if (z_center >= source_z95_m) {
            positive_w_above_source +=
                positive_w_by_k[static_cast<std::size_t>(n)];
            w2_above_source +=
                w2_by_k[static_cast<std::size_t>(n)];
        }

        const amrex::Real layer_max =
            layer_max_w[static_cast<std::size_t>(n)];

        if (layer_max >= response_threshold_mps) {
            metrics.w_threshold_top_m = z_center;
        }
        if (layer_max >= amrex::Real(0.05)) {
            metrics.w_threshold_0p05_top_m = z_center;
        }
        if (layer_max >= amrex::Real(0.10)) {
            metrics.w_threshold_0p10_top_m = z_center;
        }
        if (layer_max >= amrex::Real(0.25)) {
            metrics.w_threshold_0p25_top_m = z_center;
        }
        if (layer_max >= amrex::Real(0.50)) {
            metrics.w_threshold_0p50_top_m = z_center;
        }
    }

    metrics.positive_w_fraction_above_source_z95 =
        positive_w_above_source
        / metrics.positive_w_volume_integral_m4ps;
    metrics.w2_fraction_above_source_z95 =
        w2_above_source
        / metrics.delta_w_l2_volume_integral_m5ps2;

    require(
        std::isfinite(metrics.peak_delta_w_z_m)
            && std::isfinite(metrics.w_response_z50_m)
            && std::isfinite(metrics.w_response_z95_m)
            && std::isfinite(metrics.w2_response_z50_m)
            && std::isfinite(metrics.w2_response_z95_m)
            && std::isfinite(metrics.w2_fraction_above_source_z95)
            && std::isfinite(
                metrics.positive_w_fraction_above_source_z95)
            && std::isfinite(
                metrics.max_delta_w_above_source_z95_mps),
        "Derived buoyancy metrics are not finite");
    require(
        metrics.w_response_z50_m <= metrics.w_response_z95_m,
        "Upward-motion quantile heights are not ordered");

    return metrics;
}

void
print_metrics(
    const char* phase,
    const BuoyancyMetrics& metrics)
{
    amrex::Print()
        << std::setprecision(17)
        << "BUOYANCY_METRICS"
        << " phase=" << phase
        << " time_s=" << metrics.time_s
        << " max_delta_theta_K="
        << metrics.max_delta_theta_K
        << " max_delta_w_mps="
        << metrics.max_delta_w_mps
        << " peak_delta_w_z_m="
        << metrics.peak_delta_w_z_m
        << " positive_w_volume_integral_m4ps="
        << metrics.positive_w_volume_integral_m4ps
        << " delta_w_l2_volume_integral_m5ps2="
        << metrics.delta_w_l2_volume_integral_m5ps2
        << " w_response_z50_m="
        << metrics.w_response_z50_m
        << " w_response_z95_m="
        << metrics.w_response_z95_m
        << " w2_response_z50_m="
        << metrics.w2_response_z50_m
        << " w2_response_z95_m="
        << metrics.w2_response_z95_m
        << " source_z95_m="
        << source_z95(metrics.top_m)
        << " positive_w_fraction_above_source_z95="
        << metrics.positive_w_fraction_above_source_z95
        << " w2_fraction_above_source_z95="
        << metrics.w2_fraction_above_source_z95
        << " max_delta_w_above_source_z95_mps="
        << metrics.max_delta_w_above_source_z95_mps
        << " w_threshold_0p01_top_m="
        << metrics.w_threshold_top_m
        << " w_threshold_0p05_top_m="
        << metrics.w_threshold_0p05_top_m
        << " w_threshold_0p10_top_m="
        << metrics.w_threshold_0p10_top_m
        << " w_threshold_0p25_top_m="
        << metrics.w_threshold_0p25_top_m
        << " w_threshold_0p50_top_m="
        << metrics.w_threshold_0p50_top_m
        << " positive_theta_volume_integral_K_m3="
        << metrics.positive_theta_volume_integral_K_m3
        << "\n";
}

} // namespace

int
main(int argc, char** argv)
{
    amrex::Initialize(argc, argv);

    int result = 0;
    try {
        amrex::ParmParse pp("analysis");

        std::string early_one_way;
        std::string early_two_way;
        std::string late_one_way;
        std::string late_two_way;
        std::string one_way_final_perimeter;
        std::string two_way_final_perimeter;
        std::string early_reference_perimeter;
        std::string late_reference_perimeter;
        std::string early_one_way_u;
        std::string early_one_way_v;
        std::string early_two_way_u;
        std::string early_two_way_v;
        std::string late_one_way_u;
        std::string late_one_way_v;
        std::string late_two_way_u;
        std::string late_two_way_v;

        pp.get("early_one_way", early_one_way);
        pp.get("early_two_way", early_two_way);
        pp.get("late_one_way", late_one_way);
        pp.get("late_two_way", late_two_way);
        pp.get("one_way_final_perimeter", one_way_final_perimeter);
        pp.get("two_way_final_perimeter", two_way_final_perimeter);
        pp.get("early_reference_perimeter", early_reference_perimeter);
        pp.get("late_reference_perimeter", late_reference_perimeter);
        pp.get("early_one_way_u", early_one_way_u);
        pp.get("early_one_way_v", early_one_way_v);
        pp.get("early_two_way_u", early_two_way_u);
        pp.get("early_two_way_v", early_two_way_v);
        pp.get("late_one_way_u", late_one_way_u);
        pp.get("late_one_way_v", late_one_way_v);
        pp.get("late_two_way_u", late_two_way_u);
        pp.get("late_two_way_v", late_two_way_v);

        const BuoyancyMetrics early =
            analyze_pair(early_one_way, early_two_way);
        const BuoyancyMetrics late =
            analyze_pair(late_one_way, late_two_way);

        require(
            std::abs(early.time_s - amrex::Real(1.0))
                <= scaled_tolerance(amrex::Real(1.0)),
            "Early checkpoint is not at 1 s");
        require(
            std::abs(late.time_s - amrex::Real(5.0))
                <= scaled_tolerance(amrex::Real(5.0)),
            "Late checkpoint is not at 5 s");

        print_metrics("early", early);
        print_metrics("late", late);

        const RadialWindMetrics early_radial =
            analyze_radial_wind(
                early_one_way,
                early_two_way,
                early_reference_perimeter,
                amrex::Real(10.0));
        const RadialWindMetrics late_radial =
            analyze_radial_wind(
                late_one_way,
                late_two_way,
                late_reference_perimeter,
                amrex::Real(10.0));

        print_radial_wind_metrics("early", early_radial);
        print_radial_wind_metrics("late", late_radial);

        const RadialWindMetrics early_exact_sampler =
            analyze_exact_fire_sampler_radial_wind(
                early_one_way,
                early_one_way_u,
                early_one_way_v,
                early_two_way,
                early_two_way_u,
                early_two_way_v,
                early_reference_perimeter,
                amrex::Real(10.0));
        const RadialWindMetrics late_exact_sampler =
            analyze_exact_fire_sampler_radial_wind(
                late_one_way,
                late_one_way_u,
                late_one_way_v,
                late_two_way,
                late_two_way_u,
                late_two_way_v,
                late_reference_perimeter,
                amrex::Real(10.0));

        print_exact_fire_sampler_metrics(
            "early",
            early_exact_sampler,
            early_radial);
        print_exact_fire_sampler_metrics(
            "late",
            late_exact_sampler,
            late_radial);

        require(
            std::abs(early_exact_sampler.one_mean_radial_mps)
                <= amrex::Real(1.0e-12)
                && std::abs(late_exact_sampler.one_mean_radial_mps)
                    <= amrex::Real(1.0e-12),
            "Exact Fire sampler one-way control developed nonzero mean radial flow");
        require(
            early_exact_sampler.delta_mean_radial_mps
                >= amrex::Real(0.06),
            "Exact Fire sampler early radial response is not sufficiently outward");
        require(
            early_exact_sampler.inward_fraction
                <= amrex::Real(0.05),
            "Exact Fire sampler early response is not predominantly outward");
        require(
            late_exact_sampler.delta_mean_radial_mps
                <= -amrex::Real(0.12),
            "Exact Fire sampler late radial response is not sufficiently inward");
        require(
            late_exact_sampler.inward_fraction
                >= amrex::Real(0.95),
            "Exact Fire sampler late response is not predominantly inward");
        require(
            early_exact_sampler.delta_rms_radial_mps
                >= amrex::Real(10.0)
                    * early_exact_sampler.delta_rms_tangential_mps,
            "Exact Fire sampler early horizontal feedback is not radially dominant");
        require(
            late_exact_sampler.delta_rms_radial_mps
                >= amrex::Real(10.0)
                    * late_exact_sampler.delta_rms_tangential_mps,
            "Exact Fire sampler late horizontal feedback is not radially dominant");

        require(
            std::abs(early_radial.one_mean_radial_mps)
                <= amrex::Real(1.0e-12)
                && std::abs(late_radial.one_mean_radial_mps)
                    <= amrex::Real(1.0e-12),
            "One-way zero-wind control developed nonzero mean radial flow");
        require(
            early_radial.delta_mean_radial_mps
                >= amrex::Real(0.05),
            "Early two-way radial-flow response is not sufficiently outward");
        require(
            early_radial.inward_fraction
                <= amrex::Real(0.05),
            "Early perimeter response is not predominantly outward");
        require(
            late_radial.delta_mean_radial_mps
                <= -amrex::Real(0.10),
            "Late two-way radial-flow response is not sufficiently inward");
        require(
            late_radial.inward_fraction
                >= amrex::Real(0.95),
            "Late perimeter response is not predominantly inward");
        require(
            early_radial.delta_rms_radial_mps
                >= amrex::Real(10.0)
                    * early_radial.delta_rms_tangential_mps,
            "Early horizontal feedback is not radially dominant");
        require(
            late_radial.delta_rms_radial_mps
                >= amrex::Real(10.0)
                    * late_radial.delta_rms_tangential_mps,
            "Late horizontal feedback is not radially dominant");

        require(
            late.max_delta_w_mps >= amrex::Real(1.0),
            "Late coupled response does not reach 1 m/s upward velocity");
        require(
            late.max_delta_w_mps
                >= amrex::Real(5.0) * early.max_delta_w_mps,
            "Late upward response is not at least five times the 1 s response");
        require(
            late.delta_w_l2_volume_integral_m5ps2
                >= amrex::Real(4.0)
                    * early.delta_w_l2_volume_integral_m5ps2,
            "Vertical-velocity L2 response did not grow by at least fourfold from 1 s to 5 s");
        require(
            late.max_delta_w_above_source_z95_mps
                >= amrex::Real(0.1),
            "Late updraft above the source z95 does not reach 0.1 m/s");
        require(
            late.max_delta_w_above_source_z95_mps
                >= amrex::Real(2.0)
                    * early.max_delta_w_above_source_z95_mps,
            "Updraft above the source z95 did not at least double from 1 s to 5 s");
        require(
            late.positive_theta_volume_integral_K_m3
                > early.positive_theta_volume_integral_K_m3,
            "Positive thermal response did not grow from 1 s to 5 s");

        const amrex::Real late_source_z95_m =
            source_z95(late.top_m);

        require(
            late.max_delta_w_above_source_z95_mps
                >= amrex::Real(0.15),
            "Late updraft above the source z95 does not reach 0.15 m/s");
        require(
            late.w_threshold_0p10_top_m
                >= late_source_z95_m + amrex::Real(15.0),
            "0.10 m/s updraft does not extend at least 15 m above the source z95");
        require(
            late.w_threshold_0p05_top_m
                >= late_source_z95_m + amrex::Real(30.0),
            "0.05 m/s updraft does not extend at least 30 m above the source z95");
        require(
            late.w_threshold_0p05_top_m
                >= late.w_threshold_0p10_top_m,
            "Strong-updraft threshold heights are not ordered");

        const PerimeterGeometryMetrics one_fire =
            read_perimeter_geometry(one_way_final_perimeter);
        const PerimeterGeometryMetrics two_fire =
            read_perimeter_geometry(two_way_final_perimeter);

        require(
            one_fire.vertices == two_fire.vertices,
            "Final one-way/two-way perimeter vertex counts differ");

        const amrex::Real area_difference_m2 =
            two_fire.area_m2 - one_fire.area_m2;
        const amrex::Real mean_radius_difference_m =
            two_fire.mean_radius_m - one_fire.mean_radius_m;

        require(
            std::abs(area_difference_m2) >= amrex::Real(0.1),
            "Final Fire perimeter enclosed-area difference is below 0.1 m^2");
        require(
            std::abs(mean_radius_difference_m) >= amrex::Real(0.002),
            "Final Fire perimeter mean-radius difference is below 2 mm");

        amrex::Print()
            << std::setprecision(17)
            << "LOOP_CLOSURE_METRICS"
            << " one_area_m2=" << one_fire.area_m2
            << " two_area_m2=" << two_fire.area_m2
            << " delta_area_m2=" << area_difference_m2
            << " one_perimeter_m=" << one_fire.perimeter_m
            << " two_perimeter_m=" << two_fire.perimeter_m
            << " delta_perimeter_m="
            << two_fire.perimeter_m - one_fire.perimeter_m
            << " one_mean_radius_m=" << one_fire.mean_radius_m
            << " two_mean_radius_m=" << two_fire.mean_radius_m
            << " delta_mean_radius_m=" << mean_radius_difference_m
            << "\n";

    } catch (const std::exception& error) {
        amrex::Print()
            << "Developed-buoyancy analysis error: "
            << error.what()
            << "\n";
        result = 2;
    }

    amrex::Finalize();
    return result;
}
