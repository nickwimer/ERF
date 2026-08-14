#include <AMReX.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
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
    amrex::Real positive_w_fraction_above_source_z95{};
    amrex::Real max_delta_w_above_source_z95_mps{};
    amrex::Real w_threshold_top_m{};
    amrex::Real positive_theta_volume_integral_K_m3{};
};

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
        "M12c1 analyzer requires level-0-only plotfiles");
    require(
        one_way.spaceDim() == 3 && two_way.spaceDim() == 3,
        "M12c1 analyzer requires 3-D plotfiles");

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

                    metrics.delta_w_l2_volume_integral_m5ps2 +=
                        delta_w * delta_w * cell_volume;

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

    amrex::Real positive_w_above_source = amrex::Real(0.0);
    for (int n = 0; n < nz; ++n) {
        const amrex::Real z_center =
            lo[2] + (amrex::Real(n) + amrex::Real(0.5)) * dz;
        if (z_center >= source_z95_m) {
            positive_w_above_source +=
                positive_w_by_k[static_cast<std::size_t>(n)];
        }
        if (layer_max_w[static_cast<std::size_t>(n)]
            >= response_threshold_mps) {
            metrics.w_threshold_top_m = z_center;
        }
    }

    metrics.positive_w_fraction_above_source_z95 =
        positive_w_above_source
        / metrics.positive_w_volume_integral_m4ps;

    require(
        std::isfinite(metrics.peak_delta_w_z_m)
            && std::isfinite(metrics.w_response_z50_m)
            && std::isfinite(metrics.w_response_z95_m)
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
        << " source_z95_m="
        << source_z95(metrics.top_m)
        << " positive_w_fraction_above_source_z95="
        << metrics.positive_w_fraction_above_source_z95
        << " max_delta_w_above_source_z95_mps="
        << metrics.max_delta_w_above_source_z95_mps
        << " w_threshold_0p01_top_m="
        << metrics.w_threshold_top_m
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

        pp.get("early_one_way", early_one_way);
        pp.get("early_two_way", early_two_way);
        pp.get("late_one_way", late_one_way);
        pp.get("late_two_way", late_two_way);

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

        amrex::Print()
            << "BUOYANT_ACCELERATION_PASS=1\n";
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
