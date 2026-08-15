#include <AMReX.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr amrex::Real H_m = amrex::Real(50.0);

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

struct ResponseMetrics
{
    amrex::Real dz_m{};
    amrex::Real top_m{};
    amrex::Real max_delta_rhotheta{};
    amrex::Real max_delta_rhoqv{};
    amrex::Real max_delta_theta_K{};
    amrex::Real max_delta_w_mps{};
    amrex::Real peak_delta_w_z_m{};
    amrex::Real theta_response_z50_m{};
    amrex::Real theta_response_z95_m{};
    amrex::Real theta_below_25m_fraction{};
    amrex::Real theta_below_50m_fraction{};
    amrex::Real positive_theta_volume_integral_K_m3{};
};

amrex::Real
expected_quantile_height(
    amrex::Real quantile,
    amrex::Real top_m)
{
    const amrex::Real normalization =
        amrex::Real(1.0) - std::exp(-top_m / H_m);
    return
        -H_m
        * std::log(
            amrex::Real(1.0)
            - quantile * normalization);
}

amrex::Real
expected_below_fraction(
    amrex::Real cutoff_m,
    amrex::Real top_m)
{
    const amrex::Real bounded_cutoff =
        std::clamp(
            cutoff_m,
            amrex::Real(0.0),
            top_m);
    const amrex::Real normalization =
        amrex::Real(1.0) - std::exp(-top_m / H_m);
    return
        (amrex::Real(1.0)
         - std::exp(-bounded_cutoff / H_m))
        / normalization;
}

ResponseMetrics
analyze_response(
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
        "One-way and two-way plotfiles are at different times");

    const auto& one_names = one_way.varNames();
    const auto& two_names = two_way.varNames();
    for (const std::string& name :
         {"rhotheta", "rhoQ1", "theta", "z_velocity"}) {
        require(
            has_variable(one_names, name),
            "One-way plotfile is missing required variable " + name);
        require(
            has_variable(two_names, name),
            "Two-way plotfile is missing required variable " + name);
    }

    const auto one_domain = one_way.probDomain(0);
    const auto two_domain = two_way.probDomain(0);
    require(
        one_domain == two_domain,
        "Matched plotfile domains differ");

    const auto one_dx = one_way.cellSize(0);
    const auto two_dx = two_way.cellSize(0);
    const auto one_lo = one_way.probLo();
    const auto two_lo = two_way.probLo();
    const auto one_hi = one_way.probHi();
    const auto two_hi = two_way.probHi();

    for (int dir = 0; dir < 3; ++dir) {
        require(
            std::abs(one_dx[dir] - two_dx[dir])
                <= scaled_tolerance(one_dx[dir]),
            "Matched plotfile cell sizes differ");
        require(
            std::abs(one_lo[dir] - two_lo[dir])
                <= scaled_tolerance(one_lo[dir]),
            "Matched plotfile lower bounds differ");
        require(
            std::abs(one_hi[dir] - two_hi[dir])
                <= scaled_tolerance(one_hi[dir]),
            "Matched plotfile upper bounds differ");
    }

    two_way.syncDistributionMap(one_way);

    auto one_rhotheta = one_way.get(0, "rhotheta");
    auto two_rhotheta = two_way.get(0, "rhotheta");
    auto one_rhoqv = one_way.get(0, "rhoQ1");
    auto two_rhoqv = two_way.get(0, "rhoQ1");
    auto one_theta = one_way.get(0, "theta");
    auto two_theta = two_way.get(0, "theta");
    auto one_w = one_way.get(0, "z_velocity");
    auto two_w = two_way.get(0, "z_velocity");

    const int klo = one_domain.smallEnd(2);
    const int nz = one_domain.length(2);
    const amrex::Real dz = one_dx[2];
    const amrex::Real cell_volume =
        one_dx[0] * one_dx[1] * one_dx[2];

    std::vector<amrex::Real> positive_theta_by_k(
        static_cast<std::size_t>(nz),
        amrex::Real(0.0));

    ResponseMetrics metrics;
    metrics.dz_m = dz;
    metrics.top_m = one_hi[2] - one_lo[2];
    metrics.max_delta_rhotheta =
        -std::numeric_limits<amrex::Real>::infinity();
    metrics.max_delta_rhoqv =
        -std::numeric_limits<amrex::Real>::infinity();
    metrics.max_delta_theta_K =
        -std::numeric_limits<amrex::Real>::infinity();
    metrics.max_delta_w_mps =
        -std::numeric_limits<amrex::Real>::infinity();
    metrics.peak_delta_w_z_m =
        std::numeric_limits<amrex::Real>::quiet_NaN();

    for (amrex::MFIter mfi(one_theta); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.validbox();

        const auto one_rhotheta_arr =
            one_rhotheta.const_array(mfi);
        const auto two_rhotheta_arr =
            two_rhotheta.const_array(mfi);
        const auto one_rhoqv_arr =
            one_rhoqv.const_array(mfi);
        const auto two_rhoqv_arr =
            two_rhoqv.const_array(mfi);
        const auto one_theta_arr =
            one_theta.const_array(mfi);
        const auto two_theta_arr =
            two_theta.const_array(mfi);
        const auto one_w_arr =
            one_w.const_array(mfi);
        const auto two_w_arr =
            two_w.const_array(mfi);

        for (int k = box.smallEnd(2); k <= box.bigEnd(2); ++k) {
            const amrex::Real z_m =
                one_lo[2]
                + (amrex::Real(k - klo) + amrex::Real(0.5)) * dz;

            for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
                for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                    const amrex::Real delta_rhotheta =
                        two_rhotheta_arr(i, j, k)
                        - one_rhotheta_arr(i, j, k);
                    const amrex::Real delta_rhoqv =
                        two_rhoqv_arr(i, j, k)
                        - one_rhoqv_arr(i, j, k);
                    const amrex::Real delta_theta =
                        two_theta_arr(i, j, k)
                        - one_theta_arr(i, j, k);
                    const amrex::Real delta_w =
                        two_w_arr(i, j, k)
                        - one_w_arr(i, j, k);

                    require(
                        std::isfinite(delta_rhotheta)
                            && std::isfinite(delta_rhoqv)
                            && std::isfinite(delta_theta)
                            && std::isfinite(delta_w),
                        "Response contains a non-finite difference");

                    metrics.max_delta_rhotheta =
                        std::max(
                            metrics.max_delta_rhotheta,
                            delta_rhotheta);
                    metrics.max_delta_rhoqv =
                        std::max(
                            metrics.max_delta_rhoqv,
                            delta_rhoqv);
                    metrics.max_delta_theta_K =
                        std::max(
                            metrics.max_delta_theta_K,
                            delta_theta);

                    if (delta_w > metrics.max_delta_w_mps) {
                        metrics.max_delta_w_mps = delta_w;
                        metrics.peak_delta_w_z_m = z_m;
                    }

                    if (delta_theta > amrex::Real(0.0)) {
                        const amrex::Real contribution =
                            delta_theta * cell_volume;
                        positive_theta_by_k[
                            static_cast<std::size_t>(k - klo)]
                            += contribution;
                        metrics.positive_theta_volume_integral_K_m3
                            += contribution;
                    }
                }
            }
        }
    }

    require(
        metrics.max_delta_rhotheta > amrex::Real(0.0),
        "Response has no positive rhotheta perturbation");
    require(
        metrics.max_delta_rhoqv > amrex::Real(0.0),
        "Response has no positive rhoqv perturbation");
    require(
        metrics.max_delta_theta_K > amrex::Real(0.0),
        "Response has no positive theta perturbation");
    require(
        metrics.max_delta_w_mps > amrex::Real(0.0),
        "Response has no upward vertical-velocity perturbation");
    require(
        metrics.positive_theta_volume_integral_K_m3
            > amrex::Real(0.0),
        "Response has no positive volume-integrated theta perturbation");

    auto positive_theta_below =
        [&](amrex::Real cutoff_m)
    {
        amrex::Real sum = amrex::Real(0.0);
        for (int n = 0; n < nz; ++n) {
            const amrex::Real zlo_m =
                one_lo[2] + amrex::Real(n) * dz;
            const amrex::Real zhi_m = zlo_m + dz;
            const amrex::Real overlap_m =
                std::max(
                    amrex::Real(0.0),
                    std::min(zhi_m, cutoff_m) - zlo_m);
            const amrex::Real fraction =
                std::clamp(
                    overlap_m / dz,
                    amrex::Real(0.0),
                    amrex::Real(1.0));
            sum +=
                fraction
                * positive_theta_by_k[static_cast<std::size_t>(n)];
        }
        return
            sum
            / metrics.positive_theta_volume_integral_K_m3;
    };

    auto response_height =
        [&](amrex::Real quantile)
    {
        const amrex::Real target =
            quantile
            * metrics.positive_theta_volume_integral_K_m3;
        amrex::Real cumulative = amrex::Real(0.0);

        for (int n = 0; n < nz; ++n) {
            const amrex::Real layer =
                positive_theta_by_k[static_cast<std::size_t>(n)];
            const amrex::Real next = cumulative + layer;

            if (next >= target && layer > amrex::Real(0.0)) {
                const amrex::Real within_layer =
                    std::clamp(
                        (target - cumulative) / layer,
                        amrex::Real(0.0),
                        amrex::Real(1.0));
                return
                    one_lo[2]
                    + (amrex::Real(n) + within_layer) * dz;
            }
            cumulative = next;
        }
        return one_hi[2];
    };

    metrics.theta_response_z50_m =
        response_height(amrex::Real(0.50));
    metrics.theta_response_z95_m =
        response_height(amrex::Real(0.95));
    metrics.theta_below_25m_fraction =
        positive_theta_below(one_lo[2] + amrex::Real(25.0));
    metrics.theta_below_50m_fraction =
        positive_theta_below(one_lo[2] + amrex::Real(50.0));

    require(
        std::isfinite(metrics.peak_delta_w_z_m)
            && std::isfinite(metrics.theta_response_z50_m)
            && std::isfinite(metrics.theta_response_z95_m)
            && std::isfinite(metrics.theta_below_25m_fraction)
            && std::isfinite(metrics.theta_below_50m_fraction),
        "Response metrics are not finite");
    require(
        metrics.theta_response_z50_m
            <= metrics.theta_response_z95_m,
        "Theta response quantile heights are not ordered");

    return metrics;
}

amrex::Real
relative_difference(
    amrex::Real a,
    amrex::Real b)
{
    return
        std::abs(a - b)
        / std::max(
            amrex::Real(1.0e-30),
            amrex::Real(0.5) * (std::abs(a) + std::abs(b)));
}

void
print_metrics(
    const char* label,
    const ResponseMetrics& metrics)
{
    const amrex::Real expected_z50 =
        expected_quantile_height(
            amrex::Real(0.50),
            metrics.top_m);
    const amrex::Real expected_z95 =
        expected_quantile_height(
            amrex::Real(0.95),
            metrics.top_m);
    const amrex::Real expected_below_25 =
        expected_below_fraction(
            amrex::Real(25.0),
            metrics.top_m);
    const amrex::Real expected_below_50 =
        expected_below_fraction(
            amrex::Real(50.0),
            metrics.top_m);

    amrex::Print()
        << std::setprecision(17)
        << "VERTICAL_METRICS"
        << " resolution=" << label
        << " dz_m=" << metrics.dz_m
        << " max_delta_rhotheta="
        << metrics.max_delta_rhotheta
        << " max_delta_rhoqv="
        << metrics.max_delta_rhoqv
        << " max_delta_theta_K="
        << metrics.max_delta_theta_K
        << " max_delta_w_mps="
        << metrics.max_delta_w_mps
        << " peak_delta_w_z_m="
        << metrics.peak_delta_w_z_m
        << " theta_response_z50_m="
        << metrics.theta_response_z50_m
        << " theta_response_z95_m="
        << metrics.theta_response_z95_m
        << " analytic_z50_m="
        << expected_z50
        << " analytic_z95_m="
        << expected_z95
        << " theta_below_25m_fraction="
        << metrics.theta_below_25m_fraction
        << " analytic_below_25m_fraction="
        << expected_below_25
        << " theta_below_50m_fraction="
        << metrics.theta_below_50m_fraction
        << " analytic_below_50m_fraction="
        << expected_below_50
        << " positive_theta_volume_integral_K_m3="
        << metrics.positive_theta_volume_integral_K_m3
        << "\n";
}

void
require_profile_consistency(
    const ResponseMetrics& metrics)
{
    const amrex::Real expected_z50 =
        expected_quantile_height(
            amrex::Real(0.50),
            metrics.top_m);
    const amrex::Real expected_z95 =
        expected_quantile_height(
            amrex::Real(0.95),
            metrics.top_m);
    const amrex::Real expected_below_25 =
        expected_below_fraction(
            amrex::Real(25.0),
            metrics.top_m);
    const amrex::Real expected_below_50 =
        expected_below_fraction(
            amrex::Real(50.0),
            metrics.top_m);

    require(
        std::abs(
            metrics.theta_response_z50_m
            - expected_z50)
            <= amrex::Real(0.5) * metrics.dz_m,
        "z50 response lies outside half a local vertical cell of the analytic source profile");
    require(
        std::abs(
            metrics.theta_response_z95_m
            - expected_z95)
            <= amrex::Real(0.5) * metrics.dz_m,
        "z95 response lies outside half a local vertical cell of the analytic source profile");
    require(
        std::abs(
            metrics.theta_below_25m_fraction
            - expected_below_25)
            <= amrex::Real(0.02),
        "Theta fraction below 25 m is inconsistent with the analytic source profile");
    require(
        std::abs(
            metrics.theta_below_50m_fraction
            - expected_below_50)
            <= amrex::Real(0.02),
        "Theta fraction below 50 m is inconsistent with the analytic source profile");
}

} // namespace

int
main(int argc, char** argv)
{
    amrex::Initialize(argc, argv);

    int result = 0;
    try {
        amrex::ParmParse pp("analysis");

        std::array<std::string, 3> one_way_paths;
        std::array<std::string, 3> two_way_paths;

        pp.get("one_way_coarse", one_way_paths[0]);
        pp.get("two_way_coarse", two_way_paths[0]);
        pp.get("one_way_medium", one_way_paths[1]);
        pp.get("two_way_medium", two_way_paths[1]);
        pp.get("one_way_fine", one_way_paths[2]);
        pp.get("two_way_fine", two_way_paths[2]);

        constexpr std::array<const char*, 3> labels{
            "coarse",
            "medium",
            "fine"};
        constexpr std::array<amrex::Real, 3> expected_dz{
            amrex::Real(10.0),
            amrex::Real(5.0),
            amrex::Real(2.5)};

        std::array<ResponseMetrics, 3> metrics{};
        for (std::size_t n = 0; n < metrics.size(); ++n) {
            metrics[n] =
                analyze_response(
                    one_way_paths[n],
                    two_way_paths[n]);

            require(
                std::abs(metrics[n].dz_m - expected_dz[n])
                    <= scaled_tolerance(expected_dz[n]),
                "Plotfile vertical spacing does not match requested resolution");

            require_profile_consistency(metrics[n]);
            print_metrics(labels[n], metrics[n]);
        }

        const auto& coarse = metrics[0];
        const auto& medium = metrics[1];
        const auto& fine = metrics[2];

        require(
            std::abs(
                coarse.theta_response_z50_m
                - medium.theta_response_z50_m)
                <= coarse.dz_m,
            "Coarse/medium z50 response differs by more than one coarse vertical cell");
        require(
            std::abs(
                coarse.theta_response_z95_m
                - medium.theta_response_z95_m)
                <= coarse.dz_m,
            "Coarse/medium z95 response differs by more than one coarse vertical cell");
        require(
            std::abs(
                medium.theta_response_z50_m
                - fine.theta_response_z50_m)
                <= fine.dz_m,
            "Medium/fine z50 response differs by more than one fine vertical cell");
        require(
            std::abs(
                medium.theta_response_z95_m
                - fine.theta_response_z95_m)
                <= fine.dz_m,
            "Medium/fine z95 response differs by more than one fine vertical cell");

        require(
            std::abs(
                coarse.theta_below_25m_fraction
                - medium.theta_below_25m_fraction)
                <= amrex::Real(0.05),
            "Coarse/medium below-25m theta fraction is not stable");
        require(
            std::abs(
                coarse.theta_below_50m_fraction
                - medium.theta_below_50m_fraction)
                <= amrex::Real(0.05),
            "Coarse/medium below-50m theta fraction is not stable");
        require(
            std::abs(
                medium.theta_below_25m_fraction
                - fine.theta_below_25m_fraction)
                <= amrex::Real(0.02),
            "Medium/fine below-25m theta fraction is not stable");
        require(
            std::abs(
                medium.theta_below_50m_fraction
                - fine.theta_below_50m_fraction)
                <= amrex::Real(0.02),
            "Medium/fine below-50m theta fraction is not stable");

        require(
            relative_difference(
                coarse.positive_theta_volume_integral_K_m3,
                medium.positive_theta_volume_integral_K_m3)
                <= amrex::Real(0.01),
            "Coarse/medium positive theta integral differs by more than 1 percent");
        require(
            relative_difference(
                medium.positive_theta_volume_integral_K_m3,
                fine.positive_theta_volume_integral_K_m3)
                <= amrex::Real(0.005),
            "Medium/fine positive theta integral differs by more than 0.5 percent");

    } catch (const std::exception& error) {
        amrex::Print()
            << "Vertical-resolution sensitivity analysis error: "
            << error.what()
            << "\n";
        result = 2;
    }

    amrex::Finalize();
    return result;
}
