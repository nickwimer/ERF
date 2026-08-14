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

} // namespace

int
main(int argc, char** argv)
{
    amrex::Initialize(argc, argv);

    int result = 0;
    try {
        amrex::ParmParse pp("analysis");
        std::string one_way_path;
        std::string two_way_path;
        pp.get("one_way_plot", one_way_path);
        pp.get("two_way_plot", two_way_path);

        amrex::PlotFileData one_way(one_way_path);
        amrex::PlotFileData two_way(two_way_path);

        require(
            one_way.finestLevel() == 0 && two_way.finestLevel() == 0,
            "M12b1 analyzer requires level-0-only plotfiles");
        require(
            one_way.spaceDim() == 3 && two_way.spaceDim() == 3,
            "M12b1 analyzer requires 3-D plotfiles");

        const amrex::Real time_scale =
            std::max(
                amrex::Real(1.0),
                std::max(std::abs(one_way.time()), std::abs(two_way.time())));
        require(
            std::abs(one_way.time() - two_way.time())
                <= scaled_tolerance(time_scale),
            "one-way and two-way plotfiles are at different times");

        const auto& one_names = one_way.varNames();
        const auto& two_names = two_way.varNames();
        for (const std::string& name :
             {"rhotheta", "rhoQ1", "theta", "z_velocity"}) {
            require(
                has_variable(one_names, name),
                "one-way plotfile is missing required variable " + name);
            require(
                has_variable(two_names, name),
                "two-way plotfile is missing required variable " + name);
        }

        const auto one_domain = one_way.probDomain(0);
        const auto two_domain = two_way.probDomain(0);
        require(
            one_domain == two_domain,
            "one-way and two-way plotfile domains differ");

        const auto one_dx = one_way.cellSize(0);
        const auto two_dx = two_way.cellSize(0);
        const auto one_lo = one_way.probLo();
        const auto two_lo = two_way.probLo();
        for (int dir = 0; dir < 3; ++dir) {
            require(
                std::abs(one_dx[dir] - two_dx[dir])
                    <= scaled_tolerance(one_dx[dir]),
                "one-way and two-way plotfile cell sizes differ");
            require(
                std::abs(one_lo[dir] - two_lo[dir])
                    <= scaled_tolerance(one_lo[dir]),
                "one-way and two-way plotfile lower bounds differ");
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

        amrex::Real max_delta_rhotheta =
            -std::numeric_limits<amrex::Real>::infinity();
        amrex::Real max_delta_rhoqv =
            -std::numeric_limits<amrex::Real>::infinity();
        amrex::Real max_delta_theta =
            -std::numeric_limits<amrex::Real>::infinity();
        amrex::Real max_delta_w =
            -std::numeric_limits<amrex::Real>::infinity();
        amrex::Real peak_delta_w_z_m =
            std::numeric_limits<amrex::Real>::quiet_NaN();
        amrex::Real positive_theta_volume_integral =
            amrex::Real(0.0);

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
                            "M12b1 response contains a non-finite difference");

                        max_delta_rhotheta =
                            std::max(max_delta_rhotheta, delta_rhotheta);
                        max_delta_rhoqv =
                            std::max(max_delta_rhoqv, delta_rhoqv);
                        max_delta_theta =
                            std::max(max_delta_theta, delta_theta);

                        if (delta_w > max_delta_w) {
                            max_delta_w = delta_w;
                            peak_delta_w_z_m = z_m;
                        }

                        if (delta_theta > amrex::Real(0.0)) {
                            const amrex::Real contribution =
                                delta_theta * cell_volume;
                            positive_theta_by_k[
                                static_cast<std::size_t>(k - klo)]
                                += contribution;
                            positive_theta_volume_integral += contribution;
                        }
                    }
                }
            }
        }

        require(
            max_delta_rhotheta > amrex::Real(0.0),
            "two-way Fire did not produce a positive rhotheta response");
        require(
            max_delta_rhoqv > amrex::Real(0.0),
            "two-way Fire did not produce a positive rhoqv response");
        require(
            max_delta_theta > amrex::Real(0.0),
            "two-way Fire did not produce a positive theta response");
        require(
            max_delta_w > amrex::Real(0.0),
            "two-way Fire did not produce an upward vertical-velocity response");
        require(
            positive_theta_volume_integral > amrex::Real(0.0),
            "two-way Fire has no positive volume-integrated theta response");

        auto response_height =
            [&](amrex::Real quantile)
        {
            const amrex::Real target =
                quantile * positive_theta_volume_integral;
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
                    return one_lo[2]
                        + (amrex::Real(n) + within_layer) * dz;
                }
                cumulative = next;
            }
            return one_lo[2] + amrex::Real(nz) * dz;
        };

        const amrex::Real theta_z50_m =
            response_height(amrex::Real(0.50));
        const amrex::Real theta_z95_m =
            response_height(amrex::Real(0.95));

        require(
            std::isfinite(peak_delta_w_z_m)
                && std::isfinite(theta_z50_m)
                && std::isfinite(theta_z95_m),
            "M12b1 response heights are not finite");
        require(
            theta_z50_m <= theta_z95_m,
            "M12b1 response quantile heights are not ordered");

        amrex::Print()
            << std::setprecision(17)
            << "M12B1_RESPONSE_METRICS"
            << " time_s=" << one_way.time()
            << " max_delta_rhotheta="
            << max_delta_rhotheta
            << " max_delta_rhoqv="
            << max_delta_rhoqv
            << " max_delta_theta_K="
            << max_delta_theta
            << " max_delta_w_mps="
            << max_delta_w
            << " peak_delta_w_z_m="
            << peak_delta_w_z_m
            << " theta_response_z50_m="
            << theta_z50_m
            << " theta_response_z95_m="
            << theta_z95_m
            << " positive_theta_volume_integral_K_m3="
            << positive_theta_volume_integral
            << "\n";
    } catch (const std::exception& error) {
        amrex::Print()
            << "M12b1 atmospheric-response analysis error: "
            << error.what()
            << "\n";
        result = 2;
    }

    amrex::Finalize();
    return result;
}
