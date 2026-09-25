#include <AMReX.H>
#include <AMReX_Arena.H>
#include <AMReX_Gpu.H>
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
    return amrex::Real(256)
        * std::numeric_limits<amrex::Real>::epsilon()
        * std::max(amrex::Real(1), std::abs(value));
}

amrex::MultiFab
make_pinned_host_copy(const amrex::MultiFab& source)
{
    amrex::MFInfo info;
    info.SetArena(amrex::The_Pinned_Arena());

    amrex::MultiFab host(
        source.boxArray(),
        source.DistributionMap(),
        source.nComp(),
        0,
        info);

    amrex::MultiFab::Copy(
        host,
        source,
        0,
        0,
        source.nComp(),
        0);

    amrex::Gpu::streamSynchronize();
    return host;
}

struct DifferencePair
{
    amrex::Real coarse_medium{};
    amrex::Real medium_fine{};
};

DifferencePair
max_abs_differences(
    amrex::PlotFileData& coarse,
    amrex::PlotFileData& medium,
    amrex::PlotFileData& fine,
    const std::string& variable)
{
    auto coarse_mf = coarse.get(0, variable);
    auto medium_mf = medium.get(0, variable);
    auto fine_mf = fine.get(0, variable);

    auto coarse_host = make_pinned_host_copy(coarse_mf);
    auto medium_host = make_pinned_host_copy(medium_mf);
    auto fine_host = make_pinned_host_copy(fine_mf);

    amrex::Real coarse_medium = amrex::Real(0);
    amrex::Real medium_fine = amrex::Real(0);

    for (amrex::MFIter mfi(coarse_host); mfi.isValid(); ++mfi) {
        const amrex::Box& box = mfi.validbox();
        const auto coarse_arr = coarse_host.const_array(mfi);
        const auto medium_arr = medium_host.const_array(mfi);
        const auto fine_arr = fine_host.const_array(mfi);

        for (int k = box.smallEnd(2); k <= box.bigEnd(2); ++k) {
            for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
                for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                    const amrex::Real cm =
                        std::abs(
                            coarse_arr(i, j, k)
                            - medium_arr(i, j, k));
                    const amrex::Real mf =
                        std::abs(
                            medium_arr(i, j, k)
                            - fine_arr(i, j, k));

                    require(
                        std::isfinite(cm) && std::isfinite(mf),
                        "temporal-refinement difference is not finite");

                    coarse_medium = std::max(coarse_medium, cm);
                    medium_fine = std::max(medium_fine, mf);
                }
            }
        }
    }

    return {coarse_medium, medium_fine};
}

} // namespace

int
main(int argc, char** argv)
{
    amrex::Initialize(argc, argv);

    int result = 0;
    try {
        require(
            amrex::ParallelDescriptor::NProcs() == 1,
            "temporal-refinement analyzer is intentionally single-rank");

        amrex::ParmParse pp("analysis");

        std::string coarse_path;
        std::string medium_path;
        std::string fine_path;
        pp.get("coarse_plot", coarse_path);
        pp.get("medium_plot", medium_path);
        pp.get("fine_plot", fine_path);

        amrex::PlotFileData coarse(coarse_path);
        amrex::PlotFileData medium(medium_path);
        amrex::PlotFileData fine(fine_path);

        require(
            coarse.finestLevel() == 0
                && medium.finestLevel() == 0
                && fine.finestLevel() == 0,
            "temporal-refinement analyzer requires level-0-only plotfiles");
        require(
            coarse.spaceDim() == 3
                && medium.spaceDim() == 3
                && fine.spaceDim() == 3,
            "temporal-refinement analyzer requires 3-D plotfiles");

        const amrex::Real time_scale =
            std::max(
                amrex::Real(1),
                std::max(
                    std::abs(coarse.time()),
                    std::max(
                        std::abs(medium.time()),
                        std::abs(fine.time()))));
        require(
            std::abs(coarse.time() - medium.time())
                    <= scaled_tolerance(time_scale)
                && std::abs(medium.time() - fine.time())
                    <= scaled_tolerance(time_scale),
            "temporal-refinement plotfiles are at different physical times");

        require(
            coarse.probDomain(0) == medium.probDomain(0)
                && medium.probDomain(0) == fine.probDomain(0),
            "temporal-refinement plotfile domains differ");

        const auto coarse_dx = coarse.cellSize(0);
        const auto medium_dx = medium.cellSize(0);
        const auto fine_dx = fine.cellSize(0);
        const auto coarse_lo = coarse.probLo();
        const auto medium_lo = medium.probLo();
        const auto fine_lo = fine.probLo();

        for (int dir = 0; dir < 3; ++dir) {
            require(
                std::abs(coarse_dx[dir] - medium_dx[dir])
                        <= scaled_tolerance(coarse_dx[dir])
                    && std::abs(medium_dx[dir] - fine_dx[dir])
                        <= scaled_tolerance(medium_dx[dir]),
                "temporal-refinement plotfile cell sizes differ");
            require(
                std::abs(coarse_lo[dir] - medium_lo[dir])
                        <= scaled_tolerance(coarse_lo[dir])
                    && std::abs(medium_lo[dir] - fine_lo[dir])
                        <= scaled_tolerance(medium_lo[dir]),
                "temporal-refinement plotfile lower bounds differ");
        }

        medium.syncDistributionMap(coarse);
        fine.syncDistributionMap(coarse);

        constexpr std::array<const char*, 7> variables{
            "density",
            "rhotheta",
            "rhoQ1",
            "x_velocity",
            "y_velocity",
            "z_velocity",
            "theta"};

        for (const char* variable : variables) {
            require(
                has_variable(coarse.varNames(), variable)
                    && has_variable(medium.varNames(), variable)
                    && has_variable(fine.varNames(), variable),
                std::string("temporal-refinement plotfile is missing variable ")
                    + variable);

            const DifferencePair difference =
                max_abs_differences(
                    coarse,
                    medium,
                    fine,
                    variable);

            amrex::Real apparent_order =
                std::numeric_limits<amrex::Real>::quiet_NaN();
            if (difference.coarse_medium > amrex::Real(0)
                && difference.medium_fine > amrex::Real(0)) {
                apparent_order =
                    std::log(
                        difference.coarse_medium
                        / difference.medium_fine)
                    / std::log(amrex::Real(2));
            } else if (difference.coarse_medium > amrex::Real(0)
                       && difference.medium_fine == amrex::Real(0)) {
                apparent_order =
                    std::numeric_limits<amrex::Real>::infinity();
            }

            const bool monotone =
                difference.medium_fine
                <= difference.coarse_medium;

            amrex::Print()
                << std::setprecision(17)
                << "FIRE_TEMPORAL_REFINEMENT"
                << " variable=" << variable
                << " coarse_medium_max_abs="
                << difference.coarse_medium
                << " medium_fine_max_abs="
                << difference.medium_fine
                << " apparent_order="
                << apparent_order
                << " monotone="
                << (monotone ? 1 : 0)
                << "\n";
        }
    } catch (const std::exception& error) {
        amrex::Print()
            << "temporal-refinement analysis error: "
            << error.what()
            << "\n";
        result = 2;
    }

    amrex::Finalize();
    return result;
}
