#include <AMReX.H>
#include <AMReX_MFIter.H>
#include <AMReX_ParmParse.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Print.H>

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

} // namespace

int
main(int argc, char** argv)
{
    amrex::Initialize(argc, argv);

    int result = 0;
    try {
        amrex::ParmParse pp("analysis");
        std::string reference_path;
        std::string comparison_path;
        pp.get("reference_plot", reference_path);
        pp.get("comparison_plot", comparison_path);

        amrex::PlotFileData reference(reference_path);
        amrex::PlotFileData comparison(comparison_path);

        require(
            reference.finestLevel() == 0
                && comparison.finestLevel() == 0,
            "ERF-Fire decomposition comparison requires level-0 plotfiles");
        require(
            reference.spaceDim() == 3
                && comparison.spaceDim() == 3,
            "ERF-Fire decomposition comparison requires 3-D plotfiles");
        require(
            reference.time() == comparison.time(),
            "ERF-Fire decomposition plotfile times differ");
        require(
            reference.varNames() == comparison.varNames(),
            "ERF-Fire decomposition plotfile variable inventories differ");
        require(
            reference.probDomain(0) == comparison.probDomain(0),
            "ERF-Fire decomposition plotfile domains differ");

        const auto reference_dx = reference.cellSize(0);
        const auto comparison_dx = comparison.cellSize(0);
        const auto reference_lo = reference.probLo();
        const auto comparison_lo = comparison.probLo();

        for (int direction = 0; direction < 3; ++direction) {
            require(
                reference_dx[direction] == comparison_dx[direction],
                "ERF-Fire decomposition plotfile cell sizes differ");
            require(
                reference_lo[direction] == comparison_lo[direction],
                "ERF-Fire decomposition plotfile lower bounds differ");
        }

        comparison.syncDistributionMap(reference);

        for (const std::string& name : reference.varNames()) {
            auto reference_field = reference.get(0, name);
            auto comparison_field = comparison.get(0, name);

            require(
                reference_field.boxArray() == comparison_field.boxArray(),
                "ERF-Fire decomposition field BoxArrays differ for " + name);
            require(
                reference_field.nComp() == comparison_field.nComp(),
                "ERF-Fire decomposition component counts differ for " + name);

            for (amrex::MFIter mfi(reference_field);
                 mfi.isValid(); ++mfi) {
                const amrex::Box& box = mfi.validbox();
                const auto a = reference_field.const_array(mfi);
                const auto b = comparison_field.const_array(mfi);

                for (int component = 0;
                     component < reference_field.nComp(); ++component) {
                    for (int k = box.smallEnd(2); k <= box.bigEnd(2); ++k) {
                        for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
                            for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                                if (a(i, j, k, component)
                                    != b(i, j, k, component)) {
                                    throw std::runtime_error(
                                        "ERF-Fire atmospheric decomposition mismatch in "
                                        + name + " at ("
                                        + std::to_string(i) + ","
                                        + std::to_string(j) + ","
                                        + std::to_string(k) + ")");
                                }
                            }
                        }
                    }
                }
            }
        }

        amrex::Print()
            << "ERF-Fire atmospheric plotfields are exactly decomposition invariant\n";
    } catch (const std::exception& error) {
        amrex::Print()
            << "ERF-Fire decomposition analysis error: "
            << error.what() << "\n";
        result = 1;
    }

    amrex::Finalize();
    return result;
}
