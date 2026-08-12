#include <ERF.H>
#include <ERF_FireFlatEnvironmentSampler.H>

#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <cmath>
#include <stdexcept>

int
main (int argc, char** argv)
{
    amrex::Initialize(argc, argv);

    int result = 0;
    try {
        amrex::ParmParse pp_fire("fire");
        bool expect_environment_read = false;
        pp_fire.query("environment_read", expect_environment_read);

        amrex::Real expected_reference_height_agl_m = amrex::Real(0);
        pp_fire.query(
            "reference_height_agl_m",
            expected_reference_height_agl_m);

        ERF erf;
        erf.InitData();

        if (erf.FireEnvironmentReadEnabled() != expect_environment_read) {
            throw std::runtime_error(
                "fire environment-read configuration was not preserved");
        }

        const double dt = erf.EvolveOneStep(0.0, 0.01);
        if (!(dt > 0.0)) {
            throw std::runtime_error(
                "ERF one-step identity run did not advance");
        }

        const auto* snapshot = erf.FireEnvironmentSnapshot();
        if (expect_environment_read) {
            if (snapshot == nullptr) {
                throw std::runtime_error(
                    "enabled fire environment read did not freeze a snapshot");
            }
            if (erf.FireEnvironmentSnapshotTime() != 0.0) {
                throw std::runtime_error(
                    "fire environment snapshot was not frozen at t^n");
            }
            if (snapshot->reference_height_agl_m()
                != expected_reference_height_agl_m) {
                throw std::runtime_error(
                    "fire environment snapshot reference height changed");
            }

            const auto sample = snapshot->sample(
                amrex::Real(4.0), amrex::Real(4.0));
            if (!std::isfinite(sample.horizontal_wind_mps.x)
                || !std::isfinite(sample.horizontal_wind_mps.y)) {
                throw std::runtime_error(
                    "fire environment snapshot produced non-finite wind");
            }
        } else if (snapshot != nullptr) {
            throw std::runtime_error(
                "disabled fire environment read unexpectedly created a snapshot");
        }

        // Use ERF's existing public checkpoint writer as the identity surface.
        // It copies the valid prognostic Cell/XFace/YFace/ZFace MultiFabs
        // directly from vars_new without exposing solver-private storage.
        erf.WriteCheckpointFile();
        amrex::Print() << "ERF_FIRE_IDENTITY_CHILD_OK=1\n";
    } catch (const std::exception& error) {
        amrex::Print() << "ERF fire one-step identity error: "
                       << error.what() << "\n";
        result = 2;
    }

    amrex::Finalize();
    return result;
}
