#include <ERF.H>
#include <ERF_FireFlatEnvironmentSampler.H>
#include <ERF_FireSpreadRuntime.H>

#include <AMReX_ParmParse.H>
#include <AMReX_Print.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

int
main (int argc, char** argv)
{
    amrex::Initialize(argc, argv);

    int result = 0;
    try {
        amrex::ParmParse pp_fire("fire");
        bool expect_fire_enabled = false;
        pp_fire.query("enabled", expect_fire_enabled);

        amrex::Real expected_reference_height_agl_m = amrex::Real(0);
        pp_fire.query(
            "reference_height_agl_m",
            expected_reference_height_agl_m);

        ERF erf;
        erf.InitData();

        if (erf.FireEnabled() != expect_fire_enabled) {
            throw std::runtime_error(
                "fire enabled configuration was not preserved");
        }

        double time = 0.0;
        double last_dt = 0.0;
        constexpr int steps = 5;

        for (int step = 0; step < steps; ++step) {
            last_dt = erf.EvolveOneStep(time, 0.01);
            if (!(last_dt > 0.0)) {
                throw std::runtime_error(
                    "ERF one-way identity run did not advance");
            }
            time += last_dt;
        }

        const auto* snapshot = erf.FireEnvironmentSnapshot();
        const auto* fire_runtime = erf.FireSpreadRuntime();

        if (expect_fire_enabled) {
            if (snapshot == nullptr) {
                throw std::runtime_error(
                    "enabled fire did not freeze an environment snapshot");
            }
            if (fire_runtime == nullptr) {
                throw std::runtime_error(
                    "enabled fire did not create spread runtime state");
            }

            const double expected_snapshot_time =
                time - last_dt;
            if (erf.FireEnvironmentSnapshotTime()
                != expected_snapshot_time) {
                throw std::runtime_error(
                    "fire environment snapshot was not frozen at the final t^n");
            }
            if (snapshot->reference_height_agl_m()
                != expected_reference_height_agl_m) {
                throw std::runtime_error(
                    "fire environment snapshot reference height changed");
            }
            if (fire_runtime->current_time_s()
                != static_cast<amrex::Real>(time)) {
                throw std::runtime_error(
                    "fire runtime clock did not track ERF coarse steps");
            }

            const auto sample = snapshot->sample(
                amrex::Real(4.0), amrex::Real(4.0));
            if (!std::isfinite(sample.horizontal_wind_mps.x)
                || !std::isfinite(sample.horizontal_wind_mps.y)) {
                throw std::runtime_error(
                    "fire environment snapshot produced non-finite wind");
            }
            if (std::abs(
                    sample.horizontal_wind_mps.x
                    - amrex::Real(1.0))
                > amrex::Real(1.0e-12)
                || std::abs(sample.horizontal_wind_mps.y)
                    > amrex::Real(1.0e-12)) {
                throw std::runtime_error(
                    "fire identity atmosphere did not preserve configured constant wind");
            }

            amrex::Real maximum_x =
                -std::numeric_limits<amrex::Real>::infinity();
            for (const auto& vertex :
                 fire_runtime->perimeter().vertices_m()) {
                maximum_x = std::max(maximum_x, vertex.x);
            }
            if (!(maximum_x > amrex::Real(5.0))) {
                throw std::runtime_error(
                    "one-way fire perimeter did not propagate downwind");
            }
            if (!(fire_runtime->burned_fraction_raster().burned_area_m2()
                  > amrex::Real(0.0))) {
                throw std::runtime_error(
                    "one-way fire burned history is empty");
            }
            if (fire_runtime->first_arrival_raster().arrived_cell_count()
                == 0U) {
                throw std::runtime_error(
                    "one-way fire arrival history is empty");
            }
        } else {
            if (snapshot != nullptr) {
                throw std::runtime_error(
                    "disabled fire unexpectedly created an environment snapshot");
            }
            if (fire_runtime != nullptr) {
                throw std::runtime_error(
                    "disabled fire unexpectedly created spread runtime state");
            }
        }

        // Use ERF's existing public checkpoint writer as the atmospheric
        // identity surface. The enabled process advances a real Fire perimeter
        // for five coarse steps, while prognostic ERF checkpoint bytes must
        // remain identical to the disabled control.
        erf.WriteCheckpointFile();
        amrex::Print() << "ERF_FIRE_IDENTITY_CHILD_OK=1\n";
    } catch (const std::exception& error) {
        amrex::Print() << "ERF fire one-way identity error: "
                       << error.what() << "\n";
        result = 2;
    }

    amrex::Finalize();
    return result;
}
