#include <ERF.H>
#include <ERF_FireFlatEnvironmentSampler.H>
#include <ERF_FireRuntimeOptions.H>
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

        std::string expected_coupling_mode{"one_way"};
        pp_fire.query(
            "coupling_mode",
            expected_coupling_mode);
        const bool expect_two_way =
            expect_fire_enabled
            && expected_coupling_mode == "two_way";

        std::string expected_wind_mode{"direct_reference"};
        pp_fire.query(
            "wind_mode",
            expected_wind_mode);

        amrex::Real expected_reference_height_agl_m = amrex::Real(0);
        if (expected_wind_mode == "explicit_waf_20ft") {
            expected_reference_height_agl_m =
                ERFFire::explicit_waf_20ft_reference_height_agl_m;
        } else {
            pp_fire.query(
                "reference_height_agl_m",
                expected_reference_height_agl_m);
        }

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
        const auto* fire_source =
            erf.FireAtmosphericSourceTendency();

        if (expect_fire_enabled) {
            if (snapshot != nullptr) {
                throw std::runtime_error(
                    "enabled fire unexpectedly retained a replicated environment snapshot");
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
            if (erf.FireEnvironmentReferenceHeightAGL()
                != expected_reference_height_agl_m) {
                throw std::runtime_error(
                    "fire environment reference height changed");
            }
            if (fire_runtime->current_time_s()
                != static_cast<amrex::Real>(time)) {
                throw std::runtime_error(
                    "fire runtime clock did not track ERF coarse steps");
            }

            if (expect_two_way) {
                if (fire_source == nullptr) {
                    throw std::runtime_error(
                        "two-way fire did not create a native atmospheric source");
                }
                if (erf.FireAtmosphericSourceTime()
                    != expected_snapshot_time) {
                    throw std::runtime_error(
                        "two-way fire source was not frozen at the final t^n");
                }
                if (fire_source->norm0(
                        Rho_comp, 0, true)
                    != amrex::Real(0.0)) {
                    throw std::runtime_error(
                        "two-way fire source modified dry-air density");
                }
                if (!(fire_source->norm0(
                          RhoTheta_comp, 0, true)
                      > amrex::Real(0.0))) {
                    throw std::runtime_error(
                        "two-way fire rho-theta source is not positive");
                }
                if (!(fire_source->norm0(
                          RhoQ1_comp, 0, true)
                      > amrex::Real(0.0))) {
                    throw std::runtime_error(
                        "two-way fire rho-qv source is not positive");
                }
            } else if (fire_source != nullptr) {
                throw std::runtime_error(
                    "non-two-way fire unexpectedly created an atmospheric source");
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

            const auto combustion_totals =
                fire_runtime->combustion_raster().totals();
            if (!(combustion_totals.remaining_dry_fuel_kg
                  > amrex::Real(0.0))) {
                throw std::runtime_error(
                    "one-way fire combustion has no remaining ignited fuel");
            }
            if (!(combustion_totals.consumed_dry_fuel_kg
                  > amrex::Real(0.0))
                || !(combustion_totals.sensible_energy_j
                     > amrex::Real(0.0))
                || !(combustion_totals.water_released_kg
                     > amrex::Real(0.0))) {
                throw std::runtime_error(
                    "one-way fire combustion did not consume fuel and release heat/water");
            }

            const auto& combustion_parameters =
                fire_runtime->combustion_raster().parameters();
            const amrex::Real expected_ignited_dry_fuel_kg =
                fire_runtime->burned_fraction_raster().burned_area_m2()
                * combustion_parameters.dry_fuel_load_kg_m2;
            const amrex::Real represented_ignited_dry_fuel_kg =
                combustion_totals.remaining_dry_fuel_kg
                + combustion_totals.consumed_dry_fuel_kg;
            const amrex::Real combustion_mass_tolerance =
                amrex::Real(1.0e-10)
                * std::max(
                    amrex::Real(1.0),
                    std::abs(expected_ignited_dry_fuel_kg));
            if (std::abs(
                    represented_ignited_dry_fuel_kg
                    - expected_ignited_dry_fuel_kg)
                > combustion_mass_tolerance) {
                throw std::runtime_error(
                    "one-way fire combustion mass does not match burned-area fuel loading");
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
            if (fire_source != nullptr) {
                throw std::runtime_error(
                    "disabled fire unexpectedly created atmospheric source state");
            }
        }

        // Use ERF's existing public checkpoint writer as the atmospheric
        // identity surface. The enabled process advances a real Fire perimeter
        // for five coarse steps, while prognostic ERF checkpoint bytes must
        // remain identical to the disabled control.
        erf.WriteCheckpointFile();
        amrex::Print() << "ERF_FIRE_IDENTITY_CHILD_OK=1\n";
    } catch (const std::exception& error) {
        amrex::Print() << "ERF fire integration error: "
                       << error.what() << "\n";
        result = 2;
    }

    amrex::Finalize();
    return result;
}
