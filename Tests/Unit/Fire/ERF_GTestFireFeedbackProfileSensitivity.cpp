#include <ERF_FireAtmosphericSource.H>

#include <ERF_Constants.H>
#include <ERF_FireBurnedFractionRaster.H>
#include <ERF_FireCombustion.H>
#include <ERF_FireCombustionRaster.H>
#include <ERF_FirePerimeter.H>
#include <ERF_FireSurfaceFeedback.H>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>

namespace
{

using amrex::Real;
using ERFFire::ERFFireAtmosphericSourceCell;
using ERFFire::ERFFireAtmosphericSourceOptions;
using ERFFire::FireBurnedFractionRaster;
using ERFFire::FireCartesianRasterGeometry2D;
using ERFFire::FireCombustionRaster;
using ERFFire::FireCombustionRasterOptions;
using ERFFire::FirePerimeter;
using ERFFire::FireSurfaceFeedbackCell;
using ERFFire::FireSurfaceFeedbackRaster;
using ERFFire::FireVec2;

constexpr Real fuel_moisture = Real(0.08);
constexpr Real dt_s = Real(4.0);
constexpr std::array<Real, 4> extinction_depth_m{
    Real(10.0), Real(25.0), Real(50.0), Real(100.0)};

FirePerimeter
rectangle(
    Real xlo,
    Real xhi,
    Real ylo,
    Real yhi)
{
    std::vector<FireVec2> vertices{
        {xlo, ylo},
        {xhi, ylo},
        {xhi, yhi},
        {xlo, yhi}
    };
    return FirePerimeter(std::move(vertices));
}

FireBurnedFractionRaster
burned_from(
    const FireCartesianRasterGeometry2D& geometry,
    const FirePerimeter& perimeter)
{
    FireBurnedFractionRaster burned(geometry);
    (void)burned.update_from_perimeter(perimeter);
    return burned;
}

FireSurfaceFeedbackRaster
stationary_fm1_feedback(
    const FireCartesianRasterGeometry2D& geometry,
    Real step_dt_s)
{
    const FirePerimeter full =
        rectangle(
            geometry.xlo_m - geometry.dx_m,
            geometry.xlo_m
                + static_cast<Real>(geometry.nx) * geometry.dx_m,
            geometry.ylo_m,
            geometry.ylo_m
                + static_cast<Real>(geometry.ny) * geometry.dy_m);
    const auto burned = burned_from(geometry, full);

    FireCombustionRaster before(
        geometry,
        ERFFire::make_fm1_combustion_parameters(fuel_moisture),
        FireCombustionRasterOptions{16});
    (void)before.initialize_from_burned_fraction(burned);

    FireCombustionRaster after = before;
    (void)after.advance_from_linear_sweep(
        full,
        full,
        burned,
        burned,
        step_dt_s);

    return ERFFire::make_fire_surface_feedback_increment(
        before,
        after);
}

Real
exner_from_pressure(Real pressure_pa)
{
    return std::pow(
        pressure_pa * ip_0,
        RdoCp);
}

std::vector<Real>
physical_volumes(
    const std::vector<Real>& faces_m,
    Real horizontal_area_m2)
{
    std::vector<Real> result;
    result.reserve(faces_m.size() - 1);
    for (std::size_t k = 0; k + 1 < faces_m.size(); ++k) {
        result.push_back(
            horizontal_area_m2
            * (faces_m[k + 1] - faces_m[k]));
    }
    return result;
}

std::vector<Real>
constant_pressure(std::size_t nz, Real pressure_pa = p_0)
{
    return std::vector<Real>(nz, pressure_pa);
}

Real
recovered_energy_j(
    const std::vector<ERFFireAtmosphericSourceCell>& source,
    const std::vector<Real>& volume_m3,
    const std::vector<Real>& pressure_pa,
    Real step_dt_s)
{
    Real result = Real(0.0);
    for (std::size_t k = 0; k < source.size(); ++k) {
        result +=
            source[k].rhotheta_tendency_kg_K_m3_s
            * Cp_d
            * exner_from_pressure(pressure_pa[k])
            * volume_m3[k]
            * step_dt_s;
    }
    return result;
}

Real
recovered_water_kg(
    const std::vector<ERFFireAtmosphericSourceCell>& source,
    const std::vector<Real>& volume_m3,
    Real step_dt_s)
{
    Real result = Real(0.0);
    for (std::size_t k = 0; k < source.size(); ++k) {
        result +=
            source[k].rhoqv_tendency_kg_m3_s
            * volume_m3[k]
            * step_dt_s;
    }
    return result;
}

std::vector<Real>
recovered_layer_fractions(
    const std::vector<ERFFireAtmosphericSourceCell>& source,
    const std::vector<Real>& volume_m3,
    Real step_dt_s,
    const FireSurfaceFeedbackCell& release)
{
    std::vector<Real> result(source.size(), Real(0.0));
    for (std::size_t k = 0; k < source.size(); ++k) {
        result[k] =
            source[k].rhoqv_tendency_kg_m3_s
            * volume_m3[k]
            * step_dt_s
            / release.water_released_kg;
    }
    return result;
}


std::vector<Real>
recovered_heat_layer_fractions(
    const std::vector<ERFFireAtmosphericSourceCell>& source,
    const std::vector<Real>& volume_m3,
    const std::vector<Real>& pressure_pa,
    Real step_dt_s,
    const FireSurfaceFeedbackCell& release)
{
    std::vector<Real> result(source.size(), Real(0.0));
    for (std::size_t k = 0; k < source.size(); ++k) {
        result[k] =
            source[k].rhotheta_tendency_kg_K_m3_s
            * Cp_d
            * exner_from_pressure(pressure_pa[k])
            * volume_m3[k]
            * step_dt_s
            / release.sensible_energy_j;
    }
    return result;
}

Real
expected_layer_fraction(
    Real zlo_m,
    Real zhi_m,
    Real top_m,
    Real H_m)
{
    const Real normalization =
        Real(1.0) - std::exp(-top_m / H_m);
    return (
        std::exp(-zlo_m / H_m)
        - std::exp(-zhi_m / H_m))
        / normalization;
}

Real
extensive_tolerance(Real value)
{
    return Real(2.0e-11)
        * std::max(Real(1.0), std::abs(value));
}

void
expect_extensive_closure(
    const std::vector<ERFFireAtmosphericSourceCell>& source,
    const std::vector<Real>& volume_m3,
    const std::vector<Real>& pressure_pa,
    Real step_dt_s,
    const FireSurfaceFeedbackCell& release)
{
    EXPECT_NEAR(
        recovered_energy_j(source, volume_m3, pressure_pa, step_dt_s),
        release.sensible_energy_j,
        extensive_tolerance(release.sensible_energy_j));
    EXPECT_NEAR(
        recovered_water_kg(source, volume_m3, step_dt_s),
        release.water_released_kg,
        extensive_tolerance(release.water_released_kg));
}

struct ProfileMetrics
{
    Real first_layer_fraction{};
    Real below_25m_fraction{};
    Real below_50m_fraction{};
    Real deposition_centroid_m{};
    Real bottom_rhotheta_tendency{};
    Real bottom_rhoqv_tendency{};
    Real recovered_energy_j{};
    Real recovered_water_kg{};
};

ProfileMetrics
profile_metrics(
    const std::vector<Real>& faces_m,
    const std::vector<Real>& volume_m3,
    const std::vector<Real>& pressure_pa,
    const std::vector<ERFFireAtmosphericSourceCell>& source,
    const FireSurfaceFeedbackCell& release)
{
    const auto fractions =
        recovered_layer_fractions(
            source,
            volume_m3,
            dt_s,
            release);

    Real centroid_m = Real(0.0);
    for (std::size_t k = 0; k < fractions.size(); ++k) {
        const Real midpoint_m =
            Real(0.5) * (faces_m[k] + faces_m[k + 1]);
        centroid_m += fractions[k] * midpoint_m;
    }

    return {
        fractions[0],
        fractions[0] + fractions[1],
        fractions[0] + fractions[1] + fractions[2],
        centroid_m,
        source[0].rhotheta_tendency_kg_K_m3_s,
        source[0].rhoqv_tendency_kg_m3_s,
        recovered_energy_j(source, volume_m3, pressure_pa, dt_s),
        recovered_water_kg(source, volume_m3, dt_s)};
}

void
print_profile_metrics(
    Real H_m,
    const ProfileMetrics& metrics)
{
    std::cout
        << std::setprecision(17)
        << "FIRE_FEEDBACK_PROFILE_METRICS"
        << " H_m=" << H_m
        << " first_layer_fraction=" << metrics.first_layer_fraction
        << " below_25m_fraction=" << metrics.below_25m_fraction
        << " below_50m_fraction=" << metrics.below_50m_fraction
        << " deposition_centroid_m=" << metrics.deposition_centroid_m
        << " bottom_rhotheta_tendency=" << metrics.bottom_rhotheta_tendency
        << " bottom_rhoqv_tendency=" << metrics.bottom_rhoqv_tendency
        << " recovered_energy_j=" << metrics.recovered_energy_j
        << " recovered_water_kg=" << metrics.recovered_water_kg
        << "\n";
}

} // namespace

TEST(
    FireFeedbackProfileSensitivity,
    ExtinctionDepthSweepPreservesIntegratedHeatAndWater)
{
    const FireCartesianRasterGeometry2D geometry{
        1, 1,
        Real(0.0), Real(0.0),
        Real(2.0), Real(3.0)};
    const auto feedback = stationary_fm1_feedback(geometry, dt_s);
    const auto release = feedback.cell(0, 0);
    ASSERT_GT(release.sensible_energy_j, Real(0.0));
    ASSERT_GT(release.water_released_kg, Real(0.0));

    const std::vector<Real> faces_m{
        Real(0.0), Real(10.0), Real(25.0),
        Real(50.0), Real(100.0), Real(200.0)};
    const Real horizontal_area_m2 = geometry.dx_m * geometry.dy_m;
    const auto volume_m3 = physical_volumes(faces_m, horizontal_area_m2);
    const std::vector<Real> pressure_pa{
        p_0,
        Real(0.97) * p_0,
        Real(0.93) * p_0,
        Real(0.88) * p_0,
        Real(0.80) * p_0};

    for (const Real H_m : extinction_depth_m) {
        const auto source =
            ERFFire::make_erf_fire_atmospheric_source_column(
                release,
                faces_m,
                volume_m3,
                pressure_pa,
                dt_s,
                ERFFireAtmosphericSourceOptions{H_m});
        ASSERT_EQ(source.size(), volume_m3.size());

        const auto fractions =
            recovered_layer_fractions(
                source,
                volume_m3,
                dt_s,
                release);
        const Real fraction_sum =
            std::accumulate(
                fractions.begin(),
                fractions.end(),
                Real(0.0));
        EXPECT_NEAR(fraction_sum, Real(1.0), Real(4.0e-13));

        const auto heat_fractions =
            recovered_heat_layer_fractions(
                source,
                volume_m3,
                pressure_pa,
                dt_s,
                release);
        const Real heat_fraction_sum =
            std::accumulate(
                heat_fractions.begin(),
                heat_fractions.end(),
                Real(0.0));
        EXPECT_NEAR(
            heat_fraction_sum,
            Real(1.0),
            Real(4.0e-13));

        for (std::size_t k = 0; k < fractions.size(); ++k) {
            EXPECT_NEAR(
                fractions[k],
                expected_layer_fraction(
                    faces_m[k],
                    faces_m[k + 1],
                    faces_m.back(),
                    H_m),
                Real(4.0e-13));
            EXPECT_NEAR(
                heat_fractions[k],
                expected_layer_fraction(
                    faces_m[k],
                    faces_m[k + 1],
                    faces_m.back(),
                    H_m),
                Real(4.0e-13));
            EXPECT_NEAR(
                heat_fractions[k],
                fractions[k],
                Real(4.0e-13));
        }

        expect_extensive_closure(
            source,
            volume_m3,
            pressure_pa,
            dt_s,
            release);
    }
}

TEST(
    FireFeedbackProfileSensitivity,
    ExtinctionDepthSweepMovesSourceAloftAndReducesNearSurfaceTendency)
{
    const FireCartesianRasterGeometry2D geometry{
        1, 1,
        Real(0.0), Real(0.0),
        Real(2.0), Real(3.0)};
    const auto feedback = stationary_fm1_feedback(geometry, dt_s);
    const auto release = feedback.cell(0, 0);

    const std::vector<Real> faces_m{
        Real(0.0), Real(10.0), Real(25.0),
        Real(50.0), Real(100.0), Real(200.0)};
    const auto volume_m3 =
        physical_volumes(
            faces_m,
            geometry.dx_m * geometry.dy_m);
    const auto pressure_pa =
        constant_pressure(volume_m3.size());

    std::array<ProfileMetrics, 4> metrics{};
    for (std::size_t n = 0; n < extinction_depth_m.size(); ++n) {
        const Real H_m = extinction_depth_m[n];
        const auto source =
            ERFFire::make_erf_fire_atmospheric_source_column(
                release,
                faces_m,
                volume_m3,
                pressure_pa,
                dt_s,
                ERFFireAtmosphericSourceOptions{H_m});

        metrics[n] = profile_metrics(
            faces_m,
            volume_m3,
            pressure_pa,
            source,
            release);
        print_profile_metrics(H_m, metrics[n]);
    }

    for (std::size_t n = 1; n < metrics.size(); ++n) {
        EXPECT_GT(
            metrics[n - 1].first_layer_fraction,
            metrics[n].first_layer_fraction);
        EXPECT_GT(
            metrics[n - 1].below_25m_fraction,
            metrics[n].below_25m_fraction);
        EXPECT_GT(
            metrics[n - 1].below_50m_fraction,
            metrics[n].below_50m_fraction);
        EXPECT_LT(
            metrics[n - 1].deposition_centroid_m,
            metrics[n].deposition_centroid_m);
        EXPECT_GT(
            metrics[n - 1].bottom_rhotheta_tendency,
            metrics[n].bottom_rhotheta_tendency);
        EXPECT_GT(
            metrics[n - 1].bottom_rhoqv_tendency,
            metrics[n].bottom_rhoqv_tendency);
    }
}

TEST(
    FireFeedbackProfileSensitivity,
    VerticalPartitionPreservesIntegratedFractionsAtSharedFaces)
{
    const FireCartesianRasterGeometry2D geometry{
        1, 1,
        Real(0.0), Real(0.0),
        Real(2.0), Real(3.0)};
    const auto feedback = stationary_fm1_feedback(geometry, dt_s);
    const auto release = feedback.cell(0, 0);
    const Real horizontal_area_m2 = geometry.dx_m * geometry.dy_m;

    const std::vector<Real> coarse_faces_m{
        Real(0.0), Real(50.0), Real(100.0), Real(200.0)};
    const std::vector<Real> refined_faces_m{
        Real(0.0), Real(10.0), Real(25.0), Real(50.0),
        Real(75.0), Real(100.0), Real(150.0), Real(200.0)};
    const auto coarse_volume_m3 =
        physical_volumes(coarse_faces_m, horizontal_area_m2);
    const auto refined_volume_m3 =
        physical_volumes(refined_faces_m, horizontal_area_m2);
    const auto coarse_pressure_pa =
        constant_pressure(coarse_volume_m3.size());
    const auto refined_pressure_pa =
        constant_pressure(refined_volume_m3.size());

    for (const Real H_m : extinction_depth_m) {
        const auto coarse_source =
            ERFFire::make_erf_fire_atmospheric_source_column(
                release,
                coarse_faces_m,
                coarse_volume_m3,
                coarse_pressure_pa,
                dt_s,
                ERFFireAtmosphericSourceOptions{H_m});
        const auto refined_source =
            ERFFire::make_erf_fire_atmospheric_source_column(
                release,
                refined_faces_m,
                refined_volume_m3,
                refined_pressure_pa,
                dt_s,
                ERFFireAtmosphericSourceOptions{H_m});

        const auto coarse_fraction =
            recovered_layer_fractions(
                coarse_source,
                coarse_volume_m3,
                dt_s,
                release);
        const auto refined_fraction =
            recovered_layer_fractions(
                refined_source,
                refined_volume_m3,
                dt_s,
                release);

        ASSERT_EQ(coarse_fraction.size(), 3U);
        ASSERT_EQ(refined_fraction.size(), 7U);
        EXPECT_NEAR(
            coarse_fraction[0],
            refined_fraction[0]
                + refined_fraction[1]
                + refined_fraction[2],
            Real(5.0e-13));
        EXPECT_NEAR(
            coarse_fraction[1],
            refined_fraction[3]
                + refined_fraction[4],
            Real(5.0e-13));
        EXPECT_NEAR(
            coarse_fraction[2],
            refined_fraction[5]
                + refined_fraction[6],
            Real(5.0e-13));

        expect_extensive_closure(
            coarse_source,
            coarse_volume_m3,
            coarse_pressure_pa,
            dt_s,
            release);
        expect_extensive_closure(
            refined_source,
            refined_volume_m3,
            refined_pressure_pa,
            dt_s,
            release);
    }
}

TEST(
    FireFeedbackProfileSensitivity,
    PhysicalLayerVolumesRescaleTendenciesWithoutChangingExtensiveTotals)
{
    const FireCartesianRasterGeometry2D geometry{
        1, 1,
        Real(0.0), Real(0.0),
        Real(2.0), Real(3.0)};
    const auto feedback = stationary_fm1_feedback(geometry, dt_s);
    const auto release = feedback.cell(0, 0);

    const std::vector<Real> faces_m{
        Real(0.0), Real(10.0), Real(25.0),
        Real(50.0), Real(100.0), Real(200.0)};
    const auto base_volume_m3 =
        physical_volumes(
            faces_m,
            geometry.dx_m * geometry.dy_m);
    std::vector<Real> scaled_volume_m3 = base_volume_m3;
    const std::array<Real, 5> volume_scale{
        Real(2.0), Real(0.5), Real(3.0), Real(1.5), Real(4.0)};
    for (std::size_t k = 0; k < scaled_volume_m3.size(); ++k) {
        scaled_volume_m3[k] *= volume_scale[k];
    }

    const std::vector<Real> pressure_pa{
        p_0,
        Real(0.97) * p_0,
        Real(0.93) * p_0,
        Real(0.88) * p_0,
        Real(0.80) * p_0};
    constexpr Real H_m = Real(50.0);

    const auto base_source =
        ERFFire::make_erf_fire_atmospheric_source_column(
            release,
            faces_m,
            base_volume_m3,
            pressure_pa,
            dt_s,
            ERFFireAtmosphericSourceOptions{H_m});
    const auto scaled_source =
        ERFFire::make_erf_fire_atmospheric_source_column(
            release,
            faces_m,
            scaled_volume_m3,
            pressure_pa,
            dt_s,
            ERFFireAtmosphericSourceOptions{H_m});

    ASSERT_EQ(base_source.size(), scaled_source.size());
    for (std::size_t k = 0; k < base_source.size(); ++k) {
        EXPECT_NEAR(
            base_source[k].rhoqv_tendency_kg_m3_s
                * base_volume_m3[k],
            scaled_source[k].rhoqv_tendency_kg_m3_s
                * scaled_volume_m3[k],
            Real(5.0e-13));
        EXPECT_NEAR(
            base_source[k].rhotheta_tendency_kg_K_m3_s
                * base_volume_m3[k],
            scaled_source[k].rhotheta_tendency_kg_K_m3_s
                * scaled_volume_m3[k],
            Real(5.0e-13));
    }

    expect_extensive_closure(
        base_source,
        base_volume_m3,
        pressure_pa,
        dt_s,
        release);
    expect_extensive_closure(
        scaled_source,
        scaled_volume_m3,
        pressure_pa,
        dt_s,
        release);
}
