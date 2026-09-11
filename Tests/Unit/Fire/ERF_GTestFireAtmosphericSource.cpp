#include <ERF_FireAtmosphericSource.H>

#include <ERF_Constants.H>
#include <ERF_FireBurnedFractionRaster.H>
#include <ERF_FireCombustionRaster.H>
#include <ERF_FireSurfaceFeedback.H>

#include <AMReX_Gpu.H>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{

using amrex::Real;
using ERFFire::ERFFireAtmosphericSourceOptions;
using ERFFire::FireBurnedFractionRaster;
using ERFFire::FireCartesianRasterGeometry2D;
using ERFFire::FireCombustionParameters;
using ERFFire::FireCombustionRaster;
using ERFFire::FireCombustionRasterOptions;
using ERFFire::FirePerimeter;
using ERFFire::FireSurfaceFeedbackRaster;
using ERFFire::FireVec2;

FireCombustionParameters
parameters()
{
    return {
        Real(2.0),
        Real(10.0),
        Real(0.25),
        Real(4.0),
        Real(0.5)
    };
}

FirePerimeter
rectangle(
    Real xlo,
    Real xhi,
    Real ylo = Real(0.0),
    Real yhi = Real(1.0))
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
stationary_feedback(
    FireCartesianRasterGeometry2D geometry,
    Real dt_s)
{
    const auto p = parameters();
    const FirePerimeter full =
        rectangle(
            geometry.xlo_m - geometry.dx_m,
            geometry.xlo_m
                + static_cast<Real>(geometry.nx)
                    * geometry.dx_m,
            geometry.ylo_m,
            geometry.ylo_m
                + static_cast<Real>(geometry.ny)
                    * geometry.dy_m);

    const auto burned = burned_from(geometry, full);

    FireCombustionRaster before(
        geometry,
        p,
        FireCombustionRasterOptions{8});
    (void)before.initialize_from_burned_fraction(
        burned);

    FireCombustionRaster after = before;
    (void)after.advance_from_linear_sweep(
        full,
        full,
        burned,
        burned,
        dt_s);

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

} // namespace

#ifdef AMREX_USE_GPU
namespace
{

struct DeviceAtmosphericSourceProbe
{
    Real raw_weight{};
    ERFFire::ERFFireAtmosphericSourceCell cell{};
    int raw_status{};
    int cell_status{};
    int invalid_status{};
};

DeviceAtmosphericSourceProbe
run_device_atmospheric_source_probe()
{
    amrex::Gpu::DeviceScalar<Real> device_raw_weight;
    amrex::Gpu::DeviceScalar<ERFFire::ERFFireAtmosphericSourceCell> device_cell;
    amrex::Gpu::DeviceScalar<int> device_raw_status;
    amrex::Gpu::DeviceScalar<int> device_cell_status;
    amrex::Gpu::DeviceScalar<int> device_invalid_status;

    Real* raw_weight = device_raw_weight.dataPtr();
    auto* cell = device_cell.dataPtr();
    int* raw_status = device_raw_status.dataPtr();
    int* cell_status = device_cell_status.dataPtr();
    int* invalid_status = device_invalid_status.dataPtr();

    const ERFFire::FireSurfaceFeedbackCell feedback{
        Real(0), Real(200), Real(3)};
    constexpr Real extinction_depth_m = Real(50);

    amrex::ParallelFor(
        1,
        [=] AMREX_GPU_DEVICE (int) noexcept
        {
            Real raw{};
            ERFFire::ERFFireAtmosphericSourceCell output{};
            const auto weight_result =
                ERFFire::try_erf_fire_atmospheric_source_raw_layer_weight(
                    Real(0),
                    Real(10),
                    extinction_depth_m,
                    raw);
            const auto cell_result =
                ERFFire::try_make_erf_fire_atmospheric_source_cell(
                    feedback,
                    Real(1),
                    Real(60),
                    p_0,
                    Real(2),
                    output);
            ERFFire::ERFFireAtmosphericSourceCell rejected{};
            const auto rejected_result =
                ERFFire::try_make_erf_fire_atmospheric_source_cell(
                    feedback,
                    Real(1),
                    Real(0),
                    p_0,
                    Real(2),
                    rejected);

            *raw_weight = raw;
            *cell = output;
            *raw_status = static_cast<int>(weight_result);
            *cell_status = static_cast<int>(cell_result);
            *invalid_status = static_cast<int>(rejected_result);
        });

    return {
        device_raw_weight.dataValue(),
        device_cell.dataValue(),
        device_raw_status.dataValue(),
        device_cell_status.dataValue(),
        device_invalid_status.dataValue()};
}

} // namespace

TEST(FireAtmosphericSource, DeviceSafeScalarApiMatchesHost)
{
    const ERFFire::FireSurfaceFeedbackCell feedback{
        Real(0), Real(200), Real(3)};
    const auto host =
        ERFFire::make_erf_fire_atmospheric_source_column(
            feedback,
            std::vector<Real>{Real(0), Real(10)},
            std::vector<Real>{Real(60)},
            std::vector<Real>{p_0},
            Real(2),
            ERFFireAtmosphericSourceOptions{Real(50)});
    ASSERT_EQ(host.size(), std::size_t(1));

    const DeviceAtmosphericSourceProbe actual =
        run_device_atmospheric_source_probe();

    EXPECT_EQ(
        actual.raw_status,
        static_cast<int>(
            ERFFire::ERFFireAtmosphericSourceStatus::success));
    EXPECT_EQ(
        actual.cell_status,
        static_cast<int>(
            ERFFire::ERFFireAtmosphericSourceStatus::success));
    EXPECT_EQ(
        actual.invalid_status,
        static_cast<int>(
            ERFFire::ERFFireAtmosphericSourceStatus::invalid_argument));

    const Real expected_raw =
        -std::expm1(-Real(10) / Real(50));
    EXPECT_NEAR(actual.raw_weight, expected_raw, Real(2.0e-14));
    EXPECT_NEAR(
        actual.cell.rhotheta_tendency_kg_K_m3_s,
        host[0].rhotheta_tendency_kg_K_m3_s,
        Real(2.0e-14));
    EXPECT_NEAR(
        actual.cell.rhoqv_tendency_kg_m3_s,
        host[0].rhoqv_tendency_kg_m3_s,
        Real(2.0e-14));
}
#endif

TEST(FireAtmosphericSource, OneLayerMatchesIndependentNativeUnitOracle)
{
    const FireCartesianRasterGeometry2D geometry{
        1, 1,
        Real(0.0), Real(0.0),
        Real(2.0), Real(3.0)};

    const Real dt_s = Real(4.0);
    const auto feedback =
        stationary_feedback(geometry, dt_s);
    const auto release = feedback.cell(0, 0);

    const std::vector<Real> faces{
        Real(0.0), Real(5.0)};
    const std::vector<Real> pressure{
        p_0};

    const auto source =
        ERFFire::make_erf_fire_atmospheric_source_field(
            feedback,
            faces,
            pressure,
            dt_s);

    const Real volume_m3 =
        Real(2.0) * Real(3.0) * Real(5.0);

    EXPECT_NEAR(
        source.cell(0, 0, 0)
            .rhotheta_tendency_kg_K_m3_s,
        release.sensible_energy_j
            / (volume_m3 * dt_s * Cp_d),
        Real(2.0e-14));
    EXPECT_NEAR(
        source.cell(0, 0, 0)
            .rhoqv_tendency_kg_m3_s,
        release.water_released_kg
            / (volume_m3 * dt_s),
        Real(2.0e-14));
}

TEST(FireAtmosphericSource, ExponentialWeightsMatchAnalyticLayerIntegrals)
{
    const FireCartesianRasterGeometry2D geometry{
        1, 1,
        Real(0.0), Real(0.0),
        Real(1.0), Real(1.0)};

    const Real dt_s = Real(2.0);
    const auto feedback =
        stationary_feedback(geometry, dt_s);
    const auto release = feedback.cell(0, 0);

    const std::vector<Real> faces{
        Real(0.0), Real(10.0), Real(30.0)};
    const std::vector<Real> pressure{
        p_0, p_0};
    constexpr Real H = Real(50.0);

    const auto source =
        ERFFire::make_erf_fire_atmospheric_source_field(
            feedback,
            faces,
            pressure,
            dt_s,
            ERFFireAtmosphericSourceOptions{H});

    const Real normalization =
        Real(1.0) - std::exp(-Real(30.0) / H);
    const Real w0 =
        (Real(1.0) - std::exp(-Real(10.0) / H))
        / normalization;
    const Real w1 =
        (std::exp(-Real(10.0) / H)
         - std::exp(-Real(30.0) / H))
        / normalization;

    const Real recovered_w0 =
        source.cell(0, 0, 0)
            .rhoqv_tendency_kg_m3_s
        * Real(10.0) * dt_s
        / release.water_released_kg;
    const Real recovered_w1 =
        source.cell(0, 0, 1)
            .rhoqv_tendency_kg_m3_s
        * Real(20.0) * dt_s
        / release.water_released_kg;

    EXPECT_NEAR(recovered_w0, w0, Real(2.0e-14));
    EXPECT_NEAR(recovered_w1, w1, Real(2.0e-14));
    EXPECT_NEAR(
        recovered_w0 + recovered_w1,
        Real(1.0),
        Real(2.0e-14));
}

TEST(FireAtmosphericSource, RecoversReleasedEnergyWithVaryingPressure)
{
    const FireCartesianRasterGeometry2D geometry{
        1, 1,
        Real(0.0), Real(0.0),
        Real(2.0), Real(2.0)};

    const Real dt_s = Real(3.0);
    const auto feedback =
        stationary_feedback(geometry, dt_s);
    const auto release = feedback.cell(0, 0);

    const std::vector<Real> faces{
        Real(0.0), Real(4.0), Real(12.0)};
    const std::vector<Real> pressure{
        p_0,
        Real(0.8) * p_0};

    const auto source =
        ERFFire::make_erf_fire_atmospheric_source_field(
            feedback,
            faces,
            pressure,
            dt_s);

    Real recovered_energy_j = Real(0.0);

    for (std::size_t k = 0; k < 2; ++k) {
        const Real dz = faces[k + 1] - faces[k];
        const Real volume =
            geometry.dx_m * geometry.dy_m * dz;
        const Real exner =
            exner_from_pressure(pressure[k]);

        recovered_energy_j +=
            source.cell(0, 0, k)
                .rhotheta_tendency_kg_K_m3_s
            * Cp_d
            * exner
            * volume
            * dt_s;
    }

    EXPECT_NEAR(
        recovered_energy_j,
        release.sensible_energy_j,
        Real(2.0e-10));
}

TEST(FireAtmosphericSource, RecoversReleasedWaterExactlyAcrossColumn)
{
    const FireCartesianRasterGeometry2D geometry{
        1, 1,
        Real(0.0), Real(0.0),
        Real(3.0), Real(2.0)};

    const Real dt_s = Real(5.0);
    const auto feedback =
        stationary_feedback(geometry, dt_s);
    const auto release = feedback.cell(0, 0);

    const std::vector<Real> faces{
        Real(0.0), Real(1.0), Real(3.0), Real(9.0)};
    const std::vector<Real> pressure{
        p_0, Real(0.95) * p_0, Real(0.85) * p_0};

    const auto source =
        ERFFire::make_erf_fire_atmospheric_source_field(
            feedback,
            faces,
            pressure,
            dt_s);

    Real recovered_water_kg = Real(0.0);

    for (std::size_t k = 0; k < 3; ++k) {
        const Real dz = faces[k + 1] - faces[k];
        const Real volume =
            geometry.dx_m * geometry.dy_m * dz;

        recovered_water_kg +=
            source.cell(0, 0, k)
                .rhoqv_tendency_kg_m3_s
            * volume
            * dt_s;
    }

    EXPECT_NEAR(
        recovered_water_kg,
        release.water_released_kg,
        Real(2.0e-14));
}

TEST(FireAtmosphericSource, PressureChangesRhoThetaButNotEnergyOrWaterSource)
{
    const FireCartesianRasterGeometry2D geometry{
        1, 1,
        Real(0.0), Real(0.0),
        Real(1.0), Real(1.0)};

    const Real dt_s = Real(2.0);
    const auto feedback =
        stationary_feedback(geometry, dt_s);

    const std::vector<Real> faces{
        Real(0.0), Real(2.0)};

    const auto source_p0 =
        ERFFire::make_erf_fire_atmospheric_source_field(
            feedback,
            faces,
            std::vector<Real>{p_0},
            dt_s);
    const auto source_low_p =
        ERFFire::make_erf_fire_atmospheric_source_field(
            feedback,
            faces,
            std::vector<Real>{Real(0.5) * p_0},
            dt_s);

    EXPECT_GT(
        source_low_p.cell(0, 0, 0)
            .rhotheta_tendency_kg_K_m3_s,
        source_p0.cell(0, 0, 0)
            .rhotheta_tendency_kg_K_m3_s);
    EXPECT_EQ(
        source_low_p.cell(0, 0, 0)
            .rhoqv_tendency_kg_m3_s,
        source_p0.cell(0, 0, 0)
            .rhoqv_tendency_kg_m3_s);
}

TEST(FireAtmosphericSource, ZeroFeedbackProducesExactZeroTendencies)
{
    const FireCartesianRasterGeometry2D geometry{
        1, 1,
        Real(0.0), Real(0.0),
        Real(1.0), Real(1.0)};

    const auto p = parameters();
    const FirePerimeter full =
        rectangle(
            Real(-1.0), Real(1.0));
    const auto burned =
        burned_from(geometry, full);

    FireCombustionRaster combustion(
        geometry,
        p,
        FireCombustionRasterOptions{8});
    (void)combustion.initialize_from_burned_fraction(
        burned);

    const auto feedback =
        ERFFire::make_fire_surface_feedback_increment(
            combustion,
            combustion);

    const auto source =
        ERFFire::make_erf_fire_atmospheric_source_field(
            feedback,
            std::vector<Real>{Real(0.0), Real(2.0)},
            std::vector<Real>{p_0},
            Real(1.0));

    EXPECT_EQ(
        source.cell(0, 0, 0)
            .rhotheta_tendency_kg_K_m3_s,
        Real(0.0));
    EXPECT_EQ(
        source.cell(0, 0, 0)
            .rhoqv_tendency_kg_m3_s,
        Real(0.0));
}

TEST(FireAtmosphericSource, HorizontalCellsRemainLocal)
{
    const FireCartesianRasterGeometry2D geometry{
        2, 1,
        Real(0.0), Real(0.0),
        Real(1.0), Real(1.0)};

    const auto p = parameters();
    const FirePerimeter left_only =
        rectangle(
            Real(-1.0), Real(1.0));
    const auto burned =
        burned_from(geometry, left_only);

    FireCombustionRaster before(
        geometry,
        p,
        FireCombustionRasterOptions{8});
    (void)before.initialize_from_burned_fraction(
        burned);

    FireCombustionRaster after = before;
    (void)after.advance_from_linear_sweep(
        left_only,
        left_only,
        burned,
        burned,
        Real(4.0));

    const auto feedback =
        ERFFire::make_fire_surface_feedback_increment(
            before,
            after);

    ASSERT_GT(
        feedback.cell(0, 0).sensible_energy_j,
        Real(0.0));
    ASSERT_EQ(
        feedback.cell(1, 0).sensible_energy_j,
        Real(0.0));

    const auto source =
        ERFFire::make_erf_fire_atmospheric_source_field(
            feedback,
            std::vector<Real>{Real(0.0), Real(2.0)},
            std::vector<Real>{p_0, p_0},
            Real(4.0));

    EXPECT_GT(
        source.cell(0, 0, 0)
            .rhotheta_tendency_kg_K_m3_s,
        Real(0.0));
    EXPECT_EQ(
        source.cell(1, 0, 0)
            .rhotheta_tendency_kg_K_m3_s,
        Real(0.0));
    EXPECT_EQ(
        source.cell(1, 0, 0)
            .rhoqv_tendency_kg_m3_s,
        Real(0.0));
}

TEST(FireAtmosphericSource, RejectsInvalidProjectionInputs)
{
    const FireCartesianRasterGeometry2D geometry{
        1, 1,
        Real(0.0), Real(0.0),
        Real(1.0), Real(1.0)};
    const auto feedback =
        stationary_feedback(geometry, Real(1.0));

    EXPECT_THROW(
        (void)ERFFire::make_erf_fire_atmospheric_source_field(
            feedback,
            std::vector<Real>{Real(0.0), Real(1.0)},
            std::vector<Real>{p_0},
            Real(0.0)),
        std::invalid_argument);

    EXPECT_THROW(
        (void)ERFFire::make_erf_fire_atmospheric_source_field(
            feedback,
            std::vector<Real>{Real(1.0), Real(2.0)},
            std::vector<Real>{p_0},
            Real(1.0)),
        std::invalid_argument);

    EXPECT_THROW(
        (void)ERFFire::make_erf_fire_atmospheric_source_field(
            feedback,
            std::vector<Real>{Real(0.0), Real(1.0)},
            std::vector<Real>{},
            Real(1.0)),
        std::invalid_argument);

    EXPECT_THROW(
        (void)ERFFire::make_erf_fire_atmospheric_source_field(
            feedback,
            std::vector<Real>{Real(0.0), Real(1.0)},
            std::vector<Real>{Real(0.0)},
            Real(1.0)),
        std::invalid_argument);

    EXPECT_THROW(
        (void)ERFFire::make_erf_fire_atmospheric_source_field(
            feedback,
            std::vector<Real>{Real(0.0), Real(1.0)},
            std::vector<Real>{p_0},
            Real(1.0),
            ERFFireAtmosphericSourceOptions{
                Real(0.0)}),
        std::invalid_argument);
}
