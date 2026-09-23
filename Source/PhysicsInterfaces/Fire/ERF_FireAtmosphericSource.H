#ifndef ERF_FIRE_ATMOSPHERIC_SOURCE_H_
#define ERF_FIRE_ATMOSPHERIC_SOURCE_H_

#include <ERF_Constants.H>
#include <ERF_FireRasterGeometry.H>
#include <ERF_FireSurfaceFeedback.H>

#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace ERFFire
{

/**
 * Vertical deposition options for the first ERF Fire-feedback projection.
 *
 * The exponential shape follows the ground-fire heat/moisture extinction
 * convention used by WRF-Fire/SFIRE. ERF-Fire currently normalizes that shape
 * over the finite represented ERF column so all step-integrated Fire release
 * is deposited into the model atmosphere. Use
 * erf_fire_exponential_column_coverage() to diagnose the unrepresented
 * exponential tail and the associated normalization amplification.
 */
struct ERFFireAtmosphericSourceOptions
{
    amrex::Real extinction_depth_m{amrex::Real(50.0)};
};

struct ERFFireAtmosphericColumnCoverage
{
    // Fraction of the unnormalized exp(-z/H)/H profile represented between
    // ground and the supplied model-top AGL height.
    amrex::Real represented_fraction{};

    // Complement of represented_fraction. In the equivalent upward-flux
    // interpretation this is the flux fraction still present above model top.
    amrex::Real unrepresented_tail_fraction{};

    // Factor by which the current finite-column normalization amplifies the
    // represented profile so that the deposited weights sum to one.
    amrex::Real normalization_amplification{};
};

/**
 * Diagnose finite-column coverage of the exponential Fire feedback profile.
 *
 * This does not change the production deposition policy. It makes explicit the
 * distinction between the unnormalized model-family profile and ERF-Fire's
 * current conservative finite-column normalization.
 */
[[nodiscard]] inline ERFFireAtmosphericColumnCoverage
erf_fire_exponential_column_coverage(
    amrex::Real model_top_agl_m,
    amrex::Real extinction_depth_m)
{
    if (!std::isfinite(model_top_agl_m)
        || !(model_top_agl_m > amrex::Real(0))
        || !std::isfinite(extinction_depth_m)
        || !(extinction_depth_m > amrex::Real(0))) {
        throw std::invalid_argument(
            "ERF Fire exponential column diagnostic requires positive finite heights");
    }

    const amrex::Real tail =
        std::exp(-model_top_agl_m / extinction_depth_m);
    const amrex::Real represented =
        -std::expm1(-model_top_agl_m / extinction_depth_m);
    if (!std::isfinite(tail)
        || !std::isfinite(represented)
        || tail < amrex::Real(0)
        || !(represented > amrex::Real(0))) {
        throw std::overflow_error(
            "ERF Fire exponential column diagnostic is not representable");
    }

    const amrex::Real amplification =
        amrex::Real(1) / represented;
    if (!std::isfinite(amplification)
        || !(amplification >= amrex::Real(1))) {
        throw std::overflow_error(
            "ERF Fire exponential normalization amplification is invalid");
    }

    return {represented, tail, amplification};
}

struct ERFFireAtmosphericSourceCell
{
    /**
     * Native ERF conserved-variable tendency for rho_d * theta_d.
     * Units: kg K m^-3 s^-1.
     */
    amrex::Real rhotheta_tendency_kg_K_m3_s{};

    /**
     * Native ERF conserved-variable tendency for rho_d * q_v.
     * Units: kg water m^-3 s^-1.
     */
    amrex::Real rhoqv_tendency_kg_m3_s{};
};


enum class ERFFireAtmosphericSourceStatus : int
{
    success = 0,
    invalid_argument = 1,
    overflow_error = 2
};

namespace detail
{

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
amrex::Real
fire_atmospheric_raw_layer_weight_unchecked(
    amrex::Real zlo_m,
    amrex::Real zhi_m,
    amrex::Real extinction_depth_m) noexcept
{
    const amrex::Real dz_m = zhi_m - zlo_m;
    return std::exp(-zlo_m / extinction_depth_m)
        * (-std::expm1(-dz_m / extinction_depth_m));
}

} // namespace detail

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
ERFFireAtmosphericSourceStatus
try_erf_fire_atmospheric_source_raw_layer_weight(
    amrex::Real zlo_m,
    amrex::Real zhi_m,
    amrex::Real extinction_depth_m,
    amrex::Real& result) noexcept
{
    result = amrex::Real(0);
    if (!amrex::Math::isfinite(zlo_m)
        || !amrex::Math::isfinite(zhi_m)
        || !(zhi_m > zlo_m)
        || !amrex::Math::isfinite(extinction_depth_m)
        || !(extinction_depth_m > amrex::Real(0))) {
        return ERFFireAtmosphericSourceStatus::invalid_argument;
    }

    result = detail::fire_atmospheric_raw_layer_weight_unchecked(
        zlo_m,
        zhi_m,
        extinction_depth_m);
    if (!amrex::Math::isfinite(result)
        || !(result > amrex::Real(0))) {
        result = amrex::Real(0);
        return ERFFireAtmosphericSourceStatus::overflow_error;
    }
    return ERFFireAtmosphericSourceStatus::success;
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
ERFFireAtmosphericSourceStatus
try_make_erf_fire_atmospheric_source_cell(
    const FireSurfaceFeedbackCell& surface_feedback,
    amrex::Real normalized_weight,
    amrex::Real physical_cell_volume_m3,
    amrex::Real pressure_pa,
    amrex::Real dt_s,
    ERFFireAtmosphericSourceCell& result) noexcept
{
    result = {};
    if (!amrex::Math::isfinite(surface_feedback.sensible_energy_j)
        || surface_feedback.sensible_energy_j < amrex::Real(0)
        || !amrex::Math::isfinite(surface_feedback.water_released_kg)
        || surface_feedback.water_released_kg < amrex::Real(0)
        || !amrex::Math::isfinite(normalized_weight)
        || normalized_weight < amrex::Real(0)
        || !amrex::Math::isfinite(physical_cell_volume_m3)
        || !(physical_cell_volume_m3 > amrex::Real(0))
        || !amrex::Math::isfinite(pressure_pa)
        || !(pressure_pa > amrex::Real(0))
        || !amrex::Math::isfinite(dt_s)
        || !(dt_s > amrex::Real(0))) {
        return ERFFireAtmosphericSourceStatus::invalid_argument;
    }

    const amrex::Real exner =
        std::pow(pressure_pa * ip_0, RdoCp);
    const amrex::Real volume_dt =
        physical_cell_volume_m3 * dt_s;
    if (!amrex::Math::isfinite(exner)
        || !(exner > amrex::Real(0))
        || !amrex::Math::isfinite(volume_dt)
        || !(volume_dt > amrex::Real(0))) {
        return ERFFireAtmosphericSourceStatus::overflow_error;
    }

    result.rhotheta_tendency_kg_K_m3_s =
        surface_feedback.sensible_energy_j
        * normalized_weight
        / (volume_dt * Cp_d * exner);
    result.rhoqv_tendency_kg_m3_s =
        surface_feedback.water_released_kg
        * normalized_weight
        / volume_dt;

    if (!amrex::Math::isfinite(
            result.rhotheta_tendency_kg_K_m3_s)
        || result.rhotheta_tendency_kg_K_m3_s < amrex::Real(0)
        || !amrex::Math::isfinite(
            result.rhoqv_tendency_kg_m3_s)
        || result.rhoqv_tendency_kg_m3_s < amrex::Real(0)) {
        result = {};
        return ERFFireAtmosphericSourceStatus::overflow_error;
    }

    return ERFFireAtmosphericSourceStatus::success;
}

/**
 * Project one horizontal Fire feedback cell through one atmospheric column.
 *
 * vertical_face_height_agl_m contains nz+1 local AGL faces.
 * physical_cell_volume_m3 and pressure_pa each contain nz entries for those
 * same cells. The exponential deposition shape is normalized over this finite
 * local column; the caller owns the geometry-to-volume mapping.
 *
 * This is the common thermodynamic projection used by both flat and
 * terrain-following ERF adapters. It does not mutate ERF state.
 */
[[nodiscard]] std::vector<ERFFireAtmosphericSourceCell>
make_erf_fire_atmospheric_source_column(
    const FireSurfaceFeedbackCell& surface_feedback,
    const std::vector<amrex::Real>& vertical_face_height_agl_m,
    const std::vector<amrex::Real>& physical_cell_volume_m3,
    const std::vector<amrex::Real>& pressure_pa,
    amrex::Real dt_s,
    ERFFireAtmosphericSourceOptions options = {});

/**
 * Three-dimensional, cell-centered Fire source field in native ERF tendency
 * units.
 *
 * Horizontal geometry must match the Fire feedback raster one-for-one.
 * vertical_face_height_agl_m has nz+1 common, horizontally flat AGL faces.
 * This owning field remains the flat ConstantDz/StretchedDz representation.
 * Terrain-varying level-0 columns use
 * make_erf_fire_atmospheric_source_column() through the terrain adapter.
 */
class ERFFireAtmosphericSourceField
{
public:
    ERFFireAtmosphericSourceField(
        FireCartesianRasterGeometry2D horizontal_geometry,
        std::vector<amrex::Real> vertical_face_height_agl_m);

    [[nodiscard]] const FireCartesianRasterGeometry2D&
    horizontal_geometry() const noexcept
    {
        return horizontal_geometry_;
    }

    [[nodiscard]] const std::vector<amrex::Real>&
    vertical_face_height_agl_m() const noexcept
    {
        return vertical_face_height_agl_m_;
    }

    [[nodiscard]] std::size_t nx() const noexcept
    {
        return horizontal_geometry_.nx;
    }

    [[nodiscard]] std::size_t ny() const noexcept
    {
        return horizontal_geometry_.ny;
    }

    [[nodiscard]] std::size_t nz() const noexcept
    {
        return vertical_face_height_agl_m_.size() - 1;
    }

    [[nodiscard]] std::size_t cell_count() const noexcept
    {
        return cells_.size();
    }

    [[nodiscard]] const ERFFireAtmosphericSourceCell&
    cell(std::size_t i, std::size_t j, std::size_t k) const;

private:
    friend ERFFireAtmosphericSourceField
    make_erf_fire_atmospheric_source_field(
        const FireSurfaceFeedbackRaster& feedback,
        const std::vector<amrex::Real>& vertical_face_height_agl_m,
        const std::vector<amrex::Real>& pressure_pa,
        amrex::Real dt_s,
        ERFFireAtmosphericSourceOptions options);

    [[nodiscard]] std::size_t flat_index(
        std::size_t i,
        std::size_t j,
        std::size_t k) const;

    FireCartesianRasterGeometry2D horizontal_geometry_{};
    std::vector<amrex::Real> vertical_face_height_agl_m_;
    std::vector<ERFFireAtmosphericSourceCell> cells_;
};

/**
 * Project one step-integrated Fire surface-feedback raster into native ERF
 * rho_d*theta_d and rho_d*q_v tendencies.
 *
 * pressure_pa is cell-centered total atmospheric pressure in row-major order
 *
 *   index = (k * ny + j) * nx + i.
 *
 * For each horizontal Fire cell, vertical deposition weights are the exact
 * layer integrals of exp(-z/H)/H over the supplied AGL faces, normalized over
 * the finite represented column. Thus sum_k weight_k = 1.
 *
 * Sensible heat uses ERF dry potential temperature:
 *
 *   Pi = (p/p0)^(R_d/Cp_d)
 *   S_(rho theta) = Qdot_volume / (Cp_d * Pi)
 *
 * while released water is a direct source of rho_d*q_v:
 *
 *   S_(rho qv) = Mdot_water / volume.
 *
 * No source is applied to dry-air density. This function does not mutate ERF
 * state and does not encode one-way/two-way policy.
 */
[[nodiscard]] ERFFireAtmosphericSourceField
make_erf_fire_atmospheric_source_field(
    const FireSurfaceFeedbackRaster& feedback,
    const std::vector<amrex::Real>& vertical_face_height_agl_m,
    const std::vector<amrex::Real>& pressure_pa,
    amrex::Real dt_s,
    ERFFireAtmosphericSourceOptions options = {});

} // namespace ERFFire

#endif
