#ifndef ERF_FIRE_ATMOSPHERIC_SOURCE_H_
#define ERF_FIRE_ATMOSPHERIC_SOURCE_H_

#include <ERF_FireRasterGeometry.H>
#include <ERF_FireSurfaceFeedback.H>

#include <AMReX_REAL.H>

#include <cstddef>
#include <vector>

namespace ERFFire
{

/**
 * Vertical deposition options for the first ERF Fire-feedback projection.
 *
 * The normalized exponential shape follows the ground-fire heat/moisture
 * extinction convention used by WRF-Fire/SFIRE. The finite ERF column is
 * explicitly renormalized so all step-integrated Fire release is deposited
 * into the represented atmospheric column.
 */
struct ERFFireAtmosphericSourceOptions
{
    amrex::Real extinction_depth_m{amrex::Real(50.0)};
};

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
