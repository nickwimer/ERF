#include "ERF_FireAtmosphericSource.H"

#include <ERF_Constants.H>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ERFFire
{
namespace
{

void
require(bool condition, const char* message)
{
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

std::size_t
checked_cell_count(
    const FireCartesianRasterGeometry2D& horizontal_geometry,
    std::size_t nz)
{
    const std::size_t horizontal_count =
        detail::validate_fire_cartesian_raster_geometry(
            horizontal_geometry);

    require(
        nz > 0,
        "ERF Fire atmospheric source requires at least one vertical cell");

    if (horizontal_count
        > std::numeric_limits<std::size_t>::max() / nz) {
        throw std::overflow_error(
            "ERF Fire atmospheric source cell count overflows size_t");
    }

    return horizontal_count * nz;
}

void
validate_vertical_faces(
    const std::vector<amrex::Real>& faces)
{
    require(
        faces.size() >= 2,
        "ERF Fire atmospheric source requires at least two vertical faces");

    for (amrex::Real value : faces) {
        require(
            std::isfinite(value),
            "ERF Fire atmospheric source vertical faces must be finite");
    }

    const amrex::Real ground_tolerance =
        amrex::Real(64)
        * std::numeric_limits<amrex::Real>::epsilon();
    require(
        std::abs(faces.front()) <= ground_tolerance,
        "ERF Fire atmospheric source first AGL face must be zero");

    for (std::size_t k = 0; k + 1 < faces.size(); ++k) {
        require(
            faces[k + 1] > faces[k],
            "ERF Fire atmospheric source vertical faces must increase strictly");
    }
}

std::vector<amrex::Real>
normalized_exponential_layer_weights(
    const std::vector<amrex::Real>& faces,
    amrex::Real extinction_depth_m)
{
    require(
        std::isfinite(extinction_depth_m)
            && extinction_depth_m > amrex::Real(0),
        "ERF Fire atmospheric extinction depth must be finite and positive");

    std::vector<amrex::Real> weights(
        faces.size() - 1,
        amrex::Real(0));

    amrex::Real total = amrex::Real(0);

    for (std::size_t k = 0; k < weights.size(); ++k) {
        const amrex::Real zlo = faces[k];
        const amrex::Real dz = faces[k + 1] - zlo;

        const amrex::Real lower =
            std::exp(-zlo / extinction_depth_m);
        const amrex::Real raw =
            lower
            * (-std::expm1(-dz / extinction_depth_m));

        if (!std::isfinite(raw)
            || !(raw > amrex::Real(0))) {
            throw std::overflow_error(
                "ERF Fire atmospheric deposition weight is not finite and positive");
        }

        weights[k] = raw;
        total += raw;
    }

    if (!std::isfinite(total)
        || !(total > amrex::Real(0))) {
        throw std::overflow_error(
            "ERF Fire atmospheric deposition normalization is invalid");
    }

    amrex::Real normalized_sum = amrex::Real(0);
    for (std::size_t k = 0; k < weights.size(); ++k) {
        if (k + 1 == weights.size()) {
            weights[k] =
                std::max(
                    amrex::Real(0),
                    amrex::Real(1) - normalized_sum);
        } else {
            weights[k] /= total;
            normalized_sum += weights[k];
        }
    }

    return weights;
}

} // namespace

ERFFireAtmosphericSourceField::ERFFireAtmosphericSourceField(
    FireCartesianRasterGeometry2D horizontal_geometry,
    std::vector<amrex::Real> vertical_face_height_agl_m)
    : horizontal_geometry_(horizontal_geometry),
      vertical_face_height_agl_m_(
          std::move(vertical_face_height_agl_m))
{
    validate_vertical_faces(
        vertical_face_height_agl_m_);

    const std::size_t count =
        checked_cell_count(
            horizontal_geometry_,
            nz());

    cells_.assign(
        count,
        ERFFireAtmosphericSourceCell{});
}

std::size_t
ERFFireAtmosphericSourceField::flat_index(
    std::size_t i,
    std::size_t j,
    std::size_t k) const
{
    const std::size_t horizontal_index =
        detail::fire_cartesian_raster_flat_index(
            horizontal_geometry_,
            i,
            j);

    if (k >= nz()) {
        throw std::out_of_range(
            "ERF Fire atmospheric source vertical index out of range");
    }

    const std::size_t horizontal_count =
        horizontal_geometry_.nx
        * horizontal_geometry_.ny;

    return k * horizontal_count + horizontal_index;
}

const ERFFireAtmosphericSourceCell&
ERFFireAtmosphericSourceField::cell(
    std::size_t i,
    std::size_t j,
    std::size_t k) const
{
    return cells_[flat_index(i, j, k)];
}


std::vector<ERFFireAtmosphericSourceCell>
make_erf_fire_atmospheric_source_column(
    const FireSurfaceFeedbackCell& surface_feedback,
    const std::vector<amrex::Real>& vertical_face_height_agl_m,
    const std::vector<amrex::Real>& physical_cell_volume_m3,
    const std::vector<amrex::Real>& pressure_pa,
    amrex::Real dt_s,
    ERFFireAtmosphericSourceOptions options)
{
    require(
        std::isfinite(dt_s) && dt_s > amrex::Real(0),
        "ERF Fire atmospheric source dt must be finite and positive");

    validate_vertical_faces(
        vertical_face_height_agl_m);

    const std::size_t nz =
        vertical_face_height_agl_m.size() - 1;

    require(
        physical_cell_volume_m3.size() == nz,
        "ERF Fire atmospheric source physical-volume column size mismatch");
    require(
        pressure_pa.size() == nz,
        "ERF Fire atmospheric source pressure column size mismatch");

    const std::vector<amrex::Real> vertical_weights =
        normalized_exponential_layer_weights(
            vertical_face_height_agl_m,
            options.extinction_depth_m);

    std::vector<ERFFireAtmosphericSourceCell> result(
        nz,
        ERFFireAtmosphericSourceCell{});

    for (std::size_t k = 0; k < nz; ++k) {
        const amrex::Real volume_m3 =
            physical_cell_volume_m3[k];
        require(
            std::isfinite(volume_m3)
                && volume_m3 > amrex::Real(0),
            "ERF Fire atmospheric source physical cell volume must be finite and positive");

        const amrex::Real pressure =
            pressure_pa[k];
        require(
            std::isfinite(pressure)
                && pressure > amrex::Real(0),
            "ERF Fire atmospheric source pressure must be finite and positive");

        const amrex::Real exner =
            std::pow(
                pressure * ip_0,
                RdoCp);

        if (!std::isfinite(exner)
            || !(exner > amrex::Real(0))) {
            throw std::overflow_error(
                "ERF Fire atmospheric source Exner function is invalid");
        }

        const amrex::Real inverse_volume_dt =
            amrex::Real(1)
            / (volume_m3 * dt_s);
        const amrex::Real weight =
            vertical_weights[k];

        ERFFireAtmosphericSourceCell& output =
            result[k];

        output.rhotheta_tendency_kg_K_m3_s =
            surface_feedback.sensible_energy_j
            * weight
            * inverse_volume_dt
            / (Cp_d * exner);

        output.rhoqv_tendency_kg_m3_s =
            surface_feedback.water_released_kg
            * weight
            * inverse_volume_dt;

        if (!std::isfinite(
                output.rhotheta_tendency_kg_K_m3_s)
            || output.rhotheta_tendency_kg_K_m3_s
                < amrex::Real(0)
            || !std::isfinite(
                output.rhoqv_tendency_kg_m3_s)
            || output.rhoqv_tendency_kg_m3_s
                < amrex::Real(0)) {
            throw std::overflow_error(
                "ERF Fire atmospheric source produced invalid tendency");
        }
    }

    return result;
}

ERFFireAtmosphericSourceField
make_erf_fire_atmospheric_source_field(
    const FireSurfaceFeedbackRaster& feedback,
    const std::vector<amrex::Real>& vertical_face_height_agl_m,
    const std::vector<amrex::Real>& pressure_pa,
    amrex::Real dt_s,
    ERFFireAtmosphericSourceOptions options)
{
    ERFFireAtmosphericSourceField result(
        feedback.geometry(),
        vertical_face_height_agl_m);

    require(
        pressure_pa.size() == result.cell_count(),
        "ERF Fire atmospheric source pressure field size mismatch");

    const amrex::Real horizontal_area_m2 =
        result.horizontal_geometry_.dx_m
        * result.horizontal_geometry_.dy_m;

    std::vector<amrex::Real> physical_cell_volume_m3(
        result.nz(),
        amrex::Real(0));

    for (std::size_t k = 0; k < result.nz(); ++k) {
        const amrex::Real dz_m =
            result.vertical_face_height_agl_m_[k + 1]
            - result.vertical_face_height_agl_m_[k];
        physical_cell_volume_m3[k] =
            horizontal_area_m2 * dz_m;
    }

    std::vector<amrex::Real> column_pressure_pa(
        result.nz(),
        amrex::Real(0));

    for (std::size_t j = 0;
         j < result.horizontal_geometry_.ny;
         ++j) {
        for (std::size_t i = 0;
             i < result.horizontal_geometry_.nx;
             ++i) {
            for (std::size_t k = 0; k < result.nz(); ++k) {
                column_pressure_pa[k] =
                    pressure_pa[
                        result.flat_index(i, j, k)];
            }

            const std::vector<ERFFireAtmosphericSourceCell>
                column =
                    make_erf_fire_atmospheric_source_column(
                        feedback.cell(i, j),
                        result.vertical_face_height_agl_m_,
                        physical_cell_volume_m3,
                        column_pressure_pa,
                        dt_s,
                        options);

            for (std::size_t k = 0; k < result.nz(); ++k) {
                result.cells_[
                    result.flat_index(i, j, k)] =
                    column[k];
            }
        }
    }

    return result;
}

} // namespace ERFFire
