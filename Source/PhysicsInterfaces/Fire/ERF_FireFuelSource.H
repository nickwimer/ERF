#ifndef ERF_FIRE_FUEL_SOURCE_H_
#define ERF_FIRE_FUEL_SOURCE_H_

#include <ERF_FireFuelRaster.H>

#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Print.H>

#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ERFFire
{
namespace detail
{

inline void
expect_fire_fuel_source_token(
    std::istream& stream,
    const char* expected,
    const std::string& filename)
{
    std::string token;
    if (!(stream >> token) || token != expected) {
        throw std::runtime_error(
            "invalid ERF-Fire fuel raster " + filename
            + "; expected token " + expected);
    }
}

template <typename T>
T
read_fire_fuel_source_value(
    std::istream& stream,
    const char* field,
    const std::string& filename)
{
    T value{};
    if (!(stream >> value)) {
        throw std::runtime_error(
            "invalid ERF-Fire fuel raster " + filename
            + "; unable to read " + field);
    }
    return value;
}

inline bool
same_fire_fuel_source_geometry(
    const FireCartesianRasterGeometry2D& lhs,
    const FireCartesianRasterGeometry2D& rhs) noexcept
{
    return lhs.nx == rhs.nx
        && lhs.ny == rhs.ny
        && lhs.xlo_m == rhs.xlo_m
        && lhs.ylo_m == rhs.ylo_m
        && lhs.dx_m == rhs.dx_m
        && lhs.dy_m == rhs.dy_m;
}

inline FireFuelRasterState
read_fire_aligned_fuel_raster_text_state_on_io_rank(
    const std::string& filename,
    const FireCartesianRasterGeometry2D& expected_geometry)
{
    std::ifstream stream(filename);
    if (!stream.good()) {
        throw std::runtime_error(
            "unable to open ERF-Fire fuel raster " + filename);
    }

    expect_fire_fuel_source_token(
        stream,
        "ERF_FIRE_FUEL_RASTER",
        filename);
    const int version =
        read_fire_fuel_source_value<int>(
            stream,
            "format version",
            filename);
    if (version != 1) {
        throw std::runtime_error(
            "unsupported ERF-Fire fuel raster format version in "
            + filename);
    }

    expect_fire_fuel_source_token(stream, "nx", filename);
    const unsigned long long nx_wire =
        read_fire_fuel_source_value<unsigned long long>(
            stream, "nx", filename);
    expect_fire_fuel_source_token(stream, "ny", filename);
    const unsigned long long ny_wire =
        read_fire_fuel_source_value<unsigned long long>(
            stream, "ny", filename);

    if (nx_wire == 0ULL
        || ny_wire == 0ULL
        || nx_wire
            > static_cast<unsigned long long>(
                std::numeric_limits<std::size_t>::max())
        || ny_wire
            > static_cast<unsigned long long>(
                std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error(
            "invalid ERF-Fire fuel raster dimensions in " + filename);
    }

    FireCartesianRasterGeometry2D file_geometry;
    file_geometry.nx = static_cast<std::size_t>(nx_wire);
    file_geometry.ny = static_cast<std::size_t>(ny_wire);

    expect_fire_fuel_source_token(stream, "xlo_m", filename);
    file_geometry.xlo_m =
        read_fire_fuel_source_value<amrex::Real>(
            stream, "xlo_m", filename);
    expect_fire_fuel_source_token(stream, "ylo_m", filename);
    file_geometry.ylo_m =
        read_fire_fuel_source_value<amrex::Real>(
            stream, "ylo_m", filename);
    expect_fire_fuel_source_token(stream, "dx_m", filename);
    file_geometry.dx_m =
        read_fire_fuel_source_value<amrex::Real>(
            stream, "dx_m", filename);
    expect_fire_fuel_source_token(stream, "dy_m", filename);
    file_geometry.dy_m =
        read_fire_fuel_source_value<amrex::Real>(
            stream, "dy_m", filename);

    (void)validate_fire_cartesian_raster_geometry(file_geometry);
    if (!same_fire_fuel_source_geometry(
            file_geometry,
            expected_geometry)) {
        throw std::runtime_error(
            "ERF-Fire fuel raster geometry does not exactly match the configured Fire grid: "
            + filename);
    }

    expect_fire_fuel_source_token(stream, "components", filename);
    const int components =
        read_fire_fuel_source_value<int>(
            stream, "components", filename);
    if (components != FireFuelRasterComponents::component_count) {
        throw std::runtime_error(
            "ERF-Fire fuel raster must declare exactly seven material components: "
            + filename);
    }

    expect_fire_fuel_source_token(stream, "cells", filename);

    const std::size_t cell_count =
        validate_fire_cartesian_raster_geometry(file_geometry);
    FireFuelRasterState state;
    state.cells.reserve(cell_count);

    constexpr int valid_mask =
        (1 << FireFuelMoisture::component_count) - 1;

    for (std::size_t flat_index = 0;
         flat_index < cell_count;
         ++flat_index) {
        const int raw_model =
            read_fire_fuel_source_value<int>(
                stream, "cell model_id", filename);
        const int moisture_mask =
            read_fire_fuel_source_value<int>(
                stream, "cell moisture_presence_mask", filename);

        std::array<amrex::Real, FireFuelMoisture::component_count>
            moisture_values{};
        for (int component = 0;
             component < FireFuelMoisture::component_count;
             ++component) {
            moisture_values[static_cast<std::size_t>(component)] =
                read_fire_fuel_source_value<amrex::Real>(
                    stream, "cell moisture value", filename);
        }

        if (raw_model != static_cast<int>(FireFuelModelId::NonBurnable)
            && raw_model != static_cast<int>(FireFuelModelId::FM1)) {
            throw std::runtime_error(
                "ERF-Fire fuel raster contains an unsupported fuel-model id in "
                + filename);
        }
        if (moisture_mask < 0 || moisture_mask > valid_mask) {
            throw std::runtime_error(
                "ERF-Fire fuel raster contains an invalid moisture-presence mask in "
                + filename);
        }

        FireFuelRasterCell cell;
        cell.model_id = static_cast<FireFuelModelId>(raw_model);

        for (int component = 0;
             component < FireFuelMoisture::component_count;
             ++component) {
            const amrex::Real value =
                moisture_values[static_cast<std::size_t>(component)];
            if (!std::isfinite(value) || value < amrex::Real(0)) {
                throw std::runtime_error(
                    "ERF-Fire fuel raster moisture values must be finite and nonnegative: "
                    + filename);
            }

            const bool present =
                (moisture_mask & (1 << component)) != 0;
            if (present) {
                cell.moisture.set(
                    fire_fuel_moisture_class_from_index(component),
                    value);
            } else if (value != amrex::Real(0)) {
                throw std::runtime_error(
                    "ERF-Fire fuel raster missing-moisture components must be encoded as zero: "
                    + filename);
            }
        }

        if (!fire_fuel_raster_cell_valid(cell)) {
            throw std::runtime_error(
                "ERF-Fire fuel raster contains invalid resolved material: "
                + filename);
        }

        state.cells.push_back(std::move(cell));
    }

    expect_fire_fuel_source_token(
        stream,
        "END_ERF_FIRE_FUEL_RASTER",
        filename);

    std::string trailing;
    if (stream >> trailing) {
        throw std::runtime_error(
            "ERF-Fire fuel raster contains trailing data after its end marker: "
            + filename);
    }

    return state;
}

} // namespace detail

/**
 * Collectively load one aligned, resolved Fire material raster from text.
 *
 * File format v1 is intentionally strict and provider-neutral:
 *
 *   ERF_FIRE_FUEL_RASTER 1
 *   nx <integer>
 *   ny <integer>
 *   xlo_m <real>
 *   ylo_m <real>
 *   dx_m <real>
 *   dy_m <real>
 *   components 7
 *   cells
 *   <model_id> <presence_mask> <dead1h> <dead10h> <dead100h> <live_herb> <live_woody>
 *   ... exactly nx*ny canonical row-major cells, j-major then i ...
 *   END_ERF_FIRE_FUEL_RASTER
 *
 * Model IDs and moisture masks use FireFuelRaster's stable encoding. Missing
 * moisture components have a cleared mask bit and must carry numerical zero;
 * this prevents multiple text encodings from resolving to the same material.
 * Geometry must exactly match the configured Fire raster. Only IOProcessor()
 * reads the file; the validated canonical material is distributed directly to
 * the active FireSurfaceLayout without retaining a full-domain mirror.
 */
[[nodiscard]] inline FireFuelRaster
read_erf_fire_aligned_fuel_raster_text_file(
    const std::string& filename,
    const FireCartesianRasterGeometry2D& expected_geometry)
{
    if (filename.empty()) {
        throw std::invalid_argument(
            "ERF-Fire fuel raster filename must not be empty");
    }

    FireFuelRasterState state;
    int read_failed = 0;

    if (amrex::ParallelDescriptor::IOProcessor()) {
        try {
            state =
                detail::read_fire_aligned_fuel_raster_text_state_on_io_rank(
                    filename,
                    expected_geometry);
        } catch (const std::exception& error) {
            amrex::Print()
                << "ERF-Fire fuel raster read error: "
                << error.what() << '\n';
            read_failed = 1;
        }
    }

    amrex::ParallelDescriptor::Bcast(
        &read_failed,
        1,
        amrex::ParallelDescriptor::IOProcessorNumber());

    if (read_failed != 0) {
        throw std::runtime_error(
            "IO rank failed to read or validate the ERF-Fire fuel raster");
    }

    return FireFuelRaster::collective_from_io_rank_state(
        expected_geometry,
        state);
}

} // namespace ERFFire

#endif // ERF_FIRE_FUEL_SOURCE_H_
