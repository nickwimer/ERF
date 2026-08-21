#ifndef ERF_TERRAIN_SOURCE_H_
#define ERF_TERRAIN_SOURCE_H_

#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_REAL.H>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

class ERFTerrainSource
{
public:
    ERFTerrainSource(
        std::vector<amrex::Real> x_m,
        std::vector<amrex::Real> y_m,
        std::vector<amrex::Real> elevation_m);

    [[nodiscard]] static ERFTerrainSource
    read_regular_text_file(
        const std::string& filename);

    [[nodiscard]] std::size_t nx() const noexcept
    {
        return x_m_.size();
    }

    [[nodiscard]] std::size_t ny() const noexcept
    {
        return y_m_.size();
    }

    [[nodiscard]] amrex::Real x_min_m() const noexcept
    {
        return x_m_.front();
    }

    [[nodiscard]] amrex::Real x_max_m() const noexcept
    {
        return x_m_.back();
    }

    [[nodiscard]] amrex::Real y_min_m() const noexcept
    {
        return y_m_.front();
    }

    [[nodiscard]] amrex::Real y_max_m() const noexcept
    {
        return y_m_.back();
    }

    [[nodiscard]] std::optional<std::uint64_t>
    source_fingerprint_fnv1a64() const noexcept
    {
        return source_fingerprint_fnv1a64_;
    }

    [[nodiscard]] bool
    contains(
        amrex::Real x_m,
        amrex::Real y_m) const noexcept;

    [[nodiscard]] amrex::Real
    sample(
        amrex::Real x_m,
        amrex::Real y_m) const;

    /**
     * Fill an ERF atmospheric nodal terrain surface.
     *
     * Atmospheric compatibility uses constant edge extension when ERF nodes
     * lie outside the retained source footprint. Direct sample() remains
     * strict so Fire terrain cannot silently extrapolate beyond its DEM.
     */
    void
    fill_nodal_surface(
        const amrex::Geometry& geometry,
        amrex::FArrayBox& terrain_fab) const;

private:
    std::vector<amrex::Real> x_m_;
    std::vector<amrex::Real> y_m_;
    std::vector<amrex::Real> elevation_m_;
    std::optional<std::uint64_t> source_fingerprint_fnv1a64_;
};

#endif
