#include <gtest/gtest.h>

#include <ERF_FireRuntimeInit.H>
#include <ERF_FireRuntimeOptions.H>
#include <ERF_FireSpreadRuntime.H>

#include <AMReX_Array.H>
#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_RealBox.H>

#include <stdexcept>

namespace
{

using amrex::Array;
using amrex::Box;
using amrex::Geometry;
using amrex::IntVect;
using amrex::Real;
using amrex::RealBox;
using ERFFire::ERFFireCouplingMode;
using ERFFire::ERFFireRuntimeOptions;
using ERFFire::make_erf_fire_spread_config;

Geometry
make_geometry()
{
    const Box domain{
        IntVect(0, 0, 0),
        IntVect(3, 2, 1)};

    const RealBox real_box{
        Real(100.0),
        Real(50.0),
        Real(0.0),
        Real(108.0),
        Real(62.0),
        Real(20.0)};

    const Array<int, AMREX_SPACEDIM> periodic{{0, 0, 0}};

    return Geometry{
        domain,
        real_box,
        amrex::CoordSys::cartesian,
        periodic};
}

ERFFireRuntimeOptions
make_options()
{
    ERFFireRuntimeOptions options;
    options.enabled = true;
    options.dead_fuel_moisture_fraction = Real(0.08);
    options.remesh_min_edge_length_m = Real(0.05);
    options.remesh_max_edge_length_m = Real(0.25);
    options.remesh_max_chord_error_m = Real(0.002);
    return options;
}

TEST(FireRuntimeInit, DefaultFireGridMatchesLevel0)
{
    const Geometry geometry = make_geometry();
    const auto config =
        make_erf_fire_spread_config(
            make_options(),
            geometry);

    EXPECT_EQ(config.raster_geometry.nx, 4u);
    EXPECT_EQ(config.raster_geometry.ny, 3u);
    EXPECT_EQ(config.raster_geometry.xlo_m, Real(100.0));
    EXPECT_EQ(config.raster_geometry.ylo_m, Real(50.0));
    EXPECT_EQ(config.raster_geometry.dx_m, Real(2.0));
    EXPECT_EQ(config.raster_geometry.dy_m, Real(4.0));
}

TEST(FireRuntimeInit, FinerFireGridUsesSamePhysicalDomain)
{
    const Geometry geometry = make_geometry();
    auto options = make_options();
    options.n_cell_x = 8;
    options.n_cell_y = 6;

    const auto config =
        make_erf_fire_spread_config(
            options,
            geometry);

    EXPECT_EQ(config.raster_geometry.nx, 8u);
    EXPECT_EQ(config.raster_geometry.ny, 6u);
    EXPECT_EQ(config.raster_geometry.xlo_m, Real(100.0));
    EXPECT_EQ(config.raster_geometry.ylo_m, Real(50.0));
    EXPECT_EQ(config.raster_geometry.dx_m, Real(1.0));
    EXPECT_EQ(config.raster_geometry.dy_m, Real(2.0));

    EXPECT_EQ(
        config.raster_geometry.xlo_m
            + Real(config.raster_geometry.nx)
                * config.raster_geometry.dx_m,
        Real(108.0));
    EXPECT_EQ(
        config.raster_geometry.ylo_m
            + Real(config.raster_geometry.ny)
                * config.raster_geometry.dy_m,
        Real(62.0));
}

TEST(FireRuntimeInit, CoarserFireGridUsesSamePhysicalDomain)
{
    const Geometry geometry = make_geometry();
    auto options = make_options();
    options.n_cell_x = 2;
    options.n_cell_y = 1;

    const auto config =
        make_erf_fire_spread_config(
            options,
            geometry);

    EXPECT_EQ(config.raster_geometry.nx, 2u);
    EXPECT_EQ(config.raster_geometry.ny, 1u);
    EXPECT_EQ(config.raster_geometry.dx_m, Real(4.0));
    EXPECT_EQ(config.raster_geometry.dy_m, Real(12.0));
}

TEST(FireRuntimeInit, RejectsNoncommensurateFireGrid)
{
    const Geometry geometry = make_geometry();
    auto options = make_options();
    options.n_cell_x = 6;
    options.n_cell_y = 6;

    EXPECT_THROW(
        (void)make_erf_fire_spread_config(
            options,
            geometry),
        std::invalid_argument);
}

TEST(FireRuntimeInit, RejectsIncompleteFireGridCounts)
{
    const Geometry geometry = make_geometry();
    auto options = make_options();
    options.n_cell_x = 8;
    options.n_cell_y = 0;

    EXPECT_THROW(
        (void)make_erf_fire_spread_config(
            options,
            geometry),
        std::invalid_argument);
}

TEST(FireRuntimeInit, TwoWayIndependentGridAllowsUniformlyNestedResolutions)
{
    const Geometry geometry = make_geometry();
    auto options = make_options();
    options.coupling_mode =
        ERFFireCouplingMode::TwoWay;

    options.n_cell_x = 8;
    options.n_cell_y = 6;
    EXPECT_NO_THROW(
        (void)make_erf_fire_spread_config(
            options,
            geometry));

    options.n_cell_x = 2;
    options.n_cell_y = 1;
    EXPECT_NO_THROW(
        (void)make_erf_fire_spread_config(
            options,
            geometry));

    options.n_cell_x = 8;
    options.n_cell_y = 1;
    EXPECT_THROW(
        (void)make_erf_fire_spread_config(
            options,
            geometry),
        std::invalid_argument);
}

}
