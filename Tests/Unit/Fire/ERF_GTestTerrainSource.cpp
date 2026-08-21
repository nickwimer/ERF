#include <ERF_TerrainSource.H>
#include <ERF_FireLevel0Environment.H>

#include <AMReX_Arena.H>
#include <AMReX_Array.H>
#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>
#include <AMReX_IntVect.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_RealBox.H>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using amrex::Real;
using ERFFire::FireCartesianRasterGeometry2D;

Real
planar_height(
    Real x,
    Real y)
{
    return Real(10)
        + Real(0.5) * x
        - Real(0.25) * y;
}

Real
test_tolerance()
{
    return Real(128)
        * std::numeric_limits<Real>::epsilon();
}

} // namespace

TEST(
    ERFTerrainSource,
    ReadsRegularTextOrderingAndSamplesClosedDomain)
{
    const std::string filename =
        "ERF_TerrainSource_regular_test.txt";

    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream file(filename);
        ASSERT_TRUE(file.is_open());

        file << std::setprecision(17);
        file << 3 << '\n';
        file << 3 << '\n';

        for (const Real x
             : std::vector<Real>{
                   Real(0),
                   Real(2),
                   Real(4)}) {
            file << x << '\n';
        }

        for (const Real y
             : std::vector<Real>{
                   Real(-1),
                   Real(1),
                   Real(3)}) {
            file << y << '\n';
        }

        for (const Real x
             : std::vector<Real>{
                   Real(0),
                   Real(2),
                   Real(4)}) {
            for (const Real y
                 : std::vector<Real>{
                       Real(-1),
                       Real(1),
                       Real(3)}) {
                file
                    << planar_height(x, y)
                    << '\n';
            }
        }
    }

    amrex::ParallelDescriptor::Barrier();

    const ERFTerrainSource source =
        ERFTerrainSource::read_regular_text_file(
            filename);

    EXPECT_EQ(source.nx(), 3U);
    EXPECT_EQ(source.ny(), 3U);

    EXPECT_NEAR(
        source.sample(
            Real(1),
            Real(0)),
        planar_height(
            Real(1),
            Real(0)),
        test_tolerance());

    EXPECT_NEAR(
        source.sample(
            Real(4),
            Real(3)),
        planar_height(
            Real(4),
            Real(3)),
        test_tolerance());

    EXPECT_NEAR(
        source.sample(
            Real(0),
            Real(-1)),
        planar_height(
            Real(0),
            Real(-1)),
        test_tolerance());

    amrex::ParallelDescriptor::Barrier();

    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::remove(filename.c_str());
    }

    amrex::ParallelDescriptor::Barrier();
}

TEST(
    ERFTerrainSource,
    FileFingerprintBelongsToLoadedSourceState)
{
    const std::string filename =
        "ERF_TerrainSource_fingerprint_test.txt";

    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream file(filename);
        ASSERT_TRUE(file.is_open());

        file << "2\n";
        file << "2\n";
        file << "0\n";
        file << "1\n";
        file << "0\n";
        file << "1\n";
        file << "0\n";
        file << "0\n";
        file << "0\n";
        file << "0\n";
    }

    amrex::ParallelDescriptor::Barrier();

    const ERFTerrainSource source =
        ERFTerrainSource::read_regular_text_file(
            filename);

    const auto loaded_fingerprint =
        source.source_fingerprint_fnv1a64();

    ASSERT_TRUE(
        loaded_fingerprint.has_value());

    amrex::ParallelDescriptor::Barrier();

    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::ofstream file(
            filename,
            std::ios::out | std::ios::app);
        ASSERT_TRUE(file.is_open());
        file << "\n";
    }

    amrex::ParallelDescriptor::Barrier();

    EXPECT_EQ(
        source.source_fingerprint_fnv1a64(),
        loaded_fingerprint);

    if (amrex::ParallelDescriptor::IOProcessor()) {
        std::remove(filename.c_str());
    }

    amrex::ParallelDescriptor::Barrier();
}

TEST(
    ERFTerrainSource,
    AtmosphericFillExtendsEdgesWhileDirectSamplingRemainsStrict)
{
    const ERFTerrainSource source(
        std::vector<Real>{
            Real(0),
            Real(1)},
        std::vector<Real>{
            Real(0),
            Real(1)},
        std::vector<Real>{
            Real(10),
            Real(20),
            Real(30),
            Real(40)});

    EXPECT_THROW(
        (void)source.sample(
            Real(-1),
            Real(0)),
        std::invalid_argument);

    const amrex::Box domain{
        amrex::IntVect(0, 0, 0),
        amrex::IntVect(1, 1, 0)};

    const amrex::RealBox real_box{
        Real(-1),
        Real(-1),
        Real(0),
        Real(2),
        Real(2),
        Real(1)};

    const amrex::Array<int, AMREX_SPACEDIM>
        periodic{{0, 0, 0}};

    const amrex::Geometry geometry{
        domain,
        real_box,
        amrex::CoordSys::cartesian,
        periodic};

    const amrex::Box surface_box =
        amrex::convert(
            domain,
            amrex::IntVect(1, 1, 0));

    amrex::FArrayBox surface(
        surface_box,
        1,
        amrex::The_Arena());

    EXPECT_NO_THROW(
        source.fill_nodal_surface(
            geometry,
            surface));

    amrex::FArrayBox host(
        surface_box,
        1,
        amrex::The_Pinned_Arena());

    amrex::Gpu::dtoh_memcpy(
        host.dataPtr(),
        surface.dataPtr(),
        static_cast<std::size_t>(
            surface_box.numPts())
            * sizeof(Real));
    amrex::Gpu::streamSynchronize();

    const auto values =
        host.const_array();

    EXPECT_EQ(
        values(0, 0, 0),
        Real(10));
    EXPECT_EQ(
        values(2, 0, 0),
        Real(30));
    EXPECT_EQ(
        values(0, 2, 0),
        Real(20));
    EXPECT_EQ(
        values(2, 2, 0),
        Real(40));
    EXPECT_NEAR(
        values(1, 1, 0),
        Real(25),
        test_tolerance());
}

TEST(
    ERFTerrainSource,
    RejectsMalformedCoordinateTopology)
{
    EXPECT_THROW(
        (void)ERFTerrainSource(
            std::vector<Real>{
                Real(0),
                Real(0)},
            std::vector<Real>{
                Real(0),
                Real(1)},
            std::vector<Real>(
                4,
                Real(0))),
        std::invalid_argument);

    EXPECT_THROW(
        (void)ERFTerrainSource(
            std::vector<Real>{
                Real(0),
                Real(1)},
            std::vector<Real>{
                Real(0),
                Real(1)},
            std::vector<Real>(
                3,
                Real(0))),
        std::invalid_argument);
}

TEST(
    ERFTerrainSource,
    DirectFireSamplingRetainsTerrainLostOnCoarseGrid)
{
    std::vector<Real> x;
    for (int i = 0; i <= 8; ++i) {
        x.push_back(
            static_cast<Real>(i));
    }

    const std::vector<Real> y{
        Real(0),
        Real(1),
        Real(2)};

    const std::vector<Real> x_profile{
        Real(0),
        Real(1),
        Real(0),
        Real(-1),
        Real(0),
        Real(1),
        Real(0),
        Real(-1),
        Real(0)};

    std::vector<Real> elevation;
    elevation.reserve(
        x.size() * y.size());

    for (const Real height : x_profile) {
        for (std::size_t j = 0;
             j < y.size();
             ++j) {
            elevation.push_back(height);
        }
    }

    const ERFTerrainSource source(
        x,
        y,
        elevation);

    const auto fine =
        ERFFire::make_erf_terrain_source_surface_on_geometry(
            source,
            FireCartesianRasterGeometry2D{
                8U,
                2U,
                Real(0),
                Real(0),
                Real(1),
                Real(1)});

    const auto coarse =
        ERFFire::make_erf_terrain_source_surface_on_geometry(
            source,
            FireCartesianRasterGeometry2D{
                2U,
                1U,
                Real(0),
                Real(0),
                Real(4),
                Real(2)});

    const auto fine_positive =
        fine.terrain_gradient_m_per_m(
            Real(0.5),
            Real(0.5));

    const auto fine_negative =
        fine.terrain_gradient_m_per_m(
            Real(2.5),
            Real(0.5));

    const auto coarse_gradient =
        coarse.terrain_gradient_m_per_m(
            Real(1),
            Real(1));

    EXPECT_NEAR(
        fine_positive.x,
        Real(1),
        test_tolerance());
    EXPECT_NEAR(
        fine_positive.y,
        Real(0),
        test_tolerance());

    EXPECT_NEAR(
        fine_negative.x,
        Real(-1),
        test_tolerance());
    EXPECT_NEAR(
        fine_negative.y,
        Real(0),
        test_tolerance());

    EXPECT_NEAR(
        coarse_gradient.x,
        Real(0),
        test_tolerance());
    EXPECT_NEAR(
        coarse_gradient.y,
        Real(0),
        test_tolerance());
}
