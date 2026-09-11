#include <ERF_FireBurnedFraction.H>

#include <gtest/gtest.h>

#include <array>
#include <limits>
#include <stdexcept>

namespace
{

TEST(FireBurnedFraction, FirstCoverageBecomesHistoryAndIncrement)
{
    const auto update =
        ERFFire::update_fire_burned_fraction(0.0, 0.25);

    EXPECT_DOUBLE_EQ(
        static_cast<double>(update.burned_fraction),
        0.25);
    EXPECT_DOUBLE_EQ(
        static_cast<double>(update.newly_burned_fraction),
        0.25);
}

TEST(FireBurnedFraction, IncreasingCoverageAdvancesByExactDifference)
{
    const auto update =
        ERFFire::update_fire_burned_fraction(0.25, 0.75);

    EXPECT_DOUBLE_EQ(
        static_cast<double>(update.burned_fraction),
        0.75);
    EXPECT_DOUBLE_EQ(
        static_cast<double>(update.newly_burned_fraction),
        0.50);
}

TEST(FireBurnedFraction, LowerCoverageCannotUnburnCell)
{
    const auto update =
        ERFFire::update_fire_burned_fraction(0.75, 0.25);

    EXPECT_DOUBLE_EQ(
        static_cast<double>(update.burned_fraction),
        0.75);
    EXPECT_DOUBLE_EQ(
        static_cast<double>(update.newly_burned_fraction),
        0.0);
}

TEST(FireBurnedFraction, EqualCoverageIsIdempotent)
{
    const auto update =
        ERFFire::update_fire_burned_fraction(0.625, 0.625);

    EXPECT_DOUBLE_EQ(
        static_cast<double>(update.burned_fraction),
        0.625);
    EXPECT_DOUBLE_EQ(
        static_cast<double>(update.newly_burned_fraction),
        0.0);
}

TEST(FireBurnedFraction, FullyBurnedCellIsAbsorbing)
{
    for (const amrex::Real coverage : {
            amrex::Real(0.0),
            amrex::Real(0.25),
            amrex::Real(0.75),
            amrex::Real(1.0)}) {
        const auto update =
            ERFFire::update_fire_burned_fraction(1.0, coverage);

        EXPECT_DOUBLE_EQ(
            static_cast<double>(update.burned_fraction),
            1.0);
        EXPECT_DOUBLE_EQ(
            static_cast<double>(update.newly_burned_fraction),
            0.0);
    }
}

TEST(FireBurnedFraction, SequentialUpdatesAreMonotoneAndIncrementsTelescope)
{
    constexpr std::array<amrex::Real, 6> coverage_history{
        0.125,
        0.500,
        0.375,
        0.750,
        0.625,
        1.000
    };

    amrex::Real burned_fraction = 0.0;
    amrex::Real cumulative_newly_burned = 0.0;

    for (const amrex::Real coverage : coverage_history) {
        const amrex::Real previous = burned_fraction;
        const auto update =
            ERFFire::update_fire_burned_fraction(
                burned_fraction, coverage);

        EXPECT_GE(
            static_cast<double>(update.burned_fraction),
            static_cast<double>(previous));
        EXPECT_GE(
            static_cast<double>(update.newly_burned_fraction),
            0.0);
        EXPECT_DOUBLE_EQ(
            static_cast<double>(
                previous + update.newly_burned_fraction),
            static_cast<double>(update.burned_fraction));

        burned_fraction = update.burned_fraction;
        cumulative_newly_burned += update.newly_burned_fraction;
    }

    EXPECT_DOUBLE_EQ(
        static_cast<double>(burned_fraction),
        1.0);
    EXPECT_DOUBLE_EQ(
        static_cast<double>(cumulative_newly_burned),
        1.0);
}

TEST(FireBurnedFraction, TinyPositiveCoverageIsNotThresholdedAway)
{
    constexpr amrex::Real tiny = amrex::Real(1.0e-15);

    const auto update =
        ERFFire::update_fire_burned_fraction(0.0, tiny);

    EXPECT_DOUBLE_EQ(
        static_cast<double>(update.burned_fraction),
        static_cast<double>(tiny));
    EXPECT_DOUBLE_EQ(
        static_cast<double>(update.newly_burned_fraction),
        static_cast<double>(tiny));
}

TEST(FireBurnedFraction, RejectsInvalidFractions)
{
    const amrex::Real nan =
        std::numeric_limits<amrex::Real>::quiet_NaN();
    const amrex::Real inf =
        std::numeric_limits<amrex::Real>::infinity();

    for (const amrex::Real invalid : {
            amrex::Real(-0.01),
            amrex::Real(1.01),
            nan,
            inf}) {
        EXPECT_THROW(
            (void)ERFFire::update_fire_burned_fraction(
                invalid, 0.5),
            std::invalid_argument);

        EXPECT_THROW(
            (void)ERFFire::update_fire_burned_fraction(
                0.5, invalid),
            std::invalid_argument);
    }
}

} // namespace
