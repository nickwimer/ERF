#include <ERF_FireTypes.H>

#include <gtest/gtest.h>

#include <type_traits>

#ifndef ERF_USE_FIRE
#error "erf_fire_unit_tests must compile with ERF_USE_FIRE"
#endif

namespace
{

TEST(FireTypes, Vec2StoresPhysicalComponents)
{
    const ERFFire::FireVec2 value{1.25, -2.5};

    EXPECT_DOUBLE_EQ(static_cast<double>(value.x), 1.25);
    EXPECT_DOUBLE_EQ(static_cast<double>(value.y), -2.5);
}

TEST(FireTypes, Vec2IsDependencyLightweight)
{
    static_assert(std::is_standard_layout_v<ERFFire::FireVec2>);
    static_assert(std::is_trivially_copyable_v<ERFFire::FireVec2>);

    SUCCEED();
}

} // namespace
