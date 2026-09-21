#include <ERF_FirePlotfile2D.H>
#include <ERF_Plotfile2DMetadata.H>

#include <gtest/gtest.h>

#include <cstddef>
#include <string>

TEST(FirePlotfile2D, ProviderOwnsDiagnosticMetadata)
{
    const amrex::Vector<std::string> names{
        "fire_burned_fraction",
        "fire_has_arrived",
        "fire_first_arrival_time_s",
        "fire_ignited_area_fraction",
        "fire_remaining_dry_fuel_kg_m2",
        "fire_consumed_dry_fuel_kg_m2",
        "fire_sensible_energy_j_m2",
        "fire_water_released_kg_m2"
    };

    const auto selection =
        ERFFire::select_fire_plotfile2d_output_descriptors(
            names);

    EXPECT_TRUE(
        selection.non_fire_plot_var_names.empty());

    ASSERT_EQ(
        selection.fire_descriptors.size(),
        names.size());

    for (std::size_t index = 0;
         index < names.size();
         ++index) {
        const auto& descriptor =
            selection.fire_descriptors[index];

        EXPECT_EQ(
            descriptor.name,
            names[index]);

        EXPECT_FALSE(
            descriptor.long_name.empty());

        EXPECT_FALSE(
            descriptor.units.empty());

        EXPECT_EQ(
            descriptor.category,
            plotfile2d::DiagnosticCategory::SurfaceState);

        EXPECT_EQ(
            descriptor.missing_policy,
            plotfile2d::MissingPolicy::AlwaysAvailable);

        EXPECT_EQ(
            descriptor.static_diagnostic,
            nullptr);

        EXPECT_FALSE(
            descriptor.sampled_level.has_value());
    }

    const std::string json =
        plotfile2d::format_2d_metadata_json(
            selection.fire_descriptors);

    EXPECT_NE(
        json.find("\"category\": \"SurfaceState\""),
        std::string::npos);
}

TEST(FirePlotfile2D, SeparatesProviderAndGenericNames)
{
    const amrex::Vector<std::string> names{
        "z_surf",
        "fire_burned_fraction",
        "temperature_2m",
        "fire_water_released_kg_m2"
    };

    const auto selection =
        ERFFire::select_fire_plotfile2d_output_descriptors(
            names);

    ASSERT_EQ(
        selection.non_fire_plot_var_names.size(),
        2U);

    EXPECT_EQ(
        selection.non_fire_plot_var_names[0],
        "z_surf");

    EXPECT_EQ(
        selection.non_fire_plot_var_names[1],
        "temperature_2m");

    ASSERT_EQ(
        selection.fire_descriptors.size(),
        2U);

    EXPECT_EQ(
        selection.fire_descriptors[0].name,
        "fire_burned_fraction");

    EXPECT_EQ(
        selection.fire_descriptors[1].name,
        "fire_water_released_kg_m2");
}
