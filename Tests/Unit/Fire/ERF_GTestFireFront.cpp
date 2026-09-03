#include <ERF_FireCellCoverage.H>
#include <ERF_FireFront.H>
#include <ERF_FireFrontCollision.H>
#include <ERF_FireFrontTopology.H>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace
{

using ERFFire::FireCartesianCell2D;
using ERFFire::FireFront;
using ERFFire::FireFrontComponent;
using ERFFire::FireFrontRole;
using ERFFire::FirePerimeter;
using ERFFire::FireVec2;

FirePerimeter
make_perimeter (std::initializer_list<FireVec2> vertices)
{
    return FirePerimeter(std::vector<FireVec2>(vertices));
}

TEST(FireFront, RepresentsOuterBoundaryAndUnburnedHole)
{
    const FireFront front(std::vector<FireFrontComponent>{
        {
            FireFrontRole::Outer,
            make_perimeter({
                {0.0, 0.0},
                {10.0, 0.0},
                {10.0, 10.0},
                {0.0, 10.0}
            })
        },
        {
            FireFrontRole::Hole,
            make_perimeter({
                {3.0, 3.0},
                {7.0, 3.0},
                {7.0, 7.0},
                {3.0, 7.0}
            })
        }
    });

    ASSERT_EQ(front.components().size(), 2U);
    EXPECT_EQ(front.components()[0].role, FireFrontRole::Outer);
    EXPECT_EQ(front.components()[1].role, FireFrontRole::Hole);

    EXPECT_NEAR(
        static_cast<double>(front.burned_area_m2()),
        84.0,
        1.0e-12);

    const FireCartesianCell2D whole_cell{
        0.0, 10.0, 0.0, 10.0};
    EXPECT_NEAR(
        static_cast<double>(
            ERFFire::fire_front_cell_coverage_fraction(
                front, whole_cell)),
        0.84,
        1.0e-12);

    const FireCartesianCell2D hole_cell{
        3.0, 7.0, 3.0, 7.0};
    EXPECT_NEAR(
        static_cast<double>(
            ERFFire::fire_front_cell_coverage_fraction(
                front, hole_cell)),
        0.0,
        1.0e-12);

    const FireVec2 outer_normal =
        front.spread_normal_unit(0, 0);
    EXPECT_LT(outer_normal.x, 0.0);
    EXPECT_LT(outer_normal.y, 0.0);

    const FireVec2 hole_normal =
        front.spread_normal_unit(1, 0);
    EXPECT_GT(hole_normal.x, 0.0);
    EXPECT_GT(hole_normal.y, 0.0);
}

TEST(FireFrontTopology, ExplicitPinchPreservesBurnedAreaAndCreatesHole)
{
    const FirePerimeter perimeter = make_perimeter({
        { 0.0,  0.0},
        {10.0,  0.0},
        {10.0,  2.0},
        { 3.0,  2.0},
        { 3.0,  8.0},
        { 8.0,  8.0},
        { 8.0,  2.2},
        {10.0,  2.2},
        {10.0, 10.0},
        { 0.0, 10.0}
    });

    ASSERT_NEAR(
        static_cast<double>(perimeter.area_m2()),
        69.6,
        1.0e-12);

    const ERFFire::FirePerimeterPinch pinch{
        2U,
        amrex::Real(2.0 / 7.0),
        6U,
        amrex::Real(0.0)
    };

    const FireFront front =
        ERFFire::split_perimeter_at_pinch(
            perimeter,
            pinch);

    ASSERT_EQ(front.components().size(), 2U);

    const FireFrontComponent* outer = nullptr;
    const FireFrontComponent* hole = nullptr;

    for (const FireFrontComponent& component : front.components()) {
        if (component.role == FireFrontRole::Outer) {
            ASSERT_EQ(outer, nullptr);
            outer = &component;
        } else if (component.role == FireFrontRole::Hole) {
            ASSERT_EQ(hole, nullptr);
            hole = &component;
        }
    }

    ASSERT_NE(outer, nullptr);
    ASSERT_NE(hole, nullptr);

    EXPECT_NEAR(
        static_cast<double>(outer->perimeter.area_m2()),
        99.6,
        1.0e-12);
    EXPECT_NEAR(
        static_cast<double>(hole->perimeter.area_m2()),
        30.0,
        1.0e-12);
    EXPECT_NEAR(
        static_cast<double>(front.burned_area_m2()),
        static_cast<double>(perimeter.area_m2()),
        1.0e-12);
    EXPECT_NEAR(
        static_cast<double>(
            outer->perimeter.area_m2()
            - hole->perimeter.area_m2()),
        static_cast<double>(perimeter.area_m2()),
        1.0e-12);
}

TEST(FireFrontCollision, LocatesFirstContactDuringLinearVertexMotion)
{
    const FirePerimeter start = make_perimeter({
        { 0.0,  0.0},
        {10.0,  0.0},
        {10.0,  2.0},
        { 3.0,  2.0},
        { 3.0,  8.0},
        { 8.0,  8.0},
        { 8.0,  2.2},
        {10.0,  2.2},
        {10.0, 10.0},
        { 0.0, 10.0}
    });

    std::vector<FireVec2> end_vertices =
        start.vertices_m();
    end_vertices[6].y = amrex::Real(1.8);

    const auto collision =
        ERFFire::locate_first_perimeter_collision(
            start,
            end_vertices);

    ASSERT_TRUE(collision.has_value());

    EXPECT_NEAR(
        static_cast<double>(collision->motion_fraction),
        0.5,
        1.0e-12);

    EXPECT_EQ(collision->pinch.first_edge_index, 2U);
    EXPECT_NEAR(
        static_cast<double>(
            collision->pinch.first_edge_fraction),
        2.0 / 7.0,
        1.0e-12);

    EXPECT_EQ(collision->pinch.second_edge_index, 6U);
    EXPECT_NEAR(
        static_cast<double>(
            collision->pinch.second_edge_fraction),
        0.0,
        1.0e-12);

    EXPECT_NEAR(
        static_cast<double>(collision->contact_point_m.x),
        8.0,
        1.0e-12);
    EXPECT_NEAR(
        static_cast<double>(collision->contact_point_m.y),
        2.0,
        1.0e-12);
}

TEST(FireFrontCollision, RigidTranslationHasNoCollision)
{
    const FirePerimeter start = make_perimeter({
        {0.0, 0.0},
        {4.0, 0.0},
        {4.0, 3.0},
        {0.0, 3.0}
    });

    std::vector<FireVec2> end_vertices;
    end_vertices.reserve(start.size());

    for (const FireVec2& vertex : start.vertices_m()) {
        end_vertices.push_back({
            vertex.x + amrex::Real(7.25),
            vertex.y - amrex::Real(3.5)
        });
    }

    const auto collision =
        ERFFire::locate_first_perimeter_collision(
            start,
            end_vertices);

    EXPECT_FALSE(collision.has_value());
}

TEST(FireFrontCollision, ChoosesEarliestOfMultipleContacts)
{
    const FirePerimeter start = make_perimeter({
        { 0.0,  0.0},
        {20.0,  0.0},
        {20.0,  2.0},
        {15.0,  2.0},
        {15.0,  4.0},
        {18.0,  4.0},
        {18.0,  2.4},
        {20.0,  2.4},
        {20.0,  6.0},
        {15.0,  6.0},
        {15.0,  8.0},
        {18.0,  8.0},
        {18.0,  6.6},
        {20.0,  6.6},
        {20.0, 10.0},
        { 0.0, 10.0}
    });

    std::vector<FireVec2> end_vertices =
        start.vertices_m();

    end_vertices[6].y = amrex::Real(1.2);
    end_vertices[12].y = amrex::Real(5.8);

    const auto collision =
        ERFFire::locate_first_perimeter_collision(
            start,
            end_vertices);

    ASSERT_TRUE(collision.has_value());

    EXPECT_NEAR(
        static_cast<double>(collision->motion_fraction),
        1.0 / 3.0,
        1.0e-12);
    EXPECT_EQ(
        collision->pinch.first_edge_index,
        2U);
    EXPECT_NEAR(
        static_cast<double>(
            collision->pinch.first_edge_fraction),
        0.4,
        1.0e-12);
    EXPECT_EQ(
        collision->pinch.second_edge_index,
        6U);
    EXPECT_NEAR(
        static_cast<double>(
            collision->pinch.second_edge_fraction),
        0.0,
        1.0e-12);
    EXPECT_NEAR(
        static_cast<double>(collision->contact_point_m.x),
        18.0,
        1.0e-12);
    EXPECT_NEAR(
        static_cast<double>(collision->contact_point_m.y),
        2.0,
        1.0e-12);
}

TEST(FireFrontCollision, IsInvariantUnderPositiveScaleAndTranslation)
{
    const FirePerimeter start = make_perimeter({
        { 0.0,  0.0},
        {10.0,  0.0},
        {10.0,  2.0},
        { 3.0,  2.0},
        { 3.0,  8.0},
        { 8.0,  8.0},
        { 8.0,  2.2},
        {10.0,  2.2},
        {10.0, 10.0},
        { 0.0, 10.0}
    });

    std::vector<FireVec2> end_vertices =
        start.vertices_m();
    end_vertices[6].y = amrex::Real(1.8);

    const auto base =
        ERFFire::locate_first_perimeter_collision(
            start,
            end_vertices);

    ASSERT_TRUE(base.has_value());

    constexpr amrex::Real scale = 37.0;
    const FireVec2 translation{
        20317.25,
        6588.5
    };

    const auto transform =
        [&] (const FireVec2& point)
    {
        return FireVec2{
            translation.x + scale * point.x,
            translation.y + scale * point.y
        };
    };

    std::vector<FireVec2> transformed_start_vertices;
    transformed_start_vertices.reserve(start.size());
    for (const FireVec2& vertex : start.vertices_m()) {
        transformed_start_vertices.push_back(
            transform(vertex));
    }

    std::vector<FireVec2> transformed_end_vertices;
    transformed_end_vertices.reserve(end_vertices.size());
    for (const FireVec2& vertex : end_vertices) {
        transformed_end_vertices.push_back(
            transform(vertex));
    }

    const FirePerimeter transformed_start(
        std::move(transformed_start_vertices));

    const auto transformed =
        ERFFire::locate_first_perimeter_collision(
            transformed_start,
            transformed_end_vertices);

    ASSERT_TRUE(transformed.has_value());

    EXPECT_NEAR(
        static_cast<double>(transformed->motion_fraction),
        static_cast<double>(base->motion_fraction),
        1.0e-12);
    EXPECT_EQ(
        transformed->pinch.first_edge_index,
        base->pinch.first_edge_index);
    EXPECT_NEAR(
        static_cast<double>(
            transformed->pinch.first_edge_fraction),
        static_cast<double>(
            base->pinch.first_edge_fraction),
        1.0e-12);
    EXPECT_EQ(
        transformed->pinch.second_edge_index,
        base->pinch.second_edge_index);
    EXPECT_NEAR(
        static_cast<double>(
            transformed->pinch.second_edge_fraction),
        static_cast<double>(
            base->pinch.second_edge_fraction),
        1.0e-12);

    const FireVec2 expected_contact =
        transform(base->contact_point_m);

    EXPECT_NEAR(
        static_cast<double>(
            transformed->contact_point_m.x),
        static_cast<double>(expected_contact.x),
        1.0e-10);
    EXPECT_NEAR(
        static_cast<double>(
            transformed->contact_point_m.y),
        static_cast<double>(expected_contact.y),
        1.0e-10);
}


TEST(FireFrontTopology, HolePinchSplitsIntoTwoPositiveAreaHoles)
{
    const std::vector<FireVec2> event_vertices{
        { 0.0, 0.0},
        { 2.0, 0.0},
        { 1.0, 1.0},
        { 0.0, 0.0},
        {-1.0, 1.0},
        {-2.0, 0.0}
    };

    const ERFFire::FirePerimeterPinch pinch{
        2U,
        amrex::Real(1.0),
        5U,
        amrex::Real(1.0)
    };

    const auto holes =
        ERFFire::split_hole_perimeter_at_pinch(
            event_vertices,
            pinch);

    ASSERT_EQ(holes.size(), 2U);

    EXPECT_NEAR(
        static_cast<double>(holes[0].area_m2()),
        1.0,
        1.0e-12);
    EXPECT_NEAR(
        static_cast<double>(holes[1].area_m2()),
        1.0,
        1.0e-12);
    EXPECT_NEAR(
        static_cast<double>(
            holes[0].area_m2()
            + holes[1].area_m2()),
        2.0,
        1.0e-12);

    EXPECT_THROW(
        (void)ERFFire::split_perimeter_at_pinch(
            event_vertices,
            pinch),
        std::invalid_argument);
}



TEST(FireFrontTopology, HolePinchAreaPartitionIsStableAtPalisadesScale)
{
    const amrex::Real tx =
        amrex::Real(20465.54522430533);
    const amrex::Real ty =
        amrex::Real(6956.3808835368473);

    const std::vector<FireVec2> event_vertices{
        {tx,                    ty},
        {tx + amrex::Real(0.2),        ty},
        {tx + amrex::Real(0.1),        ty + amrex::Real(0.1)},
        {tx,                    ty},
        {tx - amrex::Real(0.1),        ty + amrex::Real(0.1)},
        {tx - amrex::Real(0.2),        ty}
    };

    const ERFFire::FirePerimeterPinch pinch{
        2U,
        amrex::Real(1.0),
        5U,
        amrex::Real(1.0)
    };

    const auto holes =
        ERFFire::split_hole_perimeter_at_pinch(
            event_vertices,
            pinch);

    ASSERT_EQ(holes.size(), 2U);

    EXPECT_NEAR(
        static_cast<double>(holes[0].area_m2()),
        0.01,
        2.0e-13);
    EXPECT_NEAR(
        static_cast<double>(holes[1].area_m2()),
        0.01,
        2.0e-13);
    EXPECT_NEAR(
        static_cast<double>(
            holes[0].area_m2()
            + holes[1].area_m2()),
        0.02,
        4.0e-13);
}




TEST(FireFrontTopology, TinyPositiveDaughterIsNotCollapsedAsDegenerateEar)
{
    const amrex::Real epsilon =
        amrex::Real(1.0e-9);

    const std::vector<FireVec2> event_vertices{
        {amrex::Real(0.0), amrex::Real(0.0)},
        {amrex::Real(2.0), amrex::Real(0.0)},
        {amrex::Real(1.0), amrex::Real(1.0)},
        {amrex::Real(0.0), amrex::Real(0.0)},
        {-epsilon, epsilon},
        {amrex::Real(-2.0), amrex::Real(0.0)}
    };

    const ERFFire::FirePerimeterPinch pinch{
        2U,
        amrex::Real(1.0),
        5U,
        amrex::Real(1.0)
    };

    const auto holes =
        ERFFire::split_hole_perimeter_at_pinch(
            event_vertices,
            pinch);

    ASSERT_EQ(
        holes.size(),
        2U);

    const amrex::Real first_area_m2 =
        holes[0].area_m2();
    const amrex::Real second_area_m2 =
        holes[1].area_m2();

    const amrex::Real smaller_area_m2 =
        first_area_m2 < second_area_m2
            ? first_area_m2
            : second_area_m2;
    const amrex::Real larger_area_m2 =
        first_area_m2 > second_area_m2
            ? first_area_m2
            : second_area_m2;

    EXPECT_GT(
        smaller_area_m2,
        amrex::Real(0.0));
    EXPECT_GT(
        larger_area_m2,
        amrex::Real(0.0));

    EXPECT_NEAR(
        static_cast<double>(
            smaller_area_m2),
        static_cast<double>(epsilon),
        1.0e-15);

    EXPECT_NEAR(
        static_cast<double>(
            larger_area_m2),
        1.0,
        1.0e-12);

    EXPECT_NEAR(
        static_cast<double>(
            first_area_m2
            + second_area_m2),
        1.000000001,
        2.0e-12);
}


TEST(FireFrontTopology, DegeneratePinchDropsZeroAreaEarAtPalisadesScale)
{
    const std::vector<ERFFire::FireVec2> event_vertices{
        {20973.410773992484, 6693.7013210084424},
        {20956.042125351403, 6707.5008213549818},
        {20946.827780260428, 6710.3417765442173},
        {20940.957494217630, 6710.1923811519409},
        {20943.595581627596, 6697.1098235134086},
        {20936.737041029512, 6685.2394201596308},
        {20950.056630912724, 6686.8089934084946},
        {20962.069116216997, 6689.8122992397011},
        {20973.865463020204, 6693.8572325658006}
    };

    const ERFFire::FirePerimeterPinch pinch{
        0U,
        amrex::Real(0.0),
        7U,
        amrex::Real(0.96145509831943354)
    };

    const ERFFire::FireFront outer_resolution =
        ERFFire::split_perimeter_at_pinch(
            event_vertices,
            pinch);

    ASSERT_EQ(
        outer_resolution.components().size(),
        1U);
    EXPECT_EQ(
        outer_resolution.components().front().role,
        ERFFire::FireFrontRole::Outer);
    EXPECT_EQ(
        outer_resolution.components()
            .front().perimeter.size(),
        8U);
    EXPECT_NEAR(
        static_cast<double>(
            outer_resolution.components()
                .front().perimeter.area_m2()),
        502.81473121121275,
        1.0e-9);

    std::vector<ERFFire::FirePerimeter>
        hole_resolution =
            ERFFire::split_hole_perimeter_at_pinch(
                event_vertices,
                pinch);

    ASSERT_EQ(
        hole_resolution.size(),
        1U);
    EXPECT_EQ(
        hole_resolution.front().size(),
        8U);
    EXPECT_NEAR(
        static_cast<double>(
            hole_resolution.front().area_m2()),
        502.81473121121275,
        1.0e-9);
}



TEST(FireFrontTopology, PalisadesHoleBoundsOverlapButPolygonsAreDisjoint)
{
    const ERFFire::FirePerimeter new_hole({
        {20817.975002680556, 7249.5484668197605},
        {20818.745558766375, 7236.445800349642},
        {20819.993604598763, 7261.0508525906916}
    });

    const ERFFire::FirePerimeter existing_hole({
        {20817.738626514143, 7237.7364214198133},
        {20819.236018772612, 7233.4078075238203},
        {20818.733980597983, 7235.9436706781889}
    });

    EXPECT_TRUE(
        ERFFire::perimeters_are_strictly_disjoint(
            new_hole,
            existing_hole));

    EXPECT_TRUE(
        ERFFire::perimeters_are_strictly_disjoint(
            existing_hole,
            new_hole));
}

TEST(FireFrontTopology, StrictDisjointnessRejectsIntersectionContainmentAndTouch)
{
    const ERFFire::FirePerimeter reference({
        {0.0, 0.0},
        {2.0, 0.0},
        {1.0, 2.0}
    });

    const ERFFire::FirePerimeter intersecting({
        {1.0, -1.0},
        {3.0, 1.0},
        {1.0, 1.0}
    });

    const ERFFire::FirePerimeter contained({
        {0.8, 0.5},
        {1.2, 0.5},
        {1.0, 1.0}
    });

    const ERFFire::FirePerimeter touching({
        {2.0, 0.0},
        {3.0, 0.0},
        {2.0, 1.0}
    });

    EXPECT_FALSE(
        ERFFire::perimeters_are_strictly_disjoint(
            reference,
            intersecting));

    EXPECT_FALSE(
        ERFFire::perimeters_are_strictly_disjoint(
            reference,
            contained));

    EXPECT_FALSE(
        ERFFire::perimeters_are_strictly_disjoint(
            contained,
            reference));

    EXPECT_FALSE(
        ERFFire::perimeters_are_strictly_disjoint(
            reference,
            touching));
}


} // namespace
