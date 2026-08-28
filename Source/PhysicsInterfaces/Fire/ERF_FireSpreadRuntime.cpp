#include "ERF_FireSpreadRuntime.H"

#include <ERF_FireSpreadOutput.H>
#include <ERF_FireWindAdjustment.H>
#include <ERF_FireFrontPropagator.H>
#include <ERF_RichardsDirectionalSpread.H>
#include <ERF_RothermelModel.H>
#include <ERF_VectorPerimeterPropagator.H>

#include <AMReX_Arena.H>
#include <AMReX_Gpu.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>

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

FireVec2
wind_push_unit(const FireVec2& wind_mps, amrex::Real speed_mps)
{
    if (!(speed_mps > amrex::Real(0.0))) {
        return {amrex::Real(1.0), amrex::Real(0.0)};
    }
    return wind_mps / speed_mps;
}


FireVec2
terrain_upslope_unit(
    const FireVec2& gradient_m_per_m,
    amrex::Real slope_tangent)
{
    if (!(slope_tangent > amrex::Real(0.0))) {
        return {
            amrex::Real(1.0),
            amrex::Real(0.0)};
    }
    return gradient_m_per_m / slope_tangent;
}

amrex::Real
fraction_tolerance(amrex::Real scale) noexcept
{
    return amrex::Real(1024)
        * std::numeric_limits<amrex::Real>::epsilon()
        * std::max(amrex::Real(1), std::abs(scale));
}

bool
fraction_equal(amrex::Real a, amrex::Real b) noexcept
{
    return std::abs(a - b)
        <= fraction_tolerance(
            std::max(std::abs(a), std::abs(b)));
}

void
validate_runtime_scalars(
    const ERFFireSpreadConfig& config,
    amrex::Real current_time_s)
{
    require(
        std::isfinite(current_time_s)
            && current_time_s >= amrex::Real(0.0),
        "fire spread time must be finite and nonnegative");

    require(
        std::isfinite(config.arrival_time_tolerance_s)
            && config.arrival_time_tolerance_s > amrex::Real(0.0),
        "fire spread arrival tolerance must be finite and positive");

    (void)evaluate_rothermel(
        config.fuel,
        RothermelInputs{
            config.dead_fuel_moisture_fraction,
            amrex::Real(0.0),
            amrex::Real(0.0)});
}

bool
same_horizontal_geometry(
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

bool
environment_matches_geometry(
    const FireFlatEnvironmentSampler& environment,
    const FireCartesianRasterGeometry2D& geometry) noexcept
{
    const auto& layout = environment.layout();

    return layout.nx() == geometry.nx
        && layout.ny() == geometry.ny
        && layout.xlo_m() == geometry.xlo_m
        && layout.ylo_m() == geometry.ylo_m
        && layout.dx_m() == geometry.dx_m
        && layout.dy_m() == geometry.dy_m;
}

void
require_terrain_runtime_geometry(
    const FireFlatEnvironmentSampler& environment,
    const FireTerrainSurface& terrain,
    const FireCartesianRasterGeometry2D& runtime_geometry)
{
    require(
        same_horizontal_geometry(
            terrain.geometry(),
            runtime_geometry),
        "fire terrain geometry must match the Fire runtime raster geometry");

    require(
        environment_matches_geometry(
            environment,
            runtime_geometry),
        "fire terrain reference-wind geometry must match the Fire runtime raster geometry");
}

void
require_perimeter_inside_environment(
    const FirePerimeter& perimeter,
    const FireFlatEnvironmentSampler& environment)
{
    for (const FireVec2& vertex : perimeter.vertices_m()) {
        if (!environment.layout().contains_physical_point(
                vertex.x, vertex.y)) {
            throw std::out_of_range(
                "fire spread propagated perimeter leaves the physical environment domain");
        }
    }
}

FireEnvironmentBatchFunction
with_terrain_gradient(
    const FireEnvironmentBatchFunction& environment,
    const FireTerrainSurface& terrain)
{
    require(
        static_cast<bool>(environment),
        "fire batched environment sampler must be set");

    return
        [&environment, &terrain](
            const std::vector<FireVec2>& positions_m) {
            std::vector<FireEnvironmentSample> samples =
                environment(positions_m);
            require(
                samples.size() == positions_m.size(),
                "fire batched environment sampler returned the wrong size");

            for (std::size_t index = 0;
                 index < samples.size();
                 ++index) {
                samples[index].terrain_gradient_m_per_m =
                    terrain.terrain_gradient_m_per_m(
                        positions_m[index].x,
                        positions_m[index].y);
            }
            return samples;
        };
}

FireFront
restore_front_from_state(
    ERFFireSpreadRuntimeState& state)
{
    if (!state.front_components.empty()) {
        std::vector<FireFrontComponent> components;
        components.reserve(
            state.front_components.size());

        for (auto& component :
             state.front_components) {
            components.push_back({
                component.role,
                FirePerimeter(
                    std::move(
                        component.vertices_m))
            });
        }

        return FireFront(
            std::move(components));
    }

    return FireFront(
        std::vector<FireFrontComponent>{
            {
                FireFrontRole::Outer,
                FirePerimeter(
                    std::move(
                        state.perimeter_vertices_m))
            }
        });
}

void
collective_broadcast_front_state(
    ERFFireSpreadRuntimeState& state)
{
    const int io_rank =
        amrex::ParallelDescriptor::IOProcessorNumber();

    unsigned long long component_count =
        amrex::ParallelDescriptor::IOProcessor()
            ? static_cast<unsigned long long>(
                state.front_components.size())
            : 0ULL;
    amrex::ParallelDescriptor::Bcast(
        &component_count,
        1,
        io_rank);

    if (component_count
        > static_cast<unsigned long long>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error(
            "collective Fire restore component count is not representable");
    }

    if (component_count != 0ULL) {
        const std::size_t count =
            static_cast<std::size_t>(
                component_count);

        state.perimeter_vertices_m.clear();

        if (!amrex::ParallelDescriptor::IOProcessor()) {
            state.front_components.resize(count);
        }

        for (std::size_t component_index = 0;
             component_index < count;
             ++component_index) {
            int role =
                amrex::ParallelDescriptor::IOProcessor()
                    ? static_cast<int>(
                        state.front_components[
                            component_index].role)
                    : 0;
            amrex::ParallelDescriptor::Bcast(
                &role,
                1,
                io_rank);

            if (role
                    != static_cast<int>(
                        FireFrontRole::Outer)
                && role
                    != static_cast<int>(
                        FireFrontRole::Hole)) {
                throw std::invalid_argument(
                    "collective Fire restore front role is invalid");
            }

            unsigned long long vertex_count =
                amrex::ParallelDescriptor::IOProcessor()
                    ? static_cast<unsigned long long>(
                        state.front_components[
                            component_index]
                            .vertices_m.size())
                    : 0ULL;
            amrex::ParallelDescriptor::Bcast(
                &vertex_count,
                1,
                io_rank);

            if (vertex_count
                > static_cast<unsigned long long>(
                    std::numeric_limits<std::size_t>::max()
                    / 2)) {
                throw std::overflow_error(
                    "collective Fire restore component size is not representable");
            }

            const std::size_t count_vertices =
                static_cast<std::size_t>(
                    vertex_count);
            std::vector<amrex::Real> packed(
                2 * count_vertices);

            if (amrex::ParallelDescriptor::IOProcessor()) {
                const auto& vertices =
                    state.front_components[
                        component_index]
                        .vertices_m;
                for (std::size_t index = 0;
                     index < count_vertices;
                     ++index) {
                    packed[2 * index] =
                        vertices[index].x;
                    packed[2 * index + 1] =
                        vertices[index].y;
                }
            }

            if (!packed.empty()) {
                amrex::ParallelDescriptor::Bcast(
                    packed.data(),
                    packed.size(),
                    io_rank);
            }

            if (!amrex::ParallelDescriptor::IOProcessor()) {
                auto& destination =
                    state.front_components[
                        component_index];
                destination.role =
                    static_cast<FireFrontRole>(role);
                destination.vertices_m.resize(
                    count_vertices);

                for (std::size_t index = 0;
                     index < count_vertices;
                     ++index) {
                    destination.vertices_m[index] = {
                        packed[2 * index],
                        packed[2 * index + 1]
                    };
                }
            }
        }

        return;
    }

    state.front_components.clear();

    unsigned long long vertex_count =
        amrex::ParallelDescriptor::IOProcessor()
            ? static_cast<unsigned long long>(
                state.perimeter_vertices_m.size())
            : 0ULL;
    amrex::ParallelDescriptor::Bcast(
        &vertex_count,
        1,
        io_rank);

    if (vertex_count
        > static_cast<unsigned long long>(
            std::numeric_limits<std::size_t>::max()
            / 2)) {
        throw std::overflow_error(
            "collective Fire restore perimeter size is not representable");
    }

    const std::size_t count =
        static_cast<std::size_t>(vertex_count);
    std::vector<amrex::Real> packed(
        2 * count);

    if (amrex::ParallelDescriptor::IOProcessor()) {
        for (std::size_t index = 0;
             index < count;
             ++index) {
            packed[2 * index] =
                state.perimeter_vertices_m[index].x;
            packed[2 * index + 1] =
                state.perimeter_vertices_m[index].y;
        }
    }

    if (!packed.empty()) {
        amrex::ParallelDescriptor::Bcast(
            packed.data(),
            packed.size(),
            io_rank);
    }

    if (!amrex::ParallelDescriptor::IOProcessor()) {
        state.perimeter_vertices_m.resize(count);
        for (std::size_t index = 0;
             index < count;
             ++index) {
            state.perimeter_vertices_m[index] = {
                packed[2 * index],
                packed[2 * index + 1]
            };
        }
    }
}

} // namespace

ERFFireSpreadRuntime::ERFFireSpreadRuntime(
    FirePerimeter initial_perimeter,
    amrex::Real initial_time_s,
    ERFFireSpreadConfig config)
    : config_(std::move(config)),
      front_(
          std::vector<FireFrontComponent>{
              {
                  FireFrontRole::Outer,
                  std::move(initial_perimeter)
              }
          }),
      burned_fraction_(config_.raster_geometry),
      first_arrival_(config_.raster_geometry),
      combustion_(
          config_.raster_geometry,
          config_.combustion_parameters,
          config_.combustion_options),
      current_time_s_(initial_time_s)
{
    validate_runtime_scalars(
        config_,
        current_time_s_);

    auto initial_remesh =
        remesh_front(front_, config_.remesh_options);
    front_ = std::move(initial_remesh.front);

    (void)first_arrival_.initialize_from_perimeter(
        perimeter(), current_time_s_);
    (void)burned_fraction_.update_from_perimeter(perimeter());
    (void)combustion_.initialize_from_burned_fraction(
        burned_fraction_);
}

ERFFireSpreadRuntime::ERFFireSpreadRuntime(
    FireFront initial_front,
    amrex::Real initial_time_s,
    ERFFireSpreadConfig config)
    : config_(std::move(config)),
      front_(std::move(initial_front)),
      burned_fraction_(config_.raster_geometry),
      first_arrival_(config_.raster_geometry),
      combustion_(
          config_.raster_geometry,
          config_.combustion_parameters,
          config_.combustion_options),
      current_time_s_(initial_time_s)
{
    validate_runtime_scalars(
        config_,
        current_time_s_);

    auto initial_remesh =
        remesh_front(
            front_,
            config_.remesh_options);
    front_ = std::move(initial_remesh.front);

    (void)first_arrival_.initialize_from_front(
        front_,
        current_time_s_);
    (void)burned_fraction_.update_from_front(
        front_);
    (void)combustion_.initialize_from_burned_fraction(
        burned_fraction_);
}

const FireFront&
ERFFireSpreadRuntime::front() const
{
    return front_;
}

ERFFireSpreadRuntime::ERFFireSpreadRuntime(
    ERFFireSpreadRuntimeState state,
    RestoreStateTag)
    : config_(std::move(state.config)),
      front_(restore_front_from_state(state)),
      burned_fraction_(
          config_.raster_geometry,
          std::move(state.burned_fraction)),
      first_arrival_(
          config_.raster_geometry,
          std::move(state.first_arrival)),
      combustion_(
          config_.raster_geometry,
          config_.combustion_parameters,
          config_.combustion_options,
          std::move(state.combustion)),
      current_time_s_(state.current_time_s)
{
    validate_runtime_scalars(
        config_,
        current_time_s_);

    // Validate remeshing controls without changing restored topology.
    (void)remesh_front(
        front_,
        config_.remesh_options);

    require(
        first_arrival_.has_initial_condition(),
        "restored fire runtime requires initialized first-arrival history");

    if (first_arrival_.has_committed_sweep()) {
        require(
            first_arrival_.last_sweep_end_time_s()
                == current_time_s_,
            "restored fire runtime clock does not match first-arrival history");
    } else {
        require(
            first_arrival_.initial_condition_time_s()
                == current_time_s_,
            "restored fire runtime initial clock does not match first-arrival history");
    }

    require(
        combustion_.initialized(),
        "restored fire runtime requires initialized combustion history");

    const auto& geometry = config_.raster_geometry;
    const amrex::Real xhi =
        geometry.xlo_m
        + static_cast<amrex::Real>(geometry.nx)
            * geometry.dx_m;
    const amrex::Real yhi =
        geometry.ylo_m
        + static_cast<amrex::Real>(geometry.ny)
            * geometry.dy_m;

    for (const FireFrontComponent& component :
         front_.components()) {
        for (const FireVec2& vertex :
             component.perimeter.vertices_m()) {
            require(
                std::isfinite(vertex.x)
                    && std::isfinite(vertex.y)
                    && vertex.x >= geometry.xlo_m
                    && vertex.x <= xhi
                    && vertex.y >= geometry.ylo_m
                    && vertex.y <= yhi,
                "restored fire front lies outside its raster geometry");
        }
    }

}

ERFFireSpreadRuntime::ERFFireSpreadRuntime(
    ERFFireSpreadConfig config,
    FireFront front,
    FireBurnedFractionRaster burned_fraction,
    FireFirstArrivalRaster first_arrival,
    FireCombustionRaster combustion,
    amrex::Real current_time_s,
    CollectiveRestoreStateTag)
    : config_(std::move(config)),
      front_(std::move(front)),
      burned_fraction_(std::move(burned_fraction)),
      first_arrival_(std::move(first_arrival)),
      combustion_(std::move(combustion)),
      current_time_s_(current_time_s)
{
    validate_runtime_scalars(config_, current_time_s_);
    (void)remesh_front(front_, config_.remesh_options);

    require(
        first_arrival_.has_initial_condition(),
        "restored fire runtime requires initialized first-arrival history");
    if (first_arrival_.has_committed_sweep()) {
        require(
            first_arrival_.last_sweep_end_time_s() == current_time_s_,
            "restored fire runtime clock does not match first-arrival history");
    } else {
        require(
            first_arrival_.initial_condition_time_s() == current_time_s_,
            "restored fire runtime initial clock does not match first-arrival history");
    }
    require(
        combustion_.initialized(),
        "restored fire runtime requires initialized combustion history");

    const auto& geometry = config_.raster_geometry;
    const amrex::Real xhi =
        geometry.xlo_m
        + static_cast<amrex::Real>(geometry.nx) * geometry.dx_m;
    const amrex::Real yhi =
        geometry.ylo_m
        + static_cast<amrex::Real>(geometry.ny) * geometry.dy_m;
    for (const FireFrontComponent& component :
         front_.components()) {
        for (const FireVec2& vertex :
             component.perimeter.vertices_m()) {
            require(
                std::isfinite(vertex.x)
                    && std::isfinite(vertex.y)
                    && vertex.x >= geometry.xlo_m
                    && vertex.x <= xhi
                    && vertex.y >= geometry.ylo_m
                    && vertex.y <= yhi,
                "restored fire front lies outside its raster geometry");
        }
    }
}

ERFFireSpreadRuntimeState
ERFFireSpreadRuntime::snapshot_state() const
{
    ERFFireSpreadRuntimeState state;
    state.config = config_;

    const auto& components =
        front_.components();

    if (components.size() == 1
        && components.front().role
            == FireFrontRole::Outer) {
        state.perimeter_vertices_m =
            components.front()
                .perimeter.vertices_m();
    }

    state.front_components.reserve(
        components.size());

    for (const FireFrontComponent& component :
         components) {
        state.front_components.push_back({
            component.role,
            component.perimeter.vertices_m()
        });
    }

    state.burned_fraction =
        burned_fraction_.snapshot_state();
    state.first_arrival =
        first_arrival_.snapshot_state();
    state.combustion =
        combustion_.snapshot_state();
    state.current_time_s =
        current_time_s_;

    return state;
}

ERFFireSpreadRuntimeState
ERFFireSpreadRuntime::
collective_snapshot_state_to_io_rank() const
{
    ERFFireSpreadRuntimeState state;
    state.config = config_;

    const auto& components =
        front_.components();

    if (components.size() == 1
        && components.front().role
            == FireFrontRole::Outer) {
        state.perimeter_vertices_m =
            components.front()
                .perimeter.vertices_m();
    }

    state.front_components.reserve(
        components.size());
    for (const FireFrontComponent& component :
         components) {
        state.front_components.push_back({
            component.role,
            component.perimeter.vertices_m()
        });
    }

    FireBurnedFractionRasterState burned =
        burned_fraction_
            .collective_snapshot_state_to_io_rank();
    FireFirstArrivalRasterState arrival =
        first_arrival_
            .collective_snapshot_state_to_io_rank();
    FireCombustionRasterState combustion =
        combustion_
            .collective_snapshot_state_to_io_rank();

    state.burned_fraction =
        std::move(burned);
    state.first_arrival =
        std::move(arrival);
    state.combustion =
        std::move(combustion);
    state.current_time_s =
        current_time_s_;

    return state;
}

ERFFireSpreadRuntime
ERFFireSpreadRuntime::restore_from_state(
    ERFFireSpreadRuntimeState state)
{
    if (state.burned_fraction.burned_fraction.size()
        == state.combustion.cells.size()) {
        for (std::size_t index = 0;
             index < state.burned_fraction.burned_fraction.size();
             ++index) {
            require(
                fraction_equal(
                    state.burned_fraction.burned_fraction[index],
                    state.combustion.cells[index]
                        .ignited_area_fraction),
                "restored fire combustion history is not synchronized with burned fraction");
        }
    }

    return ERFFireSpreadRuntime(
        std::move(state),
        RestoreStateTag{});
}

ERFFireSpreadRuntime
ERFFireSpreadRuntime::collective_restore_from_io_rank_state(
    ERFFireSpreadRuntimeState state)
{
    const int io_rank =
        amrex::ParallelDescriptor::IOProcessorNumber();

    unsigned long long component_count =
        amrex::ParallelDescriptor::IOProcessor()
            ? static_cast<unsigned long long>(
                state.front_components.size())
            : 0ULL;
    amrex::ParallelDescriptor::Bcast(
        &component_count,
        1,
        io_rank);

    if (component_count
        > static_cast<unsigned long long>(
            std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error(
            "collective Fire restore component count is not representable");
    }

    if (component_count != 0ULL) {
        const std::size_t count =
            static_cast<std::size_t>(
                component_count);

        std::vector<int> roles(count);
        std::vector<unsigned long long>
            vertex_counts(count);

        if (amrex::ParallelDescriptor::IOProcessor()) {
            for (std::size_t component = 0;
                 component < count;
                 ++component) {
                roles[component] =
                    static_cast<int>(
                        state.front_components[
                            component].role);
                vertex_counts[component] =
                    static_cast<unsigned long long>(
                        state.front_components[
                            component]
                            .vertices_m.size());
            }
        }

        amrex::ParallelDescriptor::Bcast(
            roles.data(),
            roles.size(),
            io_rank);
        amrex::ParallelDescriptor::Bcast(
            vertex_counts.data(),
            vertex_counts.size(),
            io_rank);

        std::size_t total_vertices = 0;
        for (std::size_t component = 0;
             component < count;
             ++component) {
            if (roles[component]
                    != static_cast<int>(
                        FireFrontRole::Outer)
                && roles[component]
                    != static_cast<int>(
                        FireFrontRole::Hole)) {
                throw std::invalid_argument(
                    "collective Fire restore front role is invalid");
            }

            if (vertex_counts[component]
                > static_cast<unsigned long long>(
                    std::numeric_limits<std::size_t>::max()
                    - total_vertices)) {
                throw std::overflow_error(
                    "collective Fire restore front vertex count is not representable");
            }

            total_vertices +=
                static_cast<std::size_t>(
                    vertex_counts[component]);
        }

        if (total_vertices
            > std::numeric_limits<std::size_t>::max()
                / 2) {
            throw std::overflow_error(
                "collective Fire restore packed front size is not representable");
        }

        std::vector<amrex::Real>
            packed_vertices(
                2 * total_vertices);

        if (amrex::ParallelDescriptor::IOProcessor()) {
            std::size_t offset = 0;
            for (std::size_t component = 0;
                 component < count;
                 ++component) {
                const auto& vertices =
                    state.front_components[
                        component].vertices_m;

                for (const FireVec2& vertex :
                     vertices) {
                    packed_vertices[
                        2 * offset] =
                            vertex.x;
                    packed_vertices[
                        2 * offset + 1] =
                            vertex.y;
                    ++offset;
                }
            }
        }

        if (!packed_vertices.empty()) {
            amrex::ParallelDescriptor::Bcast(
                packed_vertices.data(),
                packed_vertices.size(),
                io_rank);
        }

        if (!amrex::ParallelDescriptor::IOProcessor()) {
            state.front_components.resize(
                count);

            std::size_t offset = 0;
            for (std::size_t component = 0;
                 component < count;
                 ++component) {
                auto& destination =
                    state.front_components[
                        component];

                destination.role =
                    static_cast<FireFrontRole>(
                        roles[component]);

                const std::size_t vertex_count =
                    static_cast<std::size_t>(
                        vertex_counts[component]);
                destination.vertices_m.resize(
                    vertex_count);

                for (std::size_t index = 0;
                     index < vertex_count;
                     ++index) {
                    destination.vertices_m[index] = {
                        packed_vertices[
                            2 * offset],
                        packed_vertices[
                            2 * offset + 1]
                    };
                    ++offset;
                }
            }
        }
    } else {
        unsigned long long vertex_count =
            amrex::ParallelDescriptor::IOProcessor()
                ? static_cast<unsigned long long>(
                    state.perimeter_vertices_m.size())
                : 0ULL;
        amrex::ParallelDescriptor::Bcast(
            &vertex_count,
            1,
            io_rank);

        if (vertex_count
            > static_cast<unsigned long long>(
                std::numeric_limits<std::size_t>::max()
                / 2)) {
            throw std::overflow_error(
                "collective Fire restore perimeter size is not representable");
        }

        const std::size_t count =
            static_cast<std::size_t>(
                vertex_count);
        std::vector<amrex::Real>
            packed_vertices(2 * count);

        if (amrex::ParallelDescriptor::IOProcessor()) {
            for (std::size_t index = 0;
                 index < count;
                 ++index) {
                packed_vertices[2 * index] =
                    state.perimeter_vertices_m[
                        index].x;
                packed_vertices[2 * index + 1] =
                    state.perimeter_vertices_m[
                        index].y;
            }
        }

        if (!packed_vertices.empty()) {
            amrex::ParallelDescriptor::Bcast(
                packed_vertices.data(),
                packed_vertices.size(),
                io_rank);
        }

        if (!amrex::ParallelDescriptor::IOProcessor()) {
            state.perimeter_vertices_m.resize(
                count);
            for (std::size_t index = 0;
                 index < count;
                 ++index) {
                state.perimeter_vertices_m[index] = {
                    packed_vertices[2 * index],
                    packed_vertices[
                        2 * index + 1]
                };
            }
        }
    }

    int inconsistent_history = 0;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        if (state.burned_fraction.burned_fraction.size()
            == state.combustion.cells.size()) {
            for (std::size_t index = 0;
                 index < state.burned_fraction.burned_fraction.size();
                 ++index) {
                if (!fraction_equal(
                        state.burned_fraction.burned_fraction[index],
                        state.combustion.cells[index].ignited_area_fraction)) {
                    inconsistent_history = 1;
                    break;
                }
            }
        }
    }
    amrex::ParallelDescriptor::Bcast(
        &inconsistent_history, 1, io_rank);
    if (inconsistent_history != 0) {
        throw std::invalid_argument(
            "restored fire combustion history is not synchronized with burned fraction");
    }

    FireFront front =
        restore_front_from_state(state);
    FireBurnedFractionRaster burned =
        FireBurnedFractionRaster::collective_restore_from_io_rank_state(
            state.config.raster_geometry,
            state.burned_fraction);
    FireFirstArrivalRaster arrival =
        FireFirstArrivalRaster::collective_restore_from_io_rank_state(
            state.config.raster_geometry,
            state.first_arrival);
    FireCombustionRaster combustion =
        FireCombustionRaster::collective_restore_from_io_rank_state(
            state.config.raster_geometry,
            state.config.combustion_parameters,
            state.config.combustion_options,
            state.combustion);

    return ERFFireSpreadRuntime(
        std::move(state.config),
        std::move(front),
        std::move(burned),
        std::move(arrival),
        std::move(combustion),
        state.current_time_s,
        CollectiveRestoreStateTag{});
}

ERFFireSpreadRuntime
ERFFireSpreadRuntime::collective_restore_from_checkpoint_raster(
    ERFFireSpreadRuntimeState state,
    const amrex::MultiFab& checkpoint_raster)
{
    collective_broadcast_front_state(state);

    if (checkpoint_raster.nComp()
            != ERFFireCheckpointRasterComponents::component_count
        || checkpoint_raster.nGrow() != 0) {
        throw std::invalid_argument(
            "ERF-Fire distributed checkpoint raster has incompatible components or ghosts");
    }

    FireFront front =
        restore_front_from_state(state);
    FireBurnedFractionRaster burned =
        FireBurnedFractionRaster::
            collective_restore_from_checkpoint_raster(
                state.config.raster_geometry,
                checkpoint_raster,
                ERFFireCheckpointRasterComponents::burned_fraction);
    FireFirstArrivalRaster arrival =
        FireFirstArrivalRaster::
            collective_restore_from_checkpoint_raster(
                state.config.raster_geometry,
                checkpoint_raster,
                ERFFireCheckpointRasterComponents::arrived,
                ERFFireCheckpointRasterComponents::first_arrival_time_s,
                state.first_arrival);
    FireCombustionRaster combustion =
        FireCombustionRaster::
            collective_restore_from_checkpoint_raster(
                state.config.raster_geometry,
                state.config.combustion_parameters,
                state.config.combustion_options,
                checkpoint_raster,
                ERFFireCheckpointRasterComponents::ignited_area_fraction,
                state.combustion.initialized);

    const auto& burned_values =
        burned.distributed_burned_fraction();
    const auto& combustion_values =
        combustion.distributed_states();
    if (burned_values.boxArray()
            != combustion_values.boxArray()
        || burned_values.DistributionMap()
            != combustion_values.DistributionMap()) {
        throw std::logic_error(
            "restored Fire burn and combustion layouts are not co-located");
    }

#ifdef AMREX_USE_GPU
    amrex::MFInfo combustion_host_info;
    combustion_host_info.SetArena(
        amrex::The_Pinned_Arena());
    amrex::MultiFab combustion_host(
        combustion_values.boxArray(),
        combustion_values.DistributionMap(),
        1,
        0,
        combustion_host_info);
    combustion_host.ParallelCopy(
        combustion_values,
        FireCombustionRaster::
            ignited_area_fraction_comp,
        0,
        1,
        0,
        0);
    amrex::Gpu::streamSynchronize();
    const amrex::MultiFab& combustion_check =
        combustion_host;
#else
    const amrex::MultiFab& combustion_check =
        combustion_values;
#endif

    int inconsistent_history = 0;
    for (amrex::MFIter mfi(burned_values);
         mfi.isValid();
         ++mfi) {
        const amrex::Box& box = mfi.validbox();
        const auto burned_array =
            burned_values.const_array(mfi);
        const auto combustion_array =
            combustion_check.const_array(mfi);
        for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
            for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
                if (!fraction_equal(
                        burned_array(i, j, 0),
                        combustion_array(
                            i,
                            j,
                            0,
                            0))) {
                    inconsistent_history = 1;
                }
            }
        }
    }
    amrex::ParallelDescriptor::ReduceIntMax(
        inconsistent_history);
    if (inconsistent_history != 0) {
        throw std::invalid_argument(
            "restored fire combustion history is not synchronized with burned fraction");
    }

    return ERFFireSpreadRuntime(
        std::move(state.config),
        std::move(front),
        std::move(burned),
        std::move(arrival),
        std::move(combustion),
        state.current_time_s,
        CollectiveRestoreStateTag{});
}

ERFFireStepDiagnostics
ERFFireSpreadRuntime::advance_direct_reference_wind(
    const FireFlatEnvironmentSampler& environment,
    amrex::Real dt_s)
{
    return advance_wind_impl(
        environment,
        nullptr,
        WindInputMode::DirectReference,
        amrex::Real(1.0),
        dt_s);
}

ERFFireStepDiagnostics
ERFFireSpreadRuntime::advance_direct_reference_wind(
    const FireFlatEnvironmentSampler& environment,
    const FireTerrainSurface& terrain,
    amrex::Real dt_s)
{
    require_terrain_runtime_geometry(
        environment,
        terrain,
        config_.raster_geometry);

    return advance_wind_impl(
        environment,
        &terrain,
        WindInputMode::DirectReference,
        amrex::Real(1.0),
        dt_s);
}

ERFFireStepDiagnostics
ERFFireSpreadRuntime::advance_explicit_waf_20ft(
    const FireFlatEnvironmentSampler& environment,
    amrex::Real wind_adjustment_factor,
    amrex::Real dt_s)
{
    return advance_wind_impl(
        environment,
        nullptr,
        WindInputMode::ExplicitWaf20ft,
        wind_adjustment_factor,
        dt_s);
}

ERFFireStepDiagnostics
ERFFireSpreadRuntime::advance_explicit_waf_20ft(
    const FireFlatEnvironmentSampler& environment,
    const FireTerrainSurface& terrain,
    amrex::Real wind_adjustment_factor,
    amrex::Real dt_s)
{
    require_terrain_runtime_geometry(
        environment,
        terrain,
        config_.raster_geometry);

    return advance_wind_impl(
        environment,
        &terrain,
        WindInputMode::ExplicitWaf20ft,
        wind_adjustment_factor,
        dt_s);
}

ERFFireStepDiagnostics
ERFFireSpreadRuntime::advance_wind_impl(
    const FireFlatEnvironmentSampler& environment,
    const FireTerrainSurface* terrain_surface,
    WindInputMode wind_input_mode,
    amrex::Real wind_adjustment_factor,
    amrex::Real dt_s)
{
    if (wind_input_mode == WindInputMode::ExplicitWaf20ft) {
        require(
            std::isfinite(wind_adjustment_factor)
                && wind_adjustment_factor >= amrex::Real(0.0)
                && wind_adjustment_factor <= amrex::Real(1.0),
            "fire explicit 20-ft wind adjustment factor must be finite and in [0,1]");
    }

    require(
        std::isfinite(dt_s) && dt_s > amrex::Real(0.0),
        "fire spread dt must be finite and positive");
    require(
        config_.arrival_time_tolerance_s <= dt_s,
        "fire spread arrival tolerance must not exceed the fire dt");

    const amrex::Real start_time_s = current_time_s_;
    const amrex::Real end_time_s = start_time_s + dt_s;
    if (!std::isfinite(end_time_s) || !(end_time_s > start_time_s)) {
        throw std::overflow_error(
            "fire spread end time must be finite and representably later");
    }

    const auto normal_speed =
        [this,
         &environment,
         terrain_surface,
         wind_input_mode,
         wind_adjustment_factor](
            const FireVec2& position_m,
            const FireVec2& outward_normal,
            amrex::Real) -> amrex::Real
    {
        const FireEnvironmentSample sample =
            environment.sample(position_m.x, position_m.y);
        const FireVec2 reference_wind_mps =
            sample.horizontal_wind_mps;

        if (!std::isfinite(reference_wind_mps.x)
            || !std::isfinite(reference_wind_mps.y)) {
            throw std::overflow_error(
                "fire spread sampled reference wind is not finite");
        }

        const FireVec2 wind_mps =
            wind_input_mode == WindInputMode::ExplicitWaf20ft
                ? fire_midflame_wind_from_20ft_reference(
                    reference_wind_mps,
                    wind_adjustment_factor)
                : reference_wind_mps;

        const amrex::Real speed_mps = norm(wind_mps);
        if (!std::isfinite(speed_mps)) {
            throw std::overflow_error(
                "fire spread model wind magnitude is not finite");
        }

        FireVec2 terrain_gradient_m_per_m{};
        if (terrain_surface != nullptr) {
            terrain_gradient_m_per_m =
                terrain_surface->terrain_gradient_m_per_m(
                    position_m.x,
                    position_m.y);
        }

        const amrex::Real slope_tangent =
            norm(terrain_gradient_m_per_m);
        if (!std::isfinite(slope_tangent)) {
            throw std::overflow_error(
                "fire spread sampled terrain slope magnitude is not finite");
        }

        const FireVec2 upslope_unit =
            terrain_upslope_unit(
                terrain_gradient_m_per_m,
                slope_tangent);

        const RothermelResult behavior =
            evaluate_rothermel(
                config_.fuel,
                RothermelInputs{
                    config_.dead_fuel_moisture_fraction,
                    speed_mps,
                    slope_tangent});

        const RichardsDirectionalSpread spread =
            make_richards_directional_spread(
                behavior,
                wind_push_unit(
                    wind_mps,
                    speed_mps),
                upslope_unit);

        return richards_normal_speed_mps(
            spread.ellipse, outward_normal);
    };

    FirePerimeter advanced =
        advance_perimeter_rk2(
            perimeter(), start_time_s, dt_s, normal_speed);

    // RK2 samples current and midpoint locations. Explicitly reject a final
    // perimeter outside the supported physical environment before committing
    // history, since the endpoint is not itself an RK sampling location.
    require_perimeter_inside_environment(advanced, environment);

    FireFirstArrivalRaster next_arrival = first_arrival_;
    const FireFirstArrivalRasterUpdate arrival_update =
        next_arrival.update_from_sweep(
            perimeter(),
            advanced,
            start_time_s,
            end_time_s,
            config_.arrival_time_tolerance_s);

    FireBurnedFractionRaster next_burned = burned_fraction_;
    const FireRasterBurnedAreaUpdate burned_update =
        next_burned.update_from_linear_sweep(
            perimeter(),
            advanced,
            config_.combustion_options.temporal_substeps);

    FireCombustionRaster next_combustion = combustion_;
    const FireCombustionRasterAdvance combustion_update =
        next_combustion.advance_from_linear_sweep(
            perimeter(),
            advanced,
            burned_fraction_,
            next_burned,
            dt_s);

    const std::size_t pre_remesh_vertex_count = advanced.size();
    FirePerimeterRemeshResult remeshed =
        remesh_perimeter(advanced, config_.remesh_options);

    ERFFireStepDiagnostics diagnostics{
        start_time_s,
        end_time_s,
        pre_remesh_vertex_count,
        remeshed.perimeter.size(),
        remeshed.stats.vertices_removed,
        remeshed.stats.vertices_added,
        arrival_update.newly_arrived_cell_count,
        arrival_update.arrived_cell_count,
        burned_update.newly_burned_area_m2,
        burned_update.burned_area_m2,
        combustion_update.newly_consumed_dry_fuel_kg,
        combustion_update.totals.remaining_dry_fuel_kg,
        combustion_update.totals.consumed_dry_fuel_kg,
        combustion_update.sensible_energy_increment_j,
        combustion_update.totals.sensible_energy_j,
        combustion_update.water_released_increment_kg,
        combustion_update.totals.water_released_kg};

    front_ =
        FireFront(
            std::vector<FireFrontComponent>{
                {
                    FireFrontRole::Outer,
                    std::move(remeshed.perimeter)
                }
            });
    first_arrival_ = std::move(next_arrival);
    burned_fraction_ = std::move(next_burned);
    combustion_ = std::move(next_combustion);
    current_time_s_ = end_time_s;

    return diagnostics;
}

ERFFireStepDiagnostics
ERFFireSpreadRuntime::advance_direct_reference_wind_batched(
    const FireEnvironmentBatchFunction& environment,
    amrex::Real dt_s)
{
    return advance_wind_batched_impl(
        environment,
        WindInputMode::DirectReference,
        amrex::Real(1.0),
        dt_s);
}

ERFFireStepDiagnostics
ERFFireSpreadRuntime::advance_direct_reference_wind_batched(
    const FireEnvironmentBatchFunction& environment,
    const FireTerrainSurface& terrain,
    amrex::Real dt_s)
{
    require(
        same_horizontal_geometry(
            terrain.geometry(),
            config_.raster_geometry),
        "fire terrain geometry must match the Fire runtime raster geometry");

    const FireEnvironmentBatchFunction terrain_environment =
        with_terrain_gradient(
            environment,
            terrain);
    return advance_wind_batched_impl(
        terrain_environment,
        WindInputMode::DirectReference,
        amrex::Real(1.0),
        dt_s);
}

ERFFireStepDiagnostics
ERFFireSpreadRuntime::advance_explicit_waf_20ft_batched(
    const FireEnvironmentBatchFunction& environment,
    amrex::Real wind_adjustment_factor,
    amrex::Real dt_s)
{
    return advance_wind_batched_impl(
        environment,
        WindInputMode::ExplicitWaf20ft,
        wind_adjustment_factor,
        dt_s);
}

ERFFireStepDiagnostics
ERFFireSpreadRuntime::advance_explicit_waf_20ft_batched(
    const FireEnvironmentBatchFunction& environment,
    const FireTerrainSurface& terrain,
    amrex::Real wind_adjustment_factor,
    amrex::Real dt_s)
{
    require(
        same_horizontal_geometry(
            terrain.geometry(),
            config_.raster_geometry),
        "fire terrain geometry must match the Fire runtime raster geometry");

    const FireEnvironmentBatchFunction terrain_environment =
        with_terrain_gradient(
            environment,
            terrain);
    return advance_wind_batched_impl(
        terrain_environment,
        WindInputMode::ExplicitWaf20ft,
        wind_adjustment_factor,
        dt_s);
}

ERFFireStepDiagnostics
ERFFireSpreadRuntime::advance_wind_batched_impl(
    const FireEnvironmentBatchFunction& environment,
    WindInputMode wind_input_mode,
    amrex::Real wind_adjustment_factor,
    amrex::Real dt_s)
{
    require(
        static_cast<bool>(environment),
        "fire batched environment sampler must be set");

    if (wind_input_mode == WindInputMode::ExplicitWaf20ft) {
        require(
            std::isfinite(wind_adjustment_factor)
                && wind_adjustment_factor >= amrex::Real(0.0)
                && wind_adjustment_factor <= amrex::Real(1.0),
            "fire explicit 20-ft wind adjustment factor must be finite and in [0,1]");
    }

    require(
        std::isfinite(dt_s) && dt_s > amrex::Real(0.0),
        "fire spread dt must be finite and positive");
    require(
        config_.arrival_time_tolerance_s <= dt_s,
        "fire spread arrival tolerance must not exceed the fire dt");

    const amrex::Real start_time_s = current_time_s_;
    const amrex::Real end_time_s = start_time_s + dt_s;
    if (!std::isfinite(end_time_s)
        || !(end_time_s > start_time_s)) {
        throw std::overflow_error(
            "fire spread end time must be finite and representably later");
    }

    const auto normal_speeds =
        [this,
         &environment,
         wind_input_mode,
         wind_adjustment_factor](
            const std::vector<FireVec2>& positions_m,
            const std::vector<FireVec2>& outward_normals,
            amrex::Real) -> std::vector<amrex::Real>
    {
        require(
            positions_m.size() == outward_normals.size(),
            "fire batched positions/normals size mismatch");

        const std::vector<FireEnvironmentSample> samples =
            environment(positions_m);
        require(
            samples.size() == positions_m.size(),
            "fire batched environment sampler returned the wrong size");

        std::vector<amrex::Real> speeds;
        speeds.reserve(samples.size());

        for (std::size_t index = 0;
             index < samples.size();
             ++index) {
            const FireVec2 reference_wind_mps =
                samples[index].horizontal_wind_mps;

            if (!std::isfinite(reference_wind_mps.x)
                || !std::isfinite(reference_wind_mps.y)) {
                throw std::overflow_error(
                    "fire spread sampled reference wind is not finite");
            }

            const FireVec2 wind_mps =
                wind_input_mode
                        == WindInputMode::ExplicitWaf20ft
                    ? fire_midflame_wind_from_20ft_reference(
                        reference_wind_mps,
                        wind_adjustment_factor)
                    : reference_wind_mps;

            const amrex::Real speed_mps =
                norm(wind_mps);
            if (!std::isfinite(speed_mps)) {
                throw std::overflow_error(
                    "fire spread model wind magnitude is not finite");
            }

            const FireVec2 terrain_gradient_m_per_m =
                samples[index].terrain_gradient_m_per_m;

            const amrex::Real slope_tangent =
                norm(terrain_gradient_m_per_m);
            if (!std::isfinite(slope_tangent)) {
                throw std::overflow_error(
                    "fire spread sampled terrain slope magnitude is not finite");
            }

            const FireVec2 upslope_unit =
                terrain_upslope_unit(
                    terrain_gradient_m_per_m,
                    slope_tangent);

            const RothermelResult behavior =
                evaluate_rothermel(
                    config_.fuel,
                    RothermelInputs{
                        config_.dead_fuel_moisture_fraction,
                        speed_mps,
                        slope_tangent});

            const RichardsDirectionalSpread spread =
                make_richards_directional_spread(
                    behavior,
                    wind_push_unit(
                        wind_mps,
                        speed_mps),
                    upslope_unit);

            speeds.push_back(
                richards_normal_speed_mps(
                    spread.ellipse,
                    outward_normals[index]));
        }

        return speeds;
    };

    const bool single_outer =
        front_.components().size() == 1
        && front_.components().front().role
            == FireFrontRole::Outer;

    if (!single_outer) {
        FireFront advanced_front =
            advance_front_rk2_batched(
                front_,
                start_time_s,
                dt_s,
                normal_speeds);

        const auto& geometry =
            config_.raster_geometry;
        const amrex::Real xhi_m =
            geometry.xlo_m
            + static_cast<amrex::Real>(
                geometry.nx)
                * geometry.dx_m;
        const amrex::Real yhi_m =
            geometry.ylo_m
            + static_cast<amrex::Real>(
                geometry.ny)
                * geometry.dy_m;

        for (const FireFrontComponent& component :
             advanced_front.components()) {
            for (const FireVec2& vertex :
                 component.perimeter.vertices_m()) {
                if (!std::isfinite(vertex.x)
                    || !std::isfinite(vertex.y)
                    || vertex.x < geometry.xlo_m
                    || vertex.x > xhi_m
                    || vertex.y < geometry.ylo_m
                    || vertex.y > yhi_m) {
                    throw std::out_of_range(
                        "fire spread propagated front leaves the physical environment domain");
                }
            }
        }

        FireFirstArrivalRaster next_arrival =
            first_arrival_;
        const FireFirstArrivalRasterUpdate
            arrival_update =
                next_arrival
                    .update_from_front_linear_sweep(
                        front_,
                        advanced_front,
                        start_time_s,
                        end_time_s,
                        config_
                            .arrival_time_tolerance_s,
                        config_
                            .combustion_options
                            .temporal_substeps);

        FireBurnedFractionRaster next_burned =
            burned_fraction_;
        const FireRasterBurnedAreaUpdate
            burned_update =
                next_burned
                    .update_from_front_linear_sweep(
                        front_,
                        advanced_front,
                        config_
                            .combustion_options
                            .temporal_substeps);

        FireCombustionRaster next_combustion =
            combustion_;
        const FireCombustionRasterAdvance
            combustion_update =
                next_combustion
                    .advance_from_front_linear_sweep(
                        front_,
                        advanced_front,
                        burned_fraction_,
                        next_burned,
                        dt_s);

        const auto vertex_count =
            [](const FireFront& front) {
                std::size_t count = 0;
                for (const FireFrontComponent& component :
                     front.components()) {
                    count +=
                        component.perimeter.size();
                }
                return count;
            };

        const std::size_t
            pre_remesh_vertex_count =
                vertex_count(advanced_front);

        FireFrontRemeshResult remeshed =
            remesh_front(
                advanced_front,
                config_.remesh_options);

        const std::size_t
            post_remesh_vertex_count =
                vertex_count(remeshed.front);

        ERFFireStepDiagnostics diagnostics{
            start_time_s,
            end_time_s,
            pre_remesh_vertex_count,
            post_remesh_vertex_count,
            remeshed.stats.vertices_removed,
            remeshed.stats.vertices_added,
            arrival_update
                .newly_arrived_cell_count,
            arrival_update.arrived_cell_count,
            burned_update.newly_burned_area_m2,
            burned_update.burned_area_m2,
            combustion_update
                .newly_consumed_dry_fuel_kg,
            combustion_update
                .totals.remaining_dry_fuel_kg,
            combustion_update
                .totals.consumed_dry_fuel_kg,
            combustion_update
                .sensible_energy_increment_j,
            combustion_update
                .totals.sensible_energy_j,
            combustion_update
                .water_released_increment_kg,
            combustion_update
                .totals.water_released_kg};

        front_ = std::move(remeshed.front);
        first_arrival_ =
            std::move(next_arrival);
        burned_fraction_ =
            std::move(next_burned);
        combustion_ =
            std::move(next_combustion);
        current_time_s_ =
            end_time_s;

        return diagnostics;
    }

    const FirePerimeter& start_perimeter =
        perimeter();

    FireFrontAdvanceResult topology_advance =
        advance_perimeter_rk2_batched_until_topology_event(
            start_perimeter,
            start_time_s,
            dt_s,
            normal_speeds);

    if (topology_advance.topology_event.has_value()) {
        const amrex::Real event_dt_s =
            topology_advance.advanced_dt_s;

        require(
            std::isfinite(event_dt_s)
                && event_dt_s > amrex::Real(0.0)
                && event_dt_s <= dt_s,
            "fire topology event consumed an invalid timestep");

        const amrex::Real event_time_s =
            start_time_s + event_dt_s;

        require(
            std::isfinite(event_time_s)
                && event_time_s > start_time_s
                && event_time_s <= end_time_s,
            "fire topology event time is invalid");

        const FireFront event_front =
            topology_advance.front;

        FireFirstArrivalRaster next_arrival =
            first_arrival_;
        const FireFirstArrivalRasterUpdate
            event_arrival_update =
                next_arrival
                    .update_from_topology_event_sweep(
                        start_perimeter,
                        topology_advance
                            .terminal_vertices_m,
                        event_front,
                        start_time_s,
                        event_time_s,
                        std::min(
                            config_
                                .arrival_time_tolerance_s,
                            event_dt_s));

        FireBurnedFractionRaster next_burned =
            burned_fraction_;
        const FireRasterBurnedAreaUpdate
            event_burned_update =
                next_burned
                    .update_from_topology_event_sweep(
                        start_perimeter,
                        topology_advance
                            .terminal_vertices_m,
                        event_front,
                        config_
                            .combustion_options
                            .temporal_substeps);

        FireCombustionRaster next_combustion =
            combustion_;
        const FireCombustionRasterAdvance
            event_combustion_update =
                next_combustion
                    .advance_from_topology_event_sweep(
                        start_perimeter,
                        topology_advance
                            .terminal_vertices_m,
                        event_front,
                        burned_fraction_,
                        next_burned,
                        event_dt_s);

        std::size_t newly_arrived_cell_count =
            event_arrival_update
                .newly_arrived_cell_count;
        std::size_t arrived_cell_count =
            event_arrival_update
                .arrived_cell_count;

        amrex::Real newly_burned_area_m2 =
            event_burned_update
                .newly_burned_area_m2;

        amrex::Real
            newly_consumed_dry_fuel_kg =
                event_combustion_update
                    .newly_consumed_dry_fuel_kg;
        amrex::Real sensible_energy_increment_j =
            event_combustion_update
                .sensible_energy_increment_j;
        amrex::Real
            water_released_increment_kg =
                event_combustion_update
                    .water_released_increment_kg;

        FireFront final_front =
            event_front;

        const amrex::Real remaining_dt_s =
            end_time_s - event_time_s;

        require(
            std::isfinite(remaining_dt_s)
                && remaining_dt_s
                    >= amrex::Real(0.0),
            "fire topology-event remainder is invalid");

        if (remaining_dt_s > amrex::Real(0.0)) {
            final_front =
                advance_front_rk2_batched(
                    event_front,
                    event_time_s,
                    remaining_dt_s,
                    normal_speeds);

            const FireFirstArrivalRasterUpdate
                remainder_arrival_update =
                    next_arrival
                        .update_from_front_linear_sweep(
                            event_front,
                            final_front,
                            event_time_s,
                            end_time_s,
                            std::min(
                                config_
                                    .arrival_time_tolerance_s,
                                remaining_dt_s),
                            config_
                                .combustion_options
                                .temporal_substeps);

            newly_arrived_cell_count +=
                remainder_arrival_update
                    .newly_arrived_cell_count;
            arrived_cell_count =
                remainder_arrival_update
                    .arrived_cell_count;

            const FireBurnedFractionRaster
                burned_at_event =
                    next_burned;

            const FireRasterBurnedAreaUpdate
                remainder_burned_update =
                    next_burned
                        .update_from_front_linear_sweep(
                            event_front,
                            final_front,
                            config_
                                .combustion_options
                                .temporal_substeps);

            newly_burned_area_m2 +=
                remainder_burned_update
                    .newly_burned_area_m2;

            const FireCombustionRasterAdvance
                remainder_combustion_update =
                    next_combustion
                        .advance_from_front_linear_sweep(
                            event_front,
                            final_front,
                            burned_at_event,
                            next_burned,
                            remaining_dt_s);

            newly_consumed_dry_fuel_kg +=
                remainder_combustion_update
                    .newly_consumed_dry_fuel_kg;
            sensible_energy_increment_j +=
                remainder_combustion_update
                    .sensible_energy_increment_j;
            water_released_increment_kg +=
                remainder_combustion_update
                    .water_released_increment_kg;
        }

        const auto& geometry =
            config_.raster_geometry;
        const amrex::Real xhi_m =
            geometry.xlo_m
            + static_cast<amrex::Real>(
                geometry.nx)
                * geometry.dx_m;
        const amrex::Real yhi_m =
            geometry.ylo_m
            + static_cast<amrex::Real>(
                geometry.ny)
                * geometry.dy_m;

        for (const FireFrontComponent& component :
             final_front.components()) {
            for (const FireVec2& vertex :
                 component.perimeter.vertices_m()) {
                if (!std::isfinite(vertex.x)
                    || !std::isfinite(vertex.y)
                    || vertex.x < geometry.xlo_m
                    || vertex.x > xhi_m
                    || vertex.y < geometry.ylo_m
                    || vertex.y > yhi_m) {
                    throw std::out_of_range(
                        "fire spread propagated front leaves the physical environment domain");
                }
            }
        }

        const auto vertex_count =
            [](const FireFront& front) {
                std::size_t count = 0;
                for (const FireFrontComponent& component :
                     front.components()) {
                    count +=
                        component.perimeter.size();
                }
                return count;
            };

        const std::size_t
            pre_remesh_vertex_count =
                vertex_count(final_front);

        FireFrontRemeshResult remeshed =
            remesh_front(
                final_front,
                config_.remesh_options);

        const std::size_t
            post_remesh_vertex_count =
                vertex_count(remeshed.front);

        const auto combustion_totals =
            next_combustion.totals();

        ERFFireStepDiagnostics diagnostics{
            start_time_s,
            end_time_s,
            pre_remesh_vertex_count,
            post_remesh_vertex_count,
            remeshed.stats.vertices_removed,
            remeshed.stats.vertices_added,
            newly_arrived_cell_count,
            arrived_cell_count,
            newly_burned_area_m2,
            next_burned.burned_area_m2(),
            newly_consumed_dry_fuel_kg,
            combustion_totals
                .remaining_dry_fuel_kg,
            combustion_totals
                .consumed_dry_fuel_kg,
            sensible_energy_increment_j,
            combustion_totals
                .sensible_energy_j,
            water_released_increment_kg,
            combustion_totals
                .water_released_kg};

        front_ = std::move(remeshed.front);
        first_arrival_ =
            std::move(next_arrival);
        burned_fraction_ =
            std::move(next_burned);
        combustion_ =
            std::move(next_combustion);
        current_time_s_ =
            end_time_s;

        return diagnostics;
    }

    const FirePerimeter& advanced =
        topology_advance.front
            .components()
            .front()
            .perimeter;

    const auto& geometry = config_.raster_geometry;
    const amrex::Real xhi_m =
        geometry.xlo_m
        + static_cast<amrex::Real>(geometry.nx)
            * geometry.dx_m;
    const amrex::Real yhi_m =
        geometry.ylo_m
        + static_cast<amrex::Real>(geometry.ny)
            * geometry.dy_m;
    for (const FireVec2& vertex : advanced.vertices_m()) {
        if (!std::isfinite(vertex.x)
            || !std::isfinite(vertex.y)
            || vertex.x < geometry.xlo_m
            || vertex.x > xhi_m
            || vertex.y < geometry.ylo_m
            || vertex.y > yhi_m) {
            throw std::out_of_range(
                "fire spread propagated perimeter leaves the physical environment domain");
        }
    }

    FireFirstArrivalRaster next_arrival = first_arrival_;
    const FireFirstArrivalRasterUpdate arrival_update =
        next_arrival.update_from_sweep(
            perimeter(),
            advanced,
            start_time_s,
            end_time_s,
            config_.arrival_time_tolerance_s);

    FireBurnedFractionRaster next_burned = burned_fraction_;
    const FireRasterBurnedAreaUpdate burned_update =
        next_burned.update_from_linear_sweep(
            perimeter(),
            advanced,
            config_.combustion_options.temporal_substeps);

    FireCombustionRaster next_combustion = combustion_;
    const FireCombustionRasterAdvance combustion_update =
        next_combustion.advance_from_linear_sweep(
            perimeter(),
            advanced,
            burned_fraction_,
            next_burned,
            dt_s);

    const std::size_t pre_remesh_vertex_count =
        advanced.size();
    FirePerimeterRemeshResult remeshed =
        remesh_perimeter(
            advanced,
            config_.remesh_options);

    ERFFireStepDiagnostics diagnostics{
        start_time_s,
        end_time_s,
        pre_remesh_vertex_count,
        remeshed.perimeter.size(),
        remeshed.stats.vertices_removed,
        remeshed.stats.vertices_added,
        arrival_update.newly_arrived_cell_count,
        arrival_update.arrived_cell_count,
        burned_update.newly_burned_area_m2,
        burned_update.burned_area_m2,
        combustion_update.newly_consumed_dry_fuel_kg,
        combustion_update.totals.remaining_dry_fuel_kg,
        combustion_update.totals.consumed_dry_fuel_kg,
        combustion_update.sensible_energy_increment_j,
        combustion_update.totals.sensible_energy_j,
        combustion_update.water_released_increment_kg,
        combustion_update.totals.water_released_kg};

    front_ =
        FireFront(
            std::vector<FireFrontComponent>{
                {
                    FireFrontRole::Outer,
                    std::move(remeshed.perimeter)
                }
            });
    first_arrival_ = std::move(next_arrival);
    burned_fraction_ = std::move(next_burned);
    combustion_ = std::move(next_combustion);
    current_time_s_ = end_time_s;

    return diagnostics;
}

} // namespace ERFFire
