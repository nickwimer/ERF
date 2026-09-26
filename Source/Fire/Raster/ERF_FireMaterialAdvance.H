#ifndef ERF_FIRE_MATERIAL_ADVANCE_H_
#define ERF_FIRE_MATERIAL_ADVANCE_H_

#include <ERF_FireFuelBarrierSweep.H>
#include <ERF_FireFuelSpread.H>
#include <ERF_RothermelModel.H>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>
#include <utility>

namespace ERFFire
{
namespace detail
{

inline bool same_fire_spread_material(const FireFuelRasterCell& a,
                                      const FireFuelRasterCell& b)
{
    if (a.model_id != b.model_id) { return false; }
    const auto contract = fire_fuel_model_moisture_contract(a.model_id);
    if (!contract.valid_model) {
        throw std::invalid_argument("Fire material event has an invalid model");
    }
    for (int i = 0; i < FireFuelMoisture::component_count; ++i) {
        const auto c = static_cast<FireFuelMoistureClass>(i);
        if ((contract.required_mask & fire_fuel_moisture_component_bit(c)) == 0u) {
            continue;
        }
        amrex::Real x{}, y{};
        if (!a.moisture.try_get(c, x) || !b.moisture.try_get(c, y)) {
            throw std::invalid_argument("Fire material event requires complete moisture");
        }
        if (x != y) { return false; }
    }
    return true;
}

inline bool fire_material_has_zero_spread(const FireFuelRasterCell& cell)
{
    if (cell.model_id == FireFuelModelId::NonBurnable) { return true; }
    if (cell.model_id == FireFuelModelId::FM1) {
        amrex::Real moisture{};
        if (!cell.moisture.try_get(FireFuelMoistureClass::Dead1h, moisture)) {
            throw std::invalid_argument("Fire material requires dead 1-h moisture");
        }
        return evaluate_rothermel(make_fm1_fuel_parameters(),
            RothermelInputs{moisture, amrex::Real(0), amrex::Real(0)})
            .no_wind_no_slope_ros_mps == amrex::Real(0);
    }
    const auto input = make_fire_fuel_spread_inputs(
        cell.model_id, cell.moisture, amrex::Real(0), amrex::Real(0));
    return evaluate_rothermel_multiclass(
        make_anderson13_fuel_parameters(input.anderson13_model_number),
        input.rothermel).no_wind_no_slope_ros_mps == amrex::Real(0);
}

// Snap only at roundoff scale. This is not a user-scale contact tolerance.
inline amrex::Real fire_material_snap_axis(amrex::Real x, amrex::Real lo,
                                           amrex::Real h, std::size_t count)
{
    const amrex::Real index = std::round((x - lo) / h);
    if (index < amrex::Real(0) || index > static_cast<amrex::Real>(count)) { return x; }
    const amrex::Real face = lo + index*h;
    const amrex::Real scale = std::max(h, std::max(std::abs(x), std::abs(face)));
    const amrex::Real tolerance = std::min(h * amrex::Real(1.e-4),
        amrex::Real(32) * std::numeric_limits<amrex::Real>::epsilon() * scale);
    return std::abs(x - face) <= tolerance ? face : x;
}

inline FireVec2 fire_material_snap(const FireVec2& p,
                                   const FireCartesianRasterGeometry2D& g)
{
    return {fire_material_snap_axis(p.x, g.xlo_m, g.dx_m, g.nx),
            fire_material_snap_axis(p.y, g.ylo_m, g.dy_m, g.ny)};
}

/** First change along every complete motion segment, not just RK samples. */
inline amrex::Real first_fire_material_change(
    const FireFuelRaster& raster,
    const std::vector<FireVec2>& starts,
    const std::vector<FireVec2>& ends,
    const std::vector<FireFuelRasterCell>& initial)
{
    if (starts.size() != ends.size() || starts.size() != initial.size()) {
        throw std::logic_error("Fire material motion batches differ in size");
    }
    std::vector<FireVec2> queries;
    std::vector<FireFuelBarrierInterval> intervals;
    for (std::size_t i = 0; i < starts.size(); ++i) {
        append_fire_fuel_barrier_segment_intervals(
            raster.geometry(), i, starts[i], ends[i], queries, intervals);
    }
    const auto materials = raster.collective_sample_points(queries);
    if (materials.size() != queries.size()) {
        throw std::logic_error("Fire material traversal returned the wrong batch size");
    }
    amrex::Real first = amrex::Real(1);
    for (const auto& interval : intervals) {
        // A path on a shared face may continue on its current material side.
        bool same_side = false;
        for (std::size_t n = 0; n < interval.sample_count; ++n) {
            same_side = same_side || same_fire_spread_material(
                initial[interval.segment_index], materials[interval.sample_begin + n]);
        }
        if (!same_side) { first = std::min(first, interval.entry_fraction); }
    }
    return first;
}

inline FireVec2 fire_material_rk2_dense_point(
    const FireVec2& start,
    const FireVec2& midpoint_stage,
    const FireVec2& end,
    amrex::Real alpha)
{
    return start
        + (amrex::Real(2) * alpha * (amrex::Real(1) - alpha))
            * (midpoint_stage - start)
        + (alpha * alpha) * (end - start);
}

inline std::size_t fire_material_rk2_dense_substeps(
    const std::vector<FireVec2>& starts,
    const std::vector<FireVec2>& midpoint_stage,
    const std::vector<FireVec2>& ends,
    const FireCartesianRasterGeometry2D& g)
{
    if (starts.size() != midpoint_stage.size()
        || starts.size() != ends.size()) {
        throw std::logic_error("Fire RK2 material path batches differ in size");
    }
    amrex::Real max_quadratic = amrex::Real(0);
    for (std::size_t i = 0; i < starts.size(); ++i) {
        const FireVec2 quadratic =
            (ends[i] - starts[i])
            - amrex::Real(2) * (midpoint_stage[i] - starts[i]);
        max_quadratic = std::max(
            max_quadratic,
            std::hypot(quadratic.x, quadratic.y));
    }
    const amrex::Real tolerance =
        amrex::Real(1.e-4) * std::min(g.dx_m, g.dy_m);
    if (!(tolerance > amrex::Real(0))) {
        throw std::logic_error("Fire RK2 material path has invalid raster spacing");
    }
    const amrex::Real required =
        max_quadratic > amrex::Real(0)
        ? std::sqrt(max_quadratic / (amrex::Real(4) * tolerance))
        : amrex::Real(1);
    const std::size_t substeps = std::max<std::size_t>(
        1, static_cast<std::size_t>(std::ceil(required)));
    if (substeps > 2048) {
        throw std::runtime_error(
            "Fire RK2 material trajectory is too curved for raster-safe event localization; reduce the timestep");
    }
    return substeps;
}

inline amrex::Real first_fire_material_change_rk2(
    const FireFuelRaster& raster,
    const std::vector<FireVec2>& starts,
    const std::vector<FireVec2>& midpoint_stage,
    const std::vector<FireVec2>& ends,
    const std::vector<FireFuelRasterCell>& initial)
{
    if (starts.size() != initial.size()) {
        throw std::logic_error("Fire RK2 material path/material batches differ in size");
    }
    const std::size_t substeps =
        fire_material_rk2_dense_substeps(
            starts, midpoint_stage, ends, raster.geometry());

    std::vector<FireVec2> queries;
    std::vector<FireFuelBarrierInterval> intervals;
    std::vector<std::size_t> segment_vertex;
    std::vector<std::size_t> segment_step;
    segment_vertex.reserve(starts.size() * substeps);
    segment_step.reserve(starts.size() * substeps);

    for (std::size_t step = 0; step < substeps; ++step) {
        const amrex::Real a0 =
            static_cast<amrex::Real>(step)
            / static_cast<amrex::Real>(substeps);
        const amrex::Real a1 =
            static_cast<amrex::Real>(step + 1)
            / static_cast<amrex::Real>(substeps);
        for (std::size_t i = 0; i < starts.size(); ++i) {
            const FireVec2 p0 = fire_material_rk2_dense_point(
                starts[i], midpoint_stage[i], ends[i], a0);
            const FireVec2 p1 = fire_material_rk2_dense_point(
                starts[i], midpoint_stage[i], ends[i], a1);
            const std::size_t segment = segment_vertex.size();
            segment_vertex.push_back(i);
            segment_step.push_back(step);
            append_fire_fuel_barrier_segment_intervals(
                raster.geometry(), segment, p0, p1, queries, intervals);
        }
    }

    const auto materials = raster.collective_sample_points(queries);
    if (materials.size() != queries.size()) {
        throw std::logic_error("Fire RK2 material traversal returned the wrong batch size");
    }

    amrex::Real first = amrex::Real(1);
    for (const auto& interval : intervals) {
        const std::size_t segment = interval.segment_index;
        const std::size_t vertex = segment_vertex[segment];
        bool same_side = false;
        for (std::size_t n = 0; n < interval.sample_count; ++n) {
            same_side = same_side || same_fire_spread_material(
                initial[vertex],
                materials[interval.sample_begin + n]);
        }
        if (!same_side) {
            const amrex::Real alpha =
                (static_cast<amrex::Real>(segment_step[segment])
                 + interval.entry_fraction)
                / static_cast<amrex::Real>(substeps);
            first = std::min(first, alpha);
        }
    }
    return first;
}

inline void require_fire_material_sweep_not_extinguished(
    const FireFront& start,
    const std::vector<std::vector<FireVec2>>& end,
    const FireFuelRaster& raster)
{
    std::vector<FireVec2> queries;
    std::vector<FireCartesianRasterIndex2D> cells;
    std::vector<FireFuelBarrierSweptEnvelope> envelopes;
    const auto& components = start.components();
    if (components.size() != end.size()) {
        throw std::logic_error("Fire material sweep lost components");
    }
    for (std::size_t c = 0; c < components.size(); ++c) {
        const auto& a = components[c].perimeter.vertices_m();
        if (a.size() != end[c].size()) {
            throw std::logic_error("Fire material sweep lost vertices");
        }
        for (std::size_t i = 0; i < a.size(); ++i) {
            const std::size_t j = (i + 1) % a.size();
            append_fire_barrier_swept_envelope_samples(raster,
                {c, i, components[c].role}, a[i], a[j], end[c][i], end[c][j],
                queries, cells, envelopes);
        }
    }
    if (envelopes.empty()) { return; }
    const auto materials = raster.collective_sample_points(queries);
    if (materials.size() != queries.size()) {
        throw std::logic_error("Fire material sweep returned the wrong batch size");
    }
    for (const auto& envelope : envelopes) {
        for (std::size_t i = 0; i < envelope.sample_count; ++i) {
            const std::size_t q = envelope.sample_begin + i;
            if (!fire_material_has_zero_spread(materials[q])) { continue; }
            const auto cell = fire_cartesian_raster_cell_bounds(
                raster.geometry(), cells[q].i, cells[q].j);
            const amrex::Real area = fire_perimeter_cell_intersection_area_m2(
                envelope.perimeter, cell);
            const amrex::Real tolerance = amrex::Real(1024)
                * std::numeric_limits<amrex::Real>::epsilon()
                * fire_cartesian_cell_area_m2(cell);
            if (area > tolerance) {
                throw std::runtime_error(
                    "Fire edge sweep enters zero-spread material; refine the front or timestep. "
                    "Barrier sliding/rerouting is not implemented");
            }
        }
    }
}

inline void require_fire_material_rk2_sweep_not_extinguished(
    const FireFront& start,
    const std::vector<std::vector<FireVec2>>& midpoint_stage,
    const std::vector<std::vector<FireVec2>>& end,
    const FireFuelRaster& raster)
{
    std::vector<std::vector<FireVec2>> start_nested;
    start_nested.reserve(start.components().size());
    for (const auto& component : start.components()) {
        start_nested.push_back(component.perimeter.vertices_m());
    }
    const auto starts =
        fire_barrier_flatten_component_vertices(start_nested);
    const auto mids =
        fire_barrier_flatten_component_vertices(midpoint_stage);
    const auto ends =
        fire_barrier_flatten_component_vertices(end);
    const std::size_t substeps =
        fire_material_rk2_dense_substeps(
            starts, mids, ends, raster.geometry());

    std::vector<FireVec2> queries;
    std::vector<FireCartesianRasterIndex2D> cells;
    std::vector<FireFuelBarrierSweptEnvelope> envelopes;

    const auto& components = start.components();
    std::vector<std::size_t> component_offsets;
    component_offsets.reserve(components.size());
    std::size_t total = 0;
    for (const auto& component : components) {
        component_offsets.push_back(total);
        total += component.perimeter.size();
    }

    for (std::size_t step = 0; step < substeps; ++step) {
        const amrex::Real a0 =
            static_cast<amrex::Real>(step)
            / static_cast<amrex::Real>(substeps);
        const amrex::Real a1 =
            static_cast<amrex::Real>(step + 1)
            / static_cast<amrex::Real>(substeps);

        for (std::size_t cidx = 0; cidx < components.size(); ++cidx) {
            const std::size_t count = components[cidx].perimeter.size();
            const std::size_t offset = component_offsets[cidx];
            for (std::size_t i = 0; i < count; ++i) {
                const std::size_t j = (i + 1) % count;
                const FireVec2 a0i = fire_material_rk2_dense_point(
                    starts[offset+i], mids[offset+i], ends[offset+i], a0);
                const FireVec2 a0j = fire_material_rk2_dense_point(
                    starts[offset+j], mids[offset+j], ends[offset+j], a0);
                const FireVec2 a1i = fire_material_rk2_dense_point(
                    starts[offset+i], mids[offset+i], ends[offset+i], a1);
                const FireVec2 a1j = fire_material_rk2_dense_point(
                    starts[offset+j], mids[offset+j], ends[offset+j], a1);
                append_fire_barrier_swept_envelope_samples(
                    raster,
                    {cidx, i, components[cidx].role},
                    a0i, a0j, a1i, a1j,
                    queries, cells, envelopes);
            }
        }
    }

    if (envelopes.empty()) { return; }
    const auto materials = raster.collective_sample_points(queries);
    if (materials.size() != queries.size()) {
        throw std::logic_error("Fire RK2 material sweep returned the wrong batch size");
    }
    for (const auto& envelope : envelopes) {
        for (std::size_t i = 0; i < envelope.sample_count; ++i) {
            const std::size_t q = envelope.sample_begin + i;
            if (!fire_material_has_zero_spread(materials[q])) { continue; }
            const auto cell = fire_cartesian_raster_cell_bounds(
                raster.geometry(), cells[q].i, cells[q].j);
            const amrex::Real area = fire_perimeter_cell_intersection_area_m2(
                envelope.perimeter, cell);
            const amrex::Real tolerance_area = amrex::Real(1024)
                * std::numeric_limits<amrex::Real>::epsilon()
                * fire_cartesian_cell_area_m2(cell);
            if (area > tolerance_area) {
                throw std::runtime_error(
                    "Fire RK2 edge sweep enters zero-spread material; refine the front or timestep. "
                    "Barrier sliding/rerouting is not implemented");
            }
        }
    }
}

} // namespace detail

/**
 * Insert collinear vertices where an edge spans different materials. This
 * changes neither represented area nor topology. A midpoint in each crossed
 * cell interval prevents a thin wet region from lying only between vertices.
 * Homogeneous edges are retained exactly. All ranks supply the same front.
 */
[[nodiscard]] inline FireFront resolve_fire_front_material_edges(
    const FireFront& front, const FireFuelRaster& raster)
{
    struct Edge { std::size_t component, vertex, begin, count; };
    std::vector<Edge> edges;
    std::vector<FireVec2> queries;
    std::vector<detail::FireFuelBarrierInterval> intervals;
    const auto& components = front.components();
    for (std::size_t c = 0; c < components.size(); ++c) {
        const auto& v = components[c].perimeter.vertices_m();
        for (std::size_t i = 0; i < v.size(); ++i) {
            const auto begin = intervals.size();
            detail::append_fire_fuel_barrier_segment_intervals(raster.geometry(), edges.size(),
                v[i], v[(i+1)%v.size()], queries, intervals);
            edges.push_back({c, i, begin, intervals.size()-begin});
        }
    }
    const auto materials = raster.collective_sample_points(queries);
    if (materials.size() != queries.size()) {
        throw std::logic_error("Fire edge material query returned the wrong batch size");
    }
    std::vector<std::vector<FireVec2>> vertices(components.size());
    for (const auto& edge : edges) {
        const auto& v = components[edge.component].perimeter.vertices_m();
        const FireVec2 a = v[edge.vertex], b = v[(edge.vertex+1)%v.size()];
        auto& result = vertices[edge.component];
        result.push_back(a);
        const auto& reference = materials[intervals[edge.begin].sample_begin];
        bool heterogeneous = false;
        for (std::size_t k = 0; k < edge.count; ++k) {
            const auto& interval = intervals[edge.begin+k];
            for (std::size_t q = 0; q < interval.sample_count; ++q) {
                heterogeneous = heterogeneous || !detail::same_fire_spread_material(
                    reference, materials[interval.sample_begin+q]);
            }
        }
        if (!heterogeneous) { continue; }
        for (std::size_t k = 0; k < edge.count; ++k) {
            const amrex::Real lower = intervals[edge.begin+k].entry_fraction;
            const amrex::Real upper = k+1 < edge.count
                ? intervals[edge.begin+k+1].entry_fraction : amrex::Real(1);
            if (lower > amrex::Real(0)) {
                result.push_back(detail::fire_material_snap(a + lower*(b-a), raster.geometry()));
            }
            if (upper > lower) { result.push_back(a + ((lower+upper)*amrex::Real(0.5))*(b-a)); }
        }
    }
    std::vector<FireFrontComponent> rebuilt;
    for (std::size_t c = 0; c < components.size(); ++c) {
        rebuilt.push_back({components[c].role, FirePerimeter(std::move(vertices[c]))});
    }
    return FireFront(std::move(rebuilt));
}

/**
 * RK2 with a material-local trial and event-limited history segment.
 *
 * The caller's speed callback must use stage_materials rather than resampling
 * material at trial positions. Wind/terrain may still be sampled there. The
 * complete trial trajectories are traversed through the raster; at a material
 * change the step is shortened and RK2 is recomputed on the original side.
 * The returned advanced_dt_s can be smaller than requested even without a
 * topology event. The caller MUST commit arrival/combustion only over that
 * duration, then continue the remainder using the receiving material.
 *
 * At an exact face, outgoing-normal ownership handles both propagation signs.
 * Zero-rate wet fuels remain fuels, not reclassified NonBurnable. Continuous
 * edge penetration into either zero-rate fuel or NonBurnable is rejected.
 * There is no claim of general obstacle sliding or discontinuous-Hamiltonian
 * convergence: unresolved corner/contact geometry fails before state commit.
 */
[[nodiscard]] inline FireFrontTopologyAdvanceResult
advance_fire_front_material_segment(
    const FireFront& front, amrex::Real time_s, amrex::Real requested_dt_s,
    const NormalSpeedBatchFunction& normal_speed,
    const FireFuelRaster& raster,
    std::vector<FireFuelRasterCell>& stage_materials)
{
    if (!std::isfinite(requested_dt_s) || !(requested_dt_s > amrex::Real(0))) {
        throw std::invalid_argument("Fire material step requires positive finite dt");
    }
    std::vector<FireVec2> positions, normals;
    detail::fire_barrier_flatten_front(front, positions, normals);
    std::vector<FireVec2> material_queries;
    material_queries.reserve(positions.size());
    const auto& g = raster.geometry();
    for (std::size_t i = 0; i < positions.size(); ++i) {
        auto cell = detail::fire_cartesian_raster_locate_cell(g, positions[i].x, positions[i].y);
        const auto bounds = detail::fire_cartesian_raster_cell_bounds(g, cell.i, cell.j);
        if (positions[i].x == bounds.xlo_m && normals[i].x < amrex::Real(0) && cell.i > 0) { --cell.i; }
        if (positions[i].y == bounds.ylo_m && normals[i].y < amrex::Real(0) && cell.j > 0) { --cell.j; }
        const auto outgoing = detail::fire_cartesian_raster_cell_bounds(g, cell.i, cell.j);
        material_queries.push_back({outgoing.xlo_m, outgoing.ylo_m});
    }
    stage_materials = raster.collective_sample_points(material_queries);
    if (stage_materials.size() != positions.size()) {
        throw std::logic_error("Fire stage material batch has the wrong size");
    }

    const auto speeds = detail::fire_barrier_checked_normal_speeds(
        normal_speed, positions, normals, time_s);
    amrex::Real dt = requested_dt_s;
    // Do not send an unbounded first trial outside the environment. A later
    // accelerating trial that leaves the represented domain still fails closed.
    const auto upper = detail::fire_cartesian_raster_cell_bounds(g, g.nx-1, g.ny-1);
    for (std::size_t i = 0; i < positions.size(); ++i) {
        const FireVec2 velocity = normals[i] * speeds[i];
        if (velocity.x > amrex::Real(0)) { dt = std::min(dt, (upper.xhi_m-positions[i].x)/velocity.x); }
        if (velocity.x < amrex::Real(0)) { dt = std::min(dt, (g.xlo_m-positions[i].x)/velocity.x); }
        if (velocity.y > amrex::Real(0)) { dt = std::min(dt, (upper.yhi_m-positions[i].y)/velocity.y); }
        if (velocity.y < amrex::Real(0)) { dt = std::min(dt, (g.ylo_m-positions[i].y)/velocity.y); }
    }
    if (!(dt > amrex::Real(0)) || !(time_s+dt > time_s)) {
        throw std::out_of_range("Fire material front has reached the represented domain boundary");
    }
    std::vector<FireVec2> predictor;
    predictor.reserve(positions.size());
    for (std::size_t i = 0; i < positions.size(); ++i) {
        predictor.push_back(detail::fire_material_snap(
            positions[i] + normals[i] * (dt * speeds[i]), g));
    }
    const amrex::Real predictor_fraction = detail::first_fire_material_change(
        raster, positions, predictor, stage_materials);
    if (!(predictor_fraction > amrex::Real(0))) {
        throw std::runtime_error("Fire outgoing material does not match its predictor path");
    }
    dt *= predictor_fraction;
    if (!(dt > amrex::Real(0)) || !(time_s + dt > time_s)) {
        throw std::runtime_error("Fire predictor contact time is not representable");
    }
    for (int retry = 0; retry < 128; ++retry) {
        FireFrontTopologyAdvanceResult trial;
        try {
            trial = advance_front_rk2_batched_until_topology_event(
                front, time_s, dt, normal_speed);
        } catch (const std::invalid_argument& error) {
            throw std::invalid_argument(
                std::string("Fire material RK2 trial failed before material snapping: ")
                + error.what());
        }
        if (!trial.topology_event.has_value()) {
            for (auto& component : trial.terminal_vertices_m) {
                for (auto& p : component) { p = detail::fire_material_snap(p, g); }
            }
        }
        auto ends = detail::fire_barrier_flatten_component_vertices(trial.terminal_vertices_m);
        amrex::Real fraction = amrex::Real(1);
        if (!trial.topology_event.has_value()) {
            const auto midpoint =
                detail::fire_barrier_flatten_component_vertices(
                    trial.rk2_midpoint_vertices_m);
            fraction = detail::first_fire_material_change_rk2(
                raster, positions, midpoint, ends, stage_materials);
        } else {
            fraction = detail::first_fire_material_change(
                raster, positions, ends, stage_materials);
        }
        if (fraction < amrex::Real(1)) {
            if (!(fraction > amrex::Real(0))) {
                throw std::runtime_error(
                    "Fire material event has zero duration at a raster corner; "
                    "discontinuous corner/sliding geometry is not resolved");
            }
            const amrex::Real next_dt = trial.advanced_dt_s * fraction;
            if (!(next_dt > amrex::Real(0)) || !(time_s + next_dt > time_s) || !(next_dt < dt)) {
                throw std::runtime_error("Fire material contact time is not representable");
            }
            dt = next_dt;
            continue;
        }
        if (!trial.topology_event.has_value()) {
            // Remove roundoff-sized face gaps so outgoing ownership changes on
            // the next segment. Do not snap a topology event's raw geometry.
            for (auto& component : trial.terminal_vertices_m) {
                for (auto& p : component) { p = detail::fire_material_snap(p, g); }
            }
            const auto flat =
                detail::fire_barrier_flatten_component_vertices(
                    trial.terminal_vertices_m);
            try {
                trial.completed_front =
                    detail::fire_barrier_rebuild_front(
                        front,
                        flat);
            } catch (const std::invalid_argument& error) {
                throw std::invalid_argument(
                    std::string("Fire material snap/rebuild failed: ")
                    + error.what());
            }
        }
        if (!trial.topology_event.has_value()) {
            detail::require_fire_material_rk2_sweep_not_extinguished(
                front,
                trial.rk2_midpoint_vertices_m,
                trial.terminal_vertices_m,
                raster);
        } else {
            detail::require_fire_material_sweep_not_extinguished(
                front, trial.terminal_vertices_m, raster);
        }
        return trial;
    }
    throw std::runtime_error("Fire material contact localization did not converge");
}

} // namespace ERFFire

#endif
