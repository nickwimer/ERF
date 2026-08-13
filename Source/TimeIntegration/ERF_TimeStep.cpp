#include <ERF.H>
#include <ERF_Utils.H>
#include <ERF_ReadFromWRFBdy.H>
#include <ERF_ReadFromERFBdy.H>

#ifdef ERF_USE_FIRE
#include <ERF_FireLevel0Environment.H>
#include <ERF_FireLevel0SourceCoupling.H>
#include <ERF_FireRuntimeInit.H>
#include <ERF_FireSpreadOutput.H>
#include <ERF_FireSurfaceFeedback.H>
#endif

using namespace amrex;

/**
 * Function that coordinates the evolution across levels -- this calls Advance to do the
 * actual advance at this level,  then recursively calls itself at finer levels
 *
 * @param[in] lev level of refinement (coarsest level is 0)
 * @param[in] time start time for time advance
 * @param[in] iteration time step counter
 */

void
ERF::timeStep (int lev, double time, int /*iteration*/)
{
    //
    // We need to FillPatch the coarse level before assessing whether to regrid
    // We have not done the swap yet so we fill the "new" which will become the "old"
    //
    MultiFab& S_new = vars_new[lev][Vars::cons];
    MultiFab& U_new = vars_new[lev][Vars::xvel];
    MultiFab& V_new = vars_new[lev][Vars::yvel];
    MultiFab& W_new = vars_new[lev][Vars::zvel];

#ifdef ERF_USE_NETCDF
    //
    // Since we now only read in a subset of the time slices in wrfbdy and
    //     wrflowinp, we need to check whether it's time to read in more.
    //
    bool use_moist = (solverChoice.moisture_type != MoistureType::None);
    if (solverChoice.use_real_bcs && (lev==0))
    {
        MultiFab r_hse(base_state[lev], make_alias, BaseState::r0_comp, 1);
        Array<MultiFab*, AMREX_SPACEDIM> area_vec = {ax[lev].get(), ay[lev].get(), az[lev].get()};

        int ntimes = bdy_data_xlo.size();
        double time_since_start_bdy = time + start_time - start_bdy_time;
        int n_time_old = std::min(static_cast<int>( (time_since_start_bdy        ) /  bdy_time_interval), ntimes-1);
        int n_time_new = std::min(static_cast<int>( (time_since_start_bdy+dt[lev]) /  bdy_time_interval), ntimes-1);

        for (int itime = 0; itime < ntimes; itime++)
        {
            /*
            if (bdy_data_xlo[itime].size() > 0) {
                amrex::Print() << "HAVE  BDY DATA AT TIME " << itime << std::endl;
            } else {
                amrex::Print() << " NO   BDY DATA AT TIME " << itime << std::endl;
            }
            */

            // Note that we never release itime == 0 because it is used for the spatial interpolation at later times
            bool clear_itime = (itime > 0 && itime < n_time_old);

            if (clear_itime && bdy_data_xlo[itime].size() > 0) {
                bdy_data_xlo[itime].clear();
                bdy_data_xhi[itime].clear();
                bdy_data_ylo[itime].clear();
                bdy_data_yhi[itime].clear();
                //amrex::Print() << "CLEAR BDY DATA AT TIME " << itime << std::endl;
            }

            bool need_itime = (itime >= n_time_old && itime <= n_time_new+1);
            //if (need_itime) { amrex::Print()  << "NEED  BDY DATA AT TIME " << itime << std::endl; }

            // Handle erfbdy files (AMReX native format).
            if (use_erfbdy) {
                if (bdy_data_xlo[itime].size() == 0 && need_itime) {
                    read_from_erfbdy(itime, erfbdy_file,
                                     bdy_data_xlo, bdy_data_xhi,
                                     bdy_data_ylo, bdy_data_yhi,
                                     nvars_erfbdy, real_width);
                }
            // Handle wrfbdy files (NetCDF format).
            } else {
                if (bdy_data_xlo[itime].size() == 0 && need_itime) {
                    bool is_anelastic = (solverChoice.anelastic[0] == 1);
                    read_and_convert_from_wrfbdy(itime,nc_bdy_file,bdy_data_xlo,bdy_data_xhi,bdy_data_ylo,bdy_data_yhi,
                                                 wrf_MUB, wrf_C1H, wrf_C2H, wrf_RDNW, wrf_PHB, z_phys_cc[lev], z_phys_nd[lev],
                                                 vars_new[lev][Vars::xvel], vars_new[lev][Vars::yvel], vars_new[lev][Vars::cons],
                                                 r_hse, area_vec, geom[lev], use_moist, solverChoice.rebalance_wrf_input, domain_bcs_type,
                                                 real_width, bdy_time_interval, is_anelastic);
                }
            } // use_erfbdy
        } // itime
    } // use_real_bcs && lev == 0

    if (!nc_low_file.empty() && (lev==0))
    {
        int ntimes = low_data_zlo.size();
        double time_since_start_low = time + start_time - start_low_time;
        int n_time_old = std::min(static_cast<int>( (time_since_start_low        ) /  low_time_interval), ntimes-1);
        int n_time_new = std::min(static_cast<int>( (time_since_start_low+dt[lev]) /  low_time_interval), ntimes-1);

        for (int itime = 0; itime < ntimes; itime++)
        {
            /*
            if (low_data_zlo[itime].size() > 0) {
                amrex::Print() << "HAVE  LOW DATA AT TIME " << itime << std::endl;
            } else {
                amrex::Print() << " NO   LOW DATA AT TIME " << itime << std::endl;
            }
            */

            bool clear_itime = (itime < n_time_old);

            if (clear_itime && low_data_zlo[itime].size() > 0) {
                low_data_zlo[itime].clear();
                //amrex::Print() << "CLEAR LOW DATA AT TIME " << itime << std::endl;
            }

            bool need_itime = (itime >= n_time_old && itime <= n_time_new+1);
            //if (need_itime) { amrex::Print()  << "NEED  LOW DATA AT TIME " << itime << std::endl; }

            if (low_data_zlo[itime].size() == 0 && need_itime) {
                read_from_wrflow(itime, nc_low_file, geom[lev].Domain(), low_data_zlo);

                update_sst_tsk(itime, geom[lev], ba2d[lev],
                               sst_lev[lev], tsk_lev[lev],
                               m_SurfaceLayer, low_data_zlo,
                               S_new, *mf_PSFC[lev],
                               solverChoice.rdOcp, lmask_lev[lev][0], use_moist);
            }
        } // itime
    } // have nc_low_file && lev == 0
#endif

    //
    // NOTE: the momenta here are not fillpatched (they are only used as scratch space)
    //
    if (lev == 0) {
        FillPatchCrseLevel(lev, time, {&S_new, &U_new, &V_new, &W_new});
    } else if (lev < finest_level) {
        FillPatchFineLevel(lev, time, {&S_new, &U_new, &V_new, &W_new},
                           {&S_new, &rU_new[lev], &rV_new[lev], &rW_new[lev]},
                           base_state[lev], base_state[lev]);
    }

#ifdef ERF_USE_FIRE
    // Explicit Fire coupling sequencing:
    //   FillPatch atmosphere at t^n
    //   -> freeze one immutable t^n environment snapshot
    //   -> advance one candidate coupling-neutral Fire state across dt[0]
    //   -> for two_way only, difference combustion history, project the
    //      step-integrated release using t^n pressure, and prepare a constant
    //      native ERF source tendency for the same atmospheric step
    //   -> atomically commit the Fire candidate/source
    //   -> ordinary ERF Advance.
    //
    // one_way follows the same Fire evolution path but installs no atmospheric
    // source. VariableDz freezes local-AGL wind and uses map-plane terrain
    // slope in both coupling modes. VariableDz two_way projects the same
    // combustion feedback through local terrain-following AGL columns using
    // authoritative detJ_cc physical volumes. No WAF is applied.
    if (lev == 0 && m_fire_runtime_options.enabled) {
        ERFFire::ERFFireLevel0EnvironmentInputs fire_inputs{
            geom[0],
            U_new,
            V_new,
            *z_phys_cc[0],
            *z_phys_nd[0],
            solverChoice.mesh_type,
            solverChoice.terrain_type,
            solverChoice.buildings_type,
            max_level};

        std::unique_ptr<ERFFire::FireTerrainSurface>
            next_terrain_surface;
        std::unique_ptr<ERFFire::FireFlatEnvironmentSampler>
            next_snapshot;

        if (solverChoice.mesh_type == MeshType::VariableDz) {
            next_terrain_surface =
                std::make_unique<ERFFire::FireTerrainSurface>(
                    ERFFire::make_erf_level0_terrain_surface(
                        fire_inputs));
            next_snapshot =
                std::make_unique<ERFFire::FireFlatEnvironmentSampler>(
                    ERFFire::freeze_erf_level0_terrain_reference_wind_environment(
                        fire_inputs,
                        m_fire_runtime_options.reference_height_agl_m));
        } else {
            next_snapshot =
                std::make_unique<ERFFire::FireFlatEnvironmentSampler>(
                    ERFFire::freeze_erf_level0_environment(
                        fire_inputs,
                        m_fire_runtime_options.reference_height_agl_m));
        }

        m_fire_environment_snapshot = std::move(next_snapshot);
        m_fire_environment_snapshot_time = time;

        if (!m_fire_spread_runtime) {
            m_fire_spread_runtime =
                ERFFire::make_erf_fire_spread_runtime(
                    m_fire_runtime_options,
                    geom[0],
                    static_cast<Real>(time));
            m_fire_step_index = 0;

            ERFFire::write_erf_fire_spread_snapshot(
                *m_fire_spread_runtime,
                m_fire_runtime_options.output_dir,
                m_fire_step_index);
        }

        if (m_fire_spread_runtime->current_time_s()
            != static_cast<Real>(time)) {
            Error(
                "ERF-Fire runtime clock is not synchronized with level-0 t^n");
        }

        ERFFire::ERFFireSpreadRuntime next_fire_runtime =
            *m_fire_spread_runtime;

        if (next_terrain_surface) {
            (void)next_fire_runtime.advance_direct_reference_wind(
                *m_fire_environment_snapshot,
                *next_terrain_surface,
                static_cast<Real>(dt[0]));
        } else {
            (void)next_fire_runtime.advance_direct_reference_wind(
                *m_fire_environment_snapshot,
                static_cast<Real>(dt[0]));
        }

        std::unique_ptr<MultiFab> next_fire_source;
        double next_fire_source_time =
            std::numeric_limits<double>::quiet_NaN();

        if (m_fire_runtime_options.coupling_mode
            == ERFFire::ERFFireCouplingMode::TwoWay) {
            const ERFFire::FireSurfaceFeedbackRaster feedback =
                ERFFire::make_fire_surface_feedback_increment(
                    m_fire_spread_runtime->combustion_raster(),
                    next_fire_runtime.combustion_raster());

            const ERFFire::ERFFireAtmosphericSourceOptions
                source_options{
                    m_fire_runtime_options
                        .feedback_extinction_depth_m};

            if (solverChoice.mesh_type == MeshType::VariableDz) {
                next_fire_source =
                    ERFFire::make_erf_fire_level0_terrain_source_tendency(
                        feedback,
                        fire_inputs,
                        *detJ_cc[0],
                        S_new,
                        solverChoice.moisture_type,
                        static_cast<Real>(dt[0]),
                        source_options);
            } else {
                next_fire_source =
                    ERFFire::make_erf_fire_level0_source_tendency(
                        feedback,
                        fire_inputs,
                        S_new,
                        solverChoice.moisture_type,
                        static_cast<Real>(dt[0]),
                        source_options);
            }
            next_fire_source_time = time;
        }

        *m_fire_spread_runtime =
            std::move(next_fire_runtime);
        m_fire_atmospheric_source_tendency =
            std::move(next_fire_source);
        m_fire_atmospheric_source_time =
            next_fire_source_time;

        ++m_fire_step_index;
        if (m_fire_step_index
                % m_fire_runtime_options.output_interval_steps
            == 0) {
            ERFFire::write_erf_fire_spread_snapshot(
                *m_fire_spread_runtime,
                m_fire_runtime_options.output_dir,
                m_fire_step_index);
        }
    }
#endif

    if (regrid_int > 0)  // We may need to regrid
    {
        // help keep track of whether a level was already regridded
        // from a coarser level call to regrid
        static Vector<int> last_regrid_step(max_level+1, 0);

        // regrid changes level "lev+1" so we don't regrid on max_level
        // also make sure we don't regrid fine levels again if
        // it was taken care of during a coarser regrid
        if (lev < max_level)
        {
            if ( (istep[lev] % regrid_int == 0) && (istep[lev] > last_regrid_step[lev]) )
            {
                // regrid could add newly refine levels (if finest_level < max_level)
                // so we save the previous finest level index
                int old_finest = finest_level;

                if (solverChoice.coupling_type == CouplingType::TwoWay &&
                    solverChoice.moisture_type != MoistureType::None &&
                    Microphysics::modelType(solverChoice.moisture_type) == MoistureModelType::Lagrangian &&
                    finest_level >= 1) {
                    micro->AverageDownMicroVars(finest_level);
                    for (int flev = finest_level-1; flev >= lev; --flev) {
                        AverageDownMoistStateTo(flev);
                    }
                }

                regrid(lev, static_cast<Real>(time));

#ifdef ERF_USE_PARTICLES
                particleData.Redistribute(z_phys_nd);
#endif

                // mark that we have regridded this level already
                for (int k = lev; k <= finest_level; ++k) {
                    last_regrid_step[k] = istep[k];
                }

                // if there are newly created levels, set the time step
                for (int k = old_finest+1; k <= finest_level; ++k) {
                    dt[k] = dt[k-1] / static_cast<double>(nsubsteps[k]);
                }
            } // if
        } // lev
    }

    // Update what we call "old" and "new" time
    t_old[lev] = t_new[lev];
    t_new[lev] += dt[lev];

    if (Verbose()) {
        amrex::Print() << "[Level " << lev << " step " << istep[lev]+1 << "] ";
        amrex::Print() << std::setprecision(timeprecision)
                       << "ADVANCE from elapsed time = " << t_old[lev] << " to " << t_new[lev]
                       << " with dt = " << dt[lev] << std::endl;
    }

#ifdef ERF_USE_WW3_COUPLING
    amrex::Print() <<  " About to call send_to_ww3 from ERF_Timestep" << std::endl;
    send_to_ww3(lev);
    amrex::Print() <<  " About to call read_waves from ERF_Timestep"  << std::endl;
    read_waves(lev);
    //send_to_ww3(lev);
    //read_waves(lev);
    //send_to_ww3(lev);
#endif

    // Advance a single level for a single time step
    Advance(lev, time, dt[lev], istep[lev], nsubsteps[lev]);

    ++istep[lev];

    if (Verbose()) {
        amrex::Print() << "[Level " << lev << " step " << istep[lev] << "] ";
        amrex::Print() << "Advanced " << CountCells(lev) << " cells" << std::endl;
    }

    if (lev < finest_level)
    {
        // recursive call for next-finer level
        for (int i = 1; i <= nsubsteps[lev+1]; ++i)
        {
            double strt_time_for_fine = time + (i-1)*dt[lev+1];
            timeStep(lev+1, strt_time_for_fine, i);
        }
    }

    if ( verbose && lev == 0 && solverChoice.moisture_type != MoistureType::None) {
        amrex::Print() << "Cloud fraction " << time << "  " << cloud_fraction(time) << std::endl;
    }
}
