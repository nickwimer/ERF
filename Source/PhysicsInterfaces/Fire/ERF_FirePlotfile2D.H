#ifndef ERF_FIRE_PLOTFILE2D_H_
#define ERF_FIRE_PLOTFILE2D_H_

#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <string>

namespace ERFFire
{

struct ERFFirePlotfile2DAvailabilityInputs
{
    amrex::Vector<std::string>& available_names;
};

struct ERFFirePlotfile2DPrepareInputs
{
    const amrex::Vector<std::string>& plot_var_names;
    const amrex::Geometry& level0_geometry;
    int level0_step{};
    amrex::Real level0_time_s{};
};

struct ERFFirePlotfile2DState
{
    bool requested{false};
    amrex::MultiFab raster;
};

struct ERFFirePlotfile2DFillInputs
{
    const amrex::Vector<std::string>& plot_var_names;
    const ERFFirePlotfile2DState& state;
    int level{};
    amrex::MultiFab& output;
    int& output_component;
};

} // namespace ERFFire

#endif
