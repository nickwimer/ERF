# Scientific corrections to the September 21 fire branch

These changes address the source-level defects identified at commit
86880e309a8636df716deb2c76d983ef12896f06. Correcting a model implementation,
checking numerical identities, and validating predictions against measured
fires are different tasks. The following is not an observational validation
claim or a claim that the complete ERF/GPU/MPI test matrix has passed.

## Fuel-dependent combustion times

Production spatial combustion now uses the following SFIRE semi-empirical
surface-fuel weights: FM1-FM3: 7; FM4: 180; FM5-FM7: 100; FM8-FM13: 900.
The exponential 1/e time in seconds is weight / 0.85. The source is
`openwfm/wrf-fire/wrfv2_fire/phys/module_fr_sfire_phys.F`, `DATA weight`,
blob `4cbb42fa7b5fd36faf4df0edfbbc50e7ee3c24b5`.

Previously FM2-FM13 inherited the FM1 value (8.2353 s) simply because their
accounting copied the scalar base parameters. FM13 now has a 1058.8235 s
1/e time: about 5.51% of an instantaneously ignited reservoir is consumed
in 60 s, rather than about 99.93%. FM1 behavior is unchanged. The separate
accounting-only API still honors an explicitly supplied base burn time;
it is not the production default resolver.

This is a model-consistency correction, NOT validation against measured
fuel consumption. Fuel loading is still a scalar reservoir and the sensible
heat per dry mass and combustion-water yield retain the existing policy.
Different particle classes, flaming/smouldering partitions, moisture-dependent
burn histories, and observational calibration remain unresolved.

`FireScientificBurnTime.*` exercises the production resolver and the burn
kernel against the independent table and exponential-time reference. Correct
mass and energy accounting is checked separately from the release timescale.

## Surface-to-map propagation

Both scalar and batched runtime paths now construct the local surface metric
for z = h(x,y), with gradient g:

    G = I + g g^T,   S = sqrt(G),   B = inverse(S).

S maps a horizontal displacement into orthonormal surface coordinates. B maps
a surface displacement back to horizontal coordinates. If K is the surface
spread ellipse, its map-plane normal speed is its support evaluated at the
pulled-back covector: h_(B K)(n) = h_K(B n). The implementation normalizes B n
for the existing ellipse routine and multiplies the result by its length.
For spread directly upslope on z = s*x, this gives the required horizontal
factor 1/sqrt(1+s*s). For oblique anisotropic spread, the complete metric is
necessary; dividing by a directional cosine alone is not the general rule.

The horizontal wind azimuth is lifted with S and normalized before combining
wind and slope forcing in the surface plane. The supplied model/midflame wind
magnitude is preserved. This is an explicit convention, not an inference of
the unresolved vertical atmospheric wind. The model still requires an
appropriate reference-height/wind-adjustment choice from the user.

Reference: Finney (1998, revised 2004), FARSITE: Fire Area Simulator - Model
Development and Evaluation, USDA Forest Service RMRS-RP-4, local surface-plane
and horizontal-coordinate treatment, pp. 5-6.

Fuel loads, burned-area totals and extensive atmosphere feedback retain their
existing horizontal map-plane-area convention. No extra terrain-area factor
is inserted into the atmospheric source. The zero-slope transform explicitly
uses the old support calculation to preserve the flat limit.

`FireScientificTerrain.*` checks the flat identity, inverse-metric isotropic
solution, independent 3-D tangent-basis projection of an oblique ellipse,
rotation covariance, and invalid inputs. The existing planar-terrain runtime
wavelet reference is projected independently; its tolerance is not loosened.

## Material interfaces and contact histories

The spatial runtime no longer samples a different material opportunistically
at its RK midpoint. Each trial holds the departure material fixed, traces the
complete candidate trajectories through raster cells, shortens/recomputes the
step at a material change, and returns the actual duration. The receiving
material is selected on the next segment, with outgoing-normal ownership on
exact raster faces for both propagation signs. Wind and terrain can still
be sampled at the trial locations inside the frozen atmospheric snapshot.

The same segment duration is now used for first arrival, burned fraction,
and combustion. A front that reaches an impermeable barrier early in the
atmospheric step therefore has a moving segment followed by a stationary
segment; its history is not stretched from its starting point to the barrier
over the whole atmospheric timestep.

Heterogeneous front edges are split collinearly at raster intervals so a thin
material patch does not remain represented only between distant vertices.
Full edge-sweep envelopes are checked against both explicit NonBurnable cells
and moisture-extinguished fuels. An unresolved penetration is rejected before
persistent runtime state is committed. Moisture-extinguished fuel remains a
fuel category: it is not silently reclassified as absent fuel.

`FireScientificMaterial.*` uses independently specified piecewise constant
rates to test wet half-spaces, thin wet strips, fast-to-slow travel, reflection
of propagation direction, and contact timing. In the contact case the front
travels 0.2 m at 1 m/s before stopping. A cell entrance 0.1 m from its starting
position must be reached at 0.1 s, not halfway through a 1 s atmospheric step.
These tests exercise the integration primitive; native runtime/coupling
regressions remain necessary in addition to those focused tests. A later
re-audit added a curved RK2 dense-trajectory case in which the start/end chord
returns to its original material while the true explicit-midpoint continuous
extension enters a thin wet strip.

This is not a general solution of obstacle sliding or discontinuous
anisotropic corner propagation. Corner events that make no representable
progress fail explicitly, and the conservative edge-envelope check can reject
a folded motion larger than its exact swept region. Accuracy and performance
for complicated material boundaries require front-resolution and timestep
studies. The previous unsupported topology combinations are not enabled by
this change.

## RK2-consistent physical histories

A second adversarial pass found that the propagated front endpoint used
explicit-midpoint/RK2, while first arrival, burned area, and combustion
ignition cohorts were still reconstructed from a straight start-to-end front
interpolation. In spatially varying wind or terrain that straight chord is not
the trajectory represented by the RK2 stages.

Completed fixed-topology advances now retain the explicit-midpoint stage
geometry and use the order-2 continuous extension

    x(a) = x_n
         + 2*a*(1-a)*(x_stage - x_n)
         + a*a*(x_{n+1} - x_n),    0 <= a <= 1.

Arrival-time bisection, burned-area sampling, combustion ignition cohorts, and
ordinary material-interface traversal all use this same dense trajectory.
Material traversal approximates each quadratic vertex path by sufficiently
fine chords and batches all raster samples into one collective query. The
chord subdivision is a safety/localization approximation, not a claim of an
exact analytic raster/quadratic intersection.

Topology-changing events retain their separately localized event geometry.
General discontinuous obstacle sliding/corner rerouting remains unsupported.

## Authoritative coupling clock precision

ERF owns the coupling time as `double`. The Fire runtime previously accumulated
its authoritative clock in `amrex::Real` and compared it exactly with ERF's
`double` time after narrowing. That is safe in the tested DOUBLE build but can
desynchronize in a SINGLE build after ordinary repeated timesteps.

The Fire runtime/checkpoint metadata clock and step diagnostic start/end times
are now `double`. Geometry, rates, burned fractions, and first-arrival raster
values remain `amrex::Real`. The first-arrival history clock therefore remains
a representational `amrex::Real` clock and restart validation compares it with
the authoritative double clock using a precision-scaled tolerance. This change
prevents coupling-clock drift; it does not turn all Fire history fields into
double precision.

## Motion-adaptive history resolution

`fire.combustion_temporal_substeps` remains the persisted/configured minimum
number of ignition-history bins for restart compatibility. It is no longer
assumed to be sufficient by itself for front-history geometry.

For each completed RK2 segment the runtime estimates the largest derivative
of the explicit-midpoint dense trajectory at the two interval endpoints and
increases the history sample count until the maximum per-sample front travel
is at most one quarter of the minimum Fire-raster cell spacing. The same
selected count is used by first arrival, burned fraction, and combustion
cohorts. A request above 4096 history samples fails closed and asks for a
smaller timestep.

This is Fire-local history quadrature control. It does not replace ERF's
atmospheric CFL timestep and it does not introduce atmosphere/Fire
predictor-corrector coupling.

## Feedback-column truncation diagnostic

The production atmospheric source continues to normalize the exponential
vertical deposition shape over the finite represented ERF column, preserving
the existing conservative policy that deposits all step-integrated Fire
release into the model atmosphere.

The source API now exposes the corresponding finite-column diagnostic:

    represented = 1 - exp(-z_top/H)
    tail        = exp(-z_top/H)
    amplification = 1 / represented.

On the first two-way source construction, ERF-Fire computes the minimum
physical model-top AGL height from the actual level-0 nodal geometry and prints
the represented fraction, the unresolved exponential tail above model top,
and the normalization amplification. A tail above 5 percent is explicitly
flagged. This is a diagnostic, not a silent switch to the WRF-SFIRE
flux-divergence policy.

A projector budget oracle independently reconstructs sensible energy from
`rho*theta` tendency using local Exner/volume/dt and reconstructs released
water from `rho*q_v` tendency. Both must close to the step-integrated Fire
feedback for a nonuniform pressure/volume column. This verifies the source
projection budget itself; it is not an observational heat-flux validation or
a proof that every ERF dycore process preserves that budget in a coupled run.

## Coupled time accuracy is unchanged

ERF still supplies an immutable t^n atmospheric snapshot to the fire and
constructs explicit feedback from that step. RK2 front integration within this
snapshot does not establish second-order accuracy of the coupled system.
A changing/two-way-coupled environment generally requires atmospheric-timestep
refinement to quantify splitting error. No predictor-corrector atmosphere/fire
iteration is added here, and no second-order coupled-accuracy claim is made.

## Verification status and use

The native test sources are registered in the existing
`erf_fire_unit_tests` target and use precision-scaled tolerances where
appropriate. For MPI-enabled builds using CMake 3.29 or newer, registered
GoogleTest executables now use the configured one-rank MPI `TEST_LAUNCHER`
and `PRE_TEST` discovery. This avoids direct Cray/Slurm execution without a
PMI context while still allowing the target to be built outside a running
test allocation.

Before the second re-audit series described above, commit
`f1c5a3665804ed1efa81d5a5e98663b685155d10` was exercised natively on
Kestrel H100 GPUs. The evidence at that point included:

- native CUDA/H100 compilation and link of `erf_fire_unit_tests`;
- 14/14 targeted `FireScientific*` tests;
- 118/118 supporting Fire physics/history tests;
- 31/31 `FireSpreadRuntime.*` tests;
- two-rank CUDA/MPI smoke, decomposition invariance, and 1->2-rank restart;
- spatial-fuel/NonBurnable v4 rank-change restart;
- three coupled terrain/restart integration cases; and
- all 21 remaining Fire integration regressions in the selected suite.

A CFL-controlled two-way terrain/background-wind study at CFL 0.8, 0.4, 0.2,
and 0.1 showed monotonically decreasing Fire and atmospheric differences.
The thermodynamic `theta` and `rhoQ1` max-norm differences were close to
first-order under the later refinements; velocity max norms converged more
slowly but continued to decrease. This is numerical timestep-refinement
evidence for the explicit outer coupling, not physical validation.

The later RK2-dense-history, double-clock, adaptive-history, feedback-column,
budget-oracle, and CTest-launcher commits were source-reviewed but had NOT yet
been rebuilt or executed on Kestrel when this document was updated. They must
therefore pass a fresh native CUDA/MPI regression gate before being treated as
verified implementation.

The original isolated float/double terrain and material-event harnesses remain
useful component evidence, but they are weaker than the native Kestrel tests
and are not observational validation.

Use fresh runs for corrected-physics comparisons: old terrain or heavy-fuel
histories generated before these corrections are not equivalent baselines.
General obstacle sliding/rerouting, atmosphere AMR levels above zero,
prognostic fuel moisture, and observational fire/heat-flux validation remain
outside the established capability.
