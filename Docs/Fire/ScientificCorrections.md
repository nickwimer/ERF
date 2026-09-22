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
These tests exercise the new integration primitive; native runtime/coupling
regressions remain necessary in addition to those focused tests.

This is not a general solution of obstacle sliding or discontinuous
anisotropic corner propagation. Corner events that make no representable
progress fail explicitly, and the conservative edge-envelope check can reject
a folded motion larger than its exact swept region. Accuracy and performance
for complicated material boundaries require front-resolution and timestep
studies. The previous unsupported topology combinations are not enabled by
this change.

## Coupled time accuracy is unchanged

ERF still supplies an immutable t^n atmospheric snapshot to the fire and
constructs explicit feedback from that step. RK2 front integration within this
snapshot does not establish second-order accuracy of the coupled system.
A changing/two-way-coupled environment generally requires atmospheric-timestep
refinement to quantify splitting error. No predictor-corrector atmosphere/fire
iteration is added here, and no second-order coupled-accuracy claim is made.

## Verification status and use

The new native test sources are registered in the existing
`erf_fire_unit_tests` target and use precision-scaled tolerances where
appropriate. The GNU Make header manifest includes the added headers.

During preparation, isolated C++ terrain tests were run in float and double
with the production Richards ellipse implementation and minimal local type/
test-runner shims. The six material-event examples were also run in float and
double with AddressSanitizer/UndefinedBehaviorSanitizer, using local test
doubles for raster storage, traversal, topology and arrival dependencies.
Those isolated checks are not executions of native AMReX, MPI, CUDA, or the
complete ERF runtime. A native ERF build and its registered tests were not run
in that environment. Do not infer native integration success from the local
component checks.

Before a production stress test, rebuild the configured Fire-enabled target
and run `FireScientific*`, the existing `FireSpreadRuntime*`, combustion,
terrain-coupling, barrier and restart regressions through the platform's
configured MPI/Slurm launcher. Then perform timestep/front-resolution studies.
The changes do not add spatial-fuel checkpoint persistence where the runtime
already rejects it. Use fresh runs for corrected-physics comparisons: old
terrain or heavy-fuel histories were generated with a different model and
must not be treated as equivalent corrected-physics baselines.
