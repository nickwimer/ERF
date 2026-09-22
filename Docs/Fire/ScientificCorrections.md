# Scientific corrections to the September 21 fire branch

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
