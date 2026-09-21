if(NOT DEFINED RESPONSE_EXE OR "${RESPONSE_EXE}" STREQUAL "")
  message(FATAL_ERROR "RESPONSE_EXE is required")
endif()
if(NOT DEFINED ANALYSIS_EXE OR "${ANALYSIS_EXE}" STREQUAL "")
  message(FATAL_ERROR "ANALYSIS_EXE is required")
endif()
if(NOT DEFINED RESPONSE_INPUT OR "${RESPONSE_INPUT}" STREQUAL "")
  message(FATAL_ERROR "RESPONSE_INPUT is required")
endif()
if(NOT DEFINED MPIEXEC OR "${MPIEXEC}" STREQUAL "")
  message(FATAL_ERROR "MPIEXEC is required")
endif()
if(NOT DEFINED MPIEXEC_NUMPROC_FLAG OR
   "${MPIEXEC_NUMPROC_FLAG}" STREQUAL "")
  message(FATAL_ERROR "MPIEXEC_NUMPROC_FLAG is required")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/../../MPILauncher.cmake")

file(REAL_PATH "." test_root)
set(reference_dir "${test_root}/fire_nonburnable_v4_reference")
set(rank_change_dir "${test_root}/fire_nonburnable_v4_rank_change")
set(reference_verify_dir "${test_root}/fire_nonburnable_v4_reference_verify")
set(rank_change_verify_dir "${test_root}/fire_nonburnable_v4_rank_change_verify")
set(fuel_source "${test_root}/fire_nonburnable_v4_fuel.txt")

file(REMOVE_RECURSE
  "${reference_dir}"
  "${rank_change_dir}"
  "${reference_verify_dir}"
  "${rank_change_verify_dir}")
file(REMOVE "${fuel_source}")
file(MAKE_DIRECTORY
  "${reference_dir}"
  "${rank_change_dir}"
  "${reference_verify_dir}"
  "${rank_change_verify_dir}")

# The base MPI-decomposition input is an 8x8 Fire grid with a circular
# ignition centered at (4,4), radius 1. Column i=5 therefore begins at x=5:
# the initial front is tangent to this full-height wall but has zero positive
# area inside it. That exercises persistent barrier contact without making the
# initial combustion state invalid.
file(WRITE "${fuel_source}"
  "ERF_FIRE_FUEL_RASTER 1\n"
  "nx 8\n"
  "ny 8\n"
  "xlo_m 0\n"
  "ylo_m 0\n"
  "dx_m 1\n"
  "dy_m 1\n"
  "components 7\n"
  "cells\n")
foreach(j RANGE 0 7)
  foreach(i RANGE 0 7)
    if(i EQUAL 5)
      file(APPEND "${fuel_source}" "0 0 0 0 0 0 0\n")
    else()
      file(APPEND "${fuel_source}" "1 1 0.08 0 0 0 0\n")
    endif()
  endforeach()
endforeach()
file(APPEND "${fuel_source}" "END_ERF_FIRE_FUEL_RASTER\n")

function(run_fire ranks working_dir description use_source)
  set(extra_args ${ARGN})
  if(use_source)
    list(APPEND extra_args "fire.fuel_raster_file=${fuel_source}")
  endif()

  erf_mpi_launcher_command(run_command
    LAUNCHER "${MPIEXEC}"
    NUMPROC_FLAG "${MPIEXEC_NUMPROC_FLAG}"
    NRANKS "${ranks}"
    PREFLAGS "${MPIEXEC_PREFLAGS}"
    CONTEXT "RunFireNonBurnableV4RankChange.cmake")
  list(APPEND run_command "${RESPONSE_EXE}" "${RESPONSE_INPUT}" ${extra_args})

  execute_process(
    COMMAND ${run_command}
    WORKING_DIRECTORY "${working_dir}"
    INPUT_FILE "/dev/null"
    RESULT_VARIABLE run_result
    OUTPUT_VARIABLE run_output
    ERROR_VARIABLE run_error
  )
  if(NOT run_result EQUAL 0)
    message(FATAL_ERROR
      "${description} failed with exit code ${run_result}\n"
      "stdout:\n${run_output}\n"
      "stderr:\n${run_error}")
  endif()
endfunction()

function(compare_file expected actual description)
  if(NOT EXISTS "${expected}")
    message(FATAL_ERROR "${description}: missing ${expected}")
  endif()
  if(NOT EXISTS "${actual}")
    message(FATAL_ERROR "${description}: missing ${actual}")
  endif()

  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E compare_files
      "${expected}" "${actual}"
    RESULT_VARIABLE compare_result
  )
  if(NOT compare_result EQUAL 0)
    message(FATAL_ERROR "${description}: files differ")
  endif()
endfunction()

function(compare_tree expected_dir actual_dir description)
  if(NOT IS_DIRECTORY "${expected_dir}")
    message(FATAL_ERROR "${description}: missing ${expected_dir}")
  endif()
  if(NOT IS_DIRECTORY "${actual_dir}")
    message(FATAL_ERROR "${description}: missing ${actual_dir}")
  endif()

  file(GLOB_RECURSE expected_files
    LIST_DIRECTORIES false
    RELATIVE "${expected_dir}"
    "${expected_dir}/*")
  file(GLOB_RECURSE actual_files
    LIST_DIRECTORIES false
    RELATIVE "${actual_dir}"
    "${actual_dir}/*")
  list(REMOVE_ITEM expected_files "job_info")
  list(REMOVE_ITEM actual_files "job_info")
  list(SORT expected_files)
  list(SORT actual_files)

  if(NOT "${expected_files}" STREQUAL "${actual_files}")
    message(FATAL_ERROR
      "${description}: file inventories differ\n"
      "expected: ${expected_files}\n"
      "actual: ${actual_files}")
  endif()

  foreach(relative_path IN LISTS expected_files)
    compare_file(
      "${expected_dir}/${relative_path}"
      "${actual_dir}/${relative_path}"
      "${description} ${relative_path}")
  endforeach()
endfunction()

function(require_v4_nonburnable_metadata checkpoint description)
  set(metadata_file "${checkpoint}/FireState")
  if(NOT EXISTS "${metadata_file}")
    message(FATAL_ERROR "${description}: missing FireState")
  endif()
  if(NOT EXISTS "${checkpoint}/Level_0/FireStateRaster_H")
    message(FATAL_ERROR
      "${description}: missing distributed FireStateRaster")
  endif()

  file(READ "${metadata_file}" metadata)
  string(FIND "${metadata}" "ERF_FIRE_RUNTIME_STATE 4" version_pos)
  if(NOT version_pos EQUAL 0)
    message(FATAL_ERROR
      "${description}: checkpoint is not ERF-Fire v4")
  endif()
  string(FIND
    "${metadata}"
    "spatial_fuel_provenance resolved_raster"
    provenance_pos)
  if(provenance_pos LESS 0)
    message(FATAL_ERROR
      "${description}: missing resolved spatial-fuel provenance")
  endif()
  string(FIND
    "${metadata}"
    "spatial_fuel_fingerprint_fnv1a64"
    fingerprint_pos)
  if(fingerprint_pos LESS 0)
    message(FATAL_ERROR
      "${description}: missing spatial-fuel fingerprint")
  endif()
endfunction()

# One-rank uninterrupted reference. This reads the external source exactly
# once at initialization and writes v4 checkpoints containing resolved material.
run_fire(
  1
  "${reference_dir}"
  "one-rank NonBurnable v4 reference run"
  TRUE)

set(reference_final "${reference_dir}/chk00020")
require_v4_nonburnable_metadata(
  "${reference_final}"
  "one-rank NonBurnable v4 final checkpoint")

# Build the one-rank checkpoint that will be restored on two ranks.
run_fire(
  1
  "${rank_change_dir}"
  "one-rank NonBurnable v4 pre-restart segment"
  TRUE
  "max_step=10")

set(rank_change_checkpoint "${rank_change_dir}/chk00010")
require_v4_nonburnable_metadata(
  "${rank_change_checkpoint}"
  "one-rank NonBurnable v4 rank-change checkpoint")

# Remove the source before restart. The two-rank continuation intentionally
# does not set fire.fuel_raster_file, proving that v4 checkpoint material is
# self-contained and authoritative across decomposition changes.
file(REMOVE "${fuel_source}")
if(EXISTS "${fuel_source}")
  message(FATAL_ERROR
    "unable to remove NonBurnable source before self-contained restart")
endif()

run_fire(
  2
  "${rank_change_dir}"
  "two-rank NonBurnable v4 continuation from one-rank checkpoint"
  FALSE
  "amr.restart=${rank_change_checkpoint}")

set(rank_change_final "${rank_change_dir}/chk00020")
require_v4_nonburnable_metadata(
  "${rank_change_final}"
  "two-rank NonBurnable v4 final checkpoint")

# FireState contains front topology plus the resolved-material schema and
# fingerprint. It is decomposition independent and must match exactly.
compare_file(
  "${reference_final}/FireState"
  "${rank_change_final}/FireState"
  "NonBurnable v4 final metadata/topology/fingerprint")

# Fire output is also decomposition independent and checks the committed
# history/diagnostics across the rank-change continuation.
compare_tree(
  "${reference_dir}/fire_output"
  "${rank_change_dir}/fire_output"
  "NonBurnable v4 rank-change Fire output history")

# Atmospheric state is compared numerically after the two-way continuation.
set(reference_plot "${reference_dir}/plt00020")
set(rank_change_plot "${rank_change_dir}/plt00020")
if(NOT IS_DIRECTORY "${reference_plot}" OR
   NOT IS_DIRECTORY "${rank_change_plot}")
  message(FATAL_ERROR
    "NonBurnable v4 runs did not create final plotfiles")
endif()

erf_mpi_launcher_command(analysis_command
  LAUNCHER "${MPIEXEC}"
  NUMPROC_FLAG "${MPIEXEC_NUMPROC_FLAG}"
  NRANKS 1
  PREFLAGS "${MPIEXEC_PREFLAGS}"
  CONTEXT "RunFireNonBurnableV4RankChange.cmake")
list(APPEND analysis_command "${ANALYSIS_EXE}")

execute_process(
  COMMAND ${analysis_command}
    "analysis.reference_plot=${reference_plot}"
    "analysis.comparison_plot=${rank_change_plot}"
  WORKING_DIRECTORY "${test_root}"
  INPUT_FILE "/dev/null"
  RESULT_VARIABLE analysis_result
  OUTPUT_VARIABLE analysis_output
  ERROR_VARIABLE analysis_error
)
if(NOT analysis_result EQUAL 0)
  message(FATAL_ERROR
    "NonBurnable v4 rank-change atmospheric comparison failed\n"
    "stdout:\n${analysis_output}\n"
    "stderr:\n${analysis_error}")
endif()

# Normalize both final checkpoints back to one rank and advance one additional
# timestep. The resulting one-rank checkpoint trees must be byte-identical.
# This is a decomposition-independent exact comparison of the persisted v4
# dynamic raster after rank-change continuation, not merely its VisMF layout.
run_fire(
  1
  "${reference_verify_dir}"
  "one-rank verification continuation from reference v4 checkpoint"
  FALSE
  "amr.restart=${reference_final}"
  "max_step=21"
  "stop_time=0.042"
  "erf.check_int=1"
  "erf.plot_int_1=-1"
  "fire.output_interval_steps=1")

run_fire(
  1
  "${rank_change_verify_dir}"
  "one-rank verification continuation from rank-change v4 checkpoint"
  FALSE
  "amr.restart=${rank_change_final}"
  "max_step=21"
  "stop_time=0.042"
  "erf.check_int=1"
  "erf.plot_int_1=-1"
  "fire.output_interval_steps=1")

set(reference_verify_checkpoint "${reference_verify_dir}/chk00021")
set(rank_change_verify_checkpoint "${rank_change_verify_dir}/chk00021")
require_v4_nonburnable_metadata(
  "${reference_verify_checkpoint}"
  "one-rank reference verification checkpoint")
require_v4_nonburnable_metadata(
  "${rank_change_verify_checkpoint}"
  "one-rank rank-change verification checkpoint")

compare_tree(
  "${reference_verify_checkpoint}"
  "${rank_change_verify_checkpoint}"
  "normalized one-rank NonBurnable v4 checkpoint")
compare_tree(
  "${reference_verify_dir}/fire_output"
  "${rank_change_verify_dir}/fire_output"
  "normalized one-rank NonBurnable Fire output")

message(STATUS
  "ERF-Fire NonBurnable v4 material/topology is self-contained across one-rank-to-two-rank restart")
message(STATUS
  "ERF-Fire NonBurnable v4 continuation normalizes to an exact one-rank checkpoint after rank change")
