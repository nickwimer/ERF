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

set(mpiexec_preflags)
if(DEFINED MPIEXEC_PREFLAGS_CSV AND
   NOT "${MPIEXEC_PREFLAGS_CSV}" STREQUAL "")
  string(REPLACE "," ";" mpiexec_preflags
    "${MPIEXEC_PREFLAGS_CSV}")
endif()

set(mpiexec_postflags)
if(DEFINED MPIEXEC_POSTFLAGS_CSV AND
   NOT "${MPIEXEC_POSTFLAGS_CSV}" STREQUAL "")
  string(REPLACE "," ";" mpiexec_postflags
    "${MPIEXEC_POSTFLAGS_CSV}")
endif()

file(REAL_PATH "." test_root)
set(one_rank_dir "${test_root}/fire_mpi_one_rank")
set(two_rank_dir "${test_root}/fire_mpi_two_rank")
set(rank_change_dir "${test_root}/fire_mpi_rank_change")

file(REMOVE_RECURSE
  "${one_rank_dir}"
  "${two_rank_dir}"
  "${rank_change_dir}")
file(MAKE_DIRECTORY
  "${one_rank_dir}"
  "${two_rank_dir}"
  "${rank_change_dir}")

function(run_parallel ranks working_dir description)
  execute_process(
    COMMAND
      "${MPIEXEC}"
      "${MPIEXEC_NUMPROC_FLAG}" "${ranks}"
      ${mpiexec_preflags}
      "${RESPONSE_EXE}"
      ${mpiexec_postflags}
      "${RESPONSE_INPUT}"
      ${ARGN}
    WORKING_DIRECTORY "${working_dir}"
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
  list(SORT expected_files)
  list(SORT actual_files)

  if(NOT "${expected_files}" STREQUAL "${actual_files}")
    message(FATAL_ERROR
      "${description}: file inventories differ\n"
      "one rank: ${expected_files}\n"
      "two ranks: ${actual_files}")
  endif()

  foreach(relative_path IN LISTS expected_files)
    compare_file(
      "${expected_dir}/${relative_path}"
      "${actual_dir}/${relative_path}"
      "${description} ${relative_path}")
  endforeach()
endfunction()

run_parallel(1 "${one_rank_dir}" "one-rank decomposed Fire run")
run_parallel(2 "${two_rank_dir}" "two-rank decomposed Fire run")

set(one_plot "${one_rank_dir}/plt00020")
set(two_plot "${two_rank_dir}/plt00020")
if(NOT IS_DIRECTORY "${one_plot}" OR NOT IS_DIRECTORY "${two_plot}")
  message(FATAL_ERROR "decomposition runs did not create final plotfiles")
endif()

compare_file(
  "${one_rank_dir}/chk00020/FireState"
  "${two_rank_dir}/chk00020/FireState"
  "persistent Fire checkpoint state")

compare_tree(
  "${one_rank_dir}/fire_output"
  "${two_rank_dir}/fire_output"
  "Fire output history")

execute_process(
  COMMAND
    "${MPIEXEC}"
    "${MPIEXEC_NUMPROC_FLAG}" "1"
    ${mpiexec_preflags}
    "${ANALYSIS_EXE}"
    ${mpiexec_postflags}
    "analysis.reference_plot=${one_plot}"
    "analysis.comparison_plot=${two_plot}"
  WORKING_DIRECTORY "${test_root}"
  RESULT_VARIABLE analysis_result
  OUTPUT_VARIABLE analysis_output
  ERROR_VARIABLE analysis_error
)
if(NOT analysis_result EQUAL 0)
  message(FATAL_ERROR
    "atmospheric decomposition comparison failed\n"
    "stdout:\n${analysis_output}\n"
    "stderr:\n${analysis_error}")
endif()

run_parallel(
  1
  "${rank_change_dir}"
  "one-rank pre-restart Fire segment"
  "max_step=10")

set(rank_change_checkpoint "${rank_change_dir}/chk00010")
if(NOT IS_DIRECTORY "${rank_change_checkpoint}")
  message(FATAL_ERROR
    "rank-change pre-restart segment did not create ${rank_change_checkpoint}")
endif()
if(NOT EXISTS "${rank_change_checkpoint}/FireState")
  message(FATAL_ERROR
    "rank-change checkpoint is missing FireState metadata")
endif()
if(NOT EXISTS
   "${rank_change_checkpoint}/Level_0/FireStateRaster_H")
  message(FATAL_ERROR
    "rank-change checkpoint is missing distributed FireStateRaster")
endif()

run_parallel(
  2
  "${rank_change_dir}"
  "two-rank restart from one-rank Fire checkpoint"
  "amr.restart=${rank_change_checkpoint}")

compare_file(
  "${two_rank_dir}/chk00020/FireState"
  "${rank_change_dir}/chk00020/FireState"
  "rank-change persistent Fire checkpoint metadata")

compare_tree(
  "${two_rank_dir}/fire_output"
  "${rank_change_dir}/fire_output"
  "rank-change Fire output history")

set(rank_change_plot "${rank_change_dir}/plt00020")
if(NOT IS_DIRECTORY "${rank_change_plot}")
  message(FATAL_ERROR
    "rank-change restart did not create final plotfile")
endif()

execute_process(
  COMMAND
    "${MPIEXEC}"
    "${MPIEXEC_NUMPROC_FLAG}" "1"
    ${mpiexec_preflags}
    "${ANALYSIS_EXE}"
    ${mpiexec_postflags}
    "analysis.reference_plot=${two_plot}"
    "analysis.comparison_plot=${rank_change_plot}"
  WORKING_DIRECTORY "${test_root}"
  RESULT_VARIABLE rank_change_analysis_result
  OUTPUT_VARIABLE rank_change_analysis_output
  ERROR_VARIABLE rank_change_analysis_error
)
if(NOT rank_change_analysis_result EQUAL 0)
  message(FATAL_ERROR
    "MPI rank-change restart atmospheric comparison failed\n"
    "stdout:\n${rank_change_analysis_output}\n"
    "stderr:\n${rank_change_analysis_error}")
endif()

# File-backed heterogeneous-v4 persistence proof. The existing timestep
# identity executable has a Fire-only helper mode so this stays inside the
# already-built production-linked target while exercising the real v4 metadata
# and VisMF raster files. One rank writes the checkpoint/reference continuation;
# two ranks restore that one-rank checkpoint and continue; one rank then
# compares every persisted state/material component exactly.
get_filename_component(response_exe_dir "${RESPONSE_EXE}" DIRECTORY)
set(spatial_v4_exe "${response_exe_dir}/erf_fire_timestep_identity")
if(NOT EXISTS "${spatial_v4_exe}")
  message(FATAL_ERROR
    "spatial-v4 rank-change helper executable is missing: ${spatial_v4_exe}")
endif()

set(spatial_v4_root "${test_root}/fire_spatial_v4_rank_change")
file(REMOVE_RECURSE "${spatial_v4_root}")
file(MAKE_DIRECTORY "${spatial_v4_root}")

function(run_spatial_v4 ranks mode description)
  execute_process(
    COMMAND
      "${MPIEXEC}"
      "${MPIEXEC_NUMPROC_FLAG}" "${ranks}"
      ${mpiexec_preflags}
      "${spatial_v4_exe}"
      ${mpiexec_postflags}
      "fire_spatial_v4_test.mode=${mode}"
      "fire_spatial_v4_test.root=${spatial_v4_root}"
    WORKING_DIRECTORY "${test_root}"
    RESULT_VARIABLE spatial_result
    OUTPUT_VARIABLE spatial_output
    ERROR_VARIABLE spatial_error
  )
  if(NOT spatial_result EQUAL 0)
    message(FATAL_ERROR
      "${description} failed with exit code ${spatial_result}\n"
      "stdout:\n${spatial_output}\n"
      "stderr:\n${spatial_error}")
  endif()
endfunction()

run_spatial_v4(
  1
  write_reference
  "one-rank spatial-v4 checkpoint/reference continuation")

foreach(required_file
    "${spatial_v4_root}/checkpoint/FireState"
    "${spatial_v4_root}/checkpoint/Level_0/FireStateRaster_H"
    "${spatial_v4_root}/reference_final/FireState"
    "${spatial_v4_root}/reference_final/Level_0/FireStateRaster_H")
  if(NOT EXISTS "${required_file}")
    message(FATAL_ERROR
      "spatial-v4 one-rank writer is missing ${required_file}")
  endif()
endforeach()

file(READ "${spatial_v4_root}/checkpoint/FireState" spatial_v4_metadata)
string(FIND
  "${spatial_v4_metadata}"
  "ERF_FIRE_RUNTIME_STATE 4"
  spatial_v4_version_position)
if(NOT spatial_v4_version_position EQUAL 0)
  message(FATAL_ERROR
    "spatial-v4 one-rank checkpoint did not write format version 4")
endif()

run_spatial_v4(
  2
  restore_continue
  "two-rank spatial-v4 restore/continuation from one-rank checkpoint")

foreach(required_file
    "${spatial_v4_root}/restart_final/FireState"
    "${spatial_v4_root}/restart_final/Level_0/FireStateRaster_H")
  if(NOT EXISTS "${required_file}")
    message(FATAL_ERROR
      "spatial-v4 two-rank continuation is missing ${required_file}")
  endif()
endforeach()

run_spatial_v4(
  1
  compare
  "spatial-v4 exact post-restart comparison")

message(STATUS
  "ERF-Fire spatial v4 checkpoint is exact across one-rank-to-two-rank restart continuation")
message(STATUS
  "ERF-Fire decomposition invariance includes one-rank-to-two-rank checkpoint restart")
