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
if(NOT DEFINED MPIEXEC_NUMPROC_FLAG OR "${MPIEXEC_NUMPROC_FLAG}" STREQUAL "")
  message(FATAL_ERROR "MPIEXEC_NUMPROC_FLAG is required")
endif()

set(mpiexec_preflags)
if(DEFINED MPIEXEC_PREFLAGS_CSV AND NOT "${MPIEXEC_PREFLAGS_CSV}" STREQUAL "")
  string(REPLACE "," ";" mpiexec_preflags "${MPIEXEC_PREFLAGS_CSV}")
endif()

set(mpiexec_postflags)
if(DEFINED MPIEXEC_POSTFLAGS_CSV AND NOT "${MPIEXEC_POSTFLAGS_CSV}" STREQUAL "")
  string(REPLACE "," ";" mpiexec_postflags "${MPIEXEC_POSTFLAGS_CSV}")
endif()

file(REAL_PATH "." test_root)
set(continuous_dir "${test_root}/fire_mpi_terrain_continuous")
set(restart_dir "${test_root}/fire_mpi_terrain_restart")
file(REMOVE_RECURSE "${continuous_dir}" "${restart_dir}")
file(MAKE_DIRECTORY "${continuous_dir}" "${restart_dir}")

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
    COMMAND "${CMAKE_COMMAND}" -E compare_files "${expected}" "${actual}"
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
      "continuous: ${expected_files}\n"
      "restart: ${actual_files}")
  endif()

  foreach(relative_path IN LISTS expected_files)
    compare_file(
      "${expected_dir}/${relative_path}"
      "${actual_dir}/${relative_path}"
      "${description} ${relative_path}")
  endforeach()
endfunction()

run_parallel(2 "${continuous_dir}" "two-rank uninterrupted terrain Fire run")

run_parallel(
  2
  "${restart_dir}"
  "two-rank pre-restart terrain Fire segment"
  "max_step=10")

set(restart_checkpoint "${restart_dir}/chk00010")
if(NOT IS_DIRECTORY "${restart_checkpoint}")
  message(FATAL_ERROR
    "pre-restart terrain segment did not create ${restart_checkpoint}")
endif()
if(NOT EXISTS "${restart_checkpoint}/FireState")
  message(FATAL_ERROR
    "pre-restart terrain checkpoint is missing FireState")
endif()

run_parallel(
  2
  "${restart_dir}"
  "two-rank restarted terrain Fire segment"
  "amr.restart=${restart_checkpoint}")

compare_file(
  "${continuous_dir}/chk00020/FireState"
  "${restart_dir}/chk00020/FireState"
  "persistent terrain Fire checkpoint state")

compare_tree(
  "${continuous_dir}/fire_output"
  "${restart_dir}/fire_output"
  "terrain Fire output history")

set(continuous_plot "${continuous_dir}/plt00020")
set(restarted_plot "${restart_dir}/plt00020")
if(NOT IS_DIRECTORY "${continuous_plot}" OR NOT IS_DIRECTORY "${restarted_plot}")
  message(FATAL_ERROR
    "terrain restart comparison did not create final plotfiles")
endif()

execute_process(
  COMMAND
    "${MPIEXEC}"
    "${MPIEXEC_NUMPROC_FLAG}" "1"
    ${mpiexec_preflags}
    "${ANALYSIS_EXE}"
    ${mpiexec_postflags}
    "analysis.reference_plot=${continuous_plot}"
    "analysis.comparison_plot=${restarted_plot}"
  WORKING_DIRECTORY "${test_root}"
  RESULT_VARIABLE analysis_result
  OUTPUT_VARIABLE analysis_output
  ERROR_VARIABLE analysis_error
)
if(NOT analysis_result EQUAL 0)
  message(FATAL_ERROR
    "terrain MPI restart atmospheric comparison failed\n"
    "stdout:\n${analysis_output}\n"
    "stderr:\n${analysis_error}")
endif()

message(STATUS
  "ERF-Fire terrain coupling remains exact across checkpoint restart on a two-rank decomposition")
