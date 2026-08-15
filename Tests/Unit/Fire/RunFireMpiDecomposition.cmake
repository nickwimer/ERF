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

file(REMOVE_RECURSE "${one_rank_dir}" "${two_rank_dir}")
file(MAKE_DIRECTORY "${one_rank_dir}" "${two_rank_dir}")

function(run_parallel ranks working_dir description)
  execute_process(
    COMMAND
      "${MPIEXEC}"
      "${MPIEXEC_NUMPROC_FLAG}" "${ranks}"
      ${mpiexec_preflags}
      "${RESPONSE_EXE}"
      ${mpiexec_postflags}
      "${RESPONSE_INPUT}"
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

message(STATUS
  "ERF-Fire one-rank and two-rank decompositions produce identical Fire history and atmospheric state")
