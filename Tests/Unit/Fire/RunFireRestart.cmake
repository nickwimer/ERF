if(NOT DEFINED RESPONSE_EXE OR "${RESPONSE_EXE}" STREQUAL "")
  message(FATAL_ERROR "RESPONSE_EXE is required")
endif()
if(NOT DEFINED RESPONSE_INPUT OR "${RESPONSE_INPUT}" STREQUAL "")
  message(FATAL_ERROR "RESPONSE_INPUT is required")
endif()

file(REAL_PATH "." test_root)
set(continuous_dir "${test_root}/fire_restart_continuous")
set(split_dir "${test_root}/fire_restart_split")

file(REMOVE_RECURSE "${continuous_dir}" "${split_dir}")
file(MAKE_DIRECTORY "${continuous_dir}" "${split_dir}")

function(run_erf working_dir description)
  execute_process(
    COMMAND "${RESPONSE_EXE}" "${RESPONSE_INPUT}" ${ARGN}
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

function(compare_tree expected_dir actual_dir description)
  if(NOT IS_DIRECTORY "${expected_dir}")
    message(FATAL_ERROR
      "${description}: missing expected directory ${expected_dir}")
  endif()
  if(NOT IS_DIRECTORY "${actual_dir}")
    message(FATAL_ERROR
      "${description}: missing actual directory ${actual_dir}")
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
    execute_process(
      COMMAND "${CMAKE_COMMAND}" -E compare_files
        "${expected_dir}/${relative_path}"
        "${actual_dir}/${relative_path}"
      RESULT_VARIABLE compare_result
    )
    if(NOT compare_result EQUAL 0)
      message(FATAL_ERROR
        "${description}: files differ at ${relative_path}")
    endif()
  endforeach()
endfunction()

run_erf(
  "${continuous_dir}"
  "uninterrupted Fire run")

run_erf(
  "${split_dir}"
  "pre-restart Fire segment"
  "max_step=10")

set(restart_checkpoint "${split_dir}/chk00010")
if(NOT IS_DIRECTORY "${restart_checkpoint}")
  message(FATAL_ERROR
    "pre-restart segment did not create ${restart_checkpoint}")
endif()
if(NOT EXISTS "${restart_checkpoint}/FireState")
  message(FATAL_ERROR
    "native checkpoint is missing persistent FireState")
endif()

run_erf(
  "${split_dir}"
  "restarted Fire segment"
  "amr.restart=${restart_checkpoint}")

set(continuous_checkpoint "${continuous_dir}/chk00020")
set(split_checkpoint "${split_dir}/chk00020")

if(NOT EXISTS "${continuous_checkpoint}/FireState")
  message(FATAL_ERROR
    "uninterrupted final checkpoint is missing FireState")
endif()
if(NOT EXISTS "${split_checkpoint}/FireState")
  message(FATAL_ERROR
    "restarted final checkpoint is missing FireState")
endif()

compare_tree(
  "${continuous_checkpoint}"
  "${split_checkpoint}"
  "native final checkpoint")

compare_tree(
  "${continuous_dir}/fire_output"
  "${split_dir}/fire_output"
  "Fire output history")

message(STATUS
  "Fire restart preserves native atmospheric checkpoint state and persistent Fire history exactly")
