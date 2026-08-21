if(NOT DEFINED RESPONSE_EXE OR "${RESPONSE_EXE}" STREQUAL "")
  message(FATAL_ERROR "RESPONSE_EXE is required")
endif()
if(NOT DEFINED RESPONSE_INPUT OR "${RESPONSE_INPUT}" STREQUAL "")
  message(FATAL_ERROR "RESPONSE_INPUT is required")
endif()
if(NOT DEFINED TERRAIN_SOURCE OR "${TERRAIN_SOURCE}" STREQUAL "")
  message(FATAL_ERROR "TERRAIN_SOURCE is required")
endif()

foreach(required_path IN ITEMS
    "${RESPONSE_EXE}"
    "${RESPONSE_INPUT}"
    "${TERRAIN_SOURCE}")
  if(NOT EXISTS "${required_path}")
    message(FATAL_ERROR "missing terrain-source restart-policy input: ${required_path}")
  endif()
endforeach()

file(REAL_PATH "." test_root)
set(case_dir "${test_root}/fire_terrain_source_restart_policy")
set(source_dir "${case_dir}/source")
set(run_dir "${case_dir}/run")
set(source_file "${source_dir}/terrain.txt")

file(REMOVE_RECURSE "${case_dir}")
file(MAKE_DIRECTORY "${source_dir}" "${run_dir}")
configure_file("${TERRAIN_SOURCE}" "${source_file}" COPYONLY)

function(run_success description)
  execute_process(
    COMMAND "${RESPONSE_EXE}"
            "${RESPONSE_INPUT}"
            "erf.terrain_file_name=${source_file}"
            ${ARGN}
    WORKING_DIRECTORY "${run_dir}"
    RESULT_VARIABLE run_result
    OUTPUT_VARIABLE run_output
    ERROR_VARIABLE run_error
  )
  if(NOT "${run_result}" STREQUAL "0")
    message(FATAL_ERROR
      "${description} failed with result ${run_result}\n"
      "stdout:\n${run_output}\n"
      "stderr:\n${run_error}")
  endif()
endfunction()

run_success(
  "pre-restart terrain-source segment"
  "max_step=5"
  "erf.check_int=5"
  "fire.output_interval_steps=0"
  "erf.check_file=chk")

set(restart_checkpoint "${run_dir}/chk00005")
set(policy_file "${restart_checkpoint}/FireTerrainSourcePolicy")

if(NOT EXISTS "${policy_file}")
  message(FATAL_ERROR "terrain-source checkpoint is missing FireTerrainSourcePolicy")
endif()

file(READ "${policy_file}" policy_text)

if(NOT policy_text MATCHES "ERF_FIRE_TERRAIN_SOURCE_POLICY 1")
  message(FATAL_ERROR "terrain-source checkpoint policy has the wrong version")
endif()
if(NOT policy_text MATCHES "mode regular_file")
  message(FATAL_ERROR "terrain-source checkpoint policy did not record regular_file mode")
endif()
if(NOT policy_text MATCHES "fingerprint_fnv1a64 [0-9]+")
  message(FATAL_ERROR "terrain-source checkpoint policy is missing its fingerprint")
endif()

run_success(
  "unchanged terrain-source restart"
  "amr.restart=${restart_checkpoint}"
  "max_step=6"
  "erf.check_int=-1"
  "fire.output_interval_steps=0")
file(APPEND "${source_file}" "\n")

execute_process(
  COMMAND "${RESPONSE_EXE}"
          "${RESPONSE_INPUT}"
          "erf.terrain_file_name=${source_file}"
          "amr.restart=${restart_checkpoint}"
          "max_step=6"
          "erf.check_int=-1"
          "fire.output_interval_steps=0"
  WORKING_DIRECTORY "${run_dir}"
  RESULT_VARIABLE changed_result
  OUTPUT_VARIABLE changed_output
  ERROR_VARIABLE changed_error
)

if("${changed_result}" STREQUAL "0")
  message(FATAL_ERROR
    "restart unexpectedly accepted a terrain source whose bytes changed")
endif()

set(changed_log "${changed_output}\n${changed_error}")
if(NOT changed_log MATCHES
    "ERF-Fire checkpoint terrain-source fingerprint does not match current terrain file")
  message(FATAL_ERROR
    "changed terrain-source restart failed without the expected fingerprint diagnostic\n"
    "stdout:\n${changed_output}\n"
    "stderr:\n${changed_error}")
endif()

message(STATUS
  "Fire terrain-source restart policy: unchanged source restarts; changed source is rejected")
