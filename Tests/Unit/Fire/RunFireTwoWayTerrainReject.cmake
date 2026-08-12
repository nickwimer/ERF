if(NOT DEFINED IDENTITY_EXE OR "${IDENTITY_EXE}" STREQUAL "")
  message(FATAL_ERROR "IDENTITY_EXE is required")
endif()
if(NOT DEFINED TERRAIN_INPUT OR "${TERRAIN_INPUT}" STREQUAL "")
  message(FATAL_ERROR "TERRAIN_INPUT is required")
endif()

file(REAL_PATH "." reject_root)
set(case_dir "${reject_root}/fire_two_way_terrain_reject")
file(REMOVE_RECURSE "${case_dir}")
file(MAKE_DIRECTORY "${case_dir}")

execute_process(
  COMMAND "${IDENTITY_EXE}"
          "${TERRAIN_INPUT}"
          "fire.enabled=1"
          "fire.coupling_mode=two_way"
          "erf.check_file=chk"
  WORKING_DIRECTORY "${case_dir}"
  RESULT_VARIABLE run_result
  OUTPUT_VARIABLE run_output
  ERROR_VARIABLE run_error
)

if(run_result EQUAL 0)
  message(FATAL_ERROR
    "two-way VariableDz terrain unexpectedly ran successfully\n"
    "stdout:\n${run_output}\n"
    "stderr:\n${run_error}")
endif()

set(combined_output "${run_output}\n${run_error}")
if(NOT combined_output MATCHES
    "two_way currently defers VariableDz terrain feedback")
  message(FATAL_ERROR
    "two-way terrain rejection did not report the M10b2 scope reason\n"
    "stdout:\n${run_output}\n"
    "stderr:\n${run_error}")
endif()

message(STATUS
  "Fire two-way terrain rejection PASS: VariableDz feedback remains deferred to M10c")
