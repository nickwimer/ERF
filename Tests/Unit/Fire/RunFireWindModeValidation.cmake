if(NOT DEFINED IDENTITY_EXE OR "${IDENTITY_EXE}" STREQUAL "")
  message(FATAL_ERROR "IDENTITY_EXE is required")
endif()
if(NOT DEFINED DIRECT_INPUT OR "${DIRECT_INPUT}" STREQUAL "")
  message(FATAL_ERROR "DIRECT_INPUT is required")
endif()
if(NOT DEFINED WAF_INPUT OR "${WAF_INPUT}" STREQUAL "")
  message(FATAL_ERROR "WAF_INPUT is required")
endif()

file(REAL_PATH "." test_root)

function(expect_failure label input_file expected_text)
  set(case_dir "${test_root}/fire_wind_validation_${label}")
  file(REMOVE_RECURSE "${case_dir}")
  file(MAKE_DIRECTORY "${case_dir}")

  execute_process(
    COMMAND "${IDENTITY_EXE}"
            "${input_file}"
            ${ARGN}
    WORKING_DIRECTORY "${case_dir}"
    RESULT_VARIABLE run_result
    OUTPUT_VARIABLE run_output
    ERROR_VARIABLE run_error
  )

  if(run_result EQUAL 0)
    message(FATAL_ERROR
      "expected ERF Fire wind-input validation failure for ${label}")
  endif()

  set(combined "${run_output}\n${run_error}")
  string(FIND "${combined}" "${expected_text}" match_index)
  if(match_index EQUAL -1)
    message(FATAL_ERROR
      "wind-input validation case ${label} failed for the wrong reason\n"
      "expected text: ${expected_text}\n"
      "stdout:\n${run_output}\n"
      "stderr:\n${run_error}")
  endif()
endfunction()

expect_failure(
  unknown_mode
  "${DIRECT_INPUT}"
  "fire.wind_mode must be direct_reference or explicit_waf_20ft"
  "fire.wind_mode=not_a_fire_wind_mode")

expect_failure(
  missing_waf
  "${DIRECT_INPUT}"
  "fire.wind_mode = explicit_waf_20ft requires fire.wind_adjustment_factor"
  "fire.wind_mode=explicit_waf_20ft")

expect_failure(
  negative_waf
  "${WAF_INPUT}"
  "fire.wind_adjustment_factor must be finite and in [0,1]"
  "fire.wind_adjustment_factor=-0.01")

expect_failure(
  amplified_waf
  "${WAF_INPUT}"
  "fire.wind_adjustment_factor must be finite and in [0,1]"
  "fire.wind_adjustment_factor=1.01")

expect_failure(
  missing_direct_height
  "${WAF_INPUT}"
  "fire.wind_mode = direct_reference requires fire.reference_height_agl_m"
  "fire.wind_mode=direct_reference")

expect_failure(
  waf_reference_height_forbidden
  "${WAF_INPUT}"
  "fire.reference_height_agl_m is valid only with fire.wind_mode = direct_reference"
  "fire.reference_height_agl_m=2.75")

expect_failure(
  direct_waf_forbidden
  "${DIRECT_INPUT}"
  "fire.wind_adjustment_factor is valid only with fire.wind_mode = explicit_waf_20ft"
  "fire.wind_adjustment_factor=0.5")

message(STATUS
  "Fire wind-mode runtime validation PASS")
