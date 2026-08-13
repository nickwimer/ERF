if(NOT DEFINED IDENTITY_EXE OR "${IDENTITY_EXE}" STREQUAL "")
  message(FATAL_ERROR "IDENTITY_EXE is required")
endif()
if(NOT DEFINED DIRECT_INPUT OR "${DIRECT_INPUT}" STREQUAL "")
  message(FATAL_ERROR "DIRECT_INPUT is required")
endif()
if(NOT DEFINED WAF_INPUT OR "${WAF_INPUT}" STREQUAL "")
  message(FATAL_ERROR "WAF_INPUT is required")
endif()
if(NOT DEFINED WIND_MODE_CASE_PREFIX OR "${WIND_MODE_CASE_PREFIX}" STREQUAL "")
  set(WIND_MODE_CASE_PREFIX "fire_wind_mode")
endif()

file(REAL_PATH "." test_root)

function(run_wind_case input_file label mode factor output_var)
  set(case_dir "${test_root}/${WIND_MODE_CASE_PREFIX}_${label}")
  file(REMOVE_RECURSE "${case_dir}")
  file(MAKE_DIRECTORY "${case_dir}")

  set(output_dir "fire_output_${label}")

  if(mode STREQUAL "direct_reference")
    set(wind_args
      "fire.wind_mode=direct_reference"
      "fire.reference_height_agl_m=6.096")
  elseif(mode STREQUAL "explicit_waf_20ft")
    set(wind_args
      "fire.wind_mode=explicit_waf_20ft"
      "fire.wind_adjustment_factor=${factor}")
  else()
    message(FATAL_ERROR "unsupported comparison wind mode ${mode}")
  endif()

  execute_process(
    COMMAND "${IDENTITY_EXE}"
            "${input_file}"
            "fire.coupling_mode=one_way"
            ${wind_args}
            "fire.output_dir=${output_dir}"
            "erf.check_file=chk"
    WORKING_DIRECTORY "${case_dir}"
    RESULT_VARIABLE run_result
    OUTPUT_VARIABLE run_output
    ERROR_VARIABLE run_error
  )

  if(NOT run_result EQUAL 0)
    message(FATAL_ERROR
      "ERF Fire wind-mode child failed for ${label}\n"
      "stdout:\n${run_output}\n"
      "stderr:\n${run_error}")
  endif()

  if(NOT run_output MATCHES "ERF_FIRE_IDENTITY_CHILD_OK=1")
    message(FATAL_ERROR
      "ERF Fire wind-mode child did not report success for ${label}\n"
      "stdout:\n${run_output}\n"
      "stderr:\n${run_error}")
  endif()

  foreach(required_file IN ITEMS
      summary.csv
      perimeter_000005.csv
      raster_000005.csv)
    if(NOT EXISTS "${case_dir}/${output_dir}/${required_file}")
      message(FATAL_ERROR
        "ERF Fire wind-mode run ${label} did not produce ${required_file}")
    endif()
  endforeach()

  set(${output_var} "${case_dir}/${output_dir}" PARENT_SCOPE)
endfunction()

run_wind_case("${DIRECT_INPUT}" direct direct_reference 1.0 direct_output)
run_wind_case("${WAF_INPUT}" waf_unity explicit_waf_20ft 1.0 waf_unity_output)
run_wind_case("${WAF_INPUT}" waf_reduced explicit_waf_20ft 0.5 waf_reduced_output)

file(GLOB direct_files
  RELATIVE "${direct_output}"
  "${direct_output}/*")
file(GLOB waf_unity_files
  RELATIVE "${waf_unity_output}"
  "${waf_unity_output}/*")
list(SORT direct_files)
list(SORT waf_unity_files)

if(NOT "${direct_files}" STREQUAL "${waf_unity_files}")
  message(FATAL_ERROR
    "direct-reference and WAF=1 output file sets differ\n"
    "direct=${direct_files}\n"
    "waf_unity=${waf_unity_files}")
endif()

foreach(relative_file IN LISTS direct_files)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E compare_files
            "${direct_output}/${relative_file}"
            "${waf_unity_output}/${relative_file}"
    RESULT_VARIABLE compare_result)
  if(NOT compare_result EQUAL 0)
    message(FATAL_ERROR
      "WAF=1 did not exactly match direct_reference at 6.096 m in ${relative_file}")
  endif()
endforeach()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E compare_files
          "${direct_output}/perimeter_000005.csv"
          "${waf_reduced_output}/perimeter_000005.csv"
  RESULT_VARIABLE reduced_compare)
if(reduced_compare EQUAL 0)
  message(FATAL_ERROR
    "WAF=0.5 did not change the final Fire perimeter relative to direct_reference")
endif()

message(STATUS
  "Fire wind-mode comparison PASS: direct_reference at 6.096 m and WAF=1 "
  "are exactly identical, while WAF=0.5 changes Fire spread")
