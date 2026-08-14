if(NOT DEFINED RESPONSE_EXE OR "${RESPONSE_EXE}" STREQUAL "")
  message(FATAL_ERROR "RESPONSE_EXE is required")
endif()
if(NOT DEFINED ANALYSIS_EXE OR "${ANALYSIS_EXE}" STREQUAL "")
  message(FATAL_ERROR "ANALYSIS_EXE is required")
endif()
if(NOT DEFINED RESPONSE_INPUT OR "${RESPONSE_INPUT}" STREQUAL "")
  message(FATAL_ERROR "RESPONSE_INPUT is required")
endif()

file(REAL_PATH "." test_root)

function(run_case mode label H_m plot_var output_var)
  set(case_dir
      "${test_root}/fire_atmospheric_response_sensitivity_${label}")
  file(REMOVE_RECURSE "${case_dir}")
  file(MAKE_DIRECTORY "${case_dir}")

  set(output_dir "fire_output_${label}")

  execute_process(
    COMMAND "${RESPONSE_EXE}"
            "${RESPONSE_INPUT}"
            "fire.coupling_mode=${mode}"
            "fire.feedback_extinction_depth_m=${H_m}"
            "fire.output_dir=${output_dir}"
            "erf.plot_file_1=plt"
    WORKING_DIRECTORY "${case_dir}"
    RESULT_VARIABLE run_result
    OUTPUT_VARIABLE run_output
    ERROR_VARIABLE run_error
  )

  if(NOT run_result EQUAL 0)
    message(FATAL_ERROR
      "M12b2 ERF Fire ${mode} H=${H_m} child failed\n"
      "stdout:\n${run_output}\n"
      "stderr:\n${run_error}")
  endif()

  foreach(required_file IN ITEMS
      perimeter_000001.csv
      raster_000001.csv
      summary.csv)
    if(NOT EXISTS "${case_dir}/${output_dir}/${required_file}")
      message(FATAL_ERROR
        "M12b2 ${mode} H=${H_m} did not produce ${required_file}")
    endif()
  endforeach()

  file(GLOB plot_dirs
    LIST_DIRECTORIES true
    "${case_dir}/plt*")
  list(LENGTH plot_dirs plot_count)
  if(plot_count LESS 1)
    message(FATAL_ERROR
      "M12b2 ${mode} H=${H_m} did not produce a plotfile")
  endif()

  list(SORT plot_dirs)
  list(GET plot_dirs -1 plot_dir)
  set(${plot_var} "${plot_dir}" PARENT_SCOPE)
  set(${output_var} "${case_dir}/${output_dir}" PARENT_SCOPE)
endfunction()

function(compare_first_step control_output candidate_output label)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E compare_files
            "${control_output}/perimeter_000001.csv"
            "${candidate_output}/perimeter_000001.csv"
    RESULT_VARIABLE perimeter_compare)
  if(NOT perimeter_compare EQUAL 0)
    message(FATAL_ERROR
      "M12b2 first-step Fire perimeter differs for ${label}")
  endif()

  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E compare_files
            "${control_output}/raster_000001.csv"
            "${candidate_output}/raster_000001.csv"
    RESULT_VARIABLE raster_compare)
  if(NOT raster_compare EQUAL 0)
    message(FATAL_ERROR
      "M12b2 first-step Fire raster differs for ${label}")
  endif()
endfunction()

run_case(one_way control 50.0 one_way_plot one_way_output)
run_case(two_way h10 10.0 h10_plot h10_output)
run_case(two_way h25 25.0 h25_plot h25_output)
run_case(two_way h50 50.0 h50_plot h50_output)
run_case(two_way h100 100.0 h100_plot h100_output)

compare_first_step("${one_way_output}" "${h10_output}" "H=10 m")
compare_first_step("${one_way_output}" "${h25_output}" "H=25 m")
compare_first_step("${one_way_output}" "${h50_output}" "H=50 m")
compare_first_step("${one_way_output}" "${h100_output}" "H=100 m")

execute_process(
  COMMAND "${ANALYSIS_EXE}"
          "analysis.one_way_plot=${one_way_plot}"
          "analysis.two_way_plot_h10=${h10_plot}"
          "analysis.two_way_plot_h25=${h25_plot}"
          "analysis.two_way_plot_h50=${h50_plot}"
          "analysis.two_way_plot_h100=${h100_plot}"
  WORKING_DIRECTORY "${test_root}"
  RESULT_VARIABLE analysis_result
  OUTPUT_VARIABLE analysis_output
  ERROR_VARIABLE analysis_error
)

if(NOT analysis_result EQUAL 0)
  message(FATAL_ERROR
    "M12b2 atmospheric-response sensitivity analyzer failed\n"
    "stdout:\n${analysis_output}\n"
    "stderr:\n${analysis_error}")
endif()

if(NOT analysis_output MATCHES "M12B2_RESPONSE_ORDERING_PASS=1")
  message(FATAL_ERROR
    "M12b2 analyzer did not report ordering success\n"
    "stdout:\n${analysis_output}\n"
    "stderr:\n${analysis_error}")
endif()

string(STRIP "${analysis_output}" analysis_output_stripped)
message(STATUS "${analysis_output_stripped}")
message(STATUS
  "M12b2 extinction-depth atmospheric sensitivity PASS: "
  "all H cases retain positive thermodynamic/upward response, "
  "and increasing H shifts the positive theta response upward")
