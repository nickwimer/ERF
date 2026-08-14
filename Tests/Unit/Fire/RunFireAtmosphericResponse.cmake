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

function(run_case mode label plot_var output_var)
  set(case_dir "${test_root}/fire_atmospheric_response_${label}")
  file(REMOVE_RECURSE "${case_dir}")
  file(MAKE_DIRECTORY "${case_dir}")

  set(output_dir "fire_output_${label}")

  execute_process(
    COMMAND "${RESPONSE_EXE}"
            "${RESPONSE_INPUT}"
            "fire.coupling_mode=${mode}"
            "fire.output_dir=${output_dir}"
            "erf.plot_file_1=plt"
    WORKING_DIRECTORY "${case_dir}"
    RESULT_VARIABLE run_result
    OUTPUT_VARIABLE run_output
    ERROR_VARIABLE run_error
  )

  if(NOT run_result EQUAL 0)
    message(FATAL_ERROR
      "M12b1 ERF Fire ${mode} child failed\n"
      "stdout:\n${run_output}\n"
      "stderr:\n${run_error}")
  endif()

  foreach(required_file IN ITEMS
      perimeter_000001.csv
      raster_000001.csv
      summary.csv)
    if(NOT EXISTS "${case_dir}/${output_dir}/${required_file}")
      message(FATAL_ERROR
        "M12b1 ERF Fire ${mode} run did not produce ${required_file}")
    endif()
  endforeach()

  file(GLOB plot_dirs
    LIST_DIRECTORIES true
    "${case_dir}/plt*")
  list(LENGTH plot_dirs plot_count)
  if(plot_count LESS 1)
    message(FATAL_ERROR
      "expected at least one M12b1 plotfile for ${mode}, found none")
  endif()

  list(SORT plot_dirs)
  list(GET plot_dirs -1 plot_dir)
  set(${plot_var} "${plot_dir}" PARENT_SCOPE)
  set(${output_var} "${case_dir}/${output_dir}" PARENT_SCOPE)
endfunction()

run_case(one_way one_way one_way_plot one_way_output)
run_case(two_way two_way two_way_plot two_way_output)

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E compare_files
          "${one_way_output}/perimeter_000001.csv"
          "${two_way_output}/perimeter_000001.csv"
  RESULT_VARIABLE perimeter_compare)
if(NOT perimeter_compare EQUAL 0)
  message(FATAL_ERROR
    "M12b1 first-step Fire perimeter differs between one-way and two-way")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E compare_files
          "${one_way_output}/raster_000001.csv"
          "${two_way_output}/raster_000001.csv"
  RESULT_VARIABLE raster_compare)
if(NOT raster_compare EQUAL 0)
  message(FATAL_ERROR
    "M12b1 first-step Fire raster differs between one-way and two-way")
endif()

execute_process(
  COMMAND "${ANALYSIS_EXE}"
          "analysis.one_way_plot=${one_way_plot}"
          "analysis.two_way_plot=${two_way_plot}"
  WORKING_DIRECTORY "${test_root}"
  RESULT_VARIABLE analysis_result
  OUTPUT_VARIABLE analysis_output
  ERROR_VARIABLE analysis_error
)

if(NOT analysis_result EQUAL 0)
  message(FATAL_ERROR
    "M12b1 atmospheric-response analyzer failed\n"
    "stdout:\n${analysis_output}\n"
    "stderr:\n${analysis_error}")
endif()

if(NOT analysis_output MATCHES "M12B1_RESPONSE_METRICS")
  message(FATAL_ERROR
    "M12b1 analyzer did not emit response metrics\n"
    "stdout:\n${analysis_output}\n"
    "stderr:\n${analysis_error}")
endif()

string(STRIP "${analysis_output}" analysis_output_stripped)
message(STATUS "${analysis_output_stripped}")
message(STATUS
  "M12b1 flat gravity-on atmospheric response PASS: "
  "first-step Fire evolution matches and two-way heat/moisture forcing "
  "produces positive thermodynamic and upward-velocity response")
