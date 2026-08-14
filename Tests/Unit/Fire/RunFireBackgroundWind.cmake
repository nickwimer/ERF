if(NOT DEFINED RESPONSE_EXE OR "${RESPONSE_EXE}" STREQUAL "")
  message(FATAL_ERROR "RESPONSE_EXE is required")
endif()
if(NOT DEFINED ANALYSIS_EXE OR "${ANALYSIS_EXE}" STREQUAL "")
  message(FATAL_ERROR "ANALYSIS_EXE is required")
endif()
if(NOT DEFINED RESPONSE_INPUT OR "${RESPONSE_INPUT}" STREQUAL "")
  message(FATAL_ERROR "RESPONSE_INPUT is required")
endif()
if(NOT DEFINED CASE_PREFIX OR "${CASE_PREFIX}" STREQUAL "")
  message(FATAL_ERROR "CASE_PREFIX is required")
endif()
if(NOT DEFINED IGNITION_X_M OR "${IGNITION_X_M}" STREQUAL "")
  message(FATAL_ERROR "IGNITION_X_M is required")
endif()
if(NOT DEFINED IGNITION_Y_M OR "${IGNITION_Y_M}" STREQUAL "")
  message(FATAL_ERROR "IGNITION_Y_M is required")
endif()

file(REAL_PATH "." test_root)

function(run_case mode label output_var)
  set(case_dir
      "${test_root}/${CASE_PREFIX}_${label}")
  file(REMOVE_RECURSE "${case_dir}")
  file(MAKE_DIRECTORY "${case_dir}")

  set(output_dir "fire_output_${label}")

  execute_process(
    COMMAND "${RESPONSE_EXE}"
            "${RESPONSE_INPUT}"
            "fire.coupling_mode=${mode}"
            "fire.output_dir=${output_dir}"
    WORKING_DIRECTORY "${case_dir}"
    RESULT_VARIABLE run_result
    OUTPUT_VARIABLE run_output
    ERROR_VARIABLE run_error
  )

  if(NOT run_result EQUAL 0)
    message(FATAL_ERROR
      "ERF Fire ${mode} child failed\n"
      "stdout:\n${run_output}\n"
      "stderr:\n${run_error}")
  endif()

  set(fire_output "${case_dir}/${output_dir}")

  foreach(required_name IN ITEMS
      perimeter_000000.csv
      raster_000000.csv
      perimeter_000100.csv
      raster_000100.csv
      perimeter_000200.csv
      raster_000200.csv)
    if(NOT EXISTS "${fire_output}/${required_name}")
      message(FATAL_ERROR
        "${mode} background-wind run is missing ${required_name}")
    endif()
  endforeach()

  set(${output_var} "${fire_output}" PARENT_SCOPE)
endfunction()

run_case(one_way one_way one_way_fire_output)
run_case(two_way two_way two_way_fire_output)

foreach(initial_name IN ITEMS
    perimeter_000000.csv
    raster_000000.csv)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E compare_files
            "${one_way_fire_output}/${initial_name}"
            "${two_way_fire_output}/${initial_name}"
    RESULT_VARIABLE initial_compare)
  if(NOT initial_compare EQUAL 0)
    message(FATAL_ERROR
      "Initial one-way/two-way Fire ${initial_name} differs")
  endif()
endforeach()

foreach(diverged_name IN ITEMS
    perimeter_000100.csv
    raster_000100.csv
    perimeter_000200.csv
    raster_000200.csv)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E compare_files
            "${one_way_fire_output}/${diverged_name}"
            "${two_way_fire_output}/${diverged_name}"
    RESULT_VARIABLE diverged_compare)
  if(diverged_compare EQUAL 0)
    message(FATAL_ERROR
      "One-way/two-way Fire ${diverged_name} did not diverge")
  endif()
endforeach()

execute_process(
  COMMAND "${ANALYSIS_EXE}"
          "analysis.one_way_perimeter=${one_way_fire_output}/perimeter_000200.csv"
          "analysis.two_way_perimeter=${two_way_fire_output}/perimeter_000200.csv"
          "analysis.ignition_x_m=${IGNITION_X_M}"
          "analysis.ignition_y_m=${IGNITION_Y_M}"
  WORKING_DIRECTORY "${test_root}"
  RESULT_VARIABLE analysis_result
  OUTPUT_VARIABLE analysis_output
  ERROR_VARIABLE analysis_error
)

if(NOT analysis_result EQUAL 0)
  message(FATAL_ERROR
    "Background-wind analyzer failed\n"
    "stdout:\n${analysis_output}\n"
    "stderr:\n${analysis_error}")
endif()

string(STRIP "${analysis_output}" analysis_output_stripped)
message(STATUS "${analysis_output_stripped}")
message(STATUS
  "Background-wind closed-loop response validated: "
  "the imposed wind produces downwind Fire asymmetry and two-way feedback "
  "further advances the head while reducing backing spread")
