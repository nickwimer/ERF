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

function(run_case mode label early_plot_var late_plot_var)
  set(case_dir
      "${test_root}/fire_developed_buoyancy_${label}")
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
      "ERF Fire ${mode} child failed\n"
      "stdout:\n${run_output}\n"
      "stderr:\n${run_error}")
  endif()

  set(early_plot "${case_dir}/plt00100")
  if(NOT EXISTS "${early_plot}/Header")
    message(FATAL_ERROR
      "${mode} run did not produce the 1 s plt00100 checkpoint")
  endif()

  file(GLOB plot_dirs
    LIST_DIRECTORIES true
    "${case_dir}/plt*")
  list(SORT plot_dirs)
  list(LENGTH plot_dirs plot_count)
  if(plot_count LESS 2)
    message(FATAL_ERROR
      "${mode} run produced fewer than two plot checkpoints")
  endif()
  list(GET plot_dirs -1 late_plot)

  set(${early_plot_var} "${early_plot}" PARENT_SCOPE)
  set(${late_plot_var} "${late_plot}" PARENT_SCOPE)
endfunction()

run_case(one_way one_way early_one_way late_one_way)
run_case(two_way two_way early_two_way late_two_way)

execute_process(
  COMMAND "${ANALYSIS_EXE}"
          "analysis.early_one_way=${early_one_way}"
          "analysis.early_two_way=${early_two_way}"
          "analysis.late_one_way=${late_one_way}"
          "analysis.late_two_way=${late_two_way}"
  WORKING_DIRECTORY "${test_root}"
  RESULT_VARIABLE analysis_result
  OUTPUT_VARIABLE analysis_output
  ERROR_VARIABLE analysis_error
)

if(NOT analysis_result EQUAL 0)
  message(FATAL_ERROR
    "Developed-buoyancy analyzer failed\n"
    "stdout:\n${analysis_output}\n"
    "stderr:\n${analysis_error}")
endif()

if(NOT analysis_output MATCHES "BUOYANT_ACCELERATION_PASS=1")
  message(FATAL_ERROR
    "Analyzer did not report buoyant acceleration success\n"
    "stdout:\n${analysis_output}\n"
    "stderr:\n${analysis_error}")
endif()

string(STRIP "${analysis_output}" analysis_output_stripped)
message(STATUS "${analysis_output_stripped}")
message(STATUS
  "Developed buoyant acceleration PASS: "
  "peak and energetic upward response grow strongly from 1 s to 5 s, "
  "with significant updraft above the analytic source z95")
