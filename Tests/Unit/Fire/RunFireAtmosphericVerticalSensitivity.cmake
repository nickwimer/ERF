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

file(READ "${RESPONSE_INPUT}" base_input_text)
if(NOT base_input_text MATCHES "amr.n_cell = 16 16 48")
  message(FATAL_ERROR
    "Expected the committed vertical-grid anchor")
endif()
if(NOT base_input_text MATCHES "amr.max_grid_size = 48")
  message(FATAL_ERROR
    "Expected the committed max-grid-size anchor")
endif()

function(make_resolution_input nz output_var)
  set(case_input_text "${base_input_text}")
  string(REPLACE
    "amr.n_cell = 16 16 48"
    "amr.n_cell = 16 16 ${nz}"
    case_input_text
    "${case_input_text}")
  string(REPLACE
    "amr.max_grid_size = 48"
    "amr.max_grid_size = 96"
    case_input_text
    "${case_input_text}")

  set(input_path "${test_root}/fire_vertical_inputs_nz${nz}")
  file(WRITE "${input_path}" "${case_input_text}")
  set(${output_var} "${input_path}" PARENT_SCOPE)
endfunction()

function(run_case input_path mode label plot_var output_var)
  set(case_dir
      "${test_root}/fire_atmospheric_vertical_${label}")
  file(REMOVE_RECURSE "${case_dir}")
  file(MAKE_DIRECTORY "${case_dir}")

  set(output_dir "fire_output_${label}")

  execute_process(
    COMMAND "${RESPONSE_EXE}"
            "${input_path}"
            "fire.coupling_mode=${mode}"
            "fire.reference_height_agl_m=10.0"
            "fire.feedback_extinction_depth_m=50.0"
            "fire.output_dir=${output_dir}"
            "erf.plot_file_1=plt"
    WORKING_DIRECTORY "${case_dir}"
    RESULT_VARIABLE run_result
    OUTPUT_VARIABLE run_output
    ERROR_VARIABLE run_error
  )

  if(NOT run_result EQUAL 0)
    message(FATAL_ERROR
      "ERF Fire ${label} child failed\n"
      "stdout:\n${run_output}\n"
      "stderr:\n${run_error}")
  endif()

  foreach(required_file IN ITEMS
      perimeter_000001.csv
      raster_000001.csv
      summary.csv)
    if(NOT EXISTS "${case_dir}/${output_dir}/${required_file}")
      message(FATAL_ERROR
        "${label} did not produce ${required_file}")
    endif()
  endforeach()

  file(GLOB plot_dirs
    LIST_DIRECTORIES true
    "${case_dir}/plt*")
  list(LENGTH plot_dirs plot_count)
  if(plot_count LESS 1)
    message(FATAL_ERROR
      "${label} did not produce a plotfile")
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
      "First-step Fire perimeter differs for ${label}")
  endif()

  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E compare_files
            "${control_output}/raster_000001.csv"
            "${candidate_output}/raster_000001.csv"
    RESULT_VARIABLE raster_compare)
  if(NOT raster_compare EQUAL 0)
    message(FATAL_ERROR
      "First-step Fire raster differs for ${label}")
  endif()
endfunction()

make_resolution_input(24 coarse_input)
make_resolution_input(48 medium_input)
make_resolution_input(96 fine_input)

run_case("${coarse_input}" one_way coarse_one_way
         coarse_one_way_plot coarse_one_way_output)
run_case("${coarse_input}" two_way coarse_two_way
         coarse_two_way_plot coarse_two_way_output)

run_case("${medium_input}" one_way medium_one_way
         medium_one_way_plot medium_one_way_output)
run_case("${medium_input}" two_way medium_two_way
         medium_two_way_plot medium_two_way_output)

run_case("${fine_input}" one_way fine_one_way
         fine_one_way_plot fine_one_way_output)
run_case("${fine_input}" two_way fine_two_way
         fine_two_way_plot fine_two_way_output)

compare_first_step(
  "${coarse_one_way_output}"
  "${coarse_two_way_output}"
  "coarse resolution")
compare_first_step(
  "${medium_one_way_output}"
  "${medium_two_way_output}"
  "medium resolution")
compare_first_step(
  "${fine_one_way_output}"
  "${fine_two_way_output}"
  "fine resolution")

execute_process(
  COMMAND "${ANALYSIS_EXE}"
          "analysis.one_way_coarse=${coarse_one_way_plot}"
          "analysis.two_way_coarse=${coarse_two_way_plot}"
          "analysis.one_way_medium=${medium_one_way_plot}"
          "analysis.two_way_medium=${medium_two_way_plot}"
          "analysis.one_way_fine=${fine_one_way_plot}"
          "analysis.two_way_fine=${fine_two_way_plot}"
  WORKING_DIRECTORY "${test_root}"
  RESULT_VARIABLE analysis_result
  OUTPUT_VARIABLE analysis_output
  ERROR_VARIABLE analysis_error
)

if(NOT analysis_result EQUAL 0)
  message(FATAL_ERROR
    "Vertical-resolution analyzer failed\n"
    "stdout:\n${analysis_output}\n"
    "stderr:\n${analysis_error}")
endif()


string(STRIP "${analysis_output}" analysis_output_stripped)
message(STATUS "${analysis_output_stripped}")
message(STATUS
  "Vertical-resolution sensitivity: "
  "H=50 m atmospheric response is stable from dz=10 to 2.5 m")
