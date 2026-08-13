if(NOT DEFINED IDENTITY_EXE OR "${IDENTITY_EXE}" STREQUAL "")
  message(FATAL_ERROR "IDENTITY_EXE is required")
endif()
if(NOT DEFINED TWO_WAY_INPUT OR "${TWO_WAY_INPUT}" STREQUAL "")
  message(FATAL_ERROR "TWO_WAY_INPUT is required")
endif()

file(REAL_PATH "." test_root)

if(NOT DEFINED TWO_WAY_CASE_PREFIX OR "${TWO_WAY_CASE_PREFIX}" STREQUAL "")
  set(TWO_WAY_CASE_PREFIX "fire_two_way")
endif()

function(run_case mode label checkpoint_var output_var)
  set(case_dir "${test_root}/${TWO_WAY_CASE_PREFIX}_${label}")
  file(REMOVE_RECURSE "${case_dir}")
  file(MAKE_DIRECTORY "${case_dir}")

  set(output_dir "fire_output_${label}")

  execute_process(
    COMMAND "${IDENTITY_EXE}"
            "${TWO_WAY_INPUT}"
            "fire.coupling_mode=${mode}"
            "fire.output_dir=${output_dir}"
            "erf.check_file=chk"
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

  if(NOT run_output MATCHES "ERF_FIRE_IDENTITY_CHILD_OK=1")
    message(FATAL_ERROR
      "ERF Fire ${mode} child did not report success\n"
      "stdout:\n${run_output}\n"
      "stderr:\n${run_error}")
  endif()

  foreach(required_file IN ITEMS
      perimeter_000001.csv
      raster_000001.csv
      summary.csv)
    if(NOT EXISTS "${case_dir}/${output_dir}/${required_file}")
      message(FATAL_ERROR
        "ERF Fire ${mode} run did not produce ${required_file}")
    endif()
  endforeach()

  file(GLOB checkpoint_dirs
    LIST_DIRECTORIES true
    "${case_dir}/chk*")
  list(LENGTH checkpoint_dirs checkpoint_count)
  if(NOT checkpoint_count EQUAL 1)
    message(FATAL_ERROR
      "expected exactly one checkpoint for ${mode}, "
      "found ${checkpoint_count}")
  endif()

  list(GET checkpoint_dirs 0 checkpoint_dir)
  set(${checkpoint_var} "${checkpoint_dir}" PARENT_SCOPE)
  set(${output_var} "${case_dir}/${output_dir}" PARENT_SCOPE)
endfunction()

run_case(one_way one_way one_way_checkpoint one_way_output)
run_case(two_way two_way two_way_checkpoint two_way_output)

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E compare_files
          "${one_way_output}/perimeter_000001.csv"
          "${two_way_output}/perimeter_000001.csv"
  RESULT_VARIABLE perimeter_compare)
if(NOT perimeter_compare EQUAL 0)
  message(FATAL_ERROR
    "first-step Fire perimeter differs between one-way and two-way; "
    "feedback must not affect the already-completed t^n Fire advance")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E compare_files
          "${one_way_output}/raster_000001.csv"
          "${two_way_output}/raster_000001.csv"
  RESULT_VARIABLE raster_compare)
if(NOT raster_compare EQUAL 0)
  message(FATAL_ERROR
    "first-step Fire raster/combustion output differs between one-way and two-way")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E compare_files
          "${one_way_checkpoint}/Header"
          "${two_way_checkpoint}/Header"
  RESULT_VARIABLE header_compare)
if(NOT header_compare EQUAL 0)
  message(FATAL_ERROR
    "one-way and two-way checkpoint time/layout metadata differ")
endif()

file(GLOB one_way_cell_files
  RELATIVE "${one_way_checkpoint}/Level_0"
  "${one_way_checkpoint}/Level_0/Cell_*")
file(GLOB two_way_cell_files
  RELATIVE "${two_way_checkpoint}/Level_0"
  "${two_way_checkpoint}/Level_0/Cell_*")
list(SORT one_way_cell_files)
list(SORT two_way_cell_files)

if(NOT "${one_way_cell_files}" STREQUAL "${two_way_cell_files}")
  message(FATAL_ERROR
    "one-way and two-way checkpoints have different Cell file sets")
endif()

set(differing_cell_files 0)
foreach(relative_file IN LISTS one_way_cell_files)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E compare_files
            "${one_way_checkpoint}/Level_0/${relative_file}"
            "${two_way_checkpoint}/Level_0/${relative_file}"
    RESULT_VARIABLE compare_result)
  if(NOT compare_result EQUAL 0)
    math(EXPR differing_cell_files "${differing_cell_files}+1")
  endif()
endforeach()

if(differing_cell_files EQUAL 0)
  message(FATAL_ERROR
    "two-way Fire source did not change any cell-centered atmospheric checkpoint file")
endif()

message(STATUS
  "Fire two-way feedback PASS: first-step Fire evolution is identical, "
  "native heat/moisture source is active, and atmospheric state changes")
