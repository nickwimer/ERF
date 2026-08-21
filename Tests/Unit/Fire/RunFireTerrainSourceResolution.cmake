if(NOT DEFINED IDENTITY_EXE OR "${IDENTITY_EXE}" STREQUAL "")
  message(FATAL_ERROR "IDENTITY_EXE is required")
endif()
if(NOT DEFINED IDENTITY_INPUT OR "${IDENTITY_INPUT}" STREQUAL "")
  message(FATAL_ERROR "IDENTITY_INPUT is required")
endif()
if(NOT DEFINED FLAT_TERRAIN OR "${FLAT_TERRAIN}" STREQUAL "")
  message(FATAL_ERROR "FLAT_TERRAIN is required")
endif()
if(NOT DEFINED FINE_TERRAIN OR "${FINE_TERRAIN}" STREQUAL "")
  message(FATAL_ERROR "FINE_TERRAIN is required")
endif()

foreach(required_path IN ITEMS
    "${IDENTITY_EXE}"
    "${IDENTITY_INPUT}"
    "${FLAT_TERRAIN}"
    "${FINE_TERRAIN}")
  if(NOT EXISTS "${required_path}")
    message(FATAL_ERROR "missing terrain-source resolution test input: ${required_path}")
  endif()
endforeach()

file(REAL_PATH "." test_root)

function(run_terrain_case label terrain_file case_var checkpoint_var)
  set(case_dir "${test_root}/fire_terrain_source_${label}")
  file(REMOVE_RECURSE "${case_dir}")
  file(MAKE_DIRECTORY "${case_dir}")

  execute_process(
    COMMAND "${IDENTITY_EXE}"
            "${IDENTITY_INPUT}"
            "erf.terrain_file_name=${terrain_file}"
            "fire.output_dir=fire_output"
            "erf.check_file=chk"
    WORKING_DIRECTORY "${case_dir}"
    RESULT_VARIABLE run_result
    OUTPUT_VARIABLE run_output
    ERROR_VARIABLE run_error
  )

  if(NOT run_result EQUAL 0)
    message(FATAL_ERROR
      "terrain-source ${label} child failed\n"
      "stdout:\n${run_output}\n"
      "stderr:\n${run_error}")
  endif()

  if(NOT run_output MATCHES "ERF_FIRE_IDENTITY_CHILD_OK=1")
    message(FATAL_ERROR
      "terrain-source ${label} child did not report successful completion\n"
      "stdout:\n${run_output}\n"
      "stderr:\n${run_error}")
  endif()

  foreach(required_file IN ITEMS
      perimeter_000000.csv
      perimeter_000005.csv
      raster_000000.csv
      raster_000005.csv)
    if(NOT EXISTS "${case_dir}/fire_output/${required_file}")
      message(FATAL_ERROR
        "terrain-source ${label} case did not produce ${required_file}")
    endif()
  endforeach()

  file(GLOB checkpoint_dirs LIST_DIRECTORIES true "${case_dir}/chk*")
  list(LENGTH checkpoint_dirs checkpoint_count)
  if(NOT checkpoint_count EQUAL 1)
    message(FATAL_ERROR
      "terrain-source ${label} case expected exactly one checkpoint, found ${checkpoint_count}: ${checkpoint_dirs}")
  endif()

  list(GET checkpoint_dirs 0 checkpoint_dir)
  set(${case_var} "${case_dir}" PARENT_SCOPE)
  set(${checkpoint_var} "${checkpoint_dir}" PARENT_SCOPE)
endfunction()

function(compare_equal left right description)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E compare_files "${left}" "${right}"
    RESULT_VARIABLE compare_result
  )
  if(NOT compare_result EQUAL 0)
    message(FATAL_ERROR "${description} are not bitwise equal")
  endif()
endfunction()

function(compare_different left right description)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E compare_files "${left}" "${right}"
    RESULT_VARIABLE compare_result
  )
  if(compare_result EQUAL 0)
    message(FATAL_ERROR "${description} unexpectedly remained bitwise equal")
  endif()
endfunction()

function(collect_prognostic_files checkpoint_dir output_var)
  set(level_dir "${checkpoint_dir}/Level_0")
  if(NOT IS_DIRECTORY "${level_dir}")
    message(FATAL_ERROR "missing Level_0 in ${checkpoint_dir}")
  endif()

  set(all_files)
  foreach(prefix IN ITEMS Cell XFace YFace ZFace)
    file(GLOB prefix_files
      RELATIVE "${level_dir}"
      "${level_dir}/${prefix}_*")
    list(LENGTH prefix_files prefix_count)
    if(prefix_count EQUAL 0)
      message(FATAL_ERROR
        "terrain-source checkpoint is missing ${prefix} files: ${level_dir}")
    endif()
    list(APPEND all_files ${prefix_files})
  endforeach()

  list(REMOVE_DUPLICATES all_files)
  list(SORT all_files)
  set(${output_var} "${all_files}" PARENT_SCOPE)
endfunction()

run_terrain_case(flat "${FLAT_TERRAIN}" flat_case flat_checkpoint)
run_terrain_case(fine "${FINE_TERRAIN}" fine_case fine_checkpoint)

compare_equal(
  "${flat_case}/fire_output/perimeter_000000.csv"
  "${fine_case}/fire_output/perimeter_000000.csv"
  "initial Fire perimeters")
compare_equal(
  "${flat_case}/fire_output/raster_000000.csv"
  "${fine_case}/fire_output/raster_000000.csv"
  "initial Fire rasters")
compare_equal(
  "${flat_checkpoint}/Header"
  "${fine_checkpoint}/Header"
  "atmospheric checkpoint headers")

collect_prognostic_files("${flat_checkpoint}" flat_files)
collect_prognostic_files("${fine_checkpoint}" fine_files)

if(NOT "${flat_files}" STREQUAL "${fine_files}")
  message(FATAL_ERROR
    "flat and fine terrain cases have different atmospheric prognostic file sets")
endif()

foreach(relative_file IN LISTS flat_files)
  compare_equal(
    "${flat_checkpoint}/Level_0/${relative_file}"
    "${fine_checkpoint}/Level_0/${relative_file}"
    "atmospheric prognostic file ${relative_file}")
endforeach()

compare_different(
  "${flat_case}/fire_output/perimeter_000005.csv"
  "${fine_case}/fire_output/perimeter_000005.csv"
  "final Fire perimeters")
compare_different(
  "${flat_case}/fire_output/raster_000005.csv"
  "${fine_case}/fire_output/raster_000005.csv"
  "final Fire rasters")

message(STATUS
  "Independent terrain source resolution: atmosphere is bitwise identical while fine-grid Fire propagation responds to sub-atmospheric-grid slope")
