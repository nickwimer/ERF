if(NOT DEFINED IDENTITY_EXE OR "${IDENTITY_EXE}" STREQUAL "")
  message(FATAL_ERROR "IDENTITY_EXE is required")
endif()
if(NOT DEFINED IDENTITY_INPUT OR "${IDENTITY_INPUT}" STREQUAL "")
  message(FATAL_ERROR "IDENTITY_INPUT is required")
endif()
if(NOT DEFINED IDENTITY_CASE_PREFIX OR "${IDENTITY_CASE_PREFIX}" STREQUAL "")
  set(IDENTITY_CASE_PREFIX "fire_identity")
endif()

file(REAL_PATH "." identity_root)

function(run_identity_case mode label checkpoint_var)
  set(case_dir "${identity_root}/${IDENTITY_CASE_PREFIX}_${label}")
  file(REMOVE_RECURSE "${case_dir}")
  file(MAKE_DIRECTORY "${case_dir}")

  execute_process(
    COMMAND "${IDENTITY_EXE}"
            "${IDENTITY_INPUT}"
            "fire.enabled=${mode}"
            "erf.check_file=chk"
    WORKING_DIRECTORY "${case_dir}"
    RESULT_VARIABLE run_result
    OUTPUT_VARIABLE run_output
    ERROR_VARIABLE run_error
  )

  if(NOT run_result EQUAL 0)
    message(FATAL_ERROR
      "ERF fire identity child failed for fire.enabled=${mode}\n"
      "stdout:\n${run_output}\n"
      "stderr:\n${run_error}")
  endif()

  if(NOT run_output MATCHES "ERF_FIRE_IDENTITY_CHILD_OK=1")
    message(FATAL_ERROR
      "ERF fire identity child did not report successful completion "
      "for fire.enabled=${mode}\n"
      "stdout:\n${run_output}\n"
      "stderr:\n${run_error}")
  endif()

  if(mode EQUAL 1)
    foreach(required_file IN ITEMS
        summary.csv
        perimeter_000000.csv
        perimeter_000005.csv
        raster_000000.csv
        raster_000005.csv)
      if(NOT EXISTS "${case_dir}/fire_output/${required_file}")
        message(FATAL_ERROR
          "enabled one-way fire run did not produce ${required_file}")
      endif()
    endforeach()

    file(STRINGS
      "${case_dir}/fire_output/summary.csv"
      summary_header
      LIMIT_COUNT 1)
    if(NOT "${summary_header}" STREQUAL
        "step,time_s,vertex_count,burned_area_m2,arrived_cell_count,remaining_dry_fuel_kg,consumed_dry_fuel_kg,sensible_energy_j,water_released_kg,perimeter_file,raster_file")
      message(FATAL_ERROR
        "enabled one-way Fire combustion summary header does not match expected schema: ${summary_header}")
    endif()

    file(STRINGS
      "${case_dir}/fire_output/raster_000005.csv"
      raster_header
      LIMIT_COUNT 1)
    if(NOT "${raster_header}" STREQUAL
        "time_s,i,j,xlo_m,xhi_m,ylo_m,yhi_m,burned_fraction,has_arrived,first_arrival_time_s,ignited_area_fraction,remaining_dry_fuel_kg_m2,consumed_dry_fuel_kg_m2,sensible_energy_j_m2,water_released_kg_m2")
      message(FATAL_ERROR
        "enabled one-way Fire combustion raster header does not match expected schema: ${raster_header}")
    endif()

    if(DEFINED IDENTITY_EXPECTED_RASTER_CELLS)
      file(STRINGS
        "${case_dir}/fire_output/raster_000005.csv"
        raster_lines)
      list(LENGTH raster_lines raster_line_count)
      math(EXPR expected_raster_line_count
        "${IDENTITY_EXPECTED_RASTER_CELLS} + 1")
      if(NOT raster_line_count EQUAL expected_raster_line_count)
        message(FATAL_ERROR
          "enabled one-way Fire raster has ${raster_line_count} lines; expected ${expected_raster_line_count}")
      endif()
    endif()
  elseif(EXISTS "${case_dir}/fire_output")
    message(FATAL_ERROR
      "disabled fire control unexpectedly produced fire_output")
  endif()

  file(GLOB checkpoint_dirs LIST_DIRECTORIES true "${case_dir}/chk*")
  list(LENGTH checkpoint_dirs checkpoint_count)
  if(NOT checkpoint_count EQUAL 1)
    message(FATAL_ERROR
      "expected exactly one identity checkpoint for fire.enabled=${mode}, "
      "found ${checkpoint_count}: ${checkpoint_dirs}")
  endif()

  list(GET checkpoint_dirs 0 checkpoint_dir)
  if(NOT IS_DIRECTORY "${checkpoint_dir}")
    message(FATAL_ERROR
      "identity checkpoint path is not a directory: ${checkpoint_dir}")
  endif()

  set(${checkpoint_var} "${checkpoint_dir}" PARENT_SCOPE)
endfunction()

function(collect_prognostic_files checkpoint_dir output_var)
  set(level_dir "${checkpoint_dir}/Level_0")
  if(NOT IS_DIRECTORY "${level_dir}")
    message(FATAL_ERROR "missing Level_0 in identity checkpoint ${checkpoint_dir}")
  endif()

  set(all_files)
  foreach(prefix IN ITEMS Cell XFace YFace ZFace)
    file(GLOB prefix_files
      RELATIVE "${level_dir}"
      "${level_dir}/${prefix}_*")
    list(LENGTH prefix_files prefix_count)
    if(prefix_count EQUAL 0)
      message(FATAL_ERROR
        "identity checkpoint is missing ${prefix} prognostic files: ${level_dir}")
    endif()
    list(APPEND all_files ${prefix_files})
  endforeach()

  list(REMOVE_DUPLICATES all_files)
  list(SORT all_files)
  set(${output_var} "${all_files}" PARENT_SCOPE)
endfunction()

run_identity_case(0 control control_checkpoint)
run_identity_case(1 one_way fire_checkpoint)

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E compare_files
          "${control_checkpoint}/Header"
          "${fire_checkpoint}/Header"
  RESULT_VARIABLE header_compare_result
)
if(NOT header_compare_result EQUAL 0)
  message(FATAL_ERROR
    "one-way Fire changed multi-step checkpoint time/layout metadata")
endif()

collect_prognostic_files("${control_checkpoint}" control_files)
collect_prognostic_files("${fire_checkpoint}" fire_files)

if(NOT "${control_files}" STREQUAL "${fire_files}")
  message(FATAL_ERROR
    "control and one-way Fire checkpoints have different prognostic file sets:\n"
    "control=${control_files}\n"
    "one_way=${fire_files}")
endif()

foreach(relative_file IN LISTS control_files)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E compare_files
            "${control_checkpoint}/Level_0/${relative_file}"
            "${fire_checkpoint}/Level_0/${relative_file}"
    RESULT_VARIABLE compare_result
  )
  if(NOT compare_result EQUAL 0)
    message(FATAL_ERROR
      "one-way Fire changed atmospheric prognostic bits in ${relative_file}")
  endif()
endforeach()

message(STATUS
  "Fire one-way spread identity: Fire advances while atmospheric checkpoint prognostic files remain bitwise equal")
