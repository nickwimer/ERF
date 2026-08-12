if(NOT DEFINED IDENTITY_EXE OR "${IDENTITY_EXE}" STREQUAL "")
  message(FATAL_ERROR "IDENTITY_EXE is required")
endif()
if(NOT DEFINED IDENTITY_INPUT OR "${IDENTITY_INPUT}" STREQUAL "")
  message(FATAL_ERROR "IDENTITY_INPUT is required")
endif()

file(REAL_PATH "." identity_root)

function(run_identity_case mode label checkpoint_var)
  set(case_dir "${identity_root}/fire_identity_${label}")
  file(REMOVE_RECURSE "${case_dir}")
  file(MAKE_DIRECTORY "${case_dir}")

  execute_process(
    COMMAND "${IDENTITY_EXE}"
            "${IDENTITY_INPUT}"
            "fire.environment_read=${mode}"
            "erf.check_file=chk"
    WORKING_DIRECTORY "${case_dir}"
    RESULT_VARIABLE run_result
    OUTPUT_VARIABLE run_output
    ERROR_VARIABLE run_error
  )

  if(NOT run_result EQUAL 0)
    message(FATAL_ERROR
      "ERF fire identity child failed for environment_read=${mode}\n"
      "stdout:\n${run_output}\n"
      "stderr:\n${run_error}")
  endif()

  if(NOT run_output MATCHES "ERF_FIRE_IDENTITY_CHILD_OK=1")
    message(FATAL_ERROR
      "ERF fire identity child did not report successful completion "
      "for environment_read=${mode}\n"
      "stdout:\n${run_output}\n"
      "stderr:\n${run_error}")
  endif()

  file(GLOB checkpoint_dirs LIST_DIRECTORIES true "${case_dir}/chk*")
  list(LENGTH checkpoint_dirs checkpoint_count)
  if(NOT checkpoint_count EQUAL 1)
    message(FATAL_ERROR
      "expected exactly one identity checkpoint for environment_read=${mode}, "
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
run_identity_case(1 fire_read fire_checkpoint)

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E compare_files
          "${control_checkpoint}/Header"
          "${fire_checkpoint}/Header"
  RESULT_VARIABLE header_compare_result
)
if(NOT header_compare_result EQUAL 0)
  message(FATAL_ERROR
    "fire environment read changed one-step checkpoint time/layout metadata")
endif()

collect_prognostic_files("${control_checkpoint}" control_files)
collect_prognostic_files("${fire_checkpoint}" fire_files)

if(NOT "${control_files}" STREQUAL "${fire_files}")
  message(FATAL_ERROR
    "control and fire-read checkpoints have different prognostic file sets:\n"
    "control=${control_files}\n"
    "fire_read=${fire_files}")
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
      "fire environment read changed one-step prognostic bits in ${relative_file}")
  endif()
endforeach()

message(STATUS
  "Fire one-way read identity PASS: checkpoint prognostic files are bitwise equal")
