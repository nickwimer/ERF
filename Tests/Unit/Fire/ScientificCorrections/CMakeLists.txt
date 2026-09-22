set(_fire_science_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/ERF_GTestFireFuelBurnTime.cpp
)
if(ERF_ENABLE_CUDA)
  set_source_files_properties(${_fire_science_sources}
    TARGET_DIRECTORY erf_fire_unit_tests
    PROPERTIES LANGUAGE CUDA)
endif()
target_sources(erf_fire_unit_tests PRIVATE ${_fire_science_sources})
