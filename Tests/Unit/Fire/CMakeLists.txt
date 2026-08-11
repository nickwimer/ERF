if(NOT TARGET GTest::gtest)
  message(FATAL_ERROR
    "erf_fire_unit_tests expects the vendored googletest target from "
    "Tests/Unit/CMakeLists.txt. Configure unit tests through the top-level "
    "ERF test build so Submodules/googletest is added first.")
endif()

include(${PROJECT_SOURCE_DIR}/CMake/BuildERFExe.cmake)
include(${PROJECT_SOURCE_DIR}/CMake/SetERFCompileFlags.cmake)

add_executable(erf_fire_unit_tests
  ${PROJECT_SOURCE_DIR}/Exec/ERF_Prob.cpp
  ${PROJECT_SOURCE_DIR}/Tests/Unit/ERF_GTestMain.cpp
  ${PROJECT_SOURCE_DIR}/Tests/Unit/ERF_GTestRuntimeStubs.cpp
  ERF_GTestFireTypes.cpp
  ERF_GTestFirePerimeter.cpp
  ERF_GTestFirePropagation.cpp
  ERF_GTestFireRemeshing.cpp
  ERF_GTestRothermel.cpp
  ERF_GTestRichardsEllipse.cpp
  ERF_GTestRichardsDirectionalSpread.cpp
  ERF_GTestFireStandaloneGrowth.cpp
  ERF_GTestFireCellCoverage.cpp
)

build_erf_exe(erf_fire_unit_tests NO_ERF_MAIN)

target_link_libraries(erf_fire_unit_tests PRIVATE GTest::gtest)

erf_register_gtest_suite(erf_fire_unit_tests
  LABELS unit
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DISCOVERY_TIMEOUT 60
  EXECUTION_TIMEOUT 900
)
