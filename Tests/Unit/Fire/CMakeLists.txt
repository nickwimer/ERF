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
  ERF_GTestFireBurnedFraction.cpp
  ERF_GTestFireBurnedFractionRaster.cpp
  ERF_GTestFireCellArrival.cpp
  ERF_GTestFireFirstArrivalRaster.cpp
)

build_erf_exe(erf_fire_unit_tests NO_ERF_MAIN)

target_link_libraries(erf_fire_unit_tests PRIVATE GTest::gtest)

erf_register_gtest_suite(erf_fire_unit_tests
  LABELS unit
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  DISCOVERY_TIMEOUT 60
  EXECUTION_TIMEOUT 900
)
target_sources(erf_fire_unit_tests
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ERF_GTestFireEnvironment.cpp
)
target_sources(erf_fire_unit_tests
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ERF_GTestFireLevel0Environment.cpp
)
target_sources(erf_fire_unit_tests
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ERF_GTestFireSpreadRuntime.cpp
)
target_sources(erf_fire_unit_tests
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ERF_GTestFireCombustion.cpp
)
target_sources(erf_fire_unit_tests
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ERF_GTestFireCombustionRaster.cpp
)
target_sources(erf_fire_unit_tests
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ERF_GTestFireSurfaceFeedback.cpp
)
target_sources(erf_fire_unit_tests
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ERF_GTestFireAtmosphericSource.cpp
)
target_sources(erf_fire_unit_tests
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ERF_GTestFireLevel0SourceCoupling.cpp
)

add_executable(erf_fire_timestep_identity
  ${PROJECT_SOURCE_DIR}/Exec/ERF_Prob.cpp
  ${PROJECT_SOURCE_DIR}/Tests/Unit/ERF_GTestRuntimeStubs.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ERF_FireOneStepIdentity.cpp
)

build_erf_exe(erf_fire_timestep_identity NO_ERF_MAIN)
add_dependencies(erf_fire_unit_tests erf_fire_timestep_identity)

add_test(
  NAME FireTimeStepIntegration.OneWaySpreadPreservesPrognosticBits
  COMMAND ${CMAKE_COMMAND}
    -DIDENTITY_EXE=$<TARGET_FILE:erf_fire_timestep_identity>
    -DIDENTITY_INPUT=${CMAKE_CURRENT_SOURCE_DIR}/ERF_FireOneStepIdentity.inputs
    -P ${CMAKE_CURRENT_SOURCE_DIR}/RunFireOneStepIdentity.cmake
)
set_tests_properties(
  FireTimeStepIntegration.OneWaySpreadPreservesPrognosticBits
  PROPERTIES
    LABELS unit
    TIMEOUT 900
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_test(
  NAME FireTimeStepIntegration.TwoWayFeedbackChangesAtmosphere
  COMMAND ${CMAKE_COMMAND}
    -DIDENTITY_EXE=$<TARGET_FILE:erf_fire_timestep_identity>
    -DTWO_WAY_INPUT=${CMAKE_CURRENT_SOURCE_DIR}/ERF_FireTwoWayFeedback.inputs
    -P ${CMAKE_CURRENT_SOURCE_DIR}/RunFireTwoWayFeedback.cmake
)
set_tests_properties(
  FireTimeStepIntegration.TwoWayFeedbackChangesAtmosphere
  PROPERTIES
    LABELS unit
    TIMEOUT 900
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
