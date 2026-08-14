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
    ${CMAKE_CURRENT_SOURCE_DIR}/ERF_GTestFireWindAdjustment.cpp
)
target_sources(erf_fire_unit_tests
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ERF_GTestFireLevel0Environment.cpp
)
target_sources(erf_fire_unit_tests
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ERF_GTestFireTerrainReferenceWind.cpp
)
target_sources(erf_fire_unit_tests
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ERF_GTestFireTerrainSurface.cpp
)
target_sources(erf_fire_unit_tests
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ERF_GTestFireSpreadRuntime.cpp
)
target_sources(erf_fire_unit_tests
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ERF_GTestFireWindScience.cpp
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
    ${CMAKE_CURRENT_SOURCE_DIR}/ERF_GTestFireFeedbackProfileSensitivity.cpp
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

add_test(
  NAME FireTimeStepIntegration.OneWayTerrainSpreadPreservesPrognosticBits
  COMMAND ${CMAKE_COMMAND}
    -DIDENTITY_EXE=$<TARGET_FILE:erf_fire_timestep_identity>
    -DIDENTITY_INPUT=${CMAKE_CURRENT_SOURCE_DIR}/ERF_FireOneStepTerrain.inputs
    -DIDENTITY_CASE_PREFIX=fire_terrain_identity
    -P ${CMAKE_CURRENT_SOURCE_DIR}/RunFireOneStepIdentity.cmake
)
set_tests_properties(
  FireTimeStepIntegration.OneWayTerrainSpreadPreservesPrognosticBits
  PROPERTIES
    LABELS unit
    TIMEOUT 900
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_test(
  NAME FireTimeStepIntegration.TwoWayTerrainFeedbackChangesAtmosphere
  COMMAND ${CMAKE_COMMAND}
    -DIDENTITY_EXE=$<TARGET_FILE:erf_fire_timestep_identity>
    -DTWO_WAY_INPUT=${CMAKE_CURRENT_SOURCE_DIR}/ERF_FireTwoWayTerrain.inputs
    -DTWO_WAY_CASE_PREFIX=fire_two_way_terrain
    -P ${CMAKE_CURRENT_SOURCE_DIR}/RunFireTwoWayFeedback.cmake
)
set_tests_properties(
  FireTimeStepIntegration.TwoWayTerrainFeedbackChangesAtmosphere
  PROPERTIES
    LABELS unit
    TIMEOUT 900
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
add_test(
  NAME FireTimeStepIntegration.OneWayExplicitWafPreservesPrognosticBits
  COMMAND ${CMAKE_COMMAND}
    -DIDENTITY_EXE=$<TARGET_FILE:erf_fire_timestep_identity>
    -DIDENTITY_INPUT=${CMAKE_CURRENT_SOURCE_DIR}/ERF_FireOneStepIdentityWaf.inputs
    -DIDENTITY_CASE_PREFIX=fire_waf_identity
    -P ${CMAKE_CURRENT_SOURCE_DIR}/RunFireOneStepIdentity.cmake
)
set_tests_properties(
  FireTimeStepIntegration.OneWayExplicitWafPreservesPrognosticBits
  PROPERTIES
    LABELS unit
    TIMEOUT 900
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_test(
  NAME FireTimeStepIntegration.TwoWayExplicitWafFeedbackChangesAtmosphere
  COMMAND ${CMAKE_COMMAND}
    -DIDENTITY_EXE=$<TARGET_FILE:erf_fire_timestep_identity>
    -DTWO_WAY_INPUT=${CMAKE_CURRENT_SOURCE_DIR}/ERF_FireTwoWayFeedbackWaf.inputs
    -DTWO_WAY_CASE_PREFIX=fire_two_way_waf
    -P ${CMAKE_CURRENT_SOURCE_DIR}/RunFireTwoWayFeedback.cmake
)
set_tests_properties(
  FireTimeStepIntegration.TwoWayExplicitWafFeedbackChangesAtmosphere
  PROPERTIES
    LABELS unit
    TIMEOUT 900
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_test(
  NAME FireTimeStepIntegration.OneWayTerrainExplicitWafPreservesPrognosticBits
  COMMAND ${CMAKE_COMMAND}
    -DIDENTITY_EXE=$<TARGET_FILE:erf_fire_timestep_identity>
    -DIDENTITY_INPUT=${CMAKE_CURRENT_SOURCE_DIR}/ERF_FireOneStepTerrainWaf.inputs
    -DIDENTITY_CASE_PREFIX=fire_terrain_waf_identity
    -P ${CMAKE_CURRENT_SOURCE_DIR}/RunFireOneStepIdentity.cmake
)
set_tests_properties(
  FireTimeStepIntegration.OneWayTerrainExplicitWafPreservesPrognosticBits
  PROPERTIES
    LABELS unit
    TIMEOUT 900
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_test(
  NAME FireTimeStepIntegration.TwoWayTerrainExplicitWafFeedbackChangesAtmosphere
  COMMAND ${CMAKE_COMMAND}
    -DIDENTITY_EXE=$<TARGET_FILE:erf_fire_timestep_identity>
    -DTWO_WAY_INPUT=${CMAKE_CURRENT_SOURCE_DIR}/ERF_FireTwoWayTerrainWaf.inputs
    -DTWO_WAY_CASE_PREFIX=fire_two_way_terrain_waf
    -P ${CMAKE_CURRENT_SOURCE_DIR}/RunFireTwoWayFeedback.cmake
)
set_tests_properties(
  FireTimeStepIntegration.TwoWayTerrainExplicitWafFeedbackChangesAtmosphere
  PROPERTIES
    LABELS unit
    TIMEOUT 900
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_test(
  NAME FireWindModeIntegration.FlatIdentityAndReduction
  COMMAND ${CMAKE_COMMAND}
    -DIDENTITY_EXE=$<TARGET_FILE:erf_fire_timestep_identity>
    -DDIRECT_INPUT=${CMAKE_CURRENT_SOURCE_DIR}/ERF_FireOneStepIdentity.inputs
    -DWAF_INPUT=${CMAKE_CURRENT_SOURCE_DIR}/ERF_FireOneStepIdentityWaf.inputs
    -DWIND_MODE_CASE_PREFIX=fire_wind_mode_flat
    -P ${CMAKE_CURRENT_SOURCE_DIR}/RunFireWindModeComparison.cmake
)
set_tests_properties(
  FireWindModeIntegration.FlatIdentityAndReduction
  PROPERTIES
    LABELS unit
    TIMEOUT 900
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_test(
  NAME FireWindModeIntegration.TerrainIdentityAndReduction
  COMMAND ${CMAKE_COMMAND}
    -DIDENTITY_EXE=$<TARGET_FILE:erf_fire_timestep_identity>
    -DDIRECT_INPUT=${CMAKE_CURRENT_SOURCE_DIR}/ERF_FireOneStepTerrain.inputs
    -DWAF_INPUT=${CMAKE_CURRENT_SOURCE_DIR}/ERF_FireOneStepTerrainWaf.inputs
    -DWIND_MODE_CASE_PREFIX=fire_wind_mode_terrain
    -P ${CMAKE_CURRENT_SOURCE_DIR}/RunFireWindModeComparison.cmake
)
set_tests_properties(
  FireWindModeIntegration.TerrainIdentityAndReduction
  PROPERTIES
    LABELS unit
    TIMEOUT 900
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

add_test(
  NAME FireWindModeIntegration.RuntimeInputValidation
  COMMAND ${CMAKE_COMMAND}
    -DIDENTITY_EXE=$<TARGET_FILE:erf_fire_timestep_identity>
    -DDIRECT_INPUT=${CMAKE_CURRENT_SOURCE_DIR}/ERF_FireOneStepIdentity.inputs
    -DWAF_INPUT=${CMAKE_CURRENT_SOURCE_DIR}/ERF_FireOneStepIdentityWaf.inputs
    -P ${CMAKE_CURRENT_SOURCE_DIR}/RunFireWindModeValidation.cmake
)
set_tests_properties(
  FireWindModeIntegration.RuntimeInputValidation
  PROPERTIES
    LABELS unit
    TIMEOUT 300
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)


# FIRE_ATMOSPHERIC_RESPONSE_BEGIN
add_executable(erf_fire_atmospheric_response
  ${PROJECT_SOURCE_DIR}/Exec/ERF_Prob.cpp
)
build_erf_exe(erf_fire_atmospheric_response)
add_dependencies(erf_fire_unit_tests erf_fire_atmospheric_response)

add_executable(erf_fire_atmospheric_response_analysis
  ${CMAKE_CURRENT_SOURCE_DIR}/ERF_FireAtmosphericResponseAnalysis.cpp
)
target_link_libraries(
  erf_fire_atmospheric_response_analysis
  PRIVATE
    AMReX::amrex
)
set_erf_compile_flags(erf_fire_atmospheric_response_analysis)
add_dependencies(erf_fire_unit_tests erf_fire_atmospheric_response_analysis)

add_test(
  NAME FireAtmosphericResponse.FlatGravityOnCheckpoint
  COMMAND ${CMAKE_COMMAND}
    -DRESPONSE_EXE=$<TARGET_FILE:erf_fire_atmospheric_response>
    -DANALYSIS_EXE=$<TARGET_FILE:erf_fire_atmospheric_response_analysis>
    -DRESPONSE_INPUT=${CMAKE_CURRENT_SOURCE_DIR}/ERF_FireAtmosphericResponse.inputs
    -P ${CMAKE_CURRENT_SOURCE_DIR}/RunFireAtmosphericResponse.cmake
)
set_tests_properties(
  FireAtmosphericResponse.FlatGravityOnCheckpoint
  PROPERTIES
    LABELS unit
    TIMEOUT 900
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
# FIRE_ATMOSPHERIC_RESPONSE_END


# FIRE_ATMOSPHERIC_RESPONSE_SENSITIVITY_BEGIN
add_executable(erf_fire_atmospheric_response_sensitivity_analysis
  ${CMAKE_CURRENT_SOURCE_DIR}/ERF_FireAtmosphericResponseSensitivityAnalysis.cpp
)
target_link_libraries(
  erf_fire_atmospheric_response_sensitivity_analysis
  PRIVATE
    AMReX::amrex
)
set_erf_compile_flags(
  erf_fire_atmospheric_response_sensitivity_analysis)
add_dependencies(
  erf_fire_unit_tests
  erf_fire_atmospheric_response_sensitivity_analysis)

add_test(
  NAME FireAtmosphericResponse.ExtinctionDepthSensitivity
  COMMAND ${CMAKE_COMMAND}
    -DRESPONSE_EXE=$<TARGET_FILE:erf_fire_atmospheric_response>
    -DANALYSIS_EXE=$<TARGET_FILE:erf_fire_atmospheric_response_sensitivity_analysis>
    -DRESPONSE_INPUT=${CMAKE_CURRENT_SOURCE_DIR}/ERF_FireAtmosphericResponse.inputs
    -P ${CMAKE_CURRENT_SOURCE_DIR}/RunFireAtmosphericResponseSensitivity.cmake
)
set_tests_properties(
  FireAtmosphericResponse.ExtinctionDepthSensitivity
  PROPERTIES
    LABELS unit
    TIMEOUT 900
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
# FIRE_ATMOSPHERIC_RESPONSE_SENSITIVITY_END


# FIRE_ATMOSPHERIC_VERTICAL_SENSITIVITY_BEGIN
add_executable(erf_fire_atmospheric_vertical_sensitivity_analysis
  ${CMAKE_CURRENT_SOURCE_DIR}/ERF_FireAtmosphericVerticalSensitivityAnalysis.cpp
)
target_link_libraries(
  erf_fire_atmospheric_vertical_sensitivity_analysis
  PRIVATE
    AMReX::amrex
)
set_erf_compile_flags(
  erf_fire_atmospheric_vertical_sensitivity_analysis)
add_dependencies(
  erf_fire_unit_tests
  erf_fire_atmospheric_vertical_sensitivity_analysis)

add_test(
  NAME FireAtmosphericResponse.VerticalResolutionSensitivity
  COMMAND ${CMAKE_COMMAND}
    -DRESPONSE_EXE=$<TARGET_FILE:erf_fire_atmospheric_response>
    -DANALYSIS_EXE=$<TARGET_FILE:erf_fire_atmospheric_vertical_sensitivity_analysis>
    -DRESPONSE_INPUT=${CMAKE_CURRENT_SOURCE_DIR}/ERF_FireAtmosphericResponse.inputs
    -P ${CMAKE_CURRENT_SOURCE_DIR}/RunFireAtmosphericVerticalSensitivity.cmake
)
set_tests_properties(
  FireAtmosphericResponse.VerticalResolutionSensitivity
  PROPERTIES
    LABELS unit
    TIMEOUT 900
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
# FIRE_ATMOSPHERIC_VERTICAL_SENSITIVITY_END
