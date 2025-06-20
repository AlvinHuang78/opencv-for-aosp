# SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

file(GLOB kleidicv_api_test_sources CONFIGURE_DEPENDS "*.h" "*.cpp")

list(APPEND kleidicv_api_test_sources ${KLEIDICV_TEST_FRAMEWORK_SOURCES})

set_source_files_properties(
  ${kleidicv_api_test_sources}
  PROPERTIES COMPILE_OPTIONS "${KLEIDICV_TEST_CXX_FLAGS}"
)

add_executable(
  kleidicv-api-test
  ${kleidicv_api_test_sources}
)

set_target_properties(
  kleidicv-api-test
  PROPERTIES CXX_STANDARD 17
)

target_include_directories(
  kleidicv-api-test
  PRIVATE ${KLEIDICV_INCLUDE_DIR}
  PRIVATE ${KLEIDICV_TEST_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../kleidicv_thread/include
)

if (KLEIDICV_ALLOCATION_TESTS)
  target_link_options(
    kleidicv-api-test
    PRIVATE -Wl,--wrap,malloc
  )
endif()


target_link_libraries(
  kleidicv-api-test
  kleidicv
  kleidicv_thread
  gtest
  gmock
)
