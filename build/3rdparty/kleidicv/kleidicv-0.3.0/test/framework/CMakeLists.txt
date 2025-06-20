# SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

file(GLOB kleidicv_framework_test_sources CONFIGURE_DEPENDS "*.h" "test_*.cpp")

list(APPEND kleidicv_framework_test_sources ${KLEIDICV_TEST_FRAMEWORK_SOURCES})

set_source_files_properties(
  ${kleidicv_framework_test_sources}
  PROPERTIES COMPILE_OPTIONS "${KLEIDICV_TEST_CXX_FLAGS}"
)

add_executable(
  kleidicv-framework-test
  ${kleidicv_framework_test_sources}
)

set_target_properties(
  kleidicv-framework-test
  PROPERTIES CXX_STANDARD 17
)

target_include_directories(
  kleidicv-framework-test
  PRIVATE ${KLEIDICV_INCLUDE_DIR}
  PRIVATE ${KLEIDICV_TEST_INCLUDE_DIR}
)

if (KLEIDICV_ALLOCATION_TESTS)
  target_link_options(
    kleidicv-framework-test
    PRIVATE -Wl,--wrap,malloc
  )
endif()


target_link_libraries(
  kleidicv-framework-test
  kleidicv
  gtest_main
)
