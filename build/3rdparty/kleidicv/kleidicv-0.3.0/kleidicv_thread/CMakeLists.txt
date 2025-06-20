# SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)

project("KleidiCV thread")

file(GLOB KLEIDICV_THREAD_SOURCES
  "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp"
)

set(KLEIDICV_THREAD_CXX_FLAGS
  "-fno-exceptions"
  "-fno-rtti"
  "-Wall"
  "-Wextra"
  "-Wold-style-cast"
  "-Wno-shadow" # GCC's shadow declaration check is too sensitive for the library
)

if (KLEIDICV_CHECK_BANNED_FUNCTIONS)
  # The `SHELL:` prefix is used to turn off de-duplication of compiler flags,
  # it is necessary if other headers need to be force included.
  # https://cmake.org/cmake/help/latest/command/target_compile_options.html#option-de-duplication
  list(APPEND KLEIDICV_THREAD_CXX_FLAGS "SHELL:-include kleidicv/unsafe.h")
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  list(APPEND KLEIDICV_THREAD_CXX_FLAGS "-O0" "-g")
else()
  list(APPEND KLEIDICV_THREAD_CXX_FLAGS "-O2" "-g0")
endif()

add_library(kleidicv_thread STATIC ${KLEIDICV_THREAD_SOURCES})
target_include_directories(kleidicv_thread
  PRIVATE "${CMAKE_CURRENT_LIST_DIR}/include" $<TARGET_PROPERTY:kleidicv,INCLUDE_DIRECTORIES>
)
set_target_properties(kleidicv_thread PROPERTIES CXX_STANDARD 17)
target_compile_options(kleidicv_thread PRIVATE ${KLEIDICV_THREAD_CXX_FLAGS})
target_link_libraries(kleidicv_thread PRIVATE kleidicv)
