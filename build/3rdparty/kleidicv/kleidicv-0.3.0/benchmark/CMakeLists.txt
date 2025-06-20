# SPDX-FileCopyrightText: 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.16)

# The requirements to build Google Benchmark are higher than the
# rest of the project so disable benchmarks by default.
# Another reason to disable it by default is that the configure time
# for Google Benchmark is quite noticeable.
option(KLEIDICV_BENCHMARK "Enable KleidiCV benchmarks" OFF)

if(KLEIDICV_BENCHMARK)

include(FetchContent)

# Please update SECURITY.md if adding, removing or changing the version of third party content.
FetchContent_Declare(
  benchmark
  URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.tar.gz
  URL_HASH SHA256=6bc180a57d23d4d9515519f92b0c83d61b05b5bab188961f36ac7b06b0d9e9ce
)
FetchContent_MakeAvailable(benchmark)


set(KLEIDICV_INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/../kleidicv/include
  ${CMAKE_CURRENT_BINARY_DIR}/../kleidicv/include
)
set(KLEIDICV_BENCHMARK_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(KLEIDICV_BENCHMARK_CXX_FLAGS
  "-Wall"
  "-Wextra"
  "-Wold-style-cast"
)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  list(APPEND KLEIDICV_BENCHMARK_CXX_FLAGS "-O0" "-g")
else()
  list(APPEND KLEIDICV_BENCHMARK_CXX_FLAGS "-O2" "-g0")
endif()

file(GLOB kleidicv_benchmark_sources CONFIGURE_DEPENDS "*.h" "*.cpp")

set_source_files_properties(
  ${kleidicv_benchmark_sources}
  PROPERTIES COMPILE_OPTIONS "${KLEIDICV_BENCHMARK_CXX_FLAGS}"
)

add_executable(
  kleidicv-benchmark
  ${kleidicv_benchmark_sources}
)

set_target_properties(
  kleidicv-benchmark
  PROPERTIES CXX_STANDARD 17
)

target_include_directories(
  kleidicv-benchmark
  PRIVATE ${KLEIDICV_INCLUDE_DIR}
  PRIVATE ${KLEIDICV_BENCHMARK_INCLUDE_DIR}
)

target_link_libraries(
  kleidicv-benchmark
  kleidicv
  benchmark::benchmark
)

endif(KLEIDICV_BENCHMARK)
