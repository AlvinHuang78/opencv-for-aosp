# SPDX-FileCopyrightText: 2023 - 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
#
# SPDX-License-Identifier: Apache-2.0

include("${CMAKE_CURRENT_LIST_DIR}/../../kleidicv/CMakeLists.txt")
include("${CMAKE_CURRENT_LIST_DIR}/../../kleidicv_thread/CMakeLists.txt")

project("KleidiCV OpenCV HAL")

set(KLEIDICV_HAL_VERSION "0.3.0" CACHE INTERNAL "")
set(KLEIDICV_HAL_LIBRARIES "kleidicv_hal" CACHE INTERNAL "")
set(KLEIDICV_HAL_HEADERS "${CMAKE_CURRENT_LIST_DIR}/kleidicv_hal.h" CACHE INTERNAL "")
set(KLEIDICV_HAL_INCLUDE_DIRS "$<TARGET_PROPERTY:kleidicv_thread,INCLUDE_DIRECTORIES>" CACHE INTERNAL "")

add_library(kleidicv_hal STATIC "${CMAKE_CURRENT_LIST_DIR}/kleidicv_hal.cpp")
target_link_libraries(kleidicv_hal PUBLIC kleidicv kleidicv_thread)
target_include_directories(kleidicv_hal PRIVATE
  ${KLEIDICV_HAL_INCLUDE_DIRS}
  ${CMAKE_CURRENT_LIST_DIR}
  ${OpenCV_SOURCE_DIR}/modules/core/include
  ${OpenCV_SOURCE_DIR}/modules/imgproc/include
)
target_compile_options(
  kleidicv_hal
  PRIVATE
    $<TARGET_PROPERTY:kleidicv,COMPILE_OPTIONS>
    "-Wno-old-style-cast"
)

set_target_properties(kleidicv_hal PROPERTIES CXX_STANDARD 17)

if(NOT BUILD_SHARED_LIBS)
  ocv_install_target(kleidicv_neon
    EXPORT OpenCVModules
    ARCHIVE
    DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH}
    COMPONENT dev
  )
  if(KLEIDICV_ENABLE_SVE2)
    ocv_install_target(kleidicv_sve2
      EXPORT OpenCVModules
      ARCHIVE
      DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH}
      COMPONENT dev
    )
  endif()
  if(KLEIDICV_ENABLE_SME2)
    ocv_install_target(kleidicv_sme2
      EXPORT OpenCVModules
      ARCHIVE
      DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH}
      COMPONENT dev
    )
  endif()
  ocv_install_target(kleidicv
    EXPORT OpenCVModules
    ARCHIVE
    DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH}
    COMPONENT dev
  )
  ocv_install_target(kleidicv_thread
    EXPORT OpenCVModules
    ARCHIVE
    DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH}
    COMPONENT dev
  )
  ocv_install_target(kleidicv_hal
    EXPORT OpenCVModules
    ARCHIVE
    DESTINATION ${OPENCV_3P_LIB_INSTALL_PATH}
    COMPONENT dev
  )
endif()
