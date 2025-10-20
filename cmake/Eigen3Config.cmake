# This file exports the Eigen3::Eigen CMake target
# which should be passed to the target_link_libraries command.

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}" PATH)

set(EIGEN3_FOUND 1)
set(EIGEN3_INCLUDE_DIR ${PACKAGE_PREFIX_DIR}/include)
#include_directories(${EIGEN3_INCLUDE_DIR})

set(Eigen3_VERSION "3.4.1")

if (NOT TARGET Eigen3::Eigen)
# Create imported target Eigen3::Eigen
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${EIGEN3_INCLUDE_DIR})
endif (NOT TARGET Eigen3::Eigen)

# Cleanup temporary variables.
set(PACKAGE_PREFIX_DIR)
