# Eigen3Config.cmake

set(EIGEN3_FOUND 1)
set(Eigen3_VERSION "3.4.1")

set(Eigen3_DIR ${PROJECT_SOURCE_DIR}/include)
#include_directories(${Eigen3_DIR})

add_library(Eigen3::Eigen INTERFACE IMPORTED)
