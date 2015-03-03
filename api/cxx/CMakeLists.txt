
project(PCB_DATASET)
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
ADD_DEFINITIONS(-std=c++11)

include_directories("${PCB_DATASET_SOURCE_DIR}")

# ------------------------------------------------------------------- libraries

# Boost

set(BOOST_MIN_VERSION "1.50.0")
find_package(Boost ${BOOST_MIN_VERSION} COMPONENTS system filesystem program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

# OpenCV

set(OPENCV_MIN_VERSION "2.4.0")
find_package(OpenCV COMPONENTS core highgui imgproc REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
if(${OpenCV_VERSION} VERSION_LESS ${OPENCV_MIN_VERSION})
    message(FATAL_ERROR "OpenCV >= 2.4.0 required")
endif()

# ----------------------------------------------------------------------- build

# library
add_library(pcbdataset SHARED pcbdataset.cpp)

# example
add_executable(example_pcbdataset example.cpp)
target_link_libraries(example_pcbdataset pcbdataset ${Boost_LIBRARIES} ${OpenCV_LIBS})

# --------------------------------------------------------------------- install

INSTALL(DIRECTORY ${PCB_DATASET_SOURCE_DIR}/ DESTINATION include/pcbdataset FILES_MATCHING PATTERN "*.hpp")
INSTALL(TARGETS example_pcbdataset DESTINATION bin)
INSTALL(TARGETS pcbdataset DESTINATION lib)
