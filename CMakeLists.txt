cmake_minimum_required (VERSION 2.6)
project(fast_dqn)

option(CAFFE_CPU_ONLY "Use CPU only for Caffe" ON)

# Edit these paths to point to your caffe & minecraft installations
# when building caffe be sure and do "make distribute"
set(CAFFE_ROOT_DIR "dependencies/caffe/distribute_ff16f6e43dd718921e5203f640dd57c68f01cdb3")
set(MINECRAFT_ROOT_DIR "dependencies/minecraft_dqn_interface")

list(APPEND CMAKE_PREFIX_PATH ${CAFFE_ROOT_DIR} ${MINECRAFT_ROOT_DIR})

add_executable(fast_dqn src/fast_dqn_main.cpp src/fast_dqn.cpp src/minecraft_environment.cpp)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native -Wno-deprecated-declarations")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

find_package(Boost 1.40 COMPONENTS filesystem system iostreams regex REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(fast_dqn ${Boost_LIBRARIES})

find_package(GFLAGS REQUIRED)
include_directories(${GFLAGS_INCLUDE_DIR})
target_link_libraries(fast_dqn ${GFLAGS_LIBRARY})

find_package(GLOG REQUIRED)
include_directories(${GLOG_INCLUDE_DIRS})
target_link_libraries(fast_dqn ${GLOG_LIBRARIES})

include(FindProtobuf)
find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIRS})
target_link_libraries(fast_dqn ${PROTOBUF_LIBRARIES})

find_package(CAFFE REQUIRED)
include_directories(${CAFFE_INCLUDE_DIRS})
target_link_libraries(fast_dqn ${CAFFE_LIBRARIES})

find_package(MINECRAFT REQUIRED)
include_directories(${MINECRAFT_INCLUDE_DIRS})
target_link_libraries(fast_dqn ${MINECRAFT_LIBRARIES})

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
target_link_libraries(fast_dqn ${PYTHON_LIBRARIES})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(fast_dqn ${OpenCV_LIBS})

find_package(HDF5 COMPONENTS HL QUIET)
include_directories(SYSTEM ${HDF5_INCLUDE_DIRS} ${HDF5_HL_INCLUDE_DIR})
list(APPEND LINK_LIBS ${HDF5_LIBRARIES})


if(CAFFE_CPU_ONLY)
  add_definitions(-DCPU_ONLY)
endif()
