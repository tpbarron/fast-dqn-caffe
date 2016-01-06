# - Try to find CAFFE
#
# The following variables are optionally searched for defaults
#  CAFFE_ROOT_DIR:            Base directory where all GLOG components are found
#
# The following are set after configuration is done:
#  CAFFE_FOUND
#  CAFFE_INCLUDE_DIRS
#  CAFFE_LIBRARIES

include(FindPackageHandleStandardArgs)

set(MINECRAFT_ROOT_DIR "" CACHE PATH "Folder containing MINECRAFT")

find_path(MINECRAFT_INCLUDE_DIR minecraft_dqn_interface.hpp
  PATHS ${MINECRAFT_ROOT_DIR}
  PATH_SUFFIXES)

find_library(MINECRAFT_LIBRARY minecraft_dqn_interface
  PATHS ${MINECRAFT_ROOT_DIR}
  PATH_SUFFIXES
  build)

find_package_handle_standard_args(MINECRAFT DEFAULT_MSG
  MINECRAFT_INCLUDE_DIR MINECRAFT_LIBRARY)

if(MINECRAFT_FOUND)
  set(MINECRAFT_INCLUDE_DIRS ${MINECRAFT_INCLUDE_DIR})
  set(MINECRAFT_LIBRARIES ${MINECRAFT_LIBRARY})
endif()
