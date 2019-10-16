cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(chamfer_distance)

# find_package(Torch REQUIRED)
add_library(nativelib STATIC torch_cd/torch_cd.cpp)
add_executable(chamfer_distance chamfer_distance.cpp)
set_property(TARGET nativelib PROPERTY CXX_STANDARD 11)

include_directories(libtorch/include/torch/csrc/api/include libtorch/include)
find_package( PCL 1.8 REQUIRED PATHS /usr/lib/x86_64-linux-gnu/cmake/pcl)
# PATHS /usr/lib/x86_64-linux-gnu/cmake/pcl
if( PCL_FOUND )
  # [C/C++]>[General]>[Additional Include Directories]
  include_directories( ${PCL_INCLUDE_DIRS} )
  message(${PCL_INCLUDE_DIRS})
  # [C/C++]>[Preprocessor]>[Preprocessor Definitions]
  add_definitions( ${PCL_DEFINITIONS} )
  
  # For Use Not PreCompiled Features 
  #add_definitions( -DPCL_NO_PRECOMPILE )

  # [Linker]>[General]>[Additional Library Directories]
  link_directories( ${PCL_LIBRARY_DIRS})

  # [Linker]>[Input]>[Additional Dependencies]
endif()

set(CMAKE_PREFIX_PATH "libtorch")
find_package(Torch REQUIRED)
target_link_libraries(nativelib ${TORCH_LIBRARIES})
FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

target_link_libraries(chamfer_distance ${Boost_LIBRARIES} )
set(CMAKE_CXX_FLAGS "-lboost_program_options")

target_link_libraries(chamfer_distance nativelib ${PCL_LIBRARIES}) # ${TORCH_LIBRARIES})