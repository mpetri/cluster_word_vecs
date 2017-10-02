cmake_minimum_required (VERSION 3.2.2)
project (cluster-word-vecs CXX)

set(Boost_NO_SYSTEM_PATHS on)
set(BOOST_ROOT "/apps/boost/1.64.0-gcc4.9/")
#set(BOOST_LIBRARYDIR "/apps/boost/1.64.0-gcc4.9/lib/GNU/")
find_package(Boost 1.64.0 COMPONENTS system program_options filesystem timer REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2" )

set(CUDA_TOOLKIT_ROOT_DIR "/apps/cuda/8.0/")
set(CUDA_ARCH 60) # 61 for newer cpus

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/kmcuda/src/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/plog/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include/)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/external/kmcuda/src/)

# OUR BINS
#link_directories(${BOOST_LIBRARYDIR})
add_executable(cluster-word-vecs.x src/cluster_word_vecs.cpp)
target_link_libraries(cluster-word-vecs.x ${Boost_LIBRARIES} KMCUDA)


