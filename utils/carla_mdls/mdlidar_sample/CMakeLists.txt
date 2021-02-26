cmake_minimum_required(VERSION 2.8)
project(mdlidar_example)

# Activate c++11
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
  message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

# include_directories(include)

# protobuf
include_directories("${PROJECT_SOURCE_DIR}/protobuf/protobuf-install/include")
set(Protobuf_Static_Libraries "${PROJECT_SOURCE_DIR}/protobuf/protobuf-install/lib/libprotobuf.a")
 
file(GLOB SOURCES "src/*.cc")
 
add_executable(read_state src/read_state.cpp ${SOURCES})
target_link_libraries(read_state 
  "${Protobuf_Static_Libraries}"
)

add_executable(read_lidar src/read_lidar.cpp ${SOURCES})
target_link_libraries(read_lidar 
  "${Protobuf_Static_Libraries}"
)
