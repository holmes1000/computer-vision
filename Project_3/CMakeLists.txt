cmake_minimum_required(VERSION 3.10)

# Project details
project(Project_3 VERSION 0.1.0 LANGUAGES C CXX)

# Set OpenCV directory
set(OpenCV_DIR "C:/opencv/build")

# Set ONNX directory
set(ONNX_DIR "C:/onnxruntime-win-x64-1.17.1")

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${ONNX_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_PREFIX_PATH}/include)

# Link directories
include_directories(${CMAKE_SOURCE_DIR})
link_directories(${ONNX_DIR}/lib)
include_directories(${CMAKE_PREFIX_PATH})

# Add source files
set(SOURCES vidDisplay.cpp filters.cpp DA2Network.hpp)

# Add executable target
# add_executable(Project_3 test.cpp)        # use this to test that OpenCV is working
#add_executable(Project_3 test2.cpp)
add_executable(Project_3 ${SOURCES})
# add_executable(Project_3 da2-video.cpp DA2Network.hpp)

# Link ONNX Runtime libraries
target_link_libraries(Project_3 ${ONNX_DIR}/lib/onnxruntime.lib)

# Link OpenCV libraries
target_link_libraries(Project_3 ${OpenCV_LIBS})
