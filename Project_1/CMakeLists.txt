cmake_minimum_required(VERSION 3.10)

# Project details
project(Project_1 VERSION 0.1.0 LANGUAGES C CXX)

# Set OpenCV directory
set(OpenCV_DIR "C:/opencv/build")

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Set ONNX directory
set(ONNX_DIR "C:/onnxruntime-win-x64-1.20.1")

#Qt 5
set(CMAKE_PREFIX_PATH "C:/Qt/6.5.3/msvc2019_64")

# Find Qt5 package
find_package(Qt6 COMPONENTS Widgets REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${ONNX_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_PREFIX_PATH}/include)   # Qt5 include

# Link directories
include_directories(${CMAKE_SOURCE_DIR})
link_directories(${ONNX_DIR}/lib)
include_directories(${CMAKE_PREFIX_PATH})

# Add source files
set(SOURCES vidDisplay.cpp filters.cpp faceDetect.cpp)

# Add executable target
#add_executable(Project_1 test.cpp)
#add_executable(Project_1 imgDisplay.cpp)
#add_executable(Project_1 vidDisplay.cpp)
#add_executable(Project_1 test2.cpp)
add_executable(Project_1 ${SOURCES})
#add_executable(Project_1 da2-example.cpp DA2Network.hpp)
#add_executable(Project_1 filters.cpp timeBlur.cpp) # Use this for the time blur step

# Link OpenCV libraries
target_link_libraries(Project_1 ${OpenCV_LIBS})

# Link ONNX Runtime libraries
target_link_libraries(Project_1 ${ONNX_DIR}/lib/onnxruntime.lib)

# Link Qt6 libraries
target_link_libraries(Project_1 Qt6::Widgets)
