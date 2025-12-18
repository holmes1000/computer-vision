cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Project details
project(Project_2 VERSION 0.1.0 LANGUAGES C CXX)

# Set OpenCV directory
set(OpenCV_DIR "C:/opencv/build")

# Find OpenCV package
find_package(OpenCV REQUIRED)

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
#set(SOURCES feature_vector_writer.cpp csv_util.cpp)    # Feature writing
#set(SOURCES baseline_matcher.cpp csv_util.cpp)         # Task 1
#set(SOURCES histogram_matcher.cpp csv_util.cpp)          # Task 2 & 3
set(SOURCES texture.cpp csv_util.cpp histogram_matcher.cpp)         # Task 4
#set(SOURCES deep_network_embeddings.cpp csv_util.cpp)    # Task 5 & 6
#set(SOURCES custom.cpp csv_util.cpp faceDetect.cpp)     # Task 7 & extensions

# Add executable target
add_executable(Project_2 ${SOURCES})

# Link OpenCV libraries
target_link_libraries(Project_2 ${OpenCV_LIBS})
