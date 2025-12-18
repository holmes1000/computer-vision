cmake_minimum_required(VERSION 3.10)

# Project details
project(Project_1 VERSION 0.1.0 LANGUAGES C CXX)

# Set OpenCV directory
set(OpenCV_DIR "C:/opencv/build")

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable target
add_executable(Homework_2 laplacian.cpp)

# Link OpenCV libraries
target_link_libraries(Homework_2 ${OpenCV_LIBS})
