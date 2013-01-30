cmake_minimum_required(VERSION 2.8.3)

project(AdaptiveManifoldFilter)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIR})

file(GLOB hdrs "*.hpp" "*.h")
file(GLOB srcs "*.cpp")

add_executable(${PROJECT_NAME} ${hdrs} ${srcs})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
