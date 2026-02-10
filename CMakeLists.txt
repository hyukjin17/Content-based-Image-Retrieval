cmake_minimum_required(VERSION 3.16)
project(Content-based-Image-Retrieval LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)

add_executable(read readfiles.cpp features.cpp csv_util.cpp faceDetect.cpp)

target_include_directories(read PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(read PRIVATE ${OpenCV_LIBS})

add_executable(cbir match_image.cpp features.cpp csv_util.cpp faceDetect.cpp)

target_include_directories(cbir PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(cbir PRIVATE ${OpenCV_LIBS})