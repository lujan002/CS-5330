cmake_minimum_required(VERSION 3.16)
project(CS5330_Project3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4 REQUIRED)

add_executable(project3
    src/main.cpp
)

target_include_directories(project3 PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(project3 PRIVATE
    ${OpenCV_LIBS}
)
