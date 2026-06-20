cmake_minimum_required(VERSION 3.16)
project(makeHist LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV 4 REQUIRED)

add_executable(makeHist makeHist.cpp)
target_include_directories(makeHist PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(makeHist PRIVATE ${OpenCV_LIBS})
