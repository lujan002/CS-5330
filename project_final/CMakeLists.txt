cmake_minimum_required(VERSION 3.16)
project(CS5330_ProjectFinal LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4 REQUIRED)

add_executable(ar_card
    data/models/main.cpp
)

target_include_directories(ar_card PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ar_card PRIVATE
    ${OpenCV_LIBS}
)
