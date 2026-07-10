cmake_minimum_required(VERSION 3.16)
project(CS5330_Project4 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)

add_executable(project4
    src/main.cpp
    src/obj_loader.cpp
    src/gl_renderer.cpp
)

add_executable(harris
    src/harris_corner.cpp
)

target_include_directories(project4 PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    ${OpenCV_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
)

target_link_libraries(project4 PRIVATE
    ${OpenCV_LIBS}
    OpenGL::GL
    GLEW::GLEW
    glfw
)


target_include_directories(harris PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    ${OpenCV_INCLUDE_DIRS}
    # ${GLEW_INCLUDE_DIRS}
)

target_link_libraries(harris PRIVATE
    ${OpenCV_LIBS}
    # OpenGL::GL
    # GLEW::GLEW
    # glfw
)
