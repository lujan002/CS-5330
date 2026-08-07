cmake_minimum_required(VERSION 3.16)
project(CS5330_ProjectFinal LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(X11 REQUIRED)
find_package(assimp REQUIRED)
find_package(CURL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBZIP REQUIRED libzip)

add_executable(ar_card
    main.cpp
    src/obj_loader.cpp
    src/gl_renderer.cpp
    src/model_library.cpp
    src/model_download.cpp
)

target_include_directories(ar_card PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    ${OpenCV_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${LIBZIP_INCLUDE_DIRS}
)

target_link_libraries(ar_card PRIVATE
    ${OpenCV_LIBS}
    OpenGL::GL
    GLEW::GLEW
    glfw
    X11::X11
    assimp::assimp
    CURL::libcurl
    ${LIBZIP_LIBRARIES}
)

# Offline asset checker: renders every model and every material in isolation.
add_executable(validate_models
    tools/validate_models.cpp
    src/obj_loader.cpp
    src/gl_renderer.cpp
    src/model_library.cpp
    src/model_download.cpp
)

target_include_directories(validate_models PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    ${OpenCV_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${LIBZIP_INCLUDE_DIRS}
)

target_link_libraries(validate_models PRIVATE
    ${OpenCV_LIBS}
    OpenGL::GL
    GLEW::GLEW
    glfw
    assimp::assimp
    CURL::libcurl
    ${LIBZIP_LIBRARIES}
)
