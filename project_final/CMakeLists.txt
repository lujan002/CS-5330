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

# ONNX Runtime for the card matcher. The prebuilt tree lives beside the course
# projects; point -DONNXRUNTIME_ROOT=... at another copy if yours is elsewhere.
set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/../onnxruntime-linux-x64-1.26.0"
    CACHE PATH "Root of a prebuilt ONNX Runtime release")

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS "${ONNXRUNTIME_ROOT}/include" PATH_SUFFIXES onnxruntime)
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
    HINTS "${ONNXRUNTIME_ROOT}/lib")

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR
        "ONNX Runtime not found under ${ONNXRUNTIME_ROOT}. Download a release from "
        "https://github.com/microsoft/onnxruntime/releases and pass "
        "-DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-X.Y.Z")
endif()
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_LIBRARY}")

add_executable(ar_card
    main.cpp
    src/obj_loader.cpp
    src/gl_renderer.cpp
    src/model_library.cpp
    src/model_download.cpp
    src/poke3d_download.cpp
    src/pokemon_heights.cpp
    src/card_matcher.cpp
    src/card_net_input.cpp
    src/orientation_classifier.cpp
    src/cardness_classifier.cpp
)

target_include_directories(ar_card PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    ${OpenCV_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${LIBZIP_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
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
    ${ONNXRUNTIME_LIBRARY}
)

# libonnxruntime.so is not on the default loader path, so bake its location in.
set_target_properties(ar_card PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
    INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib"
)

# Offline asset checker: renders every model and every material in isolation.
add_executable(validate_models
    tools/validate_models.cpp
    src/obj_loader.cpp
    src/gl_renderer.cpp
    src/model_library.cpp
    src/model_download.cpp
    src/poke3d_download.cpp
    src/pokemon_heights.cpp
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

# Chessboard calibration (same flow as project4). Writes camera_intrinsics.yaml
# next to the binary for ar_card to pick up.
add_executable(calibrate_camera
    tools/calibrate_camera.cpp
)

target_include_directories(calibrate_camera PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(calibrate_camera PRIVATE
    ${OpenCV_LIBS}
)

# Runs the matcher over a still image: parity check against Python, and a way to
# debug a photo without a camera attached.
add_executable(match_image
    tools/match_image.cpp
    src/card_matcher.cpp
    src/card_net_input.cpp
)

target_include_directories(match_image PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(match_image PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARY}
)

set_target_properties(match_image PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
    INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib"
)
