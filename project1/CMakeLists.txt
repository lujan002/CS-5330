cmake_minimum_required(VERSION 3.16)
project(CS5330_Project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Adjust paths: ONNX Runtime next to your project, or use an absolute path
set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/../onnxruntime-linux-x64-1.19.2")
# set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/../onnxruntime-linux-x64-1.26.0")

find_package(OpenCV 4 REQUIRED)

add_executable(imgDisplay src/imgDisplay.cpp)
add_executable(vidDisplay src/vidDisplay.cpp)
add_executable(timeBlur src/timeblur/timeBlur.cpp)

# When filter.cpp has shared filter code (no main), use e.g.:
add_library(filters STATIC src/filter.cpp)
add_library(faceDetect STATIC src/faceDetect/faceDetect.cpp)

target_include_directories(filters PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(filters PRIVATE ${OpenCV_LIBS})
target_include_directories(faceDetect PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(faceDetect PRIVATE ${OpenCV_LIBS})

foreach(_target IN ITEMS imgDisplay vidDisplay timeBlur)
    target_include_directories(${_target} PRIVATE
        "${CMAKE_SOURCE_DIR}/src"
        "${ONNXRUNTIME_ROOT}/include"
        ${OpenCV_INCLUDE_DIRS}
    )
    target_link_libraries(${_target} PRIVATE
        ${OpenCV_LIBS}
        "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"
        filters
        faceDetect
    )
endforeach()
