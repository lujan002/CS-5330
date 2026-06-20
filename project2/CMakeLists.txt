cmake_minimum_required(VERSION 3.16)
project(CS5330_Project LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Adjust paths: ONNX Runtime next to your project, or use an absolute path
set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/../onnxruntime-linux-x64-1.19.2")
# set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/../onnxruntime-linux-x64-1.26.0")

find_package(OpenCV 4 REQUIRED)

add_library(csv_util STATIC src/csv_util/csv_util.cpp)

add_executable(dirToFeatureVec
    src/dirToFeatureVec.cpp
    "${CMAKE_SOURCE_DIR}/../project1/src/filter.cpp"
)
add_executable(imgMatch src/imgMatch.cpp)

target_include_directories(csv_util PRIVATE
    "${CMAKE_SOURCE_DIR}/src"
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(csv_util PRIVATE ${OpenCV_LIBS})

foreach(_target IN ITEMS dirToFeatureVec imgMatch)
    target_include_directories(${_target} PRIVATE
        "${CMAKE_SOURCE_DIR}/src"
        "${ONNXRUNTIME_ROOT}/include"
        "${CMAKE_SOURCE_DIR}/../project1/src"
        ${OpenCV_INCLUDE_DIRS}
    )
    target_link_libraries(${_target} PRIVATE
        ${OpenCV_LIBS}
        "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"
        csv_util
    )
endforeach()
