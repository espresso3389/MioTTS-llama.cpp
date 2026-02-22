cmake_minimum_required(VERSION 3.14)
project(miotts LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

# llama.cpp options — disable things we don't need
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)

add_subdirectory(third_party/llama.cpp)

# Optional ONNX Runtime codec backend
option(MIOTTS_ONNX "Enable ONNX Runtime codec backend" OFF)

set(MIOTTS_ONNX_SOURCES "")
set(MIOTTS_ONNX_INCLUDE_DIRS "")
set(MIOTTS_ONNX_LIBRARIES "")
set(MIOTTS_ONNX_DEFINITIONS "")

if(MIOTTS_ONNX)
    # Allow user to specify ONNXRUNTIME_DIR for custom install location
    find_library(ONNXRUNTIME_LIB onnxruntime
        HINTS ${ONNXRUNTIME_DIR}/lib
        REQUIRED)
    find_path(ONNXRUNTIME_INCLUDE onnxruntime_cxx_api.h
        HINTS ${ONNXRUNTIME_DIR}/include
        PATH_SUFFIXES onnxruntime
        REQUIRED)

    message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIB}")
    message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE}")

    set(MIOTTS_ONNX_SOURCES src/codec-backend-onnx.cpp)
    set(MIOTTS_ONNX_INCLUDE_DIRS ${ONNXRUNTIME_INCLUDE})
    set(MIOTTS_ONNX_LIBRARIES ${ONNXRUNTIME_LIB})
    set(MIOTTS_ONNX_DEFINITIONS MIOTTS_HAS_ONNX)
endif()

# Common source files
set(MIOTTS_COMMON_SOURCES
    src/miocodec.cpp
    src/istft.cpp
    src/text-normalize.cpp
    src/test-to-speech.cpp
    src/token-parser.cpp
    src/wav-writer.cpp
    ${MIOTTS_ONNX_SOURCES}
)

# Helper function to configure targets with common settings
function(miotts_configure_target target)
    target_include_directories(${target} PRIVATE
        src
        third_party/llama.cpp/include
        third_party/llama.cpp/ggml/include
        ${MIOTTS_ONNX_INCLUDE_DIRS}
        ${ARGN}
    )
    target_link_libraries(${target} PRIVATE
        llama
        ggml
        m
        ${MIOTTS_ONNX_LIBRARIES}
    )
    if(MIOTTS_ONNX_DEFINITIONS)
        target_compile_definitions(${target} PRIVATE ${MIOTTS_ONNX_DEFINITIONS})
    endif()
endfunction()

# miotts executable
add_executable(miotts
    src/main.cpp
    ${MIOTTS_COMMON_SOURCES}
)
miotts_configure_target(miotts)

add_executable(miotts-stream-device
    examples/stream-to-device.cpp
    ${MIOTTS_COMMON_SOURCES}
)
miotts_configure_target(miotts-stream-device
    third_party/llama.cpp/vendor/miniaudio
)

add_executable(miotts-stream-benchmark
    examples/stream-benchmark.cpp
    ${MIOTTS_COMMON_SOURCES}
)
miotts_configure_target(miotts-stream-benchmark)

add_executable(miotts-stream-compare
    examples/stream-compare.cpp
    ${MIOTTS_COMMON_SOURCES}
)
miotts_configure_target(miotts-stream-compare)

add_executable(miotts-codec-benchmark
    examples/codec-benchmark.cpp
    ${MIOTTS_COMMON_SOURCES}
)
miotts_configure_target(miotts-codec-benchmark)
