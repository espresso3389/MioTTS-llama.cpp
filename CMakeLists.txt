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

# miotts executable
add_executable(miotts
    src/main.cpp
    src/miocodec.cpp
    src/istft.cpp
    src/text-normalize.cpp
    src/test-to-speech.cpp
    src/token-parser.cpp
    src/wav-writer.cpp
)

target_include_directories(miotts PRIVATE
    src
    third_party/llama.cpp/include
    third_party/llama.cpp/ggml/include
)

target_link_libraries(miotts PRIVATE
    llama
    ggml
    m
)

add_executable(miotts-stream-device
    examples/stream-to-device.cpp
    src/miocodec.cpp
    src/istft.cpp
    src/text-normalize.cpp
    src/test-to-speech.cpp
    src/token-parser.cpp
    src/wav-writer.cpp
)

target_include_directories(miotts-stream-device PRIVATE
    src
    third_party/llama.cpp/include
    third_party/llama.cpp/ggml/include
    third_party/llama.cpp/vendor/miniaudio
)

target_link_libraries(miotts-stream-device PRIVATE
    llama
    ggml
    m
)
