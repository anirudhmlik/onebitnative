cmake_minimum_required(VERSION 3.22.1)
project(onebit_native C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Path to your existing BitNet fork (compile directly from there)
set(ONEBIT_BITNET_DIR "/Users/anirudhmalik/Desktop/onebit/onebit-android/bitnet")

add_library(onebit SHARED
  onebit_jni.cpp
)

if(${ANDROID_ABI} STREQUAL "arm64-v8a")
    target_compile_options(onebit PRIVATE -O3 -DNDEBUG -march=armv8.2-a+dotprod)
else()
    target_compile_options(onebit PRIVATE -O3 -DNDEBUG)
endif()

target_include_directories(onebit PRIVATE
  ${ONEBIT_BITNET_DIR}/3rdparty/llama.cpp/include
  ${ONEBIT_BITNET_DIR}/3rdparty/llama.cpp/ggml/include
)

# Build llama.cpp + ggml as part of this native build
set(BITNET_ARM_TL1 ON CACHE BOOL "Enable BitNet TL1 on ARM" FORCE)
add_subdirectory(${ONEBIT_BITNET_DIR} ${CMAKE_BINARY_DIR}/bitnet)

target_link_libraries(onebit
  llama
  ggml
  android
  log
)

