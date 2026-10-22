# The NDK toolchain must be passed to this project's top-level CMake entry.
if(NOT ANDROID_ABI STREQUAL "arm64-v8a")
    message(FATAL_ERROR "This Android profile requires arm64-v8a")
endif()
add_library(ann_platform INTERFACE)
