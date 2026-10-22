# The NDK toolchain must be passed to this project's top-level CMake entry.
if(NOT ANDROID_ABI STREQUAL "arm64-v8a")
    message(FATAL_ERROR "This Android profile requires arm64-v8a")
endif()
add_library(ann_platform INTERFACE)
# Boost 1.71 compatibility with NDK libc++ in C++17.
target_compile_definitions(ann_platform INTERFACE _LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION)
# Header-only Boost is shared by all native targets in this index.
set(ANN_BOOST_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/.deps/android" CACHE PATH "Directory containing boost/")
if(NOT EXISTS "${ANN_BOOST_INCLUDE_DIR}/boost/dynamic_bitset.hpp")
    message(FATAL_ERROR "Set ANN_BOOST_INCLUDE_DIR to the Boost headers directory")
endif()
target_include_directories(ann_platform SYSTEM INTERFACE "${ANN_BOOST_INCLUDE_DIR}")
add_library(ann_openmp INTERFACE)
target_compile_options(ann_openmp INTERFACE -fopenmp)
target_link_options(ann_openmp INTERFACE -fopenmp -static-openmp)
option(ANN_ANDROID_KNN_POOL "Pool small KNN allocations to avoid Android mapping exhaustion" ON)
