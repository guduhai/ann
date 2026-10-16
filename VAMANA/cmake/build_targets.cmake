# Keep the search executable name used by existing experiments, plus a common entry.
add_custom_command(TARGET vamana_search POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:vamana_search>" "${CMAKE_BINARY_DIR}/bin/ann_query")
find_package(OpenMP REQUIRED)
set(UPSTREAM_ROOT "${CMAKE_SOURCE_DIR}/vendor/diskann")
set(LOCAL_INCLUDE "${CMAKE_SOURCE_DIR}/vendor/deps/include")
set(LOCAL_LIB "${CMAKE_SOURCE_DIR}/vendor/deps/sysroot/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}")
find_library(ANN_AIO NAMES aio HINTS "${LOCAL_LIB}" REQUIRED)
find_library(ANN_LAPACKE NAMES lapacke HINTS "${LOCAL_LIB}" REQUIRED)
find_library(ANN_OPENBLAS NAMES openblas HINTS "${LOCAL_LIB}/openblas-pthread" "${LOCAL_LIB}" REQUIRED)
find_library(ANN_BOOST_OPTIONS NAMES boost_program_options HINTS "${LOCAL_LIB}" REQUIRED)
find_path(ANN_CBLAS cblas.h HINTS "${CMAKE_SOURCE_DIR}/vendor/deps/sysroot/usr/include" PATH_SUFFIXES ${CMAKE_LIBRARY_ARCHITECTURE}/openblas-pthread ${CMAKE_LIBRARY_ARCHITECTURE})
set(DISKANN_SOURCES
    abstract_data_store.cpp ann_exception.cpp disk_utils.cpp distance.cpp
    index.cpp in_mem_graph_store.cpp in_mem_data_store.cpp
    linux_aligned_file_reader.cpp math_utils.cpp natural_number_map.cpp
    natural_number_set.cpp memory_mapper.cpp partition.cpp pq.cpp
    pq_flash_index.cpp scratch.cpp logger.cpp utils.cpp filter_utils.cpp
    index_factory.cpp abstract_index.cpp pq_l2_distance.cpp pq_data_store.cpp)
list(TRANSFORM DISKANN_SOURCES PREPEND "${UPSTREAM_ROOT}/src/")
# The upstream byte FAST_L2 explicit instantiations do not compile on ARM64:
# their generic scalar fallback declares float pointers for T=byte. Generate
# a copy with those pointer declarations typed as T. For T=float this is
# identical; upstream graph and float L2 functions are otherwise untouched.
file(READ "${UPSTREAM_ROOT}/src/distance.cpp" DISTANCE_SOURCE_TEXT)
string(FIND "${DISTANCE_SOURCE_TEXT}" "const float *last = a + size;" LAST_POSITION)
string(FIND "${DISTANCE_SOURCE_TEXT}" "const float *unroll_group = last - 3;" GROUP_POSITION)
if(LAST_POSITION EQUAL -1 OR GROUP_POSITION EQUAL -1)
    message(FATAL_ERROR "Expected upstream FAST_L2 scalar fallback declarations missing")
endif()
string(REPLACE "const float *last = a + size;" "const T *last = a + size;"
    DISTANCE_SOURCE_TEXT "${DISTANCE_SOURCE_TEXT}")
string(REPLACE "const float *unroll_group = last - 3;" "const T *unroll_group = last - 3;"
    DISTANCE_SOURCE_TEXT "${DISTANCE_SOURCE_TEXT}")
set(ARM_DISTANCE_SOURCE "${CMAKE_CURRENT_BINARY_DIR}/distance-arm64.cpp")
file(WRITE "${ARM_DISTANCE_SOURCE}" "${DISTANCE_SOURCE_TEXT}")
list(REMOVE_ITEM DISKANN_SOURCES "${UPSTREAM_ROOT}/src/distance.cpp")
list(APPEND DISKANN_SOURCES "${ARM_DISTANCE_SOURCE}")


add_library(diskann_memory_arm64 STATIC ${DISKANN_SOURCES})
target_include_directories(diskann_memory_arm64 BEFORE PUBLIC
    "${CMAKE_SOURCE_DIR}/src/compat" "${LOCAL_INCLUDE}"
    "${CMAKE_SOURCE_DIR}/vendor/deps/sysroot/usr/include" "${UPSTREAM_ROOT}/include" "${ANN_CBLAS}")
target_compile_options(diskann_memory_arm64 PRIVATE -O3 -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_include_directories(diskann_memory_arm64 BEFORE PUBLIC "${UPSTREAM_ROOT}/arm64-compat")
    target_compile_options(diskann_memory_arm64 PUBLIC -include "${UPSTREAM_ROOT}/arm64-compat/immintrin.h")
    target_compile_definitions(diskann_memory_arm64 PRIVATE USE_AVX2)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_compile_options(diskann_memory_arm64 PUBLIC -march=native)
    target_compile_definitions(diskann_memory_arm64 PRIVATE USE_AVX2)
endif()
target_link_libraries(diskann_memory_arm64 PUBLIC OpenMP::OpenMP_CXX "${ANN_AIO}" "${ANN_LAPACKE}" "${ANN_OPENBLAS}" pthread m dl)
add_executable(ann_build "${UPSTREAM_ROOT}/apps/build_memory_index.cpp")
add_executable(ann_query_official "${UPSTREAM_ROOT}/apps/search_memory_index.cpp")
add_executable(fvecs_to_bin "${UPSTREAM_ROOT}/apps/utils/fvecs_to_bin.cpp")
foreach(target IN ITEMS ann_build ann_query_official fvecs_to_bin)
    target_link_options(${target} PRIVATE -Wl,--gc-sections -Wl,--disable-new-dtags)
    target_link_libraries(${target} PRIVATE diskann_memory_arm64 "${ANN_BOOST_OPTIONS}")
    set_target_properties(${target} PROPERTIES BUILD_RPATH "${LOCAL_LIB}")
endforeach()
