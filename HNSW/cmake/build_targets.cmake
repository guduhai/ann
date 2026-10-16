# Keep the search executable name used by existing experiments, plus a common entry.
add_custom_command(TARGET hnsw_search POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:hnsw_search>" "${CMAKE_BINARY_DIR}/bin/ann_query")
add_executable(ann_build src/build.cpp)
target_include_directories(ann_build SYSTEM PRIVATE vendor/hnswlib)
target_compile_options(ann_build PRIVATE -O3 -pthread)
target_link_libraries(ann_build PRIVATE pthread)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_options(ann_build PRIVATE -mcpu=native)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_compile_options(ann_build PRIVATE -march=native)
endif()
