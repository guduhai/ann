add_executable(ann_knn src/build_knn.cpp
    vendor/efanna_graph/src/index.cpp vendor/efanna_graph/src/index_graph.cpp
    vendor/efanna_graph/src/index_random.cpp)
target_include_directories(ann_knn PRIVATE include vendor/efanna_graph/include)
if(ANDROID AND ANN_ANDROID_KNN_POOL)
    target_sources(ann_knn PRIVATE src/knn_pooled_new_android.cpp)
endif()
add_executable(ann_build src/build_graph.cpp
    vendor/nsg/src/index.cpp vendor/nsg/src/index_nsg.cpp)
target_include_directories(ann_build PRIVATE include vendor/nsg/include)
foreach(target ann_knn ann_build)
    target_compile_definitions(${target} PRIVATE INFO)
    target_compile_options(${target} PRIVATE -O3
        -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free)
    target_link_libraries(${target} PRIVATE ann_platform ann_openmp)
endforeach()
add_executable(ann_query src/query.cpp)
# Quantization stays in the shared sibling directory; no private copy.
target_include_directories(ann_query PRIVATE include "${PROJECT_SOURCE_DIR}/..")
target_compile_options(ann_query PRIVATE -O3)
target_link_libraries(ann_query PRIVATE ann_platform)
# Compatibility name: the same query executable, not a second implementation.
add_custom_command(TARGET ann_query POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:ann_query>"
        "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ann_query_phone")
