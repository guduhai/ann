# Both platforms compile the same corrected graph sources and entry points.
add_executable(ann_knn vendor/efanna_graph/tests/test_nndescent.cpp
    vendor/efanna_graph/src/index.cpp vendor/efanna_graph/src/index_graph.cpp vendor/efanna_graph/src/index_random.cpp)
target_include_directories(ann_knn PRIVATE vendor/efanna_graph/include)
if(ANDROID AND ANN_ANDROID_KNN_POOL)
    target_sources(ann_knn PRIVATE src/knn_pooled_new_android.cpp)
endif()
add_executable(ann_build vendor/nsg/tests/test_nsg_index.cpp
    vendor/nsg/src/index.cpp vendor/nsg/src/index_nsg.cpp)
add_executable(ann_query_official vendor/nsg/tests/test_nsg_optimized_search.cpp
    vendor/nsg/src/index.cpp vendor/nsg/src/index_nsg.cpp)
foreach(target IN ITEMS ann_build ann_query_official)
    target_include_directories(${target} PRIVATE vendor/nsg/include)
endforeach()
foreach(target IN ITEMS ann_knn ann_build ann_query_official)
    # EFANNA uses random_shuffle, removed from Android libc++ C++17.
    if(ANDROID)
        set_target_properties(${target} PROPERTIES CXX_STANDARD 14)
    endif()
    target_compile_definitions(${target} PRIVATE INFO)
    target_compile_options(${target} PRIVATE -O3)
    target_link_libraries(${target} PRIVATE ann_platform ann_openmp)
endforeach()
foreach(target IN ITEMS ann_build ann_query_official)
    target_compile_options(${target} PRIVATE -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free)
endforeach()
# Keep existing Android binary names while standardizing the CMake targets.
foreach(pair IN ITEMS "ann_knn;nsg_knn" "ann_build;nsg_build" "ann_query_official;nsg_query_official")
    list(GET pair 0 original)
    list(GET pair 1 alias)
    add_custom_command(TARGET ${original} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:${original}>" "${CMAKE_BINARY_DIR}/bin/${alias}")
endforeach()
