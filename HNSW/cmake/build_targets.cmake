add_executable(ann_build src/build.cpp)
target_include_directories(ann_build SYSTEM PRIVATE vendor/hnswlib)
target_compile_options(ann_build PRIVATE -O3)
target_link_libraries(ann_build PRIVATE ann_platform)
