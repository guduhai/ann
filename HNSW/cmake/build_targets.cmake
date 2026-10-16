# Keep the search executable name used by existing experiments, plus a common entry.
add_custom_command(TARGET hnsw_search POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:hnsw_search>" "${CMAKE_BINARY_DIR}/bin/ann_query")
find_package(Python3 COMPONENTS Interpreter REQUIRED)
configure_file(cmake/ann_build.in "${CMAKE_BINARY_DIR}/bin/ann_build" @ONLY)
file(CHMOD "${CMAKE_BINARY_DIR}/bin/ann_build" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
add_custom_target(ann_build ALL DEPENDS "${CMAKE_BINARY_DIR}/bin/ann_build")
