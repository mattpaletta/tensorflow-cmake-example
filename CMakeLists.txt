project(tensorflow-example)
cmake_minimum_required(VERSION 3.11)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include TF Library
add_subdirectory(tensorflow-cmake)


if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build, options are: Debug Release." FORCE)
endif()


if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    message("Memory and refcounting: ENABLED")
    option(ENABLE_REF_DEBUG "enable memory and refcounting debugging" ON)
else()
    message("Memory and refcounting: DISABLED")
    option(ENABLE_REF_DEBUG "enable memory and refcounting debugging" OFF)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES GNU)
    add_compile_options("-g")
endif()

SET( CMAKE_EXPORT_COMPILE_COMMANDS ON )
IF( EXISTS "${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json" )
    EXECUTE_PROCESS( COMMAND ${CMAKE_COMMAND} -E copy_if_different
         ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
         ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json)
ENDIF()

add_executable(tf_example tf_demo.cpp)
target_link_libraries(tf_example tensorflow)
