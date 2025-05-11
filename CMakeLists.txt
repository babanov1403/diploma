cmake_minimum_required(VERSION 3.12)
project(exec)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    ${CMAKE_SOURCE_DIR}/src/alglib  # For ALGLIB headers
    ${CMAKE_SOURCE_DIR}/src     # For your project headers
)

file(GLOB SRC_FILES
    "src/*.cpp"
)

file(GLOB ALGLIB_SRC_FILES
    "src/alglib/*.cpp"
)

set(ALL_SRC_FILES ${SRC_FILES} ${ALGLIB_SRC_FILES})

add_executable(${PROJECT_NAME} ${ALL_SRC_FILES})

# (Optional) Additional compiler flags
target_compile_options(${PROJECT_NAME} PRIVATE
    -std=c++23 -Wvolatile -g -march=native
)

# (Optional) For debugging
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${PROJECT_NAME} PRIVATE -g)
endif()